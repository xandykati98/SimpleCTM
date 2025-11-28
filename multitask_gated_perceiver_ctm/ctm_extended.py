import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import argparse
import math
import os
import contextlib
import wave
import time

# Audio processing
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample

# Tabular data
from sklearn.datasets import load_wine

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch
import logging

# Suppress verbose logging from MLflow and Alembic
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

# Modal support for cloud GPU training
import modal

# Modal app configuration
modal_app = modal.App("gated-ctm-training")

# Modal image with required dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")  # Required for torchaudio
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "matplotlib",
        "numpy",
        "mlflow",
        "scikit-learn",
        "torchcodec"
    )
)


def _load_wav_with_stdlib(path: str) -> tuple[torch.Tensor, int]:
    """
    Minimal WAV loader using Python stdlib.

    This is used as a fallback when torchaudio has no compiled audio backends
    (common on unsupported Python versions), so that SpeechCommands can still
    be loaded.
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        frames = wf.readframes(num_frames)

    # Speech Commands uses 16‑bit PCM WAV; enforce that here
    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width ({8 * sample_width} bits) in file: {path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).T  # (C, T)
    else:
        audio = audio[None, :]  # (1, T)

    waveform = torch.from_numpy(audio)
    return waveform, sample_rate


def _maybe_patch_torchaudio_load() -> None:
    """
    If torchaudio reports no available audio backends, patch torchaudio.load
    to use a simple WAV loader so that datasets.SPEECHCOMMANDS works locally.
    """
    try:
        backends = torchaudio.list_audio_backends()
    except Exception:
        backends = []

    if len(backends) == 0:
        torchaudio.load = _load_wav_with_stdlib


_maybe_patch_torchaudio_load()


# Task configuration
ALL_TASKS = {
    # Vision tasks (image modality)
    'mnist_digit': {'output_size': 10, 'dataset': 'mnist', 'modality': 'image', 'description': 'MNIST digit classification (0-9)'},
    'mnist_even_odd': {'output_size': 2, 'dataset': 'mnist', 'modality': 'image', 'description': 'MNIST even/odd classification'},
    'cifar_fine': {'output_size': 10, 'dataset': 'cifar', 'modality': 'image', 'description': 'CIFAR-10 fine classification (10 classes)'},
    'cifar_coarse': {'output_size': 2, 'dataset': 'cifar', 'modality': 'image', 'description': 'CIFAR-10 coarse (animal vs vehicle)'},
    'fashion_fine': {'output_size': 10, 'dataset': 'fashion', 'modality': 'image', 'description': 'Fashion MNIST classification (10 classes)'},
    'emnist_fine': {'output_size': 47, 'dataset': 'emnist', 'modality': 'image', 'description': 'EMNIST balanced classification (47 classes)'},
    # Audio tasks (audio modality)
    'speech_commands': {'output_size': 35, 'dataset': 'speech_commands', 'modality': 'audio', 'description': 'Speech Commands (35 spoken words)'},
    # Tabular tasks (tabular modality)
    'wine_type': {'output_size': 3, 'dataset': 'wine', 'modality': 'tabular', 'description': 'Wine classification (3 types)'},
}


def compute_normalized_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized entropy from predictions.
    Normalized entropy = entropy / max_entropy, where max_entropy = log(num_classes)
    Returns value in [0, 1] where 0 = certain, 1 = maximally uncertain
    """
    probs = F.softmax(predictions, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    max_entropy = math.log(predictions.shape[-1])
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


class AdaptiveCNNEncoder(nn.Module):
    """
    Adaptive CNN encoder that handles variable input shapes.
    Uses adaptive pooling to produce fixed-size output regardless of input dimensions.
    Handles both 1-channel (MNIST, Fashion, EMNIST) and 3-channel (CIFAR) images.
    """
    def __init__(self, n_representation_size: int):
        super(AdaptiveCNNEncoder, self).__init__()
        
        self.n_representation_size = n_representation_size
        
        # First conv block - handles 1 or 3 channel input
        self.conv1_1ch = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_3ch = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Shared conv blocks
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Final projection with GLU
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.GLU(),
            nn.Linear(128, n_representation_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = self.conv1_1ch(x)
        else:
            x = self.conv1_3ch(x)
        
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class PerceiverEncoder(nn.Module):
    """
    Perceiver-style encoder that handles multiple modalities uniformly.
    
    Tokenizes any input (image, audio spectrogram, tabular) into a sequence,
    then uses cross-attention from learnable latents to compress into fixed-size output.
    
    Modalities:
    - image: (B, C, H, W) → pixels + Fourier positional encoding
    - audio: (B, 1, freq, time) mel spectrogram → time-freq bins + positional encoding  
    - tabular: (B, features) → each feature as token + learnable positional encoding
    """
    
    def __init__(
        self,
        output_dim: int,
        num_latents: int,
        latent_dim: int,
        num_fourier_features: int,
        num_heads: int,
        num_cross_attn_layers: int,
        dropout: float,
    ):
        super(PerceiverEncoder, self).__init__()
        
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_fourier_features = num_fourier_features
        
        # Learnable latent array (queries for cross-attention)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.latent_pos = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        # Fourier feature matrices for 2D positional encoding (image/audio)
        self.register_buffer('fourier_matrix_2d', torch.randn(2, num_fourier_features))
        # Fourier feature matrix for 1D positional encoding (tabular)
        self.register_buffer('fourier_matrix_1d', torch.randn(1, num_fourier_features))
        
        # Token dimensions: value + Fourier features
        # Image: RGB (1 or 3) + 2*num_fourier_features
        # Audio: magnitude (1) + 2*num_fourier_features
        # Tabular: value (1) + 2*num_fourier_features
        
        # Modality-specific input projections to latent_dim
        self.image_proj_1ch = nn.Linear(1 + 2 * num_fourier_features, latent_dim)
        self.image_proj_3ch = nn.Linear(3 + 2 * num_fourier_features, latent_dim)
        self.audio_proj = nn.Linear(1 + 2 * num_fourier_features, latent_dim)
        self.tabular_proj = nn.Linear(1 + 2 * num_fourier_features, latent_dim)
        
        # Cross-attention layers (latents attend to input tokens)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention block for latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        
        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.mlp_norm = nn.LayerNorm(latent_dim)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
    
    def create_fourier_encoding_2d(
        self,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create 2D Fourier positional encoding for image/audio grids."""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Normalize to [0, 1]
        x_coords = x_coords.float() / max(width - 1, 1)
        y_coords = y_coords.float() / max(height - 1, 1)
        
        # Stack: (H, W, 2)
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        # Flatten and expand: (B, H*W, 2)
        coords = coords.view(-1, 2).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Fourier projection: (B, H*W, num_fourier_features)
        fourier_proj = torch.matmul(coords, self.fourier_matrix_2d)
        
        # Sin and cos features: (B, H*W, 2*num_fourier_features)
        fourier_features = torch.cat([
            torch.cos(2 * math.pi * fourier_proj),
            torch.sin(2 * math.pi * fourier_proj)
        ], dim=-1)
        
        return fourier_features
    
    def create_fourier_encoding_1d(
        self,
        num_features: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create 1D Fourier positional encoding for tabular data."""
        # Feature indices normalized to [0, 1]
        indices = torch.arange(num_features, device=device).float()
        indices = indices / max(num_features - 1, 1)
        
        # Expand: (B, num_features, 1)
        indices = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        
        # Fourier projection: (B, num_features, num_fourier_features)
        fourier_proj = torch.matmul(indices, self.fourier_matrix_1d)
        
        # Sin and cos features: (B, num_features, 2*num_fourier_features)
        fourier_features = torch.cat([
            torch.cos(2 * math.pi * fourier_proj),
            torch.sin(2 * math.pi * fourier_proj)
        ], dim=-1)
        
        return fourier_features
    
    def tokenize_image(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize image: (B, C, H, W) → (B, H*W, latent_dim)"""
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Reshape to tokens: (B, C, H, W) → (B, H*W, C)
        value_tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channels)
        
        # Get positional encoding: (B, H*W, 2*num_fourier_features)
        pos_encoding = self.create_fourier_encoding_2d(height, width, batch_size, device)
        
        # Concatenate value + position: (B, H*W, C + 2*num_fourier_features)
        tokens = torch.cat([value_tokens, pos_encoding], dim=-1)
        
        # Project to latent dim
        if channels == 1:
            tokens = self.image_proj_1ch(tokens)
        else:
            tokens = self.image_proj_3ch(tokens)
        
        return tokens
    
    def tokenize_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize audio spectrogram: (B, 1, freq, time) → (B, freq*time, latent_dim)"""
        batch_size, channels, freq, time = x.shape
        device = x.device
        
        # Reshape to tokens: (B, 1, freq, time) → (B, freq*time, 1)
        value_tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        
        # Get positional encoding: (B, freq*time, 2*num_fourier_features)
        pos_encoding = self.create_fourier_encoding_2d(freq, time, batch_size, device)
        
        # Concatenate: (B, freq*time, 1 + 2*num_fourier_features)
        tokens = torch.cat([value_tokens, pos_encoding], dim=-1)
        
        # Project to latent dim
        tokens = self.audio_proj(tokens)
        
        return tokens
    
    def tokenize_tabular(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize tabular data: (B, features) → (B, features, latent_dim)"""
        batch_size, num_features = x.shape
        device = x.device
        
        # Each feature becomes a token: (B, features) → (B, features, 1)
        value_tokens = x.unsqueeze(-1)
        
        # Get positional encoding: (B, features, 2*num_fourier_features)
        pos_encoding = self.create_fourier_encoding_1d(num_features, batch_size, device)
        
        # Concatenate: (B, features, 1 + 2*num_fourier_features)
        tokens = torch.cat([value_tokens, pos_encoding], dim=-1)
        
        # Project to latent dim
        tokens = self.tabular_proj(tokens)
        
        return tokens
    
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Encode input of any modality to fixed-size representation.
        
        Args:
            x: Input tensor (shape depends on modality)
            modality: 'image', 'audio', or 'tabular'
        
        Returns:
            (B, output_dim) encoded representation
        """
        batch_size = x.shape[0]
        
        # Tokenize based on modality
        if modality == 'image':
            tokens = self.tokenize_image(x)
        elif modality == 'audio':
            tokens = self.tokenize_audio(x)
        elif modality == 'tabular':
            tokens = self.tokenize_tabular(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Initialize latents with positional encoding
        latents = self.latents + self.latent_pos  # (num_latents, latent_dim)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_latents, latent_dim)
        
        # Cross-attention: latents attend to input tokens
        for cross_attn, norm in zip(self.cross_attn_layers, self.cross_attn_norms):
            attn_out, _ = cross_attn(latents, tokens, tokens)
            latents = norm(latents + attn_out)
        
        # Self-attention on latents
        self_attn_out, _ = self.self_attn(latents, latents, latents)
        latents = self.self_attn_norm(latents + self_attn_out)
        
        # MLP
        mlp_out = self.mlp(latents)
        latents = self.mlp_norm(latents + mlp_out)
        
        # Global average pooling over latents
        pooled = latents.mean(dim=1)  # (B, latent_dim)
        
        # Project to output dimension
        output = self.output_proj(pooled)  # (B, output_dim)
        
        return output


class BatchedNLM(nn.Module):
    """
    Batched Neuron-Level Model with per-neuron private parameters (paper-faithful).
    
    Each neuron has its own unique weight parameters as specified in CTM paper
    Section 3.3 "Privately-Parameterized Neuron-Level Models". Uses einsum for
    efficient batched matrix multiplication across all neurons in parallel.
    
    Architecture per neuron: Linear(memory, 256) -> GLU -> Linear(128, 1)
    
    Reference: https://arxiv.org/abs/2505.05522
    """
    
    def __init__(self, n_neurons: int, max_memory: int):
        super(BatchedNLM, self).__init__()
        self.n_neurons = n_neurons
        self.max_memory = max_memory
        
        # Per-neuron weights for fc1: each neuron has its own (max_memory -> 256) projection
        # Shape: (n_neurons, max_memory, 256) - 256 because GLU halves it to 128
        self.fc1_weight = nn.Parameter(
            torch.randn(n_neurons, max_memory, 256) * math.sqrt(2.0 / max_memory)
        )
        self.fc1_bias = nn.Parameter(torch.zeros(n_neurons, 256))
        
        # Per-neuron weights for fc2: each neuron has its own (128 -> 1) projection
        # Shape: (n_neurons, 128, 1)
        self.fc2_weight = nn.Parameter(
            torch.randn(n_neurons, 128, 1) * math.sqrt(2.0 / 128)
        )
        self.fc2_bias = nn.Parameter(torch.zeros(n_neurons, 1))
    
    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        """
        Process all neuron traces in parallel with per-neuron private weights.
        
        Args:
            state_trace: (batch, neurons, memory) tensor of neuron activation histories
        
        Returns:
            (batch, neurons) post-activations for all neurons
        """
        # fc1: (B, N, M) @ (N, M, 256) -> (B, N, 256)
        # einsum performs batched matmul where each neuron uses its own weights
        x = torch.einsum('bnm,nmh->bnh', state_trace, self.fc1_weight) + self.fc1_bias
        
        # GLU activation: splits last dim in half, applies sigmoid gate
        # (B, N, 256) -> (B, N, 128)
        x = F.glu(x, dim=-1)
        
        # fc2: (B, N, 128) @ (N, 128, 1) -> (B, N, 1)
        x = torch.einsum('bnh,nho->bno', x, self.fc2_weight) + self.fc2_bias
        
        # Squeeze output dimension: (B, N, 1) -> (B, N)
        return x.squeeze(-1)


class MultitaskGatedCTM(nn.Module):
    """
    Multitask Gated Synchronization Continuous Thought Machine.
    
    Combines:
    - Gated synchronization (multiple sync pair sets with learned routing)
    - Multitask support (different output heads per task)
    - Adaptive CNN encoder for variable input sizes
    - Load balancing loss to prevent mode collapse
    
    The key innovation: instead of hardcoding which sync pairs to use per task,
    a gating network dynamically learns which sync sets are appropriate based
    on the neural state. This allows emergent task discovery and routing.
    
    Based on: https://arxiv.org/abs/2505.05522
    Reference: https://github.com/SakanaAI/continuous-thought-machines
    """
    
    def __init__(
        self, 
        n_neurons: int, 
        max_memory: int,
        max_ticks: int,
        n_representation_size: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        task_output_sizes: list[int],
        n_sync_sets: int,
        load_balance_coef: float,
        dropout_encoder: float,
        dropout_attention: float,
        dropout_synapse: float,
        dropout_nlm: float,
        dropout_sync: float,
        dropout_output: float,
    ):
        super(MultitaskGatedCTM, self).__init__()

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.n_attention_heads = n_attention_heads
        self.n_sync_sets = n_sync_sets
        self.load_balance_coef = load_balance_coef
        self.n_tasks = len(task_output_sizes)
        self.task_output_sizes = task_output_sizes
        
        # Dropout layers for different stages
        self.dropout_encoder = nn.Dropout(dropout_encoder)
        self.dropout_synapse = nn.Dropout(dropout_synapse)
        self.dropout_nlm = nn.Dropout(dropout_nlm)
        self.dropout_sync = nn.Dropout(dropout_sync)
        self._dropout_attention_rate = dropout_attention
        self._dropout_output_rate = dropout_output
        
        # Learnable initial states (Section 3.1 - start states)
        self.register_parameter(
            'start_activated_state', 
            nn.Parameter(torch.zeros(n_neurons).uniform_(
                -math.sqrt(1/n_neurons), 
                math.sqrt(1/n_neurons)
            ))
        )
        self.register_parameter(
            'start_trace', 
            nn.Parameter(torch.zeros(n_neurons, max_memory).uniform_(
                -math.sqrt(1/(n_neurons + max_memory)), 
                math.sqrt(1/(n_neurons + max_memory))
            ))
        )
        
        # Perceiver encoder - handles all modalities uniformly
        self.perceiver_encoder = PerceiverEncoder(
            output_dim=n_representation_size,
            num_latents=32,
            latent_dim=64,
            num_fourier_features=32,
            num_heads=4,
            num_cross_attn_layers=2,
            dropout=dropout_encoder,
        )
        
        # Key-Value projection for attention (from encoded input)
        self.kv_proj = nn.Sequential(
            nn.Linear(n_representation_size, n_representation_size),
            nn.LayerNorm(n_representation_size)
        )
        
        # Project encoded input to initial neural state (input-conditioned initialization)
        self.input_to_state = nn.Linear(n_representation_size, n_neurons)
        
        # Query projection from synchronization_action (gated)
        self.q_proj = nn.Linear(n_synch_action, n_representation_size)
        
        # Multi-head attention (Section 3.4) - SHARED
        self.attention = nn.MultiheadAttention(
            embed_dim=n_representation_size,
            num_heads=n_attention_heads,
            dropout=dropout_attention,
            batch_first=True
        )
        
        # Synapse model - combines attention output with current activated state - SHARED
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),
            nn.LayerNorm(n_neurons),
        )

        # Batched Neuron-level model (NLM) - processes all neurons in parallel
        self.batched_nlm = BatchedNLM(n_neurons=n_neurons, max_memory=max_memory)

        # Initialize multiple sets of synchronization pairs - SHARED gating
        self._init_synchronization_pair_sets()
        
        # Per-set learnable decay parameters for synchronization (Appendix H)
        self.set_decay_params_action = nn.ParameterList([
            nn.Parameter(torch.zeros(n_synch_action), requires_grad=True)
            for _ in range(n_sync_sets)
        ])
        self.set_decay_params_out = nn.ParameterList([
            nn.Parameter(torch.zeros(n_synch_out), requires_grad=True)
            for _ in range(n_sync_sets)
        ])
        
        # Gating network for action synchronization - SHARED
        self.gate_network_action = nn.Sequential(
            nn.Linear(n_neurons, 64),
            nn.ReLU(),
            nn.Linear(64, n_sync_sets),
        )
        
        # Gating network for output synchronization - SHARED
        self.gate_network_out = nn.Sequential(
            nn.Linear(n_neurons, 64),
            nn.ReLU(),
            nn.Linear(64, n_sync_sets),
        )

        # Task-specific output projectors (sync_out -> predictions)
        self.task_output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_synch_out, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_output),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout_output),
                nn.Linear(64, output_size)
            )
            for output_size in task_output_sizes
        ])
    
    def _init_synchronization_pair_sets(self):
        """
        Initialize multiple sets of neuron pairs for synchronization.
        Each set has its own random pairing, allowing the gating network
        to select different "views" of the neural synchronization.
        """
        for set_idx in range(self.n_sync_sets):
            # Output synchronization pairs for this set
            out_indices_left = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
            )
            out_indices_right = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
            )
            self.register_buffer(f'set_{set_idx}_out_left', out_indices_left)
            self.register_buffer(f'set_{set_idx}_out_right', out_indices_right)
            
            # Action synchronization pairs for this set
            action_indices_left = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
            )
            action_indices_right = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
            )
            self.register_buffer(f'set_{set_idx}_action_left', action_indices_left)
            self.register_buffer(f'set_{set_idx}_action_right', action_indices_right)
    
    def compute_gated_synchronization(
        self,
        activated_state: torch.Tensor,
        decay_alphas: list[torch.Tensor | None],
        decay_betas: list[torch.Tensor | None],
        synch_type: str,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """
        Compute synchronization using gated combination of multiple pair sets.
        
        The gating network observes the current neural state and decides
        which synchronization pair sets to use (soft weighted combination).
        
        Returns: (weighted_sync, updated_alphas, updated_betas, gate_weights)
        """
        batch_size = activated_state.shape[0]
        
        # Compute gate weights based on current neural state
        if synch_type == 'action':
            gate_logits = self.gate_network_action(activated_state)
        else:
            gate_logits = self.gate_network_out(activated_state)
        
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, n_sync_sets)
        
        # Compute sync for each set
        all_syncs = []
        new_alphas = []
        new_betas = []
        
        for set_idx in range(self.n_sync_sets):
            # Get indices for this set
            left = getattr(self, f'set_{set_idx}_{synch_type}_left')
            right = getattr(self, f'set_{set_idx}_{synch_type}_right')
            
            # Get decay params for this set
            if synch_type == 'action':
                decay_params = self.set_decay_params_action[set_idx]
            else:
                decay_params = self.set_decay_params_out[set_idx]
            
            # Clamp and compute r
            decay_params_clamped = torch.clamp(decay_params, 0, 15)
            r = torch.exp(-decay_params_clamped).unsqueeze(0).expand(batch_size, -1)
            
            # Compute pairwise product
            pairwise_product = activated_state[:, left] * activated_state[:, right]
            
            # Recursive update
            if decay_alphas[set_idx] is None or decay_betas[set_idx] is None:
                alpha = pairwise_product
                beta = torch.ones_like(pairwise_product)
            else:
                alpha = r * decay_alphas[set_idx] + pairwise_product
                beta = r * decay_betas[set_idx] + 1
            
            sync = alpha / torch.sqrt(beta + 1e-8)
            
            all_syncs.append(sync)
            new_alphas.append(alpha)
            new_betas.append(beta)
        
        # Stack and weight by gate: (batch, n_sets, n_synch)
        stacked_syncs = torch.stack(all_syncs, dim=1)
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, n_sets, 1)
        weighted_sync = (stacked_syncs * gate_weights_expanded).sum(dim=1)  # (batch, n_synch)
        
        return weighted_sync, new_alphas, new_betas, gate_weights

    def compute_load_balance_loss(
        self, 
        gate_weights_action: torch.Tensor, 
        gate_weights_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss to encourage balanced use of all sync sets.
        Penalizes when one set dominates (prevents mode collapse).
        
        Uses the coefficient of variation (CV) as the balance metric.
        """
        # Mean usage across batch for action gates
        mean_usage_action = gate_weights_action.mean(dim=0)
        cv_action = mean_usage_action.std() / (mean_usage_action.mean() + 1e-8)
        
        # Mean usage across batch for output gates
        mean_usage_out = gate_weights_out.mean(dim=0)
        cv_out = mean_usage_out.std() / (mean_usage_out.mean() + 1e-8)
        
        # Combined balance loss
        balance_loss = (cv_action + cv_out) / 2.0
        
        return balance_loss
    
    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        """Compute certainty as 1 - normalized_entropy."""
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def forward(self, x: torch.Tensor, task_idx: int, modality: str, track: bool = False):
        """
        Forward pass through the Multitask Gated CTM for a specific task.
        
        Args:
            x: Input tensor (shape depends on modality)
            task_idx: Index of the task in task_output_sizes
            modality: 'image', 'audio', or 'tabular'
            track: Whether to track activations for visualization
        
        Returns predictions, certainties, and load_balance_loss for ALL ticks.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode using Perceiver (handles all modalities)
        encoded_input = self.perceiver_encoder(x, modality=modality)
        
        # Prepare key-value features for attention
        kv = self.kv_proj(encoded_input).unsqueeze(1)
        
        # Initialize recurrent state from learnable parameters
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Input-conditioned initialization: each sample starts from unique neural state
        base_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        input_contribution = self.input_to_state(encoded_input)  # (B, n_neurons)
        activated_state = base_state + input_contribution  # Now each sample starts differently!
        
        # Prepare storage for outputs per iteration
        out_dims = self.task_output_sizes[task_idx]
        predictions = torch.empty(batch_size, out_dims, self.max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.max_ticks, device=device, dtype=torch.float32)
        
        # Initialize synchronization recurrence values for each set
        decay_alphas_action: list[torch.Tensor | None] = [None] * self.n_sync_sets
        decay_betas_action: list[torch.Tensor | None] = [None] * self.n_sync_sets
        decay_alphas_out: list[torch.Tensor | None] = [None] * self.n_sync_sets
        decay_betas_out: list[torch.Tensor | None] = [None] * self.n_sync_sets
        
        # Accumulate gate weights for load balance loss
        all_gate_weights_action = []
        all_gate_weights_out = []
        
        # Initialize output sync at tick 0
        _, decay_alphas_out, decay_betas_out, gate_weights_out = self.compute_gated_synchronization(
            activated_state, decay_alphas_out, decay_betas_out, synch_type='out'
        )
        all_gate_weights_out.append(gate_weights_out)
        
        # Tracking for visualization
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            attention_tracking = []
            gate_action_tracking = []
            gate_out_tracking = []
        
        # Recurrent loop over internal ticks
        for tick in range(self.max_ticks):
            # Calculate gated synchronization for action (attention modulation)
            sync_action, decay_alphas_action, decay_betas_action, gate_weights_action = self.compute_gated_synchronization(
                activated_state, decay_alphas_action, decay_betas_action, synch_type='action'
            )
            sync_action = self.dropout_sync(sync_action)
            all_gate_weights_action.append(gate_weights_action)
            
            # Create query from action synchronization
            q = self.q_proj(sync_action).unsqueeze(1)
            
            # Multi-head attention: sync_action queries the encoded input
            attn_out, attn_weights = self.attention(q, kv, kv, need_weights=True)
            attn_out = attn_out.squeeze(1)
            
            # Combine attention output with current activated state for synapse model
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            
            # Apply synapse model to get new pre-activations
            new_state = self.synapse_model(pre_synapse_input)
            new_state = self.dropout_synapse(new_state)
            
            # Update state trace (FIFO buffer of pre-activations)
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply batched neuron-level model to get post-activations (all neurons in parallel)
            activated_state = self.batched_nlm(state_trace)
            activated_state = self.dropout_nlm(activated_state)
            
            # Calculate gated synchronization for output predictions
            sync_out, decay_alphas_out, decay_betas_out, gate_weights_out = self.compute_gated_synchronization(
                activated_state, decay_alphas_out, decay_betas_out, synch_type='out'
            )
            sync_out = self.dropout_sync(sync_out)
            all_gate_weights_out.append(gate_weights_out)
            
            # Get predictions using task-specific output projector
            current_prediction = self.task_output_projectors[task_idx](sync_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            # Store predictions and certainties for this tick
            predictions[..., tick] = current_prediction
            certainties[..., tick] = current_certainty
            
            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(sync_out.detach().cpu().numpy())
                synch_action_tracking.append(sync_action.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                gate_action_tracking.append(gate_weights_action.detach().cpu().numpy())
                gate_out_tracking.append(gate_weights_out.detach().cpu().numpy())
        
        # Compute load balance loss across all ticks
        stacked_gate_action = torch.stack(all_gate_weights_action, dim=0).mean(dim=0)
        stacked_gate_out = torch.stack(all_gate_weights_out, dim=0).mean(dim=0)
        load_balance_loss = self.compute_load_balance_loss(stacked_gate_action, stacked_gate_out)
        
        if track:
            return (
                predictions, 
                certainties,
                load_balance_loss,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
                (np.array(gate_action_tracking), np.array(gate_out_tracking)),
            )
        
        return predictions, certainties, load_balance_loss


class PerceiverIODecoder(nn.Module):
    """
    Perceiver IO-style decoder for flexible multimodal/multitask output.
    
    Instead of fixed output heads per task, uses learnable query arrays that
    cross-attend to the latent representation (sync_out) to produce outputs.
    
    Key insight from Perceiver IO (https://arxiv.org/abs/2107.14795):
    - Output queries "ask" the latent space for specific outputs
    - Task embeddings tell which task/modality to decode
    - Position embeddings handle structured outputs (e.g., sequence positions)
    
    This enables:
    - Shared learning across tasks via cross-attention
    - Dynamic output sizes without architecture changes
    - True multimodal output capability
    """
    
    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        num_heads: int,
        num_tasks: int,
        max_output_tokens: int,
        dropout: float,
    ):
        super(PerceiverIODecoder, self).__init__()
        
        self.query_dim = query_dim
        self.num_tasks = num_tasks
        self.max_output_tokens = max_output_tokens
        
        # Project latent to key/value space for cross-attention
        self.latent_to_kv = nn.Linear(latent_dim, query_dim)
        
        # Learnable task embeddings - identify which task to decode
        self.task_embeddings = nn.Embedding(num_tasks, query_dim)
        
        # Learnable position embeddings for structured outputs
        self.output_pos_embeddings = nn.Parameter(
            torch.randn(max_output_tokens, query_dim) * 0.02
        )
        
        # Cross-attention: output queries attend to latent representation
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(query_dim)
        
        # Feed-forward network after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        
        # Final projection to single logit per output token
        self.output_proj = nn.Linear(query_dim, 1)
    
    def forward(
        self, 
        latent: torch.Tensor,
        task_idx: int,
        num_outputs: int,
    ) -> torch.Tensor:
        """
        Decode outputs by querying the latent representation.
        
        Args:
            latent: CTM's internal state (B, latent_dim) - typically sync_out
            task_idx: Which task to decode for
            num_outputs: Number of output tokens (e.g., num_classes)
        
        Returns:
            (B, num_outputs) - decoded class logits
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Project latent to key/value space and expand to sequence
        kv = self.latent_to_kv(latent).unsqueeze(1)  # (B, 1, query_dim)
        
        # Build output queries: task embedding + position embeddings
        task_emb = self.task_embeddings(
            torch.tensor(task_idx, device=device)
        )  # (query_dim,)
        
        # Select position embeddings for requested number of outputs
        pos_embs = self.output_pos_embeddings[:num_outputs]  # (num_outputs, query_dim)
        
        # Combine: broadcast task embedding + position embeddings
        queries = task_emb.unsqueeze(0) + pos_embs  # (num_outputs, query_dim)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_outputs, query_dim)
        
        # Cross-attention: queries attend to latent
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = self.cross_attn_norm(queries + attn_out)  # Residual
        
        # Feed-forward
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)  # Residual
        
        # Project to single logit per output token
        output = self.output_proj(queries)  # (B, num_outputs, 1)
        output = output.squeeze(-1)  # (B, num_outputs)
        
        return output


class MultitaskPerceiverCTM(nn.Module):
    """
    Multitask CTM with Perceiver encoder (multimodal input) and Perceiver IO decoder (flexible output).
    
    This version removes gating entirely and uses:
    - Single set of synchronization pairs (like original CTM paper)
    - Perceiver encoder for unified multimodal input handling
    - Perceiver IO-style decoder for query-based output generation
    
    Architecture:
    1. Input (any modality) → PerceiverEncoder → uniform representation
    2. CTM recurrent loop with sync-modulated attention
    3. sync_out → PerceiverIODecoder(task_queries) → task-specific outputs
    
    Based on:
    - CTM: https://arxiv.org/abs/2505.05522
    - Perceiver: https://arxiv.org/abs/2103.03206
    - Perceiver IO: https://arxiv.org/abs/2107.14795
    """
    
    def __init__(
        self, 
        n_neurons: int, 
        max_memory: int,
        max_ticks: int,
        n_representation_size: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        task_output_sizes: list[int],
        dropout_encoder: float,
        dropout_attention: float,
        dropout_synapse: float,
        dropout_nlm: float,
        dropout_sync: float,
        dropout_output: float,
    ):
        super(MultitaskPerceiverCTM, self).__init__()

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.n_attention_heads = n_attention_heads
        self.n_tasks = len(task_output_sizes)
        self.task_output_sizes = task_output_sizes
        
        # Dropout layers
        self.dropout_encoder = nn.Dropout(dropout_encoder)
        self.dropout_synapse = nn.Dropout(dropout_synapse)
        self.dropout_nlm = nn.Dropout(dropout_nlm)
        self.dropout_sync = nn.Dropout(dropout_sync)
        
        # Learnable initial states (Section 3.1 - start states)
        self.register_parameter(
            'start_activated_state', 
            nn.Parameter(torch.zeros(n_neurons).uniform_(
                -math.sqrt(1/n_neurons), 
                math.sqrt(1/n_neurons)
            ))
        )
        self.register_parameter(
            'start_trace', 
            nn.Parameter(torch.zeros(n_neurons, max_memory).uniform_(
                -math.sqrt(1/(n_neurons + max_memory)), 
                math.sqrt(1/(n_neurons + max_memory))
            ))
        )
        
        # Perceiver encoder - handles all modalities uniformly
        self.perceiver_encoder = PerceiverEncoder(
            output_dim=n_representation_size,
            num_latents=32,
            latent_dim=64,
            num_fourier_features=32,
            num_heads=4,
            num_cross_attn_layers=2,
            dropout=dropout_encoder,
        )
        
        # Key-Value projection for attention (from encoded input)
        self.kv_proj = nn.Sequential(
            nn.Linear(n_representation_size, n_representation_size),
            nn.LayerNorm(n_representation_size)
        )
        
        # Project encoded input to initial neural state (input-conditioned initialization)
        self.input_to_state = nn.Linear(n_representation_size, n_neurons)
        # Initialize with larger weights for stronger input conditioning at early training
        nn.init.xavier_uniform_(self.input_to_state.weight, gain=2.0)
        nn.init.zeros_(self.input_to_state.bias)
        # Learnable scale to ensure input has meaningful impact on initial state
        self.input_scale = nn.Parameter(torch.tensor(2.0))
        
        # Query projection from synchronization_action
        self.q_proj = nn.Linear(n_synch_action, n_representation_size)
        
        # Multi-head attention (Section 3.4)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_representation_size,
            num_heads=n_attention_heads,
            dropout=dropout_attention,
            batch_first=True
        )
        
        # Synapse model - combines attention output with current activated state
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),
            nn.LayerNorm(n_neurons),
        )
        
        # Normalization for state before synapse model to prevent saturation
        self.state_norm = nn.LayerNorm(n_neurons)

        # Batched Neuron-level model (NLM) - processes all neurons in parallel
        self.batched_nlm = BatchedNLM(n_neurons=n_neurons, max_memory=max_memory)

        # Single set of synchronization pairs (no gating)
        self._init_synchronization_pairs()
        
        # Learnable decay parameters for synchronization (Appendix H)
        self.decay_params_action = nn.Parameter(
            torch.zeros(n_synch_action), requires_grad=True
        )
        self.decay_params_out = nn.Parameter(
            torch.zeros(n_synch_out), requires_grad=True
        )

        # Task-specific output projectors (sync_out -> predictions)
        # Same architecture as MultitaskGatedCTM for fair comparison
        self.task_output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_synch_out, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_output),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout_output),
                nn.Linear(64, output_size)
            )
            for output_size in task_output_sizes
        ])
    
    def _init_synchronization_pairs(self):
        """
        Initialize single set of neuron pairs for synchronization.
        Each pair computes product of two neuron activations with temporal decay.
        """
        # Output synchronization pairs
        out_indices_left = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
        )
        out_indices_right = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
        )
        self.register_buffer('out_left', out_indices_left)
        self.register_buffer('out_right', out_indices_right)
        
        # Action synchronization pairs
        action_indices_left = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
        )
        action_indices_right = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
        )
        self.register_buffer('action_left', action_indices_left)
        self.register_buffer('action_right', action_indices_right)
    
    def compute_synchronization(
        self,
        activated_state: torch.Tensor,
        decay_alpha: torch.Tensor | None,
        decay_beta: torch.Tensor | None,
        synch_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute synchronization using single pair set (original CTM approach).
        
        Implements recursive sync computation from CTM paper Appendix H:
        - alpha_t = exp(-r) * alpha_{t-1} + z_i * z_j
        - beta_t = exp(-r) * beta_{t-1} + 1
        - sync = alpha / sqrt(beta)
        
        Returns: (sync, updated_alpha, updated_beta)
        """
        batch_size = activated_state.shape[0]
        
        # Select pairs and decay params based on type
        if synch_type == 'action':
            left_indices = self.action_left
            right_indices = self.action_right
            decay_params = self.decay_params_action
        else:
            left_indices = self.out_left
            right_indices = self.out_right
            decay_params = self.decay_params_out
        
        # Compute decay rate (clamped for stability)
        decay_params_clamped = torch.clamp(decay_params, 0, 15)
        r = torch.exp(-decay_params_clamped).unsqueeze(0).expand(batch_size, -1)
        
        # Get neuron values for pairs and compute pairwise product
        left_vals = activated_state[:, left_indices]
        right_vals = activated_state[:, right_indices]
        pairwise_product = left_vals * right_vals
        
        # Recursive update (or initialize)
        if decay_alpha is None or decay_beta is None:
            alpha = pairwise_product
            beta = torch.ones_like(pairwise_product)
        else:
            alpha = r * decay_alpha + pairwise_product
            beta = r * decay_beta + 1.0
        
        # Compute normalized sync
        sync = alpha / torch.sqrt(beta + 1e-8)
        
        return sync, alpha, beta
    
    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        """Compute certainty as 1 - normalized_entropy."""
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def forward(self, x: torch.Tensor, task_idx: int, modality: str, track: bool = False):
        """
        Forward pass through MultitaskPerceiverCTM.
        
        Args:
            x: Input tensor (shape depends on modality)
            task_idx: Index of the task in task_output_sizes
            modality: 'image', 'audio', or 'tabular'
            track: Whether to track activations for visualization
        
        Returns predictions and certainties for ALL ticks (no load_balance_loss).
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode using Perceiver (handles all modalities)
        encoded_input = self.perceiver_encoder(x, modality=modality)
        
        # Prepare key-value features for attention
        kv = self.kv_proj(encoded_input).unsqueeze(1)
        
        # Initialize recurrent state from learnable parameters
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Input-conditioned initialization: each sample starts from unique neural state
        base_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        input_contribution = self.input_to_state(encoded_input)  # (B, n_neurons)
        # Apply learnable scale to ensure input has meaningful impact even at early training
        activated_state = base_state + self.input_scale * input_contribution
        
        # Prepare storage for outputs per iteration
        out_dims = self.task_output_sizes[task_idx]
        predictions = torch.empty(batch_size, out_dims, self.max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.max_ticks, device=device, dtype=torch.float32)
        
        # Initialize synchronization recurrence values
        decay_alpha_action: torch.Tensor | None = None
        decay_beta_action: torch.Tensor | None = None
        decay_alpha_out: torch.Tensor | None = None
        decay_beta_out: torch.Tensor | None = None
        
        # Initialize output sync at tick 0
        _, decay_alpha_out, decay_beta_out = self.compute_synchronization(
            activated_state, decay_alpha_out, decay_beta_out, synch_type='out'
        )
        
        # Tracking for visualization
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            attention_tracking = []
        
        # Recurrent loop over internal ticks
        for tick in range(self.max_ticks):
            # Calculate synchronization for action (attention modulation)
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, synch_type='action'
            )
            sync_action = self.dropout_sync(sync_action)
            
            # Create query from action synchronization
            q = self.q_proj(sync_action).unsqueeze(1)
            
            # Multi-head attention: sync_action queries the encoded input
            attn_out, attn_weights = self.attention(q, kv, kv, need_weights=True)
            attn_out = attn_out.squeeze(1)
            
            # Normalize activated state to prevent it from overpowering attention
            # (Crucial when using strong input-conditioned initialization)
            norm_state = self.state_norm(activated_state)
            
            # Combine attention output with current activated state for synapse model
            pre_synapse_input = torch.cat([attn_out, norm_state], dim=-1)
            
            # Apply synapse model to get new pre-activations
            new_state = self.synapse_model(pre_synapse_input)
            new_state = self.dropout_synapse(new_state)
            
            # Update state trace (FIFO buffer of pre-activations)
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply batched neuron-level model to get post-activations
            activated_state = self.batched_nlm(state_trace)
            activated_state = self.dropout_nlm(activated_state)
            
            # Calculate synchronization for output
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(
                activated_state, decay_alpha_out, decay_beta_out, synch_type='out'
            )
            sync_out = self.dropout_sync(sync_out)
            
            # Get predictions using task-specific output projector
            current_prediction = self.task_output_projectors[task_idx](sync_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            # Store predictions and certainties for this tick
            predictions[..., tick] = current_prediction
            certainties[..., tick] = current_certainty
            
            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(sync_out.detach().cpu().numpy())
                synch_action_tracking.append(sync_action.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
        
        if track:
            return (
                predictions, 
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        
        return predictions, certainties


# CIFAR-10 class groupings for coarse classification
CIFAR10_COARSE_LABELS = {
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,  # Animals
    0: 1, 1: 1, 8: 1, 9: 1  # Vehicles
}

# Speech Commands labels (35 classes)
SPEECH_COMMANDS_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]


class SpeechCommandsDataset(Dataset):
    """
    Speech Commands dataset wrapper.
    Converts audio waveforms to mel spectrograms for Perceiver processing.
    35 spoken word classes: yes, no, up, down, left, right, etc.
    
    This dataset downloads automatically via torchaudio.
    """
    
    def __init__(self, root: str, subset: str, download: bool):
        self.root = root
        self.subset = subset
        
        # Speech Commands from torchaudio (auto-downloads)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            subset=subset,
            download=download
        )
        
        # Mel spectrogram transform (16kHz is native sample rate)
        self.target_sample_rate = 16000
        self.mel_transform = MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=32,
            n_fft=400,
            hop_length=320,
        )
        
        # Build label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(SPEECH_COMMANDS_LABELS)}
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        
        # Resample if needed (should be 16kHz already)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate to 1 second (16000 samples)
        target_length = self.target_sample_rate  # 1 second
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # Convert to mel spectrogram: (1, n_mels, time)
        mel_spec = self.mel_transform(waveform)
        
        # Log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        # Get label index
        label_idx = self.label_to_idx.get(label, 0)
        
        return mel_spec, label_idx


class WineDataset(Dataset):
    """
    Wine classification dataset wrapper.
    Uses sklearn's wine dataset - 13 features, 3 classes.
    """
    
    def __init__(self, train: bool, test_ratio: float):
        wine = load_wine()
        X = torch.tensor(wine.data, dtype=torch.float32)
        y = torch.tensor(wine.target, dtype=torch.long)
        
        # Normalize features
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0) + 1e-9
        X = (X - self.mean) / self.std
        
        # Train/test split
        n_samples = len(X)
        n_test = int(n_samples * test_ratio)
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        perm = torch.randperm(n_samples)
        
        if train:
            indices = perm[n_test:]
        else:
            indices = perm[:n_test]
        
        self.X = X[indices]
        self.y = y[indices]
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.X[idx], self.y[idx].item()


class CombinedMultitaskDataset(Dataset):
    """
    Combined dataset that mixes multiple datasets across modalities.
    Uses -1 for non-applicable task labels (masked in loss computation).
    
    Supports:
    - Image datasets: MNIST, CIFAR, Fashion, EMNIST
    - Audio datasets: Speech Commands (mel spectrograms)
    - Tabular datasets: Wine
    """
    def __init__(
        self, 
        mnist_dataset: Dataset | None, 
        cifar_dataset: Dataset | None,
        fashion_dataset: Dataset | None,
        emnist_dataset: Dataset | None,
        speech_commands_dataset: Dataset | None,
        wine_dataset: Dataset | None,
    ):
        self.mnist_dataset = mnist_dataset
        self.cifar_dataset = cifar_dataset
        self.fashion_dataset = fashion_dataset
        self.emnist_dataset = emnist_dataset
        self.speech_commands_dataset = speech_commands_dataset
        self.wine_dataset = wine_dataset
        
        self.mnist_len = len(mnist_dataset) if mnist_dataset is not None else 0
        self.cifar_len = len(cifar_dataset) if cifar_dataset is not None else 0
        self.fashion_len = len(fashion_dataset) if fashion_dataset is not None else 0
        self.emnist_len = len(emnist_dataset) if emnist_dataset is not None else 0
        self.speech_commands_len = len(speech_commands_dataset) if speech_commands_dataset is not None else 0
        self.wine_len = len(wine_dataset) if wine_dataset is not None else 0
        
        self.total_len = (self.mnist_len + self.cifar_len + self.fashion_len + 
                         self.emnist_len + self.speech_commands_len + self.wine_len)
        
        self.mnist_end = self.mnist_len
        self.cifar_end = self.mnist_end + self.cifar_len
        self.fashion_end = self.cifar_end + self.fashion_len
        self.emnist_end = self.fashion_end + self.emnist_len
        self.speech_commands_end = self.emnist_end + self.speech_commands_len
        self.wine_end = self.speech_commands_end + self.wine_len
    
    def __len__(self) -> int:
        return self.total_len
    
    def _empty_labels(self) -> dict[str, int]:
        return {
            'mnist_digit': -1,
            'mnist_even_odd': -1,
            'cifar_fine': -1,
            'cifar_coarse': -1,
            'fashion_fine': -1,
            'emnist_fine': -1,
            'speech_commands': -1,
            'wine_type': -1,
        }
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int], str]:
        labels = self._empty_labels()
        
        if idx < self.mnist_end:
            image, digit_label = self.mnist_dataset[idx]
            labels['mnist_digit'] = digit_label
            labels['mnist_even_odd'] = digit_label % 2
            return image, labels, 'mnist'
        
        elif idx < self.cifar_end:
            cifar_idx = idx - self.mnist_end
            image, fine_label = self.cifar_dataset[cifar_idx]
            labels['cifar_fine'] = fine_label
            labels['cifar_coarse'] = CIFAR10_COARSE_LABELS[fine_label]
            return image, labels, 'cifar'
        
        elif idx < self.fashion_end:
            fashion_idx = idx - self.cifar_end
            image, fine_label = self.fashion_dataset[fashion_idx]
            labels['fashion_fine'] = fine_label
            return image, labels, 'fashion'
        
        elif idx < self.emnist_end:
            emnist_idx = idx - self.fashion_end
            image, char_label = self.emnist_dataset[emnist_idx]
            labels['emnist_fine'] = char_label
            return image, labels, 'emnist'
        
        elif idx < self.speech_commands_end:
            sc_idx = idx - self.emnist_end
            spectrogram, word_label = self.speech_commands_dataset[sc_idx]
            labels['speech_commands'] = word_label
            return spectrogram, labels, 'speech_commands'
        
        else:
            wine_idx = idx - self.speech_commands_end
            features, wine_label = self.wine_dataset[wine_idx]
            labels['wine_type'] = wine_label
            return features, labels, 'wine'


def custom_collate_fn(batch: list) -> tuple:
    """Custom collate function for combined dataset with mixed data types."""
    dataset_data: dict[str, list] = {
        'mnist': [], 'cifar': [], 'fashion': [], 'emnist': [],
        'speech_commands': [], 'wine': [],
    }
    labels_dict = {key: [] for key in batch[0][1].keys()}
    sources = []
    batch_indices: dict[str, list] = {
        'mnist': [], 'cifar': [], 'fashion': [], 'emnist': [],
        'speech_commands': [], 'wine': [],
    }
    
    for idx, (data, labels, source) in enumerate(batch):
        dataset_data[source].append(data)
        batch_indices[source].append(idx)
        for key, val in labels.items():
            labels_dict[key].append(val)
        sources.append(source)
    
    stacked_data = {
        key: torch.stack(items) if items else None
        for key, items in dataset_data.items()
    }
    
    return stacked_data, labels_dict, sources, batch_indices


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics across epochs."""
    task_names: list[str]
    epoch_losses: dict[str, list[float]] = field(default_factory=dict)
    epoch_accuracies: dict[str, list[float]] = field(default_factory=dict)
    batch_losses: dict[str, list[float]] = field(default_factory=dict)
    epoch_balance_losses: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        for name in self.task_names:
            self.epoch_losses[name] = []
            self.epoch_accuracies[name] = []
            self.batch_losses[name] = []


def train_multitask_gated_model(
    model: MultitaskGatedCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int],
    test_dataset: Dataset,
    output_folder: str,
) -> TrainingMetrics:
    """
    Training loop for MultitaskGatedCTM on combined multi-dataset.
    Computes loss across ALL ticks as per CTM paper Section 3.5.
    Includes load balancing loss to prevent mode collapse in gating.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    metrics = TrainingMetrics(task_names=task_names)
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
        'speech_commands': [6],
        'wine': [7],
    }
    
    # Mapping from dataset to modality for Perceiver encoder
    dataset_modality = {
        'mnist': 'image',
        'cifar': 'image',
        'fashion': 'image',
        'emnist': 'image',
        'speech_commands': 'audio',
        'wine': 'tabular',
    }
    
    for epoch in range(epochs):
        task_losses = {name: 0.0 for name in task_names}
        task_correct = {name: 0 for name in task_names}
        task_total = {name: 0 for name in task_names}
        task_batch_count = {name: 0 for name in task_names}
        total_balance_loss = 0.0
        balance_count = 0
        
        epoch_start_time = time.time()
        last_checkpoint_time = epoch_start_time
        
        for batch_idx, (stacked_data, labels, sources, batch_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_balance_loss = torch.tensor(0.0, device=device)
            
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_data[dataset_name] is not None:
                    data = stacked_data[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    modality = dataset_modality[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                # Forward pass returns predictions for ALL ticks plus load balance loss
                                predictions, certainties, load_balance_loss = model(data, task_idx=model_task_idx, modality=modality)
                                
                                # Compute loss across all ticks (Section 3.5)
                                loss = torch.tensor(0.0, device=device)
                                for tick in range(model.max_ticks):
                                    tick_predictions = predictions[..., tick]
                                    loss = loss + criterion(tick_predictions, target)
                                loss = loss / model.max_ticks
                                
                                # Add load balance loss
                                total_loss = total_loss + loss + model.load_balance_coef * load_balance_loss
                                batch_balance_loss = batch_balance_loss + load_balance_loss
                                
                                task_losses[task_name] += loss.item()
                                task_batch_count[task_name] += 1
                                metrics.batch_losses[task_name].append(loss.item())
                                
                                # Stats using final tick
                                valid_mask = target >= 0
                                if valid_mask.any():
                                    final_predictions = predictions[..., -1]
                                    _, predicted = torch.max(final_predictions[valid_mask].data, 1)
                                    task_total[task_name] += valid_mask.sum().item()
                                    task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
            
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()
                total_balance_loss += batch_balance_loss.item()
                balance_count += 1
            
            if batch_idx % 25 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_checkpoint_time
                avg_balance = batch_balance_loss.item() / max(1, sum(1 for d in dataset_task_indices if stacked_data[d] is not None))
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Total Loss: {total_loss.item():.4f}, Balance: {avg_balance:.4f}, '
                      f'Time: {elapsed_time:.2f}s')
                last_checkpoint_time = current_time
            
            # Gate inspection at batch 100
            if batch_idx % 500 == 0:
                model.eval()
                with torch.no_grad():
                    # Find first available data for inspection
                    for dataset_name, task_indices in dataset_task_indices.items():
                        data = stacked_data.get(dataset_name)
                        if data is not None:
                            modality = dataset_modality[dataset_name]
                            for original_idx in task_indices:
                                if original_idx in task_index_mapping:
                                    model_task_idx = task_index_mapping[original_idx]
                                    task_name = task_names[model_task_idx]
                                    
                                    # Forward with tracking to get gate weights
                                    results = model(data[:8].to(device), task_idx=model_task_idx, modality=modality, track=True)
                                    predictions, certainties, lb_loss, syncs, pre_acts, post_acts, attn, gates = results
                                    gate_action, gate_out = gates
                                    
                                    # gate_action shape: (ticks, batch, n_sets)
                                    # Average over batch, show per tick
                                    avg_gate_action = gate_action.mean(axis=1)  # (ticks, n_sets)
                                    avg_gate_out = gate_out.mean(axis=1)
                                    
                                    print(f'\nTask: {task_name} (modality: {modality})')
                                    print(f'Action Gate Weights (avg over batch, first/mid/last tick):')
                                    print(f'  Tick 0:  {avg_gate_action[0]}')
                                    print(f'  Tick 7:  {avg_gate_action[7]}')
                                    print(f'  Tick 14: {avg_gate_action[14]}')
                                    print(f'Output Gate Weights (avg over batch, first/mid/last tick):')
                                    print(f'  Tick 0:  {avg_gate_out[0]}')
                                    print(f'  Tick 7:  {avg_gate_out[7]}')
                                    print(f'  Tick 14: {avg_gate_out[14]}')
                                    
                                    # Per-sample variance to see if gates differentiate across inputs
                                    sample_var_action = gate_action[-1].var(axis=0)  # variance across batch at last tick
                                    sample_var_out = gate_out[-1].var(axis=0)
                                    print(f'Gate variance across samples (last tick):')
                                    print(f'  Action: {sample_var_action}')
                                    print(f'  Output: {sample_var_out}')
                                    
                                    # Log gate weights to MLflow
                                    for set_idx in range(avg_gate_out.shape[1]):
                                        mlflow.log_metric(f"gate_out_set{set_idx}_tick14", float(avg_gate_out[14, set_idx]), step=epoch + 1)
                                        mlflow.log_metric(f"gate_action_set{set_idx}_tick14", float(avg_gate_action[14, set_idx]), step=epoch + 1)
                                    mlflow.log_metric("gate_out_variance", float(sample_var_out.mean()), step=epoch + 1)
                                    mlflow.log_metric("gate_action_variance", float(sample_var_action.mean()), step=epoch + 1)
                                    
                                    # Performance on this batch
                                    target = torch.tensor([labels[original_task_names[original_idx]][i] for i in batch_indices[dataset_name][:8]]).to(device)
                                    final_preds = predictions[..., -1]
                                    _, predicted = torch.max(final_preds, 1)
                                    correct = (predicted == target).sum().item()
                                    print(f'Batch accuracy: {correct}/8 = {100*correct/8:.1f}%')
                                    
                            continue
                print('='*60 + '\n')
                model.train()
        
        avg_balance = total_balance_loss / max(1, balance_count)
        metrics.epoch_balance_losses.append(avg_balance)
        
        print(f'\n=== Epoch {epoch+1}/{epochs} Summary ===')
        print(f'  Average Balance Loss: {avg_balance:.4f}')
        
        # Log epoch-level metrics to MLflow
        mlflow.log_metric("balance_loss", avg_balance, step=epoch + 1)
        
        for task_name in task_names:
            if task_total[task_name] > 0:
                accuracy = 100. * task_correct[task_name] / task_total[task_name]
                avg_loss = task_losses[task_name] / max(1, task_batch_count[task_name])
                metrics.epoch_losses[task_name].append(avg_loss)
                metrics.epoch_accuracies[task_name].append(accuracy)
                print(f'  {task_name}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
                
                # Log per-task metrics to MLflow
                mlflow.log_metric(f"{task_name}/train_loss", avg_loss, step=epoch + 1)
                mlflow.log_metric(f"{task_name}/train_accuracy", accuracy, step=epoch + 1)
        
        # Generate neural dynamics plots for all tasks at end of each epoch
        generate_epoch_plots(
            model=model,
            test_dataset=test_dataset,
            task_names=task_names,
            task_index_mapping=task_index_mapping,
            device=device,
            output_folder=output_folder,
            epoch=epoch + 1,
        )
        model.train()
        print()
    
    return metrics


def evaluate_multitask_gated_model(
    model: MultitaskGatedCTM,
    test_loader: DataLoader,
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int]
) -> dict[str, float]:
    """Evaluate model on combined multi-dataset test set."""
    model.eval()
    
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
        'speech_commands': [6],
        'wine': [7],
    }
    
    dataset_modality = {
        'mnist': 'image',
        'cifar': 'image',
        'fashion': 'image',
        'emnist': 'image',
        'speech_commands': 'audio',
        'wine': 'tabular',
    }
    
    with torch.no_grad():
        for stacked_data, labels, sources, batch_indices in test_loader:
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_data[dataset_name] is not None:
                    data = stacked_data[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    modality = dataset_modality[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                predictions, _, _ = model(data, task_idx=model_task_idx, modality=modality)
                                final_predictions = predictions[..., -1]
                                
                                valid_mask = target >= 0
                                _, predicted = torch.max(final_predictions[valid_mask].data, 1)
                                task_total[task_name] += valid_mask.sum().item()
                                task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
    
    test_accuracies = {}
    print('\n=== Test Results ===')
    for task_name in task_names:
        if task_total[task_name] > 0:
            accuracy = 100. * task_correct[task_name] / task_total[task_name]
            test_accuracies[task_name] = accuracy
            print(f'  {task_name}: Accuracy={accuracy:.2f}%')
    print()
    
    return test_accuracies


def get_neural_dynamics(
    model: MultitaskGatedCTM,
    x: torch.Tensor,
    task_idx: int,
    modality: str,
    device: torch.device,
) -> dict[str, np.ndarray | torch.Tensor | tuple]:
    """Extract all neural dynamics for visualization (gated model)."""
    model.eval()
    
    with torch.no_grad():
        results = model(x.to(device), task_idx=task_idx, modality=modality, track=True)
        predictions, certainties, load_balance_loss, syncs, pre_acts, post_acts, attn, gates = results
    
    post_history = torch.from_numpy(post_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    pre_history = torch.from_numpy(pre_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    
    return {
        'post_history': post_history,
        'pre_history': pre_history,
        'sync_out': syncs[0],  # (ticks, B, n_synch_out)
        'sync_action': syncs[1],  # (ticks, B, n_synch_action)
        'attention': attn,  # (ticks, B, n_heads, 1, seq_len)
        'predictions': predictions.cpu(),  # (B, out_dims, ticks)
        'certainties': certainties.cpu(),  # (B, 2, ticks)
        'gates': gates,  # (gate_action, gate_out)
    }


def plot_neural_dynamics_for_task(
    model: MultitaskGatedCTM,
    sample_data: torch.Tensor,
    task_idx: int,
    task_name: str,
    modality: str,
    device: torch.device,
    output_path: str,
    epoch: int,
):
    """Generate neural dynamics plot for a single task."""
    model.eval()
    with torch.no_grad():
        dynamics = get_neural_dynamics(
            model, sample_data.unsqueeze(0), task_idx=task_idx, modality=modality, device=device
        )
    post_history = dynamics['post_history']
    gate_action, gate_out = dynamics['gates']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Neural Dynamics - {task_name} [{modality}] (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    # Input visualization (depends on modality)
    ax_input = axes[0, 0]
    data_np = sample_data.cpu().numpy()
    
    if modality == 'image':
        if data_np.shape[0] == 3:
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
            data_np = data_np * std + mean
            data_np = np.clip(data_np, 0, 1)
            data_np = np.transpose(data_np, (1, 2, 0))
            ax_input.imshow(data_np)
        else:
            data_np = data_np * 0.3081 + 0.1307
            data_np = np.clip(data_np, 0, 1)
            ax_input.imshow(data_np.squeeze(), cmap='gray')
        ax_input.set_title(f'Input Image ({task_name})')
    elif modality == 'audio':
        # Audio spectrogram visualization
        spec = data_np.squeeze()
        im = ax_input.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax_input.set_xlabel('Time')
        ax_input.set_ylabel('Frequency')
        ax_input.set_title(f'Mel Spectrogram ({task_name})')
        plt.colorbar(im, ax=ax_input, label='Log Magnitude')
    elif modality == 'tabular':
        # Bar chart for tabular features
        ax_input.bar(range(len(data_np)), data_np)
        ax_input.set_xlabel('Feature Index')
        ax_input.set_ylabel('Normalized Value')
        ax_input.set_title(f'Tabular Features ({task_name})')
    ax_input.axis('off') if modality == 'image' else None
    
    # Post-activation dynamics
    ax_activity = axes[0, 1]
    activity_data = post_history[0].numpy()
    im = ax_activity.imshow(activity_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_activity.set_xlabel('Tick')
    ax_activity.set_ylabel('Neuron')
    ax_activity.set_title('Post-Activation Dynamics')
    plt.colorbar(im, ax=ax_activity, label='Activation')
    
    # Action gate weights
    ax_gate_action = axes[1, 0]
    gate_action_data = gate_action[:, 0, :]
    for set_idx in range(gate_action_data.shape[1]):
        ax_gate_action.plot(gate_action_data[:, set_idx], label=f'Set {set_idx}', alpha=0.8)
    ax_gate_action.set_xlabel('Tick')
    ax_gate_action.set_ylabel('Gate Weight')
    ax_gate_action.set_title('Action Gate Weights Over Ticks')
    ax_gate_action.legend()
    ax_gate_action.grid(True, alpha=0.3)
    ax_gate_action.set_ylim([0, 1])
    
    # Output gate weights
    ax_gate_out = axes[1, 1]
    gate_out_data = gate_out[:, 0, :]
    for set_idx in range(gate_out_data.shape[1]):
        ax_gate_out.plot(gate_out_data[:, set_idx], label=f'Set {set_idx}', alpha=0.8)
    ax_gate_out.set_xlabel('Tick')
    ax_gate_out.set_ylabel('Gate Weight')
    ax_gate_out.set_title('Output Gate Weights Over Ticks')
    ax_gate_out.legend()
    ax_gate_out.grid(True, alpha=0.3)
    ax_gate_out.set_ylim([0, 1])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_detailed_ctm_analysis_gated(
    model: MultitaskGatedCTM,
    sample_data: torch.Tensor,
    task_idx: int,
    task_name: str,
    modality: str,
    device: torch.device,
    output_path: str,
    epoch: int,
    label: int,
):
    """
    Generate detailed CTM analysis figure for gated model.
    
    Shows:
    - Individual neuron activity traces
    - Gate weights evolution per set
    - Attention weights per head
    - Prediction evolution and certainty
    - Synchronization pair heatmaps
    
    Reference: https://arxiv.org/html/2505.05522v4
    """
    model.eval()
    with torch.no_grad():
        dynamics = get_neural_dynamics(
            model, sample_data.unsqueeze(0), task_idx=task_idx, modality=modality, device=device
        )
    
    post_history = dynamics['post_history'][0].numpy()  # (neurons, ticks)
    pre_history = dynamics['pre_history'][0].numpy()  # (neurons, ticks)
    sync_action = dynamics['sync_action'][:, 0, :]  # (ticks, n_synch_action)
    sync_out = dynamics['sync_out'][:, 0, :]  # (ticks, n_synch_out)
    attention = dynamics['attention']  # (ticks, B, n_heads, 1, seq_len)
    predictions = dynamics['predictions'][0]  # (out_dims, ticks)
    certainties = dynamics['certainties'][0]  # (2, ticks)
    gate_action, gate_out = dynamics['gates']
    gate_action = gate_action[:, 0, :]  # (ticks, n_sync_sets)
    gate_out = gate_out[:, 0, :]  # (ticks, n_sync_sets)
    
    n_neurons, n_ticks = post_history.shape
    n_sync_sets = gate_action.shape[1]
    
    # Create large figure
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f'Detailed Gated CTM Analysis - {task_name} [{modality}] (Epoch {epoch}, Label: {label})', 
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # ========== Row 1: Input and neuron traces ==========
    
    # Input visualization
    ax_input = fig.add_subplot(gs[0, 0])
    data_np = sample_data.cpu().numpy()
    if modality == 'image':
        if data_np.shape[0] == 3:
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
            data_np = data_np * std + mean
            data_np = np.clip(data_np, 0, 1)
            data_np = np.transpose(data_np, (1, 2, 0))
            ax_input.imshow(data_np)
        else:
            data_np = data_np * 0.3081 + 0.1307
            data_np = np.clip(data_np, 0, 1)
            ax_input.imshow(data_np.squeeze(), cmap='gray')
        ax_input.axis('off')
    elif modality == 'audio':
        spec = data_np.squeeze()
        ax_input.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax_input.set_xlabel('Time')
        ax_input.set_ylabel('Freq')
    elif modality == 'tabular':
        ax_input.bar(range(len(data_np)), data_np, color='#3498db')
        ax_input.set_xlabel('Feature')
        ax_input.set_ylabel('Value')
    ax_input.set_title('Input', fontsize=11, fontweight='bold')
    
    # Individual neuron traces
    ax_traces = fig.add_subplot(gs[0, 1:3])
    n_show = min(16, n_neurons)
    colors = plt.cm.tab20(np.linspace(0, 1, n_show))
    for i in range(n_show):
        neuron_idx = i * (n_neurons // n_show)
        trace = post_history[neuron_idx, :]
        ax_traces.plot(range(n_ticks), trace + i * 0.5, color=colors[i], linewidth=1.2, alpha=0.8)
    ax_traces.set_xlabel('Tick')
    ax_traces.set_ylabel('Neuron Activity (offset)')
    ax_traces.set_title('Individual Neuron Activity Traces', fontsize=11, fontweight='bold')
    ax_traces.set_xlim(0, n_ticks - 1)
    ax_traces.grid(True, alpha=0.2)
    
    # Neuron activity variance
    ax_var = fig.add_subplot(gs[0, 3])
    variance_per_tick = post_history.var(axis=0)
    mean_per_tick = post_history.mean(axis=0)
    ax_var.fill_between(range(n_ticks), mean_per_tick - np.sqrt(variance_per_tick), 
                        mean_per_tick + np.sqrt(variance_per_tick), alpha=0.3, color='#9b59b6')
    ax_var.plot(range(n_ticks), mean_per_tick, color='#9b59b6', linewidth=2)
    ax_var.set_xlabel('Tick')
    ax_var.set_ylabel('Activation')
    ax_var.set_title('Mean ± Std Activation', fontsize=11, fontweight='bold')
    ax_var.grid(True, alpha=0.3)
    
    # ========== Row 2: Post and Pre activation heatmaps ==========
    
    ax_post = fig.add_subplot(gs[1, 0:2])
    im = ax_post.imshow(post_history, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_post.set_xlabel('Tick')
    ax_post.set_ylabel('Neuron')
    ax_post.set_title('Post-Activation Dynamics', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_post, shrink=0.8)
    
    ax_pre = fig.add_subplot(gs[1, 2:4])
    im2 = ax_pre.imshow(pre_history, aspect='auto', cmap='PuOr', interpolation='nearest')
    ax_pre.set_xlabel('Tick')
    ax_pre.set_ylabel('Neuron')
    ax_pre.set_title('Pre-Activation Dynamics', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax_pre, shrink=0.8)
    
    # ========== Row 3: Gate weights ==========
    
    ax_gate_act = fig.add_subplot(gs[2, 0:2])
    for set_idx in range(n_sync_sets):
        ax_gate_act.plot(range(n_ticks), gate_action[:, set_idx], 
                        linewidth=2, label=f'Set {set_idx}', alpha=0.8)
    ax_gate_act.set_xlabel('Tick')
    ax_gate_act.set_ylabel('Gate Weight')
    ax_gate_act.set_title('Action Gate Weights (Sync Set Selection)', fontsize=11, fontweight='bold')
    ax_gate_act.legend(loc='upper right')
    ax_gate_act.grid(True, alpha=0.3)
    ax_gate_act.set_ylim(0, 1)
    
    ax_gate_out = fig.add_subplot(gs[2, 2:4])
    for set_idx in range(n_sync_sets):
        ax_gate_out.plot(range(n_ticks), gate_out[:, set_idx], 
                        linewidth=2, label=f'Set {set_idx}', alpha=0.8)
    ax_gate_out.set_xlabel('Tick')
    ax_gate_out.set_ylabel('Gate Weight')
    ax_gate_out.set_title('Output Gate Weights (Sync Set Selection)', fontsize=11, fontweight='bold')
    ax_gate_out.legend(loc='upper right')
    ax_gate_out.grid(True, alpha=0.3)
    ax_gate_out.set_ylim(0, 1)
    
    # ========== Row 4: Synchronization heatmaps ==========
    
    ax_sync_act = fig.add_subplot(gs[3, 0:2])
    n_sync_show = min(32, sync_action.shape[1])
    im3 = ax_sync_act.imshow(sync_action[:, :n_sync_show].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax_sync_act.set_xlabel('Tick')
    ax_sync_act.set_ylabel('Sync Pair')
    ax_sync_act.set_title('Action Sync Pairs (Gated Weighted)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax_sync_act, shrink=0.8)
    
    ax_sync_out = fig.add_subplot(gs[3, 2:4])
    n_sync_show = min(32, sync_out.shape[1])
    im4 = ax_sync_out.imshow(sync_out[:, :n_sync_show].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax_sync_out.set_xlabel('Tick')
    ax_sync_out.set_ylabel('Sync Pair')
    ax_sync_out.set_title('Output Sync Pairs (Gated Weighted)', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax_sync_out, shrink=0.8)
    
    # ========== Row 5: Predictions and certainty ==========
    
    ax_pred = fig.add_subplot(gs[4, 0:2])
    n_classes = predictions.shape[0]
    probs = torch.softmax(predictions, dim=0).numpy()
    
    final_probs = probs[:, -1]
    top_k = min(5, n_classes)
    top_indices = np.argsort(final_probs)[-top_k:][::-1]
    
    for idx in top_indices:
        ax_pred.plot(range(n_ticks), probs[idx, :], linewidth=2, 
                    label=f'Class {idx}' + (' ✓' if idx == label else ''))
    ax_pred.axhline(y=1.0/n_classes, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax_pred.set_xlabel('Tick')
    ax_pred.set_ylabel('Probability')
    ax_pred.set_title(f'Prediction Evolution (Top-{top_k} Classes)', fontsize=11, fontweight='bold')
    ax_pred.legend(loc='upper right', fontsize=9)
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_ylim(0, 1)
    
    ax_cert = fig.add_subplot(gs[4, 2])
    entropy = certainties[0, :].numpy()
    certainty = certainties[1, :].numpy()
    ax_cert.plot(range(n_ticks), certainty, color='#27ae60', linewidth=2, label='Certainty')
    ax_cert.plot(range(n_ticks), entropy, color='#e74c3c', linewidth=2, label='Entropy')
    ax_cert.fill_between(range(n_ticks), 0, certainty, alpha=0.2, color='#27ae60')
    ax_cert.set_xlabel('Tick')
    ax_cert.set_ylabel('Value')
    ax_cert.set_title('Certainty vs Entropy', fontsize=11, fontweight='bold')
    ax_cert.legend()
    ax_cert.grid(True, alpha=0.3)
    ax_cert.set_ylim(0, 1)
    
    # Final prediction summary
    ax_summary = fig.add_subplot(gs[4, 3])
    ax_summary.axis('off')
    
    final_pred = predictions[:, -1].argmax().item()
    final_prob = probs[final_pred, -1]
    correct = '✓ Correct' if final_pred == label else '✗ Incorrect'
    
    summary_text = (
        f"Final Prediction\n"
        f"───────────────\n"
        f"Predicted: {final_pred}\n"
        f"True Label: {label}\n"
        f"Confidence: {final_prob:.1%}\n"
        f"Final Certainty: {certainty[-1]:.1%}\n"
        f"\n{correct}"
    )
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_epoch_plots(
    model: MultitaskGatedCTM,
    test_dataset: Dataset,
    task_names: list[str],
    task_index_mapping: dict[int, int],
    device: torch.device,
    output_folder: str,
    epoch: int,
):
    """
    Generate neural dynamics plots for all tasks at end of epoch.
    
    Creates two figures per task:
    1. Standard dynamics plot (compact overview)
    2. Detailed CTM analysis (inspired by CTM paper Figure 2)
    
    Reference: https://arxiv.org/html/2505.05522v4
    """
    all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    task_to_modality = {
        'mnist_digit': 'image', 'mnist_even_odd': 'image',
        'cifar_fine': 'image', 'cifar_coarse': 'image',
        'fashion_fine': 'image', 'emnist_fine': 'image',
        'speech_commands': 'audio', 'wine_type': 'tabular',
    }
    
    # Get sample data AND labels for each task
    task_samples: dict[str, tuple[torch.Tensor, int]] = {}
    
    for idx in range(len(test_dataset)):
        data, labels, source = test_dataset[idx]
        
        for task_name in task_names:
            if task_name in task_samples:
                continue
            
            if task_name in all_task_order:
                label_key = task_name
                label_val = labels.get(label_key, -1)
                
                if label_val >= 0:
                    task_samples[task_name] = (data, label_val)
        
        if len(task_samples) == len(task_names):
            break
    
    # Generate plots for each task
    model.eval()
    for task_name in task_names:
        if task_name not in task_samples:
            continue
        
        sample_data, sample_label = task_samples[task_name]
        original_idx = all_task_order.index(task_name)
        model_task_idx = task_index_mapping[original_idx]
        modality = task_to_modality.get(task_name, 'image')
        
        # 1. Standard dynamics plot (compact)
        output_path = os.path.join(output_folder, f'epoch{epoch:02d}_{task_name}_dynamics.png')
        plot_neural_dynamics_for_task(
            model=model,
            sample_data=sample_data,
            task_idx=model_task_idx,
            task_name=task_name,
            modality=modality,
            device=device,
            output_path=output_path,
            epoch=epoch,
        )
        
        # 2. Detailed CTM analysis (inspired by CTM paper Figure 2)
        detailed_path = os.path.join(output_folder, f'epoch{epoch:02d}_{task_name}_detailed.png')
        plot_detailed_ctm_analysis_gated(
            model=model,
            sample_data=sample_data,
            task_idx=model_task_idx,
            task_name=task_name,
            modality=modality,
            device=device,
            output_path=detailed_path,
            epoch=epoch,
            label=sample_label,
        )
    
    print(f'  Saved neural dynamics plots for epoch {epoch} to {output_folder}')


# ==============================================================================
# MultitaskPerceiverCTM Functions (Perceiver IO output, no gating)
# ==============================================================================

def get_neural_dynamics_perceiver(
    model: MultitaskPerceiverCTM,
    x: torch.Tensor,
    task_idx: int,
    modality: str,
    device: torch.device,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Extract all neural dynamics for visualization (no gates)."""
    model.eval()
    
    with torch.no_grad():
        results = model(x.to(device), task_idx=task_idx, modality=modality, track=True)
        predictions, certainties, syncs, pre_acts, post_acts, attn = results
    
    post_history = torch.from_numpy(post_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    pre_history = torch.from_numpy(pre_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    
    return {
        'post_history': post_history,
        'pre_history': pre_history,
        'sync_out': syncs[0],  # (ticks, B, n_synch_out)
        'sync_action': syncs[1],  # (ticks, B, n_synch_action)
        'attention': attn,  # (ticks, B, n_heads, 1, seq_len)
        'predictions': predictions.cpu(),  # (B, out_dims, ticks)
        'certainties': certainties.cpu(),  # (B, 2, ticks)
    }


def plot_neural_dynamics_for_task_perceiver(
    model: MultitaskPerceiverCTM,
    sample_data: torch.Tensor,
    task_idx: int,
    task_name: str,
    modality: str,
    device: torch.device,
    output_path: str,
    epoch: int,
):
    """Generate neural dynamics plot for a single task (Perceiver CTM, no gates)."""
    model.eval()
    with torch.no_grad():
        dynamics = get_neural_dynamics_perceiver(
            model, sample_data.unsqueeze(0), task_idx=task_idx, modality=modality, device=device
        )
    
    post_history = dynamics['post_history']
    sync_action_data = dynamics['sync_action'][:, 0, :]  # (ticks, n_synch_action)
    sync_out_data = dynamics['sync_out'][:, 0, :]  # (ticks, n_synch_out)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Neural Dynamics - {task_name} [{modality}] (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    # Input visualization (depends on modality)
    ax_input = axes[0, 0]
    data_np = sample_data.cpu().numpy()
    
    if modality == 'image':
        if data_np.shape[0] == 3:
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
            data_np = data_np * std + mean
            data_np = np.clip(data_np, 0, 1)
            data_np = np.transpose(data_np, (1, 2, 0))
            ax_input.imshow(data_np)
        else:
            data_np = data_np * 0.3081 + 0.1307
            data_np = np.clip(data_np, 0, 1)
            ax_input.imshow(data_np.squeeze(), cmap='gray')
        ax_input.set_title(f'Input Image ({task_name})')
    elif modality == 'audio':
        spec = data_np.squeeze()
        im = ax_input.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax_input.set_xlabel('Time')
        ax_input.set_ylabel('Frequency')
        ax_input.set_title(f'Mel Spectrogram ({task_name})')
        plt.colorbar(im, ax=ax_input, label='Log Magnitude')
    elif modality == 'tabular':
        ax_input.bar(range(len(data_np)), data_np)
        ax_input.set_xlabel('Feature Index')
        ax_input.set_ylabel('Normalized Value')
        ax_input.set_title(f'Tabular Features ({task_name})')
    ax_input.axis('off') if modality == 'image' else None
    
    # Post-activation dynamics
    ax_activity = axes[0, 1]
    activity_data = post_history[0].numpy()
    im = ax_activity.imshow(activity_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_activity.set_xlabel('Tick')
    ax_activity.set_ylabel('Neuron')
    ax_activity.set_title('Post-Activation Dynamics')
    plt.colorbar(im, ax=ax_activity, label='Activation')
    
    # Action synchronization over ticks
    ax_sync_action = axes[1, 0]
    sync_mean = sync_action_data.mean(axis=1)
    sync_std = sync_action_data.std(axis=1)
    ticks = np.arange(len(sync_mean))
    ax_sync_action.plot(ticks, sync_mean, color='#3498db', linewidth=2, label='Mean')
    ax_sync_action.fill_between(ticks, sync_mean - sync_std, sync_mean + sync_std, 
                                 alpha=0.3, color='#3498db')
    ax_sync_action.set_xlabel('Tick')
    ax_sync_action.set_ylabel('Sync Value')
    ax_sync_action.set_title('Action Synchronization Over Ticks')
    ax_sync_action.legend()
    ax_sync_action.grid(True, alpha=0.3)
    
    # Output synchronization over ticks
    ax_sync_out = axes[1, 1]
    sync_mean = sync_out_data.mean(axis=1)
    sync_std = sync_out_data.std(axis=1)
    ticks = np.arange(len(sync_mean))
    ax_sync_out.plot(ticks, sync_mean, color='#e74c3c', linewidth=2, label='Mean')
    ax_sync_out.fill_between(ticks, sync_mean - sync_std, sync_mean + sync_std, 
                              alpha=0.3, color='#e74c3c')
    ax_sync_out.set_xlabel('Tick')
    ax_sync_out.set_ylabel('Sync Value')
    ax_sync_out.set_title('Output Synchronization Over Ticks')
    ax_sync_out.legend()
    ax_sync_out.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_detailed_ctm_analysis(
    model: MultitaskPerceiverCTM,
    sample_data: torch.Tensor,
    task_idx: int,
    task_name: str,
    modality: str,
    device: torch.device,
    output_path: str,
    epoch: int,
    label: int,
):
    """
    Generate detailed CTM analysis figure inspired by CTM paper Figure 2.
    
    Shows:
    - Individual neuron activity traces (like Figure 2a)
    - Attention weights per head over ticks
    - Prediction evolution and certainty
    - Pre vs post activation comparison
    - Synchronization pair heatmaps
    
    Reference: https://arxiv.org/html/2505.05522v4
    """
    model.eval()
    with torch.no_grad():
        dynamics = get_neural_dynamics_perceiver(
            model, sample_data.unsqueeze(0), task_idx=task_idx, modality=modality, device=device
        )
    
    post_history = dynamics['post_history'][0].numpy()  # (neurons, ticks)
    pre_history = dynamics['pre_history'][0].numpy()  # (neurons, ticks)
    sync_action = dynamics['sync_action'][:, 0, :]  # (ticks, n_synch_action)
    sync_out = dynamics['sync_out'][:, 0, :]  # (ticks, n_synch_out)
    attention = dynamics['attention']  # (ticks, B, n_heads, 1, seq_len)
    predictions = dynamics['predictions'][0]  # (out_dims, ticks)
    certainties = dynamics['certainties'][0]  # (2, ticks) - [entropy, 1-entropy]
    
    n_neurons, n_ticks = post_history.shape
    n_heads = attention.shape[2] if len(attention.shape) > 2 else 1
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f'Detailed CTM Analysis - {task_name} [{modality}] (Epoch {epoch}, Label: {label})', 
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # ========== Row 1: Input and neuron traces ==========
    
    # Input visualization
    ax_input = fig.add_subplot(gs[0, 0])
    data_np = sample_data.cpu().numpy()
    if modality == 'image':
        if data_np.shape[0] == 3:
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
            data_np = data_np * std + mean
            data_np = np.clip(data_np, 0, 1)
            data_np = np.transpose(data_np, (1, 2, 0))
            ax_input.imshow(data_np)
        else:
            data_np = data_np * 0.3081 + 0.1307
            data_np = np.clip(data_np, 0, 1)
            ax_input.imshow(data_np.squeeze(), cmap='gray')
        ax_input.axis('off')
    elif modality == 'audio':
        spec = data_np.squeeze()
        im = ax_input.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax_input.set_xlabel('Time')
        ax_input.set_ylabel('Freq')
    elif modality == 'tabular':
        ax_input.bar(range(len(data_np)), data_np, color='#3498db')
        ax_input.set_xlabel('Feature')
        ax_input.set_ylabel('Value')
    ax_input.set_title('Input', fontsize=11, fontweight='bold')
    
    # Individual neuron traces (like CTM paper Figure 2a)
    ax_traces = fig.add_subplot(gs[0, 1:3])
    n_show = min(16, n_neurons)  # Show subset of neurons
    colors = plt.cm.tab20(np.linspace(0, 1, n_show))
    for i in range(n_show):
        neuron_idx = i * (n_neurons // n_show)
        trace = post_history[neuron_idx, :]
        ax_traces.plot(range(n_ticks), trace + i * 0.5, color=colors[i], linewidth=1.2, alpha=0.8)
    ax_traces.set_xlabel('Tick')
    ax_traces.set_ylabel('Neuron Activity (offset)')
    ax_traces.set_title('Individual Neuron Activity Traces', fontsize=11, fontweight='bold')
    ax_traces.set_xlim(0, n_ticks - 1)
    ax_traces.grid(True, alpha=0.2)
    
    # Neuron activity variance over time
    ax_var = fig.add_subplot(gs[0, 3])
    variance_per_tick = post_history.var(axis=0)
    mean_per_tick = post_history.mean(axis=0)
    ax_var.fill_between(range(n_ticks), mean_per_tick - np.sqrt(variance_per_tick), 
                        mean_per_tick + np.sqrt(variance_per_tick), alpha=0.3, color='#9b59b6')
    ax_var.plot(range(n_ticks), mean_per_tick, color='#9b59b6', linewidth=2)
    ax_var.set_xlabel('Tick')
    ax_var.set_ylabel('Activation')
    ax_var.set_title('Mean ± Std Activation', fontsize=11, fontweight='bold')
    ax_var.grid(True, alpha=0.3)
    
    # ========== Row 2: Post-activation heatmap and attention ==========
    
    # Post-activation dynamics heatmap
    ax_post = fig.add_subplot(gs[1, 0:2])
    im = ax_post.imshow(post_history, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_post.set_xlabel('Tick')
    ax_post.set_ylabel('Neuron')
    ax_post.set_title('Post-Activation Dynamics (All Neurons)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_post, label='Activation', shrink=0.8)
    
    # Pre-activation dynamics heatmap
    ax_pre = fig.add_subplot(gs[1, 2:4])
    im2 = ax_pre.imshow(pre_history, aspect='auto', cmap='PuOr', interpolation='nearest')
    ax_pre.set_xlabel('Tick')
    ax_pre.set_ylabel('Neuron')
    ax_pre.set_title('Pre-Activation Dynamics (All Neurons)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax_pre, label='Pre-Activation', shrink=0.8)
    
    # ========== Row 3: Synchronization and attention heads ==========
    
    # Action synchronization heatmap (individual pairs)
    ax_sync_act_heat = fig.add_subplot(gs[2, 0])
    n_sync_show = min(32, sync_action.shape[1])
    im3 = ax_sync_act_heat.imshow(sync_action[:, :n_sync_show].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax_sync_act_heat.set_xlabel('Tick')
    ax_sync_act_heat.set_ylabel('Sync Pair')
    ax_sync_act_heat.set_title('Action Sync Pairs', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax_sync_act_heat, shrink=0.8)
    
    # Output synchronization heatmap (individual pairs)
    ax_sync_out_heat = fig.add_subplot(gs[2, 1])
    n_sync_show = min(32, sync_out.shape[1])
    im4 = ax_sync_out_heat.imshow(sync_out[:, :n_sync_show].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax_sync_out_heat.set_xlabel('Tick')
    ax_sync_out_heat.set_ylabel('Sync Pair')
    ax_sync_out_heat.set_title('Output Sync Pairs', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax_sync_out_heat, shrink=0.8)
    
    # Attention weights per head over ticks (if multi-head)
    ax_attn = fig.add_subplot(gs[2, 2:4])
    if len(attention.shape) >= 4 and attention.shape[2] > 1:
        # Plot attention weight evolution per head
        attn_weights = attention[:, 0, :, 0, 0]  # (ticks, n_heads) - first position attention
        for head in range(min(n_heads, 8)):
            ax_attn.plot(range(n_ticks), attn_weights[:, head], 
                        label=f'Head {head}', linewidth=1.5, alpha=0.8)
        ax_attn.set_xlabel('Tick')
        ax_attn.set_ylabel('Attention Weight')
        ax_attn.set_title('Attention Weights by Head', fontsize=11, fontweight='bold')
        ax_attn.legend(loc='upper right', fontsize=8, ncol=2)
        ax_attn.grid(True, alpha=0.3)
    else:
        # Show mean attention over ticks
        ax_attn.text(0.5, 0.5, 'Single-head attention\n(no per-head breakdown)', 
                    ha='center', va='center', transform=ax_attn.transAxes, fontsize=12)
        ax_attn.set_title('Attention', fontsize=11, fontweight='bold')
    
    # ========== Row 4: Predictions and certainty ==========
    
    # Prediction logits over ticks
    ax_pred = fig.add_subplot(gs[3, 0:2])
    n_classes = predictions.shape[0]
    probs = torch.softmax(predictions, dim=0).numpy()  # (n_classes, ticks)
    
    # Show top-k classes
    final_probs = probs[:, -1]
    top_k = min(5, n_classes)
    top_indices = np.argsort(final_probs)[-top_k:][::-1]
    
    for idx in top_indices:
        ax_pred.plot(range(n_ticks), probs[idx, :], linewidth=2, 
                    label=f'Class {idx}' + (' ✓' if idx == label else ''))
    ax_pred.axhline(y=1.0/n_classes, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax_pred.set_xlabel('Tick')
    ax_pred.set_ylabel('Probability')
    ax_pred.set_title(f'Prediction Evolution (Top-{top_k} Classes)', fontsize=11, fontweight='bold')
    ax_pred.legend(loc='upper right', fontsize=9)
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_ylim(0, 1)
    
    # Certainty/entropy over ticks
    ax_cert = fig.add_subplot(gs[3, 2])
    entropy = certainties[0, :].numpy()  # Normalized entropy
    certainty = certainties[1, :].numpy()  # 1 - normalized entropy
    ax_cert.plot(range(n_ticks), certainty, color='#27ae60', linewidth=2, label='Certainty')
    ax_cert.plot(range(n_ticks), entropy, color='#e74c3c', linewidth=2, label='Entropy')
    ax_cert.fill_between(range(n_ticks), 0, certainty, alpha=0.2, color='#27ae60')
    ax_cert.set_xlabel('Tick')
    ax_cert.set_ylabel('Value')
    ax_cert.set_title('Certainty vs Entropy', fontsize=11, fontweight='bold')
    ax_cert.legend()
    ax_cert.grid(True, alpha=0.3)
    ax_cert.set_ylim(0, 1)
    
    # Final prediction summary
    ax_summary = fig.add_subplot(gs[3, 3])
    ax_summary.axis('off')
    
    final_pred = predictions[:, -1].argmax().item()
    final_prob = probs[final_pred, -1]
    correct = '✓ Correct' if final_pred == label else '✗ Incorrect'
    color = '#27ae60' if final_pred == label else '#e74c3c'
    
    summary_text = (
        f"Final Prediction\n"
        f"───────────────\n"
        f"Predicted: {final_pred}\n"
        f"True Label: {label}\n"
        f"Confidence: {final_prob:.1%}\n"
        f"Final Certainty: {certainty[-1]:.1%}\n"
        f"\n{correct}"
    )
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_epoch_plots_perceiver(
    model: MultitaskPerceiverCTM,
    test_dataset: Dataset,
    task_names: list[str],
    task_index_mapping: dict[int, int],
    device: torch.device,
    output_folder: str,
    epoch: int,
):
    """
    Generate neural dynamics plots for all tasks at end of epoch (Perceiver CTM).
    
    Creates two figures per task:
    1. Standard dynamics plot (compact overview)
    2. Detailed CTM analysis (inspired by CTM paper Figure 2)
    
    Reference: https://arxiv.org/html/2505.05522v4
    """
    all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    task_to_modality = {
        'mnist_digit': 'image', 'mnist_even_odd': 'image',
        'cifar_fine': 'image', 'cifar_coarse': 'image',
        'fashion_fine': 'image', 'emnist_fine': 'image',
        'speech_commands': 'audio', 'wine_type': 'tabular',
    }
    
    # Get sample data AND labels for each task from test dataset
    task_samples: dict[str, tuple[torch.Tensor, int]] = {}
    
    for idx in range(len(test_dataset)):
        data, labels, source = test_dataset[idx]
        
        for task_name in task_names:
            if task_name in task_samples:
                continue
            
            if task_name in all_task_order:
                label_key = task_name
                label_val = labels.get(label_key, -1)
                
                if label_val >= 0:
                    task_samples[task_name] = (data, label_val)
        
        if len(task_samples) == len(task_names):
            break
    
    # Generate plots for each task
    model.eval()
    for task_name in task_names:
        if task_name not in task_samples:
            continue
        
        sample_data, sample_label = task_samples[task_name]
        original_idx = all_task_order.index(task_name)
        model_task_idx = task_index_mapping[original_idx]
        modality = task_to_modality.get(task_name, 'image')
        
        # 1. Standard dynamics plot (compact)
        output_path = os.path.join(output_folder, f'epoch{epoch:02d}_{task_name}_dynamics.png')
        plot_neural_dynamics_for_task_perceiver(
            model=model,
            sample_data=sample_data,
            task_idx=model_task_idx,
            task_name=task_name,
            modality=modality,
            device=device,
            output_path=output_path,
            epoch=epoch,
        )
        
        # 2. Detailed CTM analysis (inspired by CTM paper Figure 2)
        detailed_path = os.path.join(output_folder, f'epoch{epoch:02d}_{task_name}_detailed.png')
        plot_detailed_ctm_analysis(
            model=model,
            sample_data=sample_data,
            task_idx=model_task_idx,
            task_name=task_name,
            modality=modality,
            device=device,
            output_path=detailed_path,
            epoch=epoch,
            label=sample_label,
        )
    
    print(f'  Saved neural dynamics plots for epoch {epoch} to {output_folder}')


def train_multitask_perceiver_model(
    model: MultitaskPerceiverCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int],
    test_dataset: Dataset,
    output_folder: str,
) -> TrainingMetrics:
    """
    Training loop for MultitaskPerceiverCTM on combined multi-dataset.
    Computes loss across ALL ticks as per CTM paper Section 3.5.
    No load balancing loss (no gating mechanism).
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    metrics = TrainingMetrics(task_names=task_names)
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
        'speech_commands': [6],
        'wine': [7],
    }
    
    # Mapping from dataset to modality for Perceiver encoder
    dataset_modality = {
        'mnist': 'image',
        'cifar': 'image',
        'fashion': 'image',
        'emnist': 'image',
        'speech_commands': 'audio',
        'wine': 'tabular',
    }
    
    for epoch in range(epochs):
        task_losses = {name: 0.0 for name in task_names}
        task_correct = {name: 0 for name in task_names}
        task_total = {name: 0 for name in task_names}
        task_batch_count = {name: 0 for name in task_names}
        
        epoch_start_time = time.time()
        last_checkpoint_time = epoch_start_time
        
        for batch_idx, (stacked_data, labels, sources, batch_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_data[dataset_name] is not None:
                    data = stacked_data[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    modality = dataset_modality[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                # Forward pass returns predictions and certainties (no load_balance_loss)
                                predictions, certainties = model(data, task_idx=model_task_idx, modality=modality)
                                
                                # Compute loss across all ticks (Section 3.5)
                                loss = torch.tensor(0.0, device=device)
                                for tick in range(model.max_ticks):
                                    tick_predictions = predictions[..., tick]
                                    loss = loss + criterion(tick_predictions, target)
                                loss = loss / model.max_ticks
                                
                                total_loss = total_loss + loss
                                
                                task_losses[task_name] += loss.item()
                                task_batch_count[task_name] += 1
                                metrics.batch_losses[task_name].append(loss.item())
                                
                                # Stats using final tick
                                valid_mask = target >= 0
                                if valid_mask.any():
                                    final_predictions = predictions[..., -1]
                                    _, predicted = torch.max(final_predictions[valid_mask].data, 1)
                                    task_total[task_name] += valid_mask.sum().item()
                                    task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
            
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()
            
            if batch_idx % 25 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_checkpoint_time
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Total Loss: {total_loss.item():.4f}, '
                      f'Time: {elapsed_time:.2f}s')
                last_checkpoint_time = current_time
        
        print(f'\n=== Epoch {epoch+1}/{epochs} Summary ===')
        
        for task_name in task_names:
            if task_total[task_name] > 0:
                accuracy = 100. * task_correct[task_name] / task_total[task_name]
                avg_loss = task_losses[task_name] / max(1, task_batch_count[task_name])
                metrics.epoch_losses[task_name].append(avg_loss)
                metrics.epoch_accuracies[task_name].append(accuracy)
                print(f'  {task_name}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
                
                # Log per-task metrics to MLflow
                mlflow.log_metric(f"{task_name}/train_loss", avg_loss, step=epoch + 1)
                mlflow.log_metric(f"{task_name}/train_accuracy", accuracy, step=epoch + 1)
        
        # Generate neural dynamics plots for all tasks at end of each epoch
        generate_epoch_plots_perceiver(
            model=model,
            test_dataset=test_dataset,
            task_names=task_names,
            task_index_mapping=task_index_mapping,
            device=device,
            output_folder=output_folder,
            epoch=epoch + 1,
        )
        model.train()
        print()
    
    return metrics


def evaluate_multitask_perceiver_model(
    model: MultitaskPerceiverCTM,
    test_loader: DataLoader,
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int]
) -> dict[str, float]:
    """Evaluate MultitaskPerceiverCTM on combined multi-dataset test set."""
    model.eval()
    
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
        'speech_commands': [6],
        'wine': [7],
    }
    
    dataset_modality = {
        'mnist': 'image',
        'cifar': 'image',
        'fashion': 'image',
        'emnist': 'image',
        'speech_commands': 'audio',
        'wine': 'tabular',
    }
    
    with torch.no_grad():
        for stacked_data, labels, sources, batch_indices in test_loader:
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_data[dataset_name] is not None:
                    data = stacked_data[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    modality = dataset_modality[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                predictions, _ = model(data, task_idx=model_task_idx, modality=modality)
                                final_predictions = predictions[..., -1]
                                
                                valid_mask = target >= 0
                                _, predicted = torch.max(final_predictions[valid_mask].data, 1)
                                task_total[task_name] += valid_mask.sum().item()
                                task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
    
    test_accuracies = {}
    print('\n=== Test Results ===')
    for task_name in task_names:
        if task_total[task_name] > 0:
            accuracy = 100. * task_correct[task_name] / task_total[task_name]
            test_accuracies[task_name] = accuracy
            print(f'  {task_name}: Accuracy={accuracy:.2f}%')
    print()
    
    return test_accuracies


def plot_training_results(
    metrics: TrainingMetrics,
    test_accuracies: dict[str, float],
    model: MultitaskGatedCTM,
    test_dataset: Dataset,
    task_index_mapping: dict[int, int],
    device: torch.device,
    output_folder: str,
):
    """Generate comprehensive training visualization plots."""
    task_names = metrics.task_names
    color_palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(task_names)}
    
    # Figure 1: Training Loss and Accuracy
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Training Progress per Task (Gated CTM)', fontsize=14, fontweight='bold')
    
    ax_loss = axes1[0]
    for task_name in task_names:
        if metrics.epoch_losses[task_name]:
            epochs = range(1, len(metrics.epoch_losses[task_name]) + 1)
            ax_loss.plot(epochs, metrics.epoch_losses[task_name], 
                        marker='o', label=f'{task_name}', color=colors[task_name], linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    ax_acc = axes1[1]
    for task_name in task_names:
        if metrics.epoch_accuracies[task_name]:
            epochs = range(1, len(metrics.epoch_accuracies[task_name]) + 1)
            ax_acc.plot(epochs, metrics.epoch_accuracies[task_name], 
                       marker='s', label=f'{task_name}', color=colors[task_name], linewidth=2)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Training Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim([0, 105])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_folder, 'fig1_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Figure 2: Batch-level Loss + Balance Loss
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Training Dynamics (Gated CTM)', fontsize=14, fontweight='bold')
    
    ax_batch = axes2[0]
    window_size = 50
    for task_name in task_names:
        losses = metrics.batch_losses[task_name]
        if len(losses) > window_size:
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax_batch.plot(smoothed, label=f'{task_name}', color=colors[task_name], alpha=0.8)
        elif losses:
            ax_batch.plot(losses, label=f'{task_name}', color=colors[task_name], alpha=0.8)
    ax_batch.set_xlabel('Batch')
    ax_batch.set_ylabel('Loss')
    ax_batch.set_title('Batch-level Training Loss (Smoothed)')
    ax_batch.legend()
    ax_batch.grid(True, alpha=0.3)
    
    ax_balance = axes2[1]
    if metrics.epoch_balance_losses:
        epochs = range(1, len(metrics.epoch_balance_losses) + 1)
        ax_balance.plot(epochs, metrics.epoch_balance_losses, marker='o', color='#e74c3c', linewidth=2)
    ax_balance.set_xlabel('Epoch')
    ax_balance.set_ylabel('Balance Loss')
    ax_balance.set_title('Load Balance Loss (Lower = More Balanced Gating)')
    ax_balance.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_folder, 'fig2_batch_losses.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Figure 3: Final Accuracy
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.suptitle('Final Model Performance (Gated CTM)', fontsize=14, fontweight='bold')
    
    x = np.arange(len(task_names))
    width = 0.35
    
    train_accs = [metrics.epoch_accuracies[name][-1] if metrics.epoch_accuracies[name] else 0 for name in task_names]
    test_accs = [test_accuracies.get(name, 0) for name in task_names]
    
    bars1 = ax3.bar(x - width/2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, test_accs, width, label='Test', color='#9b59b6', alpha=0.8)
    
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name.replace('_', '\n') for name in task_names], fontsize=9)
    ax3.legend()
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_folder, 'fig3_final_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print(f'\nFinal plots saved to {output_folder}:')
    print(f'  - fig1_training_curves.png')
    print(f'  - fig2_batch_losses.png')
    print(f'  - fig3_final_accuracy.png')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: MultitaskGatedCTM):
    """Print detailed model information."""
    total_params = count_parameters(model)
    print(f'\n=== MultitaskGatedCTM Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Number of tasks: {model.n_tasks}')
    print(f'Number of sync sets: {model.n_sync_sets}')
    print(f'Sync pairs (output): {model.n_synch_out}')
    print(f'Sync pairs (action): {model.n_synch_action}')
    print(f'Attention heads: {model.n_attention_heads}')
    print(f'Max ticks: {model.max_ticks}')
    print(f'Memory length: {model.max_memory}')
    print(f'Load balance coefficient: {model.load_balance_coef}')
    print(f'Task output sizes: {model.task_output_sizes}')
    
    print(f'\nShared components:')
    shared = ['perceiver_encoder', 'kv_proj', 'q_proj', 'attention', 'synapse_model', 
              'batched_nlm', 'gate_network_action', 'gate_network_out']
    for name in shared:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nTask-specific components:')
    task_specific = ['task_output_projectors']
    for name in task_specific:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nLearnable states:')
    print(f'  start_activated_state: {model.start_activated_state.numel():,} elements')
    print(f'  start_trace: {model.start_trace.numel():,} elements')
    print(f'  set_decay_params_action: {sum(p.numel() for p in model.set_decay_params_action):,} elements')
    print(f'  set_decay_params_out: {sum(p.numel() for p in model.set_decay_params_out):,} elements')
    
    print(f'\nDropout rates:')
    print(f'  perceiver (internal): see perceiver_encoder')
    print(f'  attention: {model._dropout_attention_rate}')
    print(f'  synapse: {model.dropout_synapse.p}')
    print(f'  nlm: {model.dropout_nlm.p}')
    print(f'  sync: {model.dropout_sync.p}')
    print(f'  output: {model._dropout_output_rate}')
    print(f'================================================\n')


def print_model_info_perceiver(model: MultitaskPerceiverCTM):
    """Print detailed model information for MultitaskPerceiverCTM."""
    total_params = count_parameters(model)
    print(f'\n=== MultitaskPerceiverCTM Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Number of tasks: {model.n_tasks}')
    print(f'Sync pairs (output): {model.n_synch_out}')
    print(f'Sync pairs (action): {model.n_synch_action}')
    print(f'Attention heads: {model.n_attention_heads}')
    print(f'Max ticks: {model.max_ticks}')
    print(f'Memory length: {model.max_memory}')
    print(f'Task output sizes: {model.task_output_sizes}')
    
    print(f'\nShared components:')
    shared = ['perceiver_encoder', 'kv_proj', 'q_proj', 'attention', 'synapse_model', 
              'batched_nlm', 'input_to_state']
    for name in shared:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nTask-specific components:')
    task_specific = ['task_output_projectors']
    for name in task_specific:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nLearnable states:')
    print(f'  start_activated_state: {model.start_activated_state.numel():,} elements')
    print(f'  start_trace: {model.start_trace.numel():,} elements')
    print(f'  decay_params_action: {model.decay_params_action.numel():,} elements')
    print(f'  decay_params_out: {model.decay_params_out.numel():,} elements')
    print(f'  input_scale: {model.input_scale.item():.2f} (learnable)')
    
    print(f'\nDropout rates:')
    print(f'  synapse: {model.dropout_synapse.p}')
    print(f'  nlm: {model.dropout_nlm.p}')
    print(f'  sync: {model.dropout_sync.p}')
    print(f'================================================\n')


def train_combined(
    device: torch.device, 
    selected_tasks: list[str], 
    epochs: int, 
    n_neurons: int, 
    n_sync_sets: int,
    dropout_encoder: float,
    dropout_attention: float,
    dropout_synapse: float,
    dropout_nlm: float,
    dropout_sync: float,
    dropout_output: float,
    max_memory: int = 10,
    max_ticks: int = 15,
    n_representation_size: int = 32,
    n_synch_out: int = 32,
    n_synch_action: int = 16,
    n_attention_heads: int = 4,
    load_balance_coef: float = 0.0,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    data_dir: str = './data',
    output_dir: str = '.',
) -> tuple:
    """Train MultitaskGatedCTM on selected tasks."""
    print('\n' + '='*60)
    print(f'TRAINING GATED CTM ON TASKS: {", ".join(selected_tasks)}')
    print('='*60)
    
    # Create output folder for plots
    self_filename = os.path.basename(__file__)
    output_prefix = self_filename.split('.')[0] + '_' + '_'.join([t.split('_')[0][0] + t.split('_')[1][0] for t in selected_tasks]) + '_gated'
    output_folder = os.path.join(output_dir, output_prefix + '_plots')
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')
    
    # Set MLflow tracking URI to sqlite database in output folder
    mlflow_db_path = os.path.join(output_folder, 'mlflow.db')
    # Convert backslashes to forward slashes for URI compatibility on Windows
    db_uri = f'sqlite:///{os.path.abspath(mlflow_db_path).replace(os.sep, "/")}'
    mlflow.set_tracking_uri(db_uri)
    
    # Set experiment with custom artifact location
    experiment_name = "gated_ctm"
    try:
        # Try to create with custom artifact location
        artifact_uri = os.path.join(output_folder, "mlartifacts")
        # Ensure artifact path uses forward slashes for URI compatibility
        artifact_uri = f"file:///{os.path.abspath(artifact_uri).replace(os.sep, '/')}"
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        pass
        
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=output_prefix):
        # Log hyperparameters
        mlflow.log_params({
            "tasks": ",".join(selected_tasks),
            "n_tasks": len(selected_tasks),
            "epochs": epochs,
            "n_neurons": n_neurons,
            "n_sync_sets": n_sync_sets,
            "max_memory": max_memory,
            "max_ticks": max_ticks,
            "n_representation_size": n_representation_size,
            "n_synch_out": n_synch_out,
            "n_synch_action": n_synch_action,
            "n_attention_heads": n_attention_heads,
            "load_balance_coef": load_balance_coef,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout_encoder": dropout_encoder,
            "dropout_attention": dropout_attention,
            "dropout_synapse": dropout_synapse,
            "dropout_nlm": dropout_nlm,
            "dropout_sync": dropout_sync,
            "dropout_output": dropout_output,
        })
        
        needs_mnist = any(ALL_TASKS[t]['dataset'] == 'mnist' for t in selected_tasks)
        needs_cifar = any(ALL_TASKS[t]['dataset'] == 'cifar' for t in selected_tasks)
        needs_fashion = any(ALL_TASKS[t]['dataset'] == 'fashion' for t in selected_tasks)
        needs_emnist = any(ALL_TASKS[t]['dataset'] == 'emnist' for t in selected_tasks)
        needs_speech_commands = any(ALL_TASKS[t]['dataset'] == 'speech_commands' for t in selected_tasks)
        needs_wine = any(ALL_TASKS[t]['dataset'] == 'wine' for t in selected_tasks)
        
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        mnist_train = mnist_test = None
        cifar_train = cifar_test = None
        fashion_train = fashion_test = None
        emnist_train = emnist_test = None
        speech_commands_train = speech_commands_test = None
        wine_train = wine_test = None
        
        if needs_mnist:
            mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
            mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=mnist_transform)
        
        if needs_cifar:
            cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=cifar_transform)
            cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=cifar_transform)
        
        if needs_fashion:
            fashion_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
            fashion_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=mnist_transform)
        
        if needs_emnist:
            emnist_train = datasets.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=mnist_transform)
            emnist_test = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=mnist_transform)
        
        if needs_speech_commands:
            try:
                speech_commands_train = SpeechCommandsDataset(root=data_dir, subset='training', download=True)
                speech_commands_test = SpeechCommandsDataset(root=data_dir, subset='testing', download=True)
                print(f'  Loaded Speech Commands: {len(speech_commands_train)} train, {len(speech_commands_test)} test samples')
            except Exception as e:
                print(f'  Warning: Could not load Speech Commands dataset: {e}')
                speech_commands_train = speech_commands_test = None
        
        if needs_wine:
            wine_train = WineDataset(train=True, test_ratio=0.2)
            wine_test = WineDataset(train=False, test_ratio=0.2)
            print(f'  Loaded Wine: {len(wine_train)} train, {len(wine_test)} test samples')
        
        train_dataset = CombinedMultitaskDataset(
            mnist_train, cifar_train, fashion_train, emnist_train, speech_commands_train, wine_train
        )
        test_dataset = CombinedMultitaskDataset(
            mnist_test, cifar_test, fashion_test, emnist_test, speech_commands_test, wine_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
        task_names = selected_tasks
        task_output_sizes = [ALL_TASKS[t]['output_size'] for t in selected_tasks]
        
        task_index_mapping = {}
        for new_idx, task_name in enumerate(selected_tasks):
            original_idx = all_task_order.index(task_name)
            task_index_mapping[original_idx] = new_idx
        
        print(f'Task configuration:')
        for i, task_name in enumerate(task_names):
            print(f'  Task {i}: {task_name} ({ALL_TASKS[task_name]["description"]})')
        
        model = MultitaskGatedCTM(
            n_neurons=n_neurons,
            max_memory=max_memory, 
            max_ticks=max_ticks, 
            n_representation_size=n_representation_size,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            n_attention_heads=n_attention_heads,
            task_output_sizes=task_output_sizes,
            n_sync_sets=n_sync_sets,
            load_balance_coef=load_balance_coef,
            dropout_encoder=dropout_encoder,
            dropout_attention=dropout_attention,
            dropout_synapse=dropout_synapse,
            dropout_nlm=dropout_nlm,
            dropout_sync=dropout_sync,
            dropout_output=dropout_output,
        ).to(device)
        
        # Log model parameter count
        total_params = count_parameters(model)
        mlflow.log_param("total_parameters", total_params)
        
        print_model_info(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        metrics = train_multitask_gated_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            task_names=task_names,
            task_index_mapping=task_index_mapping,
            test_dataset=test_dataset,
            output_folder=output_folder,
        )
        
        test_accuracies = evaluate_multitask_gated_model(
            model, test_loader, device=device, task_names=task_names,
            task_index_mapping=task_index_mapping
        )
        
        # Log test accuracies
        for task_name, accuracy in test_accuracies.items():
            mlflow.log_metric(f"{task_name}/test_accuracy", accuracy)
        
        # Log average test accuracy
        avg_test_accuracy = sum(test_accuracies.values()) / len(test_accuracies)
        mlflow.log_metric("avg_test_accuracy", avg_test_accuracy)
        
        model_path = os.path.join(output_folder, 'multitask_ctm.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        
        # Log model artifact
        mlflow.log_artifact(model_path)
        
        print('\nGenerating final training plots...')
        plot_training_results(
            metrics=metrics,
            test_accuracies=test_accuracies,
            model=model,
            test_dataset=test_dataset,
            task_index_mapping=task_index_mapping,
            device=device,
            output_folder=output_folder,
        )
        
        # Log plot artifacts
        for plot_file in ['fig1_training_curves.png', 'fig2_batch_losses.png', 'fig3_final_accuracy.png']:
            plot_path = os.path.join(output_folder, plot_file)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path)
        
        print(f'\nMLflow tracking URI: {mlflow.get_tracking_uri()}')
        print(f'MLflow run ID: {mlflow.active_run().info.run_id}')
    
    return model, metrics, test_accuracies


def train_combined_perceiver(
    device: torch.device, 
    selected_tasks: list[str], 
    epochs: int, 
    n_neurons: int, 
    dropout_encoder: float,
    dropout_attention: float,
    dropout_synapse: float,
    dropout_nlm: float,
    dropout_sync: float,
    dropout_output: float,
    max_memory: int = 10,
    max_ticks: int = 15,
    n_representation_size: int = 32,
    n_synch_out: int = 32,
    n_synch_action: int = 16,
    n_attention_heads: int = 4,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    data_dir: str = './data',
    output_dir: str = '.',
) -> tuple:
    """Train MultitaskPerceiverCTM on selected tasks (no gating, Perceiver IO output)."""
    print('\n' + '='*60)
    print(f'TRAINING PERCEIVER CTM ON TASKS: {", ".join(selected_tasks)}')
    print('='*60)
    
    # Create output folder for plots
    self_filename = os.path.basename(__file__)
    output_prefix = self_filename.split('.')[0] + '_' + '_'.join([t.split('_')[0][0] + t.split('_')[1][0] for t in selected_tasks]) + '_perceiver'
    output_folder = os.path.join(output_dir, output_prefix + '_plots')
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')
    
    # Set MLflow tracking URI to sqlite database in output folder
    mlflow_db_path = os.path.join(output_folder, 'mlflow.db')
    db_uri = f'sqlite:///{os.path.abspath(mlflow_db_path).replace(os.sep, "/")}'
    mlflow.set_tracking_uri(db_uri)
    
    # Set experiment
    experiment_name = "perceiver_ctm"
    try:
        artifact_uri = os.path.join(output_folder, "mlartifacts")
        artifact_uri = f"file:///{os.path.abspath(artifact_uri).replace(os.sep, '/')}"
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    except mlflow.exceptions.MlflowException:
        pass
        
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=output_prefix):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "perceiver",
            "tasks": ",".join(selected_tasks),
            "n_tasks": len(selected_tasks),
            "epochs": epochs,
            "n_neurons": n_neurons,
            "max_memory": max_memory,
            "max_ticks": max_ticks,
            "n_representation_size": n_representation_size,
            "n_synch_out": n_synch_out,
            "n_synch_action": n_synch_action,
            "n_attention_heads": n_attention_heads,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout_encoder": dropout_encoder,
            "dropout_attention": dropout_attention,
            "dropout_synapse": dropout_synapse,
            "dropout_nlm": dropout_nlm,
            "dropout_sync": dropout_sync,
            "dropout_output": dropout_output,
        })
        
        needs_mnist = any(ALL_TASKS[t]['dataset'] == 'mnist' for t in selected_tasks)
        needs_cifar = any(ALL_TASKS[t]['dataset'] == 'cifar' for t in selected_tasks)
        needs_fashion = any(ALL_TASKS[t]['dataset'] == 'fashion' for t in selected_tasks)
        needs_emnist = any(ALL_TASKS[t]['dataset'] == 'emnist' for t in selected_tasks)
        needs_speech_commands = any(ALL_TASKS[t]['dataset'] == 'speech_commands' for t in selected_tasks)
        needs_wine = any(ALL_TASKS[t]['dataset'] == 'wine' for t in selected_tasks)
        
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        mnist_train = mnist_test = None
        cifar_train = cifar_test = None
        fashion_train = fashion_test = None
        emnist_train = emnist_test = None
        speech_commands_train = speech_commands_test = None
        wine_train = wine_test = None
        
        if needs_mnist:
            mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
            mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=mnist_transform)
        
        if needs_cifar:
            cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=cifar_transform)
            cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=cifar_transform)
        
        if needs_fashion:
            fashion_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
            fashion_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=mnist_transform)
        
        if needs_emnist:
            emnist_train = datasets.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=mnist_transform)
            emnist_test = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=mnist_transform)
        
        if needs_speech_commands:
            try:
                speech_commands_train = SpeechCommandsDataset(root=data_dir, subset='training', download=True)
                speech_commands_test = SpeechCommandsDataset(root=data_dir, subset='testing', download=True)
                print(f'  Loaded Speech Commands: {len(speech_commands_train)} train, {len(speech_commands_test)} test samples')
            except Exception as e:
                print(f'  Warning: Could not load Speech Commands dataset: {e}')
                speech_commands_train = speech_commands_test = None
        
        if needs_wine:
            wine_train = WineDataset(train=True, test_ratio=0.2)
            wine_test = WineDataset(train=False, test_ratio=0.2)
            print(f'  Loaded Wine: {len(wine_train)} train, {len(wine_test)} test samples')
        
        train_dataset = CombinedMultitaskDataset(
            mnist_train, cifar_train, fashion_train, emnist_train, speech_commands_train, wine_train
        )
        test_dataset = CombinedMultitaskDataset(
            mnist_test, cifar_test, fashion_test, emnist_test, speech_commands_test, wine_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine', 'speech_commands', 'wine_type']
        task_names = selected_tasks
        task_output_sizes = [ALL_TASKS[t]['output_size'] for t in selected_tasks]
        
        task_index_mapping = {}
        for new_idx, task_name in enumerate(selected_tasks):
            original_idx = all_task_order.index(task_name)
            task_index_mapping[original_idx] = new_idx
        
        print(f'Task configuration:')
        for i, task_name in enumerate(task_names):
            print(f'  Task {i}: {task_name} ({ALL_TASKS[task_name]["description"]})')
        
        model = MultitaskPerceiverCTM(
            n_neurons=n_neurons,
            max_memory=max_memory, 
            max_ticks=max_ticks, 
            n_representation_size=n_representation_size,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            n_attention_heads=n_attention_heads,
            task_output_sizes=task_output_sizes,
            dropout_encoder=dropout_encoder,
            dropout_attention=dropout_attention,
            dropout_synapse=dropout_synapse,
            dropout_nlm=dropout_nlm,
            dropout_sync=dropout_sync,
            dropout_output=dropout_output,
        ).to(device)
        
        # Log model parameter count
        total_params = count_parameters(model)
        mlflow.log_param("total_parameters", total_params)
        
        print_model_info_perceiver(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        metrics = train_multitask_perceiver_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            task_names=task_names,
            task_index_mapping=task_index_mapping,
            test_dataset=test_dataset,
            output_folder=output_folder,
        )
        
        test_accuracies = evaluate_multitask_perceiver_model(
            model, test_loader, device=device, task_names=task_names,
            task_index_mapping=task_index_mapping
        )
        
        # Log test accuracies
        for task_name, accuracy in test_accuracies.items():
            mlflow.log_metric(f"{task_name}/test_accuracy", accuracy)
        
        # Log average test accuracy
        avg_test_accuracy = sum(test_accuracies.values()) / len(test_accuracies)
        mlflow.log_metric("avg_test_accuracy", avg_test_accuracy)
        
        model_path = os.path.join(output_folder, 'multitask_perceiver_ctm.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        
        # Log model artifact
        mlflow.log_artifact(model_path)
        
        print(f'\nMLflow tracking URI: {mlflow.get_tracking_uri()}')
        print(f'MLflow run ID: {mlflow.active_run().info.run_id}')
    
    return model, metrics, test_accuracies


@modal_app.function(
    image=modal_image,
    gpu="T4",
    timeout=36000,
    volumes={"/data": modal.Volume.from_name("ctm-data", create_if_missing=True)},
)
def train_on_modal(
    selected_tasks: list[str],
    epochs: int,
    n_neurons: int,
    n_sync_sets: int,
    dropout_encoder: float,
    dropout_attention: float,
    dropout_synapse: float,
    dropout_nlm: float,
    dropout_sync: float,
    dropout_output: float,
) -> dict:
    """
    Modal GPU training function.
    Runs training on Modal's cloud infrastructure with GPU support.
    
    Returns training metrics and test accuracies.
    """
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Modal training started on device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Use Modal volume for data persistence
    data_dir = '/data/datasets'
    output_dir = '/data/outputs'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    model, metrics, test_accuracies = train_combined(
        device=device,
        selected_tasks=selected_tasks,
        epochs=epochs,
        n_neurons=n_neurons,
        n_sync_sets=n_sync_sets,
        dropout_encoder=dropout_encoder,
        dropout_attention=dropout_attention,
        dropout_synapse=dropout_synapse,
        dropout_nlm=dropout_nlm,
        dropout_sync=dropout_sync,
        dropout_output=dropout_output,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    
    # Return serializable results
    return {
        'test_accuracies': test_accuracies,
        'final_train_accuracies': {
            name: metrics.epoch_accuracies[name][-1] 
            for name in metrics.task_names 
            if metrics.epoch_accuracies[name]
        },
        'tasks': selected_tasks,
        'epochs': epochs,
        'n_neurons': n_neurons,
        'n_sync_sets': n_sync_sets,
    }


@modal_app.local_entrypoint()
def modal_entrypoint(
    tasks: str = "fashion_fine,emnist_fine,cifar_fine",
    epochs: int = 5,
    neurons: int = 64,
    sync_sets: int = 4,
    dropout_encoder: float = 0.1,
    dropout_attention: float = 0.1,
    dropout_synapse: float = 0.1,
    dropout_nlm: float = 0.1,
    dropout_sync: float = 0.1,
    dropout_output: float = 0.2,
):
    """
    Modal entrypoint for running training from command line.
    
    Usage:
        modal run ctm_extended.py --tasks "fashion_fine,cifar_fine" --epochs 10
        modal run ctm_extended.py --dropout-encoder 0.2 --dropout-output 0.3
    """
    # Parse tasks: handle both comma-separated and space-separated formats
    selected_tasks = [t.strip() for t in tasks.replace(',', ' ').split() if t.strip()]
    
    # Validate tasks
    for task in selected_tasks:
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {list(ALL_TASKS.keys())}")
    
    print(f'\n{"="*60}')
    print('STARTING MODAL GPU TRAINING')
    print(f'{"="*60}')
    print(f'Tasks: {", ".join(selected_tasks)}')
    print(f'Epochs: {epochs}')
    print(f'Neurons: {neurons}')
    print(f'Sync sets: {sync_sets}')
    print(f'Dropout - encoder: {dropout_encoder}, attention: {dropout_attention}')
    print(f'Dropout - synapse: {dropout_synapse}, nlm: {dropout_nlm}')
    print(f'Dropout - sync: {dropout_sync}, output: {dropout_output}')
    print(f'{"="*60}\n')
    
    # Run training on Modal GPU
    results = train_on_modal.remote(
        selected_tasks=selected_tasks,
        epochs=epochs,
        n_neurons=neurons,
        n_sync_sets=sync_sets,
        dropout_encoder=dropout_encoder,
        dropout_attention=dropout_attention,
        dropout_synapse=dropout_synapse,
        dropout_nlm=dropout_nlm,
        dropout_sync=dropout_sync,
        dropout_output=dropout_output,
    )
    
    print(f'\n{"="*60}')
    print('MODAL TRAINING COMPLETE')
    print(f'{"="*60}')
    print('\nResults:')
    for task, acc in results['test_accuracies'].items():
        print(f'  {task}: {acc:.2f}%')
    
    return results


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CTM on selected tasks (gated or perceiver variant)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
  mnist_digit      - MNIST digit classification (0-9)
  mnist_even_odd   - MNIST even/odd classification
  cifar_fine       - CIFAR-10 fine classification (10 classes)
  cifar_coarse     - CIFAR-10 coarse (animal vs vehicle)
  fashion_fine     - Fashion MNIST classification (10 classes)
  emnist_fine      - EMNIST balanced classification (47 classes)
  speech_commands  - Speech Commands (35 spoken words, audio)
  wine_type        - Wine classification (3 types, tabular)

Model variants:
  gated     - MultitaskGatedCTM with gated sync sets and fixed output heads
  perceiver - MultitaskPerceiverCTM with single sync set and Perceiver IO decoder

Examples (local):
  py ctm_extended.py --tasks "fashion_fine emnist_fine cifar_fine"
  py ctm_extended.py --tasks "fashion_fine,emnist_fine,cifar_fine" --model perceiver
  py ctm_extended.py --tasks "mnist_digit cifar_fine" --sync_sets 8
  py ctm_extended.py --tasks "fashion_fine speech_commands wine_type" --epochs 10 --model perceiver

Examples (Modal cloud GPU):
  modal run ctm_extended.py --tasks "fashion_fine,cifar_fine" --epochs 10
  modal run ctm_extended.py --tasks "fashion_fine,speech_commands,wine_type" --epochs 5
  modal run ctm_extended.py --tasks "mnist_digit,cifar_fine" --neurons 128
        """
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default='fashion_fine,emnist_fine,cifar_fine',
        help='Tasks to train on (comma-separated or space-separated, default: fashion_fine,emnist_fine,cifar_fine)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--neurons',
        type=int,
        default=64,
        help='Number of neurons in CTM (default: 64)'
    )
    
    parser.add_argument(
        '--sync_sets',
        type=int,
        default=4,
        help='Number of synchronization pair sets (default: 4, only for gated model)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gated',
        choices=['gated', 'perceiver'],
        help='Model variant: gated (with gate networks) or perceiver (Perceiver IO output, no gating)'
    )
    
    parser.add_argument(
        '--modal',
        action='store_true',
        help='Run training on Modal cloud GPU instead of local'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        default='T4',
        choices=['T4', 'A10G', 'A100', 'H100'],
        help='Modal GPU type (default: T4). Only used with --modal flag'
    )
    
    # Dropout arguments
    parser.add_argument(
        '--dropout_encoder',
        type=float,
        default=0.1,
        help='Dropout in Perceiver encoder (default: 0.1)'
    )
    
    parser.add_argument(
        '--dropout_attention',
        type=float,
        default=0.1,
        help='Dropout in multi-head attention (default: 0.1)'
    )
    
    parser.add_argument(
        '--dropout_synapse',
        type=float,
        default=0.1,
        help='Dropout after synapse model (default: 0.1)'
    )
    
    parser.add_argument(
        '--dropout_nlm',
        type=float,
        default=0.1,
        help='Dropout after neuron-level models (default: 0.1)'
    )
    
    parser.add_argument(
        '--dropout_sync',
        type=float,
        default=0.1,
        help='Dropout after synchronization (default: 0.1)'
    )
    
    parser.add_argument(
        '--dropout_output',
        type=float,
        default=0.1,
        help='Dropout in output projectors (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Parse tasks: handle both comma-separated and space-separated formats
    if isinstance(args.tasks, str):
        # Split by comma or space
        tasks_list = [t.strip() for t in args.tasks.replace(',', ' ').split() if t.strip()]
    else:
        tasks_list = args.tasks
    
    # Validate tasks
    valid_tasks = list(ALL_TASKS.keys())
    invalid_tasks = [t for t in tasks_list if t not in valid_tasks]
    if invalid_tasks:
        print(f"Error: Invalid tasks: {invalid_tasks}")
        print(f"Valid tasks: {', '.join(valid_tasks)}")
        return
    
    if not tasks_list:
        print("Error: At least one task must be selected")
        return
    
    if len(tasks_list) != len(set(tasks_list)):
        print("Error: Duplicate tasks detected")
        return
    
    # Replace args.tasks with parsed list
    args.tasks = tasks_list
    
    if args.modal:
        # Run on Modal cloud
        print(f'\n{"="*60}')
        print('RUNNING ON MODAL CLOUD GPU')
        print(f'{"="*60}')
        print(f'GPU type: {args.gpu}')
        print(f'Selected tasks: {", ".join(args.tasks)}')
        print(f'Epochs: {args.epochs}')
        print(f'Neurons: {args.neurons}')
        print(f'Sync sets: {args.sync_sets}')
        print(f'Dropout - encoder: {args.dropout_encoder}, attention: {args.dropout_attention}')
        print(f'Dropout - synapse: {args.dropout_synapse}, nlm: {args.dropout_nlm}')
        print(f'Dropout - sync: {args.dropout_sync}, output: {args.dropout_output}')
        
        # Dynamically update GPU type if different from default
        with modal_app.run():
            results = train_on_modal.remote(
                selected_tasks=args.tasks,
                epochs=args.epochs,
                n_neurons=args.neurons,
                n_sync_sets=args.sync_sets,
                dropout_encoder=args.dropout_encoder,
                dropout_attention=args.dropout_attention,
                dropout_synapse=args.dropout_synapse,
                dropout_nlm=args.dropout_nlm,
                dropout_sync=args.dropout_sync,
                dropout_output=args.dropout_output,
            )
        
        print(f'\n{"="*60}')
        print('MODAL TRAINING COMPLETE')
        print(f'{"="*60}')
        print('\nResults:')
        for task, acc in results['test_accuracies'].items():
            print(f'  {task}: {acc:.2f}%')
    else:
        # Run locally
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        print(f'Model type: {args.model}')
        print(f'Selected tasks: {", ".join(args.tasks)}')
        print(f'Epochs: {args.epochs}')
        print(f'Neurons: {args.neurons}')
        if args.model == 'gated':
            print(f'Sync sets: {args.sync_sets}')
        print(f'Dropout - encoder: {args.dropout_encoder}, attention: {args.dropout_attention}')
        print(f'Dropout - synapse: {args.dropout_synapse}, nlm: {args.dropout_nlm}')
        print(f'Dropout - sync: {args.dropout_sync}, output: {args.dropout_output}')
        
        if args.model == 'gated':
            train_combined(
                device=device, 
                selected_tasks=args.tasks, 
                epochs=args.epochs, 
                n_neurons=args.neurons, 
                n_sync_sets=args.sync_sets,
                dropout_encoder=args.dropout_encoder,
                dropout_attention=args.dropout_attention,
                dropout_synapse=args.dropout_synapse,
                dropout_nlm=args.dropout_nlm,
                dropout_sync=args.dropout_sync,
                dropout_output=args.dropout_output,
            )
        else:  # perceiver
            train_combined_perceiver(
                device=device, 
                selected_tasks=args.tasks, 
                epochs=args.epochs, 
                n_neurons=args.neurons, 
                dropout_encoder=args.dropout_encoder,
                dropout_attention=args.dropout_attention,
                dropout_synapse=args.dropout_synapse,
                dropout_nlm=args.dropout_nlm,
                dropout_sync=args.dropout_sync,
                dropout_output=args.dropout_output,
            )
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
