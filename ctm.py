import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import numpy as np
import wandb
from save_utils import (
    save_model_components,
    load_model_components,
    save_checkpoint,
    load_checkpoint,
    create_model_from_checkpoint,
)


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


class BatchedNLM(nn.Module):
    """
    Batched Neuron-Level Model with per-neuron private parameters (paper-faithful).
    
    Each neuron has its own unique weight parameters as specified in CTM paper
    Section 3.3 "Privately-Parameterized Neuron-Level Models". Uses einsum for
    efficient batched matrix multiplication across all neurons in parallel.
    
    Architecture per neuron: Linear(memory, 256) -> GLU -> Linear(128, 2) -> GLU -> 1
    Matches the paper's double-GLU architecture with temperature scaling.
    
    Reference: https://arxiv.org/abs/2505.05522
    """
    
    def __init__(self, n_neurons: int, max_memory: int, dropout: float = 0.0):
        super(BatchedNLM, self).__init__()
        self.n_neurons = n_neurons
        self.max_memory = max_memory
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Per-neuron weights for fc1: each neuron has its own (max_memory -> 256) projection
        # Shape: (n_neurons, max_memory, 256) - 256 because GLU halves it to 128
        self.fc1_weight = nn.Parameter(
            torch.randn(n_neurons, max_memory, 256) * math.sqrt(2.0 / max_memory)
        )
        self.fc1_bias = nn.Parameter(torch.zeros(n_neurons, 256))
        
        # Per-neuron weights for fc2: each neuron has its own (128 -> 2) projection
        # Outputs 2 values which will be gated by GLU to produce 1 value
        # Shape: (n_neurons, 128, 2)
        self.fc2_weight = nn.Parameter(
            torch.randn(n_neurons, 128, 2) * math.sqrt(2.0 / 128)
        )
        self.fc2_bias = nn.Parameter(torch.zeros(n_neurons, 2))
        
        # Learnable temperature parameter for scaling (paper Section 3.3)
        self.register_parameter('T', nn.Parameter(torch.ones(1)))
    
    def forward(self, state_trace: torch.Tensor, track: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process all neuron traces in parallel with per-neuron private weights.
        
        Args:
            state_trace: (batch, neurons, memory) tensor of neuron activation histories
            track: If True, return intermediate activations for visualization
        
        Returns:
            If track=False: (batch, neurons) post-activations for all neurons
            If track=True: tuple of (post-activations, dict of intermediate activations)
        """
        # Apply dropout to input
        x_input = self.dropout(state_trace)
        
        # fc1: (B, N, M) @ (N, M, 256) -> (B, N, 256)
        # einsum performs batched matmul where each neuron uses its own weights
        x_fc1 = torch.einsum('bnm,nmh->bnh', x_input, self.fc1_weight) + self.fc1_bias
        
        # First GLU activation: splits last dim in half, applies sigmoid gate
        # (B, N, 256) -> (B, N, 128)
        x_glu1 = F.glu(x_fc1, dim=-1)
        
        # fc2: (B, N, 128) @ (N, 128, 2) -> (B, N, 2)
        x_fc2 = torch.einsum('bnh,nho->bno', x_glu1, self.fc2_weight) + self.fc2_bias
        
        # Second GLU: gates the output (learned confidence mechanism)
        # (B, N, 2) -> (B, N, 1)
        x_glu2 = F.glu(x_fc2, dim=-1)
        
        # Apply temperature scaling and squeeze: (B, N, 1) -> (B, N)
        x_output = (x_glu2.squeeze(-1) / self.T)
        
        if track:
            activations = {
                'input': x_input,           # (B, N, M) - after dropout
                'fc1': x_fc1,               # (B, N, 256) - after first linear layer
                'glu1': x_glu1,             # (B, N, 128) - after first GLU
                'fc2': x_fc2,               # (B, N, 2) - after second linear layer
                'glu2': x_glu2,             # (B, N, 1) - after second GLU
                'output': x_output,         # (B, N) - final output after temperature scaling
            }
            return x_output, activations
        
        return x_output


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them - this is the KEY for foveated attention!
    
    For a 28x28 MNIST image with patch_size=7:
    - Creates a 4x4 = 16 patch grid
    - Each patch is 7x7 = 49 pixels
    - Attention can now focus on different patches at each tick
    """
    
    def __init__(self, input_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.n_patches = (input_size // patch_size) ** 2
        self.grid_size = input_size // patch_size
        
        # Conv2d with kernel_size=stride=patch_size acts as patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image
        Returns:
            (B, n_patches, embed_dim) patch embeddings
        """
        # (B, C, H, W) -> (B, embed_dim, grid_h, grid_w)
        x = self.proj(x)
        # (B, embed_dim, grid_h, grid_w) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbedding1D(nn.Module):
    """
    Split 1D input into patches and embed them.

    Expected input shape: (B, C, L)
    Output shape: (B, n_patches, embed_dim)
    """

    def __init__(self, input_length: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding1D, self).__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.n_patches = input_length // patch_size

        if self.n_patches < 1:
            raise ValueError(
                f"Invalid 1D patching: input_length={input_length}, patch_size={patch_size} -> n_patches={self.n_patches}"
            )

        # Conv1d with kernel_size=stride=patch_size acts as patch extraction + linear projection
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) input sequence/features
        Returns:
            (B, n_patches, embed_dim) patch embeddings
        """
        if x.dim() != 3:
            raise ValueError(f"PatchEmbedding1D expects input of shape (B, C, L), got shape={tuple(x.shape)}")

        # (B, C, L) -> (B, embed_dim, n_patches)
        x = self.proj(x)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbedding3D(nn.Module):
    """
    Split 3D point cloud into patches and embed them using PointNet-style feature extraction.
    
    Groups consecutive points into patches and uses MLP + max pooling to extract patch-level features.
    This is more efficient than byte-level tokens while preserving spatial structure.
    
    Expected input shape: (B, 3, num_points) where 3 = xyz coordinates
    Output shape: (B, n_patches, embed_dim)
    """

    def __init__(self, num_points: int, patch_size: int, embed_dim: int):
        """
        Initialize 3D patch embedding.
        
        Args:
            num_points: Number of points in the point cloud
            patch_size: Number of points per patch
            embed_dim: Embedding dimension for output tokens
        """
        super(PatchEmbedding3D, self).__init__()
        self.num_points = num_points
        self.patch_size = patch_size
        self.n_patches = num_points // patch_size
        self.embed_dim = embed_dim

        if self.n_patches < 1:
            raise ValueError(
                f"Invalid 3D patching: num_points={num_points}, patch_size={patch_size} -> n_patches={self.n_patches}"
            )

        # PointNet-style feature extraction: MLP to extract per-point features, then max pool
        # Input: 3D coordinates (xyz) -> extract features -> max pool over patch
        self.point_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert point cloud to patch embeddings.
        
        Args:
            x: (B, 3, num_points) point cloud tensor
        Returns:
            (B, n_patches, embed_dim) patch embeddings
        """
        if x.dim() != 3:
            raise ValueError(f"PatchEmbedding3D expects input of shape (B, 3, num_points), got shape={tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"PatchEmbedding3D expects 3 channels (xyz), got {x.shape[1]} channels")
        
        batch_size = x.shape[0]
        
        # Reshape to group points into patches: (B, 3, num_points) -> (B, n_patches, patch_size, 3)
        # First transpose to (B, num_points, 3)
        x = x.transpose(1, 2)  # (B, num_points, 3)
        
        # Reshape to patches: (B, num_points, 3) -> (B, n_patches, patch_size, 3)
        x = x[:, :self.n_patches * self.patch_size, :]  # Trim to exact multiple
        x = x.reshape(batch_size, self.n_patches, self.patch_size, 3)
        
        # Extract features for each point in each patch: (B, n_patches, patch_size, 3) -> (B, n_patches, patch_size, embed_dim)
        x = self.point_mlp(x)
        
        # Max pool over patch dimension to get patch-level features: (B, n_patches, patch_size, embed_dim) -> (B, n_patches, embed_dim)
        patch_embeddings = x.max(dim=2)[0]  # Max pooling over patch_size dimension
        
        # Apply final normalization
        patch_embeddings = self.norm(patch_embeddings)
        
        return patch_embeddings


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable 2D positional embeddings for patches.
    Helps the model understand spatial relationships between patches.
    """
    
    def __init__(self, n_patches: int, embed_dim: int):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, embed_dim) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class PerceiverPatchEmbedding(nn.Module):
    """
    Perceiver-style patch embedding with byte-level tokenization and Fourier positional encodings.
    
    Based on the original Perceiver paper (Jaegle et al., 2021):
    - Converts any input to byte-level tokens (flatten all dimensions except batch to bytes)
    - Uses Fourier positional encodings (sinusoidal) based on byte positions
    - Projects to embedding dimension
    
    Supports inputs of any dimensionality:
    - 1D: (B, C, L) -> flattens to (B, C*L) bytes
    - 2D: (B, C, H, W) -> flattens to (B, C*H*W) bytes
    - etc.
    
    Reference: https://arxiv.org/abs/2103.03206
    """
    
    def __init__(self, input_shape: tuple[int, ...], embed_dim: int):
        """
        Initialize Perceiver patch embedding.
        
        Args:
            input_shape: Shape of input tensor excluding batch dimension, e.g., (C, H, W) or (C, L)
            embed_dim: Embedding dimension for output tokens
        """
        super(PerceiverPatchEmbedding, self).__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        
        # Total number of bytes: product of all dimensions
        self.n_bytes = int(np.prod(input_shape))
        
        # Linear projection from byte values to embedding dimension
        self.byte_proj = nn.Linear(1, embed_dim)
        
        # Layer norm after projection
        self.norm = nn.LayerNorm(embed_dim)
        
        # Pre-compute Fourier positional encodings (Perceiver-style)
        # Using sinusoidal encodings based on byte positions
        self.register_buffer('fourier_pos_embed', self._create_fourier_pos_encoding())
    
    def _create_fourier_pos_encoding(self) -> torch.Tensor:
        """
        Create Fourier positional encodings for byte positions.
        Uses sinusoidal functions like in the Perceiver paper.
        
        Returns:
            (1, n_bytes, embed_dim) positional encoding tensor
        """
        pos_embed = torch.zeros(1, self.n_bytes, self.embed_dim)
        position = torch.arange(0, self.n_bytes, dtype=torch.float32).unsqueeze(1)
        
        # Create sinusoidal encodings
        # Using different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * 
                            -(math.log(10000.0) / self.embed_dim))
        
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert any input to byte-level tokens with Fourier positional encodings.
        
        Args:
            x: Input tensor of shape (B, ...) where ... can be any dimensions
               Examples:
               - 1D: (B, C, L)
               - 2D: (B, C, H, W)
        Returns:
            (B, n_bytes, embed_dim) byte token embeddings with positional encoding
        """
        batch_size = x.shape[0]
        
        # Flatten all dimensions except batch to byte-level tokens
        # (B, ...) -> (B, n_bytes) where n_bytes = product of all non-batch dimensions
        byte_tokens = x.flatten(1)  # (B, n_bytes)
        
        # Reshape to treat each byte as a separate token: (B, n_bytes, 1)
        byte_tokens = byte_tokens.unsqueeze(-1)  # (B, n_bytes, 1)
        
        # Project bytes to embedding dimension: (B, n_bytes, 1) -> (B, n_bytes, embed_dim)
        byte_embeddings = self.byte_proj(byte_tokens)
        
        # Add Fourier positional encodings (Perceiver-style)
        byte_embeddings = byte_embeddings + self.fourier_pos_embed
        
        # Apply layer norm
        byte_embeddings = self.norm(byte_embeddings)
        
        return byte_embeddings


class SimplifiedCTM(nn.Module):
    """
    Simplified Continuous Thought Machine with PROPER spatial attention.
    
    Based on: https://arxiv.org/abs/2505.05522
    Reference: https://github.com/SakanaAI/continuous-thought-machines
    
    Key components:
    1. Internal temporal axis (ticks) for thought unfolding
    2. Neuron-level models (NLMs) with private parameters per neuron
    3. Neural synchronization as representation for output and action
    4. Recursive synchronization computation (O(1) per tick)
    5. **SPATIAL PATCHES** for foveated attention - the key difference!
    """
    
    def __init__(
        self, 
        n_neurons: int, 
        max_memory: int,
        max_ticks: int,
        d_input: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        out_dims: int,
        input_size: int,
        patch_size: int,
        in_channels: int,
        input_ndim: int = 2,
        dropout: float = 0.0,
        dropout_nlm: float = 0.0,
        use_perceiver: bool = False,
    ):
        super(SimplifiedCTM, self).__init__()

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.out_dims = out_dims
        self.n_attention_heads = n_attention_heads
        self.patch_size = patch_size
        self.input_size = input_size
        self.input_ndim = input_ndim
        self.use_perceiver = use_perceiver

        if self.use_perceiver:
            # Perceiver uses byte-level tokens: flatten all dimensions to bytes
            # Compute input shape based on input_ndim
            if self.input_ndim == 1:
                # 1D: (C, L) where L = input_size
                input_shape = (in_channels, input_size)
            elif self.input_ndim == 2:
                # 2D: (C, H, W) where H = W = input_size
                input_shape = (in_channels, input_size, input_size)
            elif self.input_ndim == 3:
                # 3D: (3, num_points) where num_points = input_size
                input_shape = (in_channels, input_size)
            else:
                raise ValueError(f"Invalid input_ndim: {input_ndim}. Expected 1, 2, or 3.")
            self.n_patches = int(np.prod(input_shape))
        elif self.input_ndim == 2:
            self.n_patches = (input_size // patch_size) ** 2
        elif self.input_ndim == 1:
            self.n_patches = input_size // patch_size
            if self.n_patches < 1:
                raise ValueError(
                    f"Invalid 1D patching: input_size(input_length)={input_size}, patch_size={patch_size} -> n_patches={self.n_patches}"
                )
        elif self.input_ndim == 3:
            # 3D point cloud: group points into patches
            self.n_patches = input_size // patch_size
            if self.n_patches < 1:
                raise ValueError(
                    f"Invalid 3D patching: num_points={input_size}, patch_size={patch_size} -> n_patches={self.n_patches}"
                )
        else:
            raise ValueError(f"Invalid input_ndim: {input_ndim}. Expected 1, 2, or 3.")
        
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

        if self.use_perceiver:
            # Perceiver patch embedding with byte-level tokenization and Fourier positional encodings
            # Compute input shape based on input_ndim
            if self.input_ndim == 1:
                input_shape = (in_channels, input_size)
            elif self.input_ndim == 2:
                input_shape = (in_channels, input_size, input_size)
            elif self.input_ndim == 3:
                input_shape = (in_channels, input_size)
            else:
                raise ValueError(f"Invalid input_ndim: {input_ndim}. Expected 1, 2, or 3.")
            
            self.patch_embed = PerceiverPatchEmbedding(
                input_shape=input_shape,
                embed_dim=d_input,
            )
            # Perceiver embedding already includes positional encoding, so no separate pos_embed needed
            self.pos_embed = None
        elif self.input_ndim == 2:
            self.patch_embed = PatchEmbedding(
                input_size=input_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=d_input,
            )
            # Positional embeddings help model understand patch locations
            self.pos_embed = LearnablePositionalEmbedding(
                n_patches=self.n_patches,
                embed_dim=d_input
            )
        elif self.input_ndim == 3:
            # 3D point cloud patch embedding
            self.patch_embed = PatchEmbedding3D(
                num_points=input_size,
                patch_size=patch_size,
                embed_dim=d_input,
            )
            # Positional embeddings help model understand patch locations
            self.pos_embed = LearnablePositionalEmbedding(
                n_patches=self.n_patches,
                embed_dim=d_input
            )
        else:
            self.patch_embed = PatchEmbedding1D(
                input_length=input_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=d_input,
            )
            # Positional embeddings help model understand patch locations
            self.pos_embed = LearnablePositionalEmbedding(
                n_patches=self.n_patches,
                embed_dim=d_input
            )
        
        # Key-Value projection for attention (from patch embeddings)
        self.kv_proj = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.LayerNorm(d_input)
        )
        
        # Query projection from synchronization_action
        self.q_proj = nn.Linear(n_synch_action, d_input)
        
        # Multi-head attention (Section 3.4)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_input,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Synapse model - combines attention output with current activated state
        synapse_input_size = d_input + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),
            nn.LayerNorm(n_neurons),
        )

        # Batched Neuron-level model (NLM) with dropout support
        dropout_nlm_actual = dropout if dropout_nlm == 0.0 else dropout_nlm
        self.batched_nlm = BatchedNLM(n_neurons=n_neurons, max_memory=max_memory, dropout=dropout_nlm_actual)

        # Synchronization neuron pairs - Section 3.4.1 (random-pairing strategy)
        self._init_synchronization_pairs()
        
        # Learnable decay parameters for synchronization (Appendix H)
        self.register_parameter(
            'decay_params_action',
            nn.Parameter(torch.zeros(n_synch_action), requires_grad=True)
        )
        self.register_parameter(
            'decay_params_out',
            nn.Parameter(torch.zeros(n_synch_out), requires_grad=True)
        )

        # Output projector - from synchronization_out to predictions
        # Paper uses simple single linear layer (sync representation is already rich)
        self.output_projector = nn.Linear(n_synch_out, out_dims)
    
    def _init_synchronization_pairs(self):
        """
        Initialize neuron pairs for synchronization using random-pairing strategy.
        Creates separate pairs for output (S_out) and action (S_action).
        """
        # Output synchronization pairs
        out_indices_left = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
        )
        out_indices_right = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out, replace=True)
        )
        self.register_buffer('out_neuron_indices_left', out_indices_left)
        self.register_buffer('out_neuron_indices_right', out_indices_right)
        
        # Action synchronization pairs  
        action_indices_left = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
        )
        action_indices_right = torch.from_numpy(
            np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action, replace=True)
        )
        self.register_buffer('action_neuron_indices_left', action_indices_left)
        self.register_buffer('action_neuron_indices_right', action_indices_right)
    
    def compute_synchronization(
        self,
        activated_state: torch.Tensor,
        decay_alpha: torch.Tensor,
        decay_beta: torch.Tensor,
        r: torch.Tensor,
        synch_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute synchronization using efficient recursive formula (Appendix H).
        """
        if synch_type == 'action':
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")
        
        left = activated_state[:, neuron_indices_left]
        right = activated_state[:, neuron_indices_right]
        pairwise_product = left * right
        
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronization = decay_alpha / torch.sqrt(decay_beta + 1e-8)
        
        return synchronization, decay_alpha, decay_beta
    
    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute certainty as 1 - normalized_entropy.
        """
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial patch features for attention.
        This is where the FOVEATED VIEW comes from!
        
        Returns:
            kv: (batch_size, n_patches, d_input) - multiple tokens to attend over
        """
        # Extract and embed patches: (B, C, H, W) -> (B, n_patches, d_input)
        patch_embeddings = self.patch_embed(x)
        
        # Add positional information (skip for Perceiver as it's already included)
        if self.pos_embed is not None:
            patch_embeddings = self.pos_embed(patch_embeddings)
        
        # Project to key-value space
        kv = self.kv_proj(patch_embeddings)  # (B, n_patches, d_input)
        
        return kv
    
    def forward(self, x: torch.Tensor, track: bool = False):
        """
        Forward pass through the CTM with proper spatial attention.
        """
        batch_size = x.shape[0]
        device = x.device
        
        kv = self.compute_features(x)  # (batch_size, n_patches, d_input)
        
        # Initialize recurrent state from learnable parameters
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Prepare storage for outputs
        predictions = torch.empty(batch_size, self.out_dims, self.max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.max_ticks, device=device, dtype=torch.float32)
        
        # Initialize synchronization recurrence values
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        
        # Clamp decay parameters
        self.decay_params_action.data = torch.clamp(self.decay_params_action.data, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out.data, 0, 15)
        
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).expand(batch_size, -1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(batch_size, -1)
        
        # Initialize output sync
        _, decay_alpha_out, decay_beta_out = self.compute_synchronization(
            activated_state, None, None, r_out, synch_type='out'
        )
        
        # Tracking for visualization
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            attention_tracking = []
            nlm_activations_tracking = []
        
        # Recurrent loop over internal ticks
        for tick in range(self.max_ticks):
            # Calculate synchronization for action
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # Query from action synchronization
            q = self.q_proj(sync_action).unsqueeze(1)  # (batch_size, 1, d_input)
            
            # ============================================================
            # FOVEATED ATTENTION: Query attends over ALL patches
            # At each tick, the attention weights change based on sync_action
            # This creates the "looking around" behavior!
            # ============================================================
            attn_out, attn_weights = self.attention(
                q, kv, kv, 
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights for visualization
            )
            attn_out = attn_out.squeeze(1)  # (batch_size, d_input)
            
            # Combine attention output with current state
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            
            # Apply synapse model
            new_state = self.synapse_model(pre_synapse_input)
            
            # Update state trace (FIFO buffer)
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply NLM
            if track:
                activated_state, nlm_activations = self.batched_nlm(state_trace, track=True)
                nlm_activations_tracking.append({
                    key: value.detach().cpu().numpy() 
                    for key, value in nlm_activations.items()
                })
            else:
                activated_state = self.batched_nlm(state_trace)
            
            # Calculate synchronization for output
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Get predictions
            current_prediction = self.output_projector(sync_out)
            current_certainty = self.compute_certainty(current_prediction)
            
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
                nlm_activations_tracking
            )
        
        return predictions, certainties


def visualize_attention(model: SimplifiedCTM, image: torch.Tensor, device: torch.device, use_wandb: bool = False, epoch: int | None = None) -> str:
    """
    Visualize how attention weights evolve across ticks.
    This shows the foveated attention behavior!
    
    Args:
        model: The CTM model
        image: Input image tensor
        device: Device to run on
        use_wandb: If True, log visualization to wandb
        epoch: Optional epoch number for filename
    
    Returns:
        Filename of saved visualization
    """
    import matplotlib.pyplot as plt

    if model.input_ndim != 2:
        raise ValueError("visualize_attention currently supports only 2D image inputs (input_ndim=2).")
    
    model.eval()
    with torch.no_grad():
        predictions, certainties, synch, pre_act, post_act, attention, nlm_activations = model(
            image.unsqueeze(0).to(device), 
            track=True
        )
    
    # Calculate class probabilities
    probs = F.softmax(predictions, dim=1)
    
    # attention shape: (n_ticks, batch, n_heads, 1, n_patches)
    attention = attention[:, 0, :, 0, :]  # (n_ticks, n_heads, n_patches)
    
    # Average across heads
    attention_avg = attention.mean(axis=1)  # (n_ticks, n_patches)
    
    n_ticks_to_show = model.max_ticks
    tick_indices = np.arange(model.max_ticks)
    
    fig, axes = plt.subplots(3, n_ticks_to_show, figsize=(2 * n_ticks_to_show, 6))
    
    # Show original image
    img_np = image.squeeze().cpu().numpy()
    
    # Handle different embedding types
    if model.use_perceiver:
        # Perceiver: reshape attention to match image dimensions (H, W, C) -> (H, W)
        # For byte-level tokens, we need to reshape from flattened (H*W*C) back to (H, W)
        # Assuming single channel, reshape to (H, W)
        h, w = model.input_size, model.input_size
        grid_size = h
        patch_size = 1  # Each byte is a pixel
    else:
        # Regular patch embedding: square grid
        grid_size = int(np.sqrt(model.n_patches))
        patch_size = model.patch_size
    
    for i, tick_idx in enumerate(tick_indices):
        # Top row: attention heatmap overlaid on image
        axes[0, i].imshow(img_np, cmap='gray', alpha=0.5)
        
        if model.use_perceiver:
            # Reshape attention from (H*W*C) to (H, W) for visualization
            # For single channel, we can reshape directly
            attn_flat = attention_avg[tick_idx]
            if len(attn_flat) == model.input_size * model.input_size:
                # Single channel case: reshape to (H, W)
                attn_map = attn_flat.reshape(model.input_size, model.input_size)
            else:
                # Multi-channel: take average across channels
                attn_map = attn_flat.reshape(model.input_size, model.input_size, -1).mean(axis=-1)
            attn_resized = attn_map  # Already pixel-level
        else:
            attn_map = attention_avg[tick_idx].reshape(grid_size, grid_size)
            attn_resized = np.kron(attn_map, np.ones((patch_size, patch_size)))
        
        axes[0, i].imshow(attn_resized, cmap='hot', alpha=0.5)
        axes[0, i].set_title(f'Tick {tick_idx}')
        axes[0, i].axis('off')
        
        # Middle row: just attention heatmap
        axes[1, i].imshow(attn_map, cmap='hot')
        axes[1, i].set_title(f'Attn')
        axes[1, i].axis('off')
        
        # Bottom row: class probabilities
        tick_probs = probs[0, :, tick_idx].cpu().numpy()
        axes[2, i].bar(range(model.out_dims), tick_probs)
        axes[2, i].set_ylim(0, 1)
        axes[2, i].set_title(f'Pred: {tick_probs.argmax()}')
        axes[2, i].set_xticks([])
        if i > 0:
            axes[2, i].set_yticks([])
    
    title = 'Foveated Attention and Predictions Over Ticks'
    if model.use_perceiver:
        title += ' (Perceiver Byte-Level)'
    if epoch is not None:
        title += f' - Epoch {epoch+1}'
    plt.suptitle(title)
    plt.tight_layout()
    filename = f'attention_visualization_epoch_{epoch+1}.png' if epoch is not None else 'attention_visualization.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")
    
    # Log to wandb if requested
    if use_wandb:
        wandb.log({
            "attention_visualization": wandb.Image(filename),
        })
    
    return filename


def visualize_nlm_activations(model: SimplifiedCTM, image: torch.Tensor, device: torch.device, epoch: int | None = None) -> str:
    """
    Visualize NLM activations per tick - simple grid of line plots for each neuron.
    
    Args:
        model: The CTM model
        image: Input image tensor
        device: Device to run on
        epoch: Optional epoch number for filename
    
    Returns:
        Filename of saved visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    model.eval()
    with torch.no_grad():
        predictions, certainties, synch, pre_act, post_act, attention, nlm_activations = model(
            image.unsqueeze(0).to(device), 
            track=True
        )
    
    # Extract output activations: shape (n_ticks, batch, neurons) -> (n_ticks, neurons)
    n_ticks = len(nlm_activations)
    n_neurons = model.n_neurons
    output_activations = np.array([nlm_activations[tick]['output'][0] for tick in range(n_ticks)])
    # Shape: (n_ticks, n_neurons)
    
    # Create grid layout
    n_cols = 18
    n_rows = int(np.ceil(n_neurons / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 1.2))
    axes = axes.flatten() if n_neurons > 1 else [axes]
    
    # Color map for different neurons
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_neurons))
    
    # Get synchronization pair indices
    out_left = model.out_neuron_indices_left.cpu().numpy()
    out_right = model.out_neuron_indices_right.cpu().numpy()
    action_left = model.action_neuron_indices_left.cpu().numpy()
    action_right = model.action_neuron_indices_right.cpu().numpy()
    
    # Create sets for quick lookup
    out_neurons = set(out_left) | set(out_right)
    action_neurons = set(action_left) | set(action_right)
    
    ticks = np.arange(n_ticks)
    
    for neuron_idx in range(n_neurons):
        ax = axes[neuron_idx]
        neuron_data = output_activations[:, neuron_idx]
        
        # Plot line with color
        ax.plot(ticks, neuron_data, color=colors[neuron_idx % len(colors)], linewidth=1.5, alpha=0.8)
        
        # Fill area under curve
        ax.fill_between(ticks, neuron_data, alpha=0.3, color=colors[neuron_idx % len(colors)])
        
        # Add border for synchronization pairs
        if neuron_idx in out_neurons:
            # Red border for output synchronization pairs
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('red')
                spine.set_linewidth(2.5)
        elif neuron_idx in action_neurons:
            # Blue border for action synchronization pairs
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('blue')
                spine.set_linewidth(2.5)
        else:
            # No border for non-synchronization neurons
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Simple styling
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_neurons, len(axes)):
        axes[idx].axis('off')
    
    title = 'NLM Activations Over Ticks (One Plot Per Neuron)'
    if epoch is not None:
        title += f' - Epoch {epoch+1}'
    title += '\nRed border: Output sync pairs | Blue border: Action sync pairs'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    filename = f'nlm_activations_epoch_{epoch+1}.png' if epoch is not None else 'nlm_activations.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")
    return filename


def evaluate(
    model: SimplifiedCTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: The CTM model to evaluate
        loader: DataLoader for validation/test data
        device: Device to run evaluation on
    
    Returns:
        Tuple of (average_loss, accuracy, average_certainty, average_tick_index)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_certainty = 0.0
    total_tick_index = 0.0
    
    # Handle empty loader
    if len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            predictions, certainties = model(data)
            
            # Use adaptive compute: select prediction at most certain tick
            most_certain_indices = certainties[:, 1, :].argmax(dim=1)
            batch_indices = torch.arange(data.size(0), device=device)
            final_predictions = predictions[batch_indices, :, most_certain_indices]
            
            loss = criterion(final_predictions, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(final_predictions, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Track certainty and tick index
            batch_certainty = certainties[batch_indices, 1, most_certain_indices]
            total_certainty += batch_certainty.sum().item()
            total_tick_index += most_certain_indices.float().sum().item()
    
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return (
        total_loss / len(loader), 
        100. * correct / total,
        total_certainty / total,
        total_tick_index / total
    )


def train_model(
    model: SimplifiedCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    val_loader: DataLoader | None = None,
    use_wandb: bool = False,
):
    """
    Training loop with adaptive compute loss (Section 3.5 & Appendix E.2).
    
    The loss function encourages the model to:
    1. Find the correct answer eventually (min_loss across ticks)
    2. Be confident about the correct answer (loss at most certain tick)
    
    Args:
        model: The CTM model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to run training on
        val_loader: Optional DataLoader for validation data
        use_wandb: If True, log metrics to wandb
    """
    model.train()
    # reduction='none' allows us to inspect loss per tick per sample
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Get a sample image for visualization (from first batch)
    sample_image = None
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Store first image from first batch for visualization
            if epoch == 0 and batch_idx == 0:
                sample_image = data[0]
            
            optimizer.zero_grad()
            
            # predictions: (B, C, T), certainties: (B, 2, T)
            predictions, certainties = model(data)
            
            # Expand targets to match temporal dimension: (B) -> (B, T)
            targets_expanded = target.unsqueeze(-1).expand(-1, model.max_ticks)
            
            # Compute loss for every tick: (B, C, T) vs (B, T) -> (B, T)
            loss_all_ticks = criterion(predictions, targets_expanded)
            
            # 1. Best possible tick loss (Minimum loss across time)
            # This encourages the model to find the answer *at some point*
            loss_min, _ = loss_all_ticks.min(dim=1)
            
            # 2. Loss at the step the model is MOST CERTAIN about
            # certainties[:, 1, :] is the confidence score (1 - normalized_entropy)
            most_certain_indices = certainties[:, 1, :].argmax(dim=1)
            
            # Select the loss values corresponding to the most certain ticks
            batch_indices = torch.arange(data.size(0), device=device)
            loss_selected = loss_all_ticks[batch_indices, most_certain_indices]
            
            # Combine losses (Official Formula): Average of best-case and chosen-case
            loss = (loss_min.mean() + loss_selected.mean()) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy based on the "adaptive compute" choice (most certain tick)
            # This reflects how the model would actually be used in inference
            final_predictions = predictions[batch_indices, :, most_certain_indices]
            _, predicted = torch.max(final_predictions, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                # Average certainty at the chosen tick
                avg_certainty = certainties[batch_indices, 1, most_certain_indices].mean().item()
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}% (Adaptive), '
                      f'Certainty: {avg_certainty:.3f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # Evaluate on validation set if provided
        val_loss = None
        val_acc = None
        val_certainty = None
        val_tick_index = None
        if val_loader is not None:
            val_loss, val_acc, val_certainty, val_tick_index = evaluate(model, val_loader, device)
            print(f'Epoch {epoch+1}/{epochs} completed - '
                  f'Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Val Certainty: {val_certainty:.3f}, Val Avg Tick: {val_tick_index:.2f}')
        else:
            print(f'Epoch {epoch+1}/{epochs} completed - '
                  f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Log epoch metrics to wandb
        if use_wandb:
            log_dict = {
                "train/epoch_loss": avg_loss,
                "train/epoch_accuracy": accuracy,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            if val_loss is not None and val_acc is not None:
                log_dict["val/loss"] = val_loss
                log_dict["val/accuracy"] = val_acc
            if val_certainty is not None:
                log_dict["val/certainty"] = val_certainty
            if val_tick_index is not None:
                log_dict["val/avg_tick_index"] = val_tick_index
            wandb.log(log_dict, step=epoch)
        
        # Visualize NLM activations at end of each epoch
        if sample_image is not None:
            print(f'Generating NLM activations visualization for epoch {epoch+1}...')
            nlm_filename = visualize_nlm_activations(model, sample_image, device, epoch=epoch)
            
            # Log NLM activations visualization to wandb
            if use_wandb and nlm_filename:
                wandb.log({
                    "nlm_activations": wandb.Image(nlm_filename),
                }, step=epoch)
            
            # Visualize attention at end of each epoch
            print(f'Generating attention visualization for epoch {epoch+1}...')
            attention_filename = visualize_attention(model, sample_image, device, use_wandb=False, epoch=epoch)
            
            # Log attention visualization to wandb
            if use_wandb and attention_filename:
                wandb.log({
                    "attention_visualization": wandb.Image(attention_filename),
                }, step=epoch)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: SimplifiedCTM):
    total_params = count_parameters(model)
    print(f'\n=== Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Memory length: {model.max_memory}')
    print(f'Max ticks: {model.max_ticks}')
    if model.use_perceiver:
        print(f'Embedding type: Perceiver (byte-level tokens with Fourier positional encoding)')
        print(f'Input ndim: {model.input_ndim}')
        if model.input_ndim == 2:
            print(f'Input size: {model.input_size}x{model.input_size}')
        elif model.input_ndim == 3:
            print(f'Point cloud size: {model.input_size} points')
        else:
            print(f'Sequence length: {model.input_size}')
        print(f'Number of byte tokens: {model.n_patches}')
    elif model.input_ndim == 3:
        print(f'Input ndim: 3 (Point Cloud)')
        print(f'Number of points: {model.input_size}')
        print(f'Number of patches: {model.n_patches} (patch_size={model.patch_size})')
    elif model.input_ndim == 2:
        print(f'Input ndim: 2')
        print(f'Input size: {model.input_size}x{model.input_size}')
        print(f'Number of patches: {model.n_patches} ({model.input_size//model.patch_size}x{model.input_size//model.patch_size})')
        print(f'Patch size: {model.patch_size}x{model.patch_size}')
    else:
        print(f'Input ndim: 1')
        print(f'Sequence length: {model.input_size}')
        print(f'Number of patches: {model.n_patches}')
        print(f'Patch size: {model.patch_size}')
    print(f'Sync pairs (output): {model.n_synch_out}')
    print(f'Sync pairs (action): {model.n_synch_action}')
    print(f'Attention heads: {model.n_attention_heads}')
    print(f'=========================\n')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model hyperparameters
    n_neurons = 64
    max_memory = 10
    max_ticks = 15
    d_input = 64
    n_synch_out = 32
    n_synch_action = 16
    n_attention_heads = 4
    out_dims = 10
    input_size = 28
    patch_size = 7
    in_channels = 1
    dropout = 0.0
    dropout_nlm = 0.0
    batch_size = 32
    epochs = 3
    lr = 0.001
    
    # Initialize model with spatial patches for foveated attention
    model = SimplifiedCTM(
        n_neurons=n_neurons,
        max_memory=max_memory,
        max_ticks=max_ticks,
        d_input=d_input,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        n_attention_heads=n_attention_heads,
        out_dims=out_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dropout=dropout,
        dropout_nlm=dropout_nlm,
    ).to(device)
    
    print_model_info(model)
    
    # Compute total parameters
    total_params = count_parameters(model)
    
    # Wandb config
    wandb_config = {
        "input_size": input_size,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": "Adam",
        "n_neurons": n_neurons,
        "max_memory": max_memory,
        "max_ticks": max_ticks,
        "d_input": d_input,
        "n_synch_out": n_synch_out,
        "n_synch_action": n_synch_action,
        "n_attention_heads": n_attention_heads,
        "out_dims": out_dims,
        "in_channels": in_channels,
        "dropout": dropout,
        "dropout_nlm": dropout_nlm,
        "total_params": total_params,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "device": str(device),
        "use_perceiver": model.use_perceiver,
        "input_ndim": model.input_ndim,
    }
    
    # Initialize wandb
    wandb.init(
        project="mnist",
        name=f"ctm_mnist_ep{epochs}",
        config=wandb_config
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train for a few epochs
    train_model(model, train_loader, optimizer, epochs=epochs, device=device, val_loader=val_loader, use_wandb=True)
    
    # Visualize the foveated attention behavior
    print("\nVisualizing attention patterns...")
    test_image, _ = train_dataset[0]
    attention_filename = visualize_attention(model, test_image, device, use_wandb=True)
    
    torch.save(model.state_dict(), 'ctm_spatial_model.pth')
    print('Model saved to ctm_spatial_model.pth')
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()

