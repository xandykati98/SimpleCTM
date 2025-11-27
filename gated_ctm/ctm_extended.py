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

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch

# Modal support for cloud GPU training
import modal

# Modal app configuration
modal_app = modal.App("gated-ctm-training")

# Modal image with required dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib",
        "numpy",
        "mlflow",
    )
)


# Task configuration
ALL_TASKS = {
    'mnist_digit': {'output_size': 10, 'dataset': 'mnist', 'description': 'MNIST digit classification (0-9)'},
    'mnist_even_odd': {'output_size': 2, 'dataset': 'mnist', 'description': 'MNIST even/odd classification'},
    'cifar_fine': {'output_size': 10, 'dataset': 'cifar', 'description': 'CIFAR-10 fine classification (10 classes)'},
    'cifar_coarse': {'output_size': 2, 'dataset': 'cifar', 'description': 'CIFAR-10 coarse (animal vs vehicle)'},
    'fashion_fine': {'output_size': 10, 'dataset': 'fashion', 'description': 'Fashion MNIST classification (10 classes)'},
    'emnist_fine': {'output_size': 47, 'dataset': 'emnist', 'description': 'EMNIST balanced classification (47 classes)'},
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
        
        # Adaptive CNN encoder - SHARED across all tasks
        self.image_encoder = AdaptiveCNNEncoder(n_representation_size)
        
        # Key-Value projection for attention (from encoded input)
        self.kv_proj = nn.Sequential(
            nn.Linear(n_representation_size, n_representation_size),
            nn.LayerNorm(n_representation_size)
        )
        
        # Query projection from synchronization_action (gated)
        self.q_proj = nn.Linear(n_synch_action, n_representation_size)
        
        # Multi-head attention (Section 3.4) - SHARED
        self.attention = nn.MultiheadAttention(
            embed_dim=n_representation_size,
            num_heads=n_attention_heads,
            dropout=0.0,
            batch_first=True
        )
        
        # Synapse model - combines attention output with current activated state - SHARED
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),
            nn.LayerNorm(n_neurons),
        )

        # Neuron-level models (NLMs) - SHARED, each neuron has private params
        self.neuron_level_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_memory, 128 * 2),
                nn.GLU(),
                nn.Linear(128, 1),
            )
            for _ in range(n_neurons)
        ])

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
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
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
    
    def forward(self, x: torch.Tensor, task_idx: int, track: bool = False):
        """
        Forward pass through the Multitask Gated CTM for a specific task.
        
        Returns predictions, certainties, and load_balance_loss for ALL ticks.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode the image using adaptive CNN encoder
        encoded_image = self.image_encoder(x)
        
        # Prepare key-value features for attention
        kv = self.kv_proj(encoded_image).unsqueeze(1)
        
        # Initialize recurrent state from learnable parameters
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        
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
            
            # Update state trace (FIFO buffer of pre-activations)
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply neuron-level models to get post-activations
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_trace = state_trace[:, neuron_idx, :]
                post_activation = self.neuron_level_models[neuron_idx](neuron_trace)
                post_activations_list.append(post_activation)
            
            activated_state = torch.cat(post_activations_list, dim=-1)
            
            # Calculate gated synchronization for output predictions
            sync_out, decay_alphas_out, decay_betas_out, gate_weights_out = self.compute_gated_synchronization(
                activated_state, decay_alphas_out, decay_betas_out, synch_type='out'
            )
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


# CIFAR-10 class groupings for coarse classification
CIFAR10_COARSE_LABELS = {
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,  # Animals
    0: 1, 1: 1, 8: 1, 9: 1  # Vehicles
}


class CombinedMultitaskDataset(Dataset):
    """
    Combined dataset that mixes multiple image datasets.
    Uses -1 for non-applicable task labels (masked in loss computation).
    """
    def __init__(
        self, 
        mnist_dataset: Dataset | None, 
        cifar_dataset: Dataset | None,
        fashion_dataset: Dataset | None,
        emnist_dataset: Dataset | None
    ):
        self.mnist_dataset = mnist_dataset
        self.cifar_dataset = cifar_dataset
        self.fashion_dataset = fashion_dataset
        self.emnist_dataset = emnist_dataset
        
        self.mnist_len = len(mnist_dataset) if mnist_dataset is not None else 0
        self.cifar_len = len(cifar_dataset) if cifar_dataset is not None else 0
        self.fashion_len = len(fashion_dataset) if fashion_dataset is not None else 0
        self.emnist_len = len(emnist_dataset) if emnist_dataset is not None else 0
        
        self.total_len = self.mnist_len + self.cifar_len + self.fashion_len + self.emnist_len
        
        self.mnist_end = self.mnist_len
        self.cifar_end = self.mnist_end + self.cifar_len
        self.fashion_end = self.cifar_end + self.fashion_len
        self.emnist_end = self.fashion_end + self.emnist_len
    
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
        
        else:
            emnist_idx = idx - self.fashion_end
            image, char_label = self.emnist_dataset[emnist_idx]
            labels['emnist_fine'] = char_label
            return image, labels, 'emnist'


def custom_collate_fn(batch: list) -> tuple:
    """Custom collate function for combined dataset with mixed image sizes."""
    dataset_images: dict[str, list] = {
        'mnist': [], 'cifar': [], 'fashion': [], 'emnist': [],
    }
    labels_dict = {key: [] for key in batch[0][1].keys()}
    sources = []
    batch_indices: dict[str, list] = {
        'mnist': [], 'cifar': [], 'fashion': [], 'emnist': [],
    }
    
    for idx, (image, labels, source) in enumerate(batch):
        dataset_images[source].append(image)
        batch_indices[source].append(idx)
        for key, val in labels.items():
            labels_dict[key].append(val)
        sources.append(source)
    
    stacked_images = {
        key: torch.stack(images) if images else None
        for key, images in dataset_images.items()
    }
    
    return stacked_images, labels_dict, sources, batch_indices


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
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
    }
    
    for epoch in range(epochs):
        task_losses = {name: 0.0 for name in task_names}
        task_correct = {name: 0 for name in task_names}
        task_total = {name: 0 for name in task_names}
        task_batch_count = {name: 0 for name in task_names}
        total_balance_loss = 0.0
        balance_count = 0
        
        for batch_idx, (stacked_images, labels, sources, batch_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_balance_loss = torch.tensor(0.0, device=device)
            
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_images[dataset_name] is not None:
                    data = stacked_images[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                # Forward pass returns predictions for ALL ticks plus load balance loss
                                predictions, certainties, load_balance_loss = model(data, task_idx=model_task_idx)
                                
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
                avg_balance = batch_balance_loss.item() / max(1, sum(1 for d in dataset_task_indices if stacked_images[d] is not None))
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Total Loss: {total_loss.item():.4f}, Balance: {avg_balance:.4f}')
            
            # Gate inspection at batch 100
            if batch_idx == 100:
                print('\n' + '='*60)
                print('GATE INSPECTION AT BATCH 100')
                print('='*60)
                model.eval()
                with torch.no_grad():
                    # Find first available data for inspection
                    for dataset_name, task_indices in dataset_task_indices.items():
                        data = stacked_images.get(dataset_name)
                        if data is not None:
                            for original_idx in task_indices:
                                if original_idx in task_index_mapping:
                                    model_task_idx = task_index_mapping[original_idx]
                                    task_name = task_names[model_task_idx]
                                    
                                    # Forward with tracking to get gate weights
                                    results = model(data[:8].to(device), task_idx=model_task_idx, track=True)
                                    predictions, certainties, lb_loss, syncs, pre_acts, post_acts, attn, gates = results
                                    gate_action, gate_out = gates
                                    
                                    # gate_action shape: (ticks, batch, n_sets)
                                    # Average over batch, show per tick
                                    avg_gate_action = gate_action.mean(axis=1)  # (ticks, n_sets)
                                    avg_gate_out = gate_out.mean(axis=1)
                                    
                                    print(f'\nTask: {task_name}')
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
                                    break
                            break
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
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine']
    
    dataset_task_indices = {
        'mnist': [0, 1],
        'cifar': [2, 3],
        'fashion': [4],
        'emnist': [5],
    }
    
    with torch.no_grad():
        for stacked_images, labels, sources, batch_indices in test_loader:
            for dataset_name, task_indices in dataset_task_indices.items():
                if stacked_images[dataset_name] is not None:
                    data = stacked_images[dataset_name].to(device)
                    data_batch_indices = batch_indices[dataset_name]
                    
                    for original_idx in task_indices:
                        if original_idx in task_index_mapping:
                            model_task_idx = task_index_mapping[original_idx]
                            original_task_name = original_task_names[original_idx]
                            task_name = task_names[model_task_idx]
                            
                            target = torch.tensor([labels[original_task_name][i] for i in data_batch_indices]).to(device)
                            
                            if (target >= 0).any():
                                predictions, _, _ = model(data, task_idx=model_task_idx)
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
    device: torch.device
) -> tuple[torch.Tensor, dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Extract neural dynamics, synchronizations, and gate weights for visualization."""
    model.eval()
    
    with torch.no_grad():
        results = model(x.to(device), task_idx=task_idx, track=True)
        predictions, certainties, load_balance_loss, syncs, pre_acts, post_acts, attn, gates = results
    
    post_history = torch.from_numpy(post_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    
    return post_history, {'out': syncs[0], 'action': syncs[1]}, gates


def plot_neural_dynamics_for_task(
    model: MultitaskGatedCTM,
    sample_image: torch.Tensor,
    task_idx: int,
    task_name: str,
    device: torch.device,
    output_path: str,
    epoch: int,
):
    """Generate neural dynamics plot for a single task."""
    model.eval()
    with torch.no_grad():
        post_history, task_syncs, gates = get_neural_dynamics(
            model, sample_image.unsqueeze(0), task_idx=task_idx, device=device
        )
    gate_action, gate_out = gates
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Neural Dynamics - {task_name} (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    # Input image
    ax_img = axes[0, 0]
    img_np = sample_image.cpu().numpy()
    if img_np.shape[0] == 3:
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        ax_img.imshow(img_np)
    else:
        img_np = img_np * 0.3081 + 0.1307
        img_np = np.clip(img_np, 0, 1)
        ax_img.imshow(img_np.squeeze(), cmap='gray')
    ax_img.set_title(f'Input Image ({task_name})')
    ax_img.axis('off')
    
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


def generate_epoch_plots(
    model: MultitaskGatedCTM,
    test_dataset: Dataset,
    task_names: list[str],
    task_index_mapping: dict[int, int],
    device: torch.device,
    output_folder: str,
    epoch: int,
):
    """Generate neural dynamics plots for all tasks at end of epoch."""
    all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine']
    
    # Get sample images for each task from test dataset
    task_samples: dict[str, torch.Tensor] = {}
    
    for idx in range(len(test_dataset)):
        image, labels, source = test_dataset[idx]
        
        for task_name in task_names:
            if task_name in task_samples:
                continue
            
            original_idx = all_task_order.index(task_name)
            label_key = all_task_order[original_idx]
            
            if labels.get(label_key, -1) >= 0:
                task_samples[task_name] = image
        
        if len(task_samples) == len(task_names):
            break
    
    # Generate plot for each task
    model.eval()
    for task_name in task_names:
        if task_name not in task_samples:
            continue
        
        sample_image = task_samples[task_name]
        original_idx = all_task_order.index(task_name)
        model_task_idx = task_index_mapping[original_idx]
        
        output_path = os.path.join(output_folder, f'epoch{epoch:02d}_{task_name}_dynamics.png')
        plot_neural_dynamics_for_task(
            model=model,
            sample_image=sample_image,
            task_idx=model_task_idx,
            task_name=task_name,
            device=device,
            output_path=output_path,
            epoch=epoch,
        )
    
    print(f'  Saved neural dynamics plots for epoch {epoch} to {output_folder}')


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
    shared = ['image_encoder', 'kv_proj', 'q_proj', 'attention', 'synapse_model', 
              'neuron_level_models', 'gate_network_action', 'gate_network_out']
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
    print(f'================================================\n')


def train_combined(
    device: torch.device, 
    selected_tasks: list[str], 
    epochs: int, 
    n_neurons: int, 
    n_sync_sets: int,
    data_dir: str = './data',
    output_dir: str = '.',
) -> tuple:
    """Train MultitaskGatedCTM on selected tasks."""
    print('\n' + '='*60)
    print(f'TRAINING GATED CTM ON TASKS: {", ".join(selected_tasks)}')
    print('='*60)
    
    # Create output folder for plots
    output_prefix = '_'.join([t.split('_')[0][0] + t.split('_')[1][0] for t in selected_tasks]) + '_gated'
    output_folder = os.path.join(output_dir, output_prefix + '_plots')
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')
    
    # Set MLflow tracking URI to output folder
    mlflow_dir = os.path.join(output_folder, 'mlruns')
    mlflow.set_tracking_uri(f'file://{os.path.abspath(mlflow_dir)}')
    mlflow.set_experiment("gated_ctm")
    
    # Start MLflow run
    with mlflow.start_run(run_name=output_prefix):
        # Log hyperparameters
        mlflow.log_params({
            "tasks": ",".join(selected_tasks),
            "n_tasks": len(selected_tasks),
            "epochs": epochs,
            "n_neurons": n_neurons,
            "n_sync_sets": n_sync_sets,
            "max_memory": 10,
            "max_ticks": 15,
            "n_representation_size": 32,
            "n_synch_out": 32,
            "n_synch_action": 16,
            "n_attention_heads": 4,
            "load_balance_coef": 0.0,
            "learning_rate": 0.001,
            "batch_size": 16,
        })
        
        needs_mnist = any(ALL_TASKS[t]['dataset'] == 'mnist' for t in selected_tasks)
        needs_cifar = any(ALL_TASKS[t]['dataset'] == 'cifar' for t in selected_tasks)
        needs_fashion = any(ALL_TASKS[t]['dataset'] == 'fashion' for t in selected_tasks)
        needs_emnist = any(ALL_TASKS[t]['dataset'] == 'emnist' for t in selected_tasks)
        
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
        
        train_dataset = CombinedMultitaskDataset(mnist_train, cifar_train, fashion_train, emnist_train)
        test_dataset = CombinedMultitaskDataset(mnist_test, cifar_test, fashion_test, emnist_test)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
        
        all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse', 'fashion_fine', 'emnist_fine']
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
            max_memory=10, 
            max_ticks=15, 
            n_representation_size=32,
            n_synch_out=32,
            n_synch_action=16,
            n_attention_heads=4,
            task_output_sizes=task_output_sizes,
            n_sync_sets=n_sync_sets,
            load_balance_coef=0.0,  # Disabled to test if gates differentiate without balance pressure
        ).to(device)
        
        # Log model parameter count
        total_params = count_parameters(model)
        mlflow.log_param("total_parameters", total_params)
        
        print_model_info(model)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
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
):
    """
    Modal entrypoint for running training from command line.
    
    Usage:
        modal run ctm_extended.py --tasks "fashion_fine,cifar_fine" --epochs 10
    """
    selected_tasks = [t.strip() for t in tasks.split(',')]
    
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
    print(f'{"="*60}\n')
    
    # Run training on Modal GPU
    results = train_on_modal.remote(
        selected_tasks=selected_tasks,
        epochs=epochs,
        n_neurons=neurons,
        n_sync_sets=sync_sets,
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
        description='Train MultitaskGatedCTM on selected tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
  mnist_digit      - MNIST digit classification (0-9)
  mnist_even_odd   - MNIST even/odd classification
  cifar_fine       - CIFAR-10 fine classification (10 classes)
  cifar_coarse     - CIFAR-10 coarse (animal vs vehicle)
  fashion_fine     - Fashion MNIST classification (10 classes)
  emnist_fine      - EMNIST balanced classification (47 classes)

Examples (local):
  py ctm_extended.py --tasks fashion_fine emnist_fine cifar_fine
  py ctm_extended.py --tasks mnist_digit cifar_fine --sync_sets 8
  py ctm_extended.py --tasks fashion_fine emnist_fine cifar_fine --epochs 10 --neurons 64

Examples (Modal cloud GPU):
  modal run ctm_extended.py --tasks "fashion_fine,cifar_fine" --epochs 10
  modal run ctm_extended.py --tasks "mnist_digit,cifar_fine" --neurons 128
        """
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=list(ALL_TASKS.keys()),
        default=['fashion_fine', 'emnist_fine', 'cifar_fine'],
        help='Tasks to train on (default: fashion_fine emnist_fine cifar_fine)'
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
        help='Number of synchronization pair sets (default: 4)'
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
    
    args = parser.parse_args()
    
    if not args.tasks:
        print("Error: At least one task must be selected")
        return
    
    if len(args.tasks) != len(set(args.tasks)):
        print("Error: Duplicate tasks detected")
        return
    
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
        
        # Dynamically update GPU type if different from default
        with modal_app.run():
            results = train_on_modal.remote(
                selected_tasks=args.tasks,
                epochs=args.epochs,
                n_neurons=args.neurons,
                n_sync_sets=args.sync_sets,
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
        print(f'Selected tasks: {", ".join(args.tasks)}')
        print(f'Epochs: {args.epochs}')
        print(f'Neurons: {args.neurons}')
        print(f'Sync sets: {args.sync_sets}')
        
        train_combined(
            device=device, 
            selected_tasks=args.tasks, 
            epochs=args.epochs, 
            n_neurons=args.neurons, 
            n_sync_sets=args.sync_sets,
        )
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
