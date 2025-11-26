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


class MultitaskExtendedCTM(nn.Module):
    """
    Extended Continuous Thought Machine with multitask support.
    
    Combines:
    - Multitask architecture (different neuron pairs per task)
    - Recursive synchronization computation (O(1) per tick)
    - Separate S_out and S_action synchronization per task
    - Multi-head attention
    - GLU nonlinearities
    - Loss across all ticks
    - Certainty computation
    
    Based on: https://arxiv.org/abs/2505.05522
    Reference: https://github.com/SakanaAI/continuous-thought-machines
    """
    def __init__(
        self, 
        n_neurons: int, 
        max_memory: int, 
        max_ticks: int, 
        n_representation_size: int, 
        task_output_sizes: list[int],
        n_attention_heads: int,
    ):
        super(MultitaskExtendedCTM, self).__init__()

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_tasks = len(task_output_sizes)
        self.task_output_sizes = task_output_sizes
        self.n_representation_size = n_representation_size
        self.n_attention_heads = n_attention_heads
        
        # Calculate sync pairs per task
        self.n_synch_out_per_task = max(n_neurons // 4, 8)
        self.n_synch_action_per_task = max(n_neurons // 8, 4)
        
        # Learnable initial states (Section 3.1)
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
        
        # Synapse model - SHARED across all tasks
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
        
        # Initialize synchronization pairs for each task
        self._init_task_synchronization_pairs()
        
        # Task-specific decay parameters for synchronization
        self.task_decay_params_action = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_synch_action_per_task), requires_grad=True)
            for _ in range(self.n_tasks)
        ])
        self.task_decay_params_out = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_synch_out_per_task), requires_grad=True)
            for _ in range(self.n_tasks)
        ])
        
        # Task-specific query projections (sync_action -> attention query)
        self.task_query_projections = nn.ModuleList([
            nn.Linear(self.n_synch_action_per_task, n_representation_size)
            for _ in range(self.n_tasks)
        ])
        
        # Multi-head attention per task
        self.task_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=n_representation_size,
                num_heads=n_attention_heads,
                dropout=0.0,
                batch_first=True
            )
            for _ in range(self.n_tasks)
        ])
        
        # Task-specific output projectors (sync_out -> predictions)
        self.task_output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_synch_out_per_task, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            )
            for output_size in task_output_sizes
        ])
    
    def _init_task_synchronization_pairs(self):
        """Initialize separate neuron pairs for each task (output and action)."""
        for task_idx in range(self.n_tasks):
            # Output sync pairs
            out_left = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out_per_task, replace=True)
            )
            out_right = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_out_per_task, replace=True)
            )
            self.register_buffer(f'task_{task_idx}_out_left', out_left)
            self.register_buffer(f'task_{task_idx}_out_right', out_right)
            
            # Action sync pairs
            action_left = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action_per_task, replace=True)
            )
            action_right = torch.from_numpy(
                np.random.choice(np.arange(self.n_neurons), size=self.n_synch_action_per_task, replace=True)
            )
            self.register_buffer(f'task_{task_idx}_action_left', action_left)
            self.register_buffer(f'task_{task_idx}_action_right', action_right)
    
    def get_task_sync_indices(self, task_idx: int, synch_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get neuron pair indices for a specific task and sync type."""
        left = getattr(self, f'task_{task_idx}_{synch_type}_left')
        right = getattr(self, f'task_{task_idx}_{synch_type}_right')
        return left, right
    
    def compute_synchronization(
        self,
        activated_state: torch.Tensor,
        decay_alpha: torch.Tensor,
        decay_beta: torch.Tensor,
        r: torch.Tensor,
        task_idx: int,
        synch_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute synchronization using efficient recursive formula (Appendix H).
        
        S_ij^t = α_ij^t / sqrt(β_ij^t)
        
        Where:
        - α_ij^(t+1) = e^(-r_ij) * α_ij^t + z_i^(t+1) * z_j^(t+1)
        - β_ij^(t+1) = e^(-r_ij) * β_ij^t + 1
        """
        indices_left, indices_right = self.get_task_sync_indices(task_idx, synch_type)
        
        left = activated_state[:, indices_left]
        right = activated_state[:, indices_right]
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
        """Compute certainty as 1 - normalized_entropy."""
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def forward(self, x: torch.Tensor, task_idx: int, track: bool = False):
        """
        Forward pass for a specific task.
        Returns predictions and certainties for ALL ticks.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode image (SHARED)
        encoded_image = self.image_encoder(x)
        
        # Prepare key-value for attention
        kv = self.kv_proj(encoded_image).unsqueeze(1)
        
        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Storage for all tick outputs
        out_dims = self.task_output_sizes[task_idx]
        predictions = torch.empty(batch_size, out_dims, self.max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.max_ticks, device=device, dtype=torch.float32)
        
        # Initialize sync recurrence
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        
        # Clamp and compute decay factors for this task
        self.task_decay_params_action[task_idx].data = torch.clamp(
            self.task_decay_params_action[task_idx].data, 0, 15
        )
        self.task_decay_params_out[task_idx].data = torch.clamp(
            self.task_decay_params_out[task_idx].data, 0, 15
        )
        
        r_action = torch.exp(-self.task_decay_params_action[task_idx]).unsqueeze(0).expand(batch_size, -1)
        r_out = torch.exp(-self.task_decay_params_out[task_idx]).unsqueeze(0).expand(batch_size, -1)
        
        # Initialize output sync
        _, decay_alpha_out, decay_beta_out = self.compute_synchronization(
            activated_state, None, None, r_out, task_idx, synch_type='out'
        )
        
        # Tracking
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            attention_tracking = []
        
        # Recurrent loop
        for tick in range(self.max_ticks):
            # Compute action sync
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, r_action, task_idx, synch_type='action'
            )
            
            # Create query and attend
            q = self.task_query_projections[task_idx](sync_action).unsqueeze(1)
            attn_out, attn_weights = self.task_attention[task_idx](q, kv, kv, need_weights=True)
            attn_out = attn_out.squeeze(1)
            
            # Synapse model
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            new_state = self.synapse_model(pre_synapse_input)
            
            # Update trace
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply NLMs
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_trace = state_trace[:, neuron_idx, :]
                post_activation = self.neuron_level_models[neuron_idx](neuron_trace)
                post_activations_list.append(post_activation)
            
            activated_state = torch.cat(post_activations_list, dim=-1)
            
            # Compute output sync
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(
                activated_state, decay_alpha_out, decay_beta_out, r_out, task_idx, synch_type='out'
            )
            
            # Get prediction
            current_prediction = self.task_output_projectors[task_idx](sync_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            predictions[..., tick] = current_prediction
            certainties[..., tick] = current_certainty
            
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
                np.array(attention_tracking)
            )
        
        return predictions, certainties
    
    def forward_all_tasks(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass computing predictions for ALL tasks.
        Shared neural dynamics computed once, then each task reads its sync.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode image (SHARED)
        encoded_image = self.image_encoder(x)
        kv = self.kv_proj(encoded_image).unsqueeze(1)
        
        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Run neural dynamics (SHARED)
        for tick in range(self.max_ticks):
            # Use task 0's action sync for shared dynamics (or could average across tasks)
            r_action = torch.exp(-self.task_decay_params_action[0]).unsqueeze(0).expand(batch_size, -1)
            
            if tick == 0:
                decay_alpha_action, decay_beta_action = None, None
            
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, r_action, task_idx=0, synch_type='action'
            )
            
            q = self.task_query_projections[0](sync_action).unsqueeze(1)
            attn_out, _ = self.task_attention[0](q, kv, kv)
            attn_out = attn_out.squeeze(1)
            
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            new_state = self.synapse_model(pre_synapse_input)
            
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_trace = state_trace[:, neuron_idx, :]
                post_activation = self.neuron_level_models[neuron_idx](neuron_trace)
                post_activations_list.append(post_activation)
            
            activated_state = torch.cat(post_activations_list, dim=-1)
        
        # Compute final predictions for each task
        results = []
        for task_idx in range(self.n_tasks):
            r_out = torch.exp(-self.task_decay_params_out[task_idx]).unsqueeze(0).expand(batch_size, -1)
            
            # Initialize and compute sync_out for this task
            indices_left, indices_right = self.get_task_sync_indices(task_idx, 'out')
            left = activated_state[:, indices_left]
            right = activated_state[:, indices_right]
            pairwise_product = left * right
            
            # Simple final sync (not recursive for efficiency in eval)
            sync_out = pairwise_product / torch.sqrt(torch.ones_like(pairwise_product) + 1e-8)
            
            prediction = self.task_output_projectors[task_idx](sync_out)
            certainty = self.compute_certainty(prediction)
            results.append((prediction, certainty))
        
        return results


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics across epochs."""
    task_names: list[str]
    epoch_losses: dict[str, list[float]] = field(default_factory=dict)
    epoch_accuracies: dict[str, list[float]] = field(default_factory=dict)
    batch_losses: dict[str, list[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        for name in self.task_names:
            self.epoch_losses[name] = []
            self.epoch_accuracies[name] = []
            self.batch_losses[name] = []


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


def train_combined_multitask_model(
    model: MultitaskExtendedCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int]
) -> TrainingMetrics:
    """
    Training loop for MultitaskExtendedCTM on combined multi-dataset.
    Computes loss across ALL ticks as per CTM paper Section 3.5.
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
        
        for batch_idx, (stacked_images, labels, sources, batch_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
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
                                # Forward pass returns predictions for ALL ticks
                                predictions, certainties = model(data, task_idx=model_task_idx)
                                
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
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f}')
        
        print(f'\n=== Epoch {epoch+1}/{epochs} Summary ===')
        for task_name in task_names:
            if task_total[task_name] > 0:
                accuracy = 100. * task_correct[task_name] / task_total[task_name]
                avg_loss = task_losses[task_name] / max(1, task_batch_count[task_name])
                metrics.epoch_losses[task_name].append(avg_loss)
                metrics.epoch_accuracies[task_name].append(accuracy)
                print(f'  {task_name}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')
        print()
    
    return metrics


def evaluate_combined_multitask_model(
    model: MultitaskExtendedCTM,
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
                                predictions, _ = model(data, task_idx=model_task_idx)
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
    model: MultitaskExtendedCTM,
    x: torch.Tensor,
    task_idx: int,
    device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Extract neural dynamics and synchronizations for visualization."""
    model.eval()
    
    with torch.no_grad():
        predictions, certainties, syncs, pre_acts, post_acts, attn = model(
            x.to(device), task_idx=task_idx, track=True
        )
    
    post_history = torch.from_numpy(post_acts).permute(1, 2, 0)  # (B, neurons, ticks)
    
    return post_history, {'out': syncs[0], 'action': syncs[1]}


def plot_training_results(
    metrics: TrainingMetrics,
    test_accuracies: dict[str, float],
    model: MultitaskExtendedCTM,
    sample_image: torch.Tensor,
    device: torch.device,
    output_prefix: str
):
    """Generate comprehensive training visualization plots."""
    task_names = metrics.task_names
    color_palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(task_names)}
    
    # Figure 1: Training Loss and Accuracy
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Training Progress per Task (Extended CTM)', fontsize=14, fontweight='bold')
    
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
    fig1.savefig(f'{output_prefix}_fig1_training_curves.png', dpi=150, bbox_inches='tight')
    
    # Figure 2: Batch-level Loss
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.suptitle('Batch-level Training Loss (Smoothed)', fontsize=14, fontweight='bold')
    
    window_size = 50
    for task_name in task_names:
        losses = metrics.batch_losses[task_name]
        if len(losses) > window_size:
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(smoothed, label=f'{task_name}', color=colors[task_name], alpha=0.8)
        elif losses:
            ax2.plot(losses, label=f'{task_name}', color=colors[task_name], alpha=0.8)
    
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_fig2_batch_losses.png', dpi=150, bbox_inches='tight')
    
    # Figure 3: Neural Dynamics
    post_history, task_syncs = get_neural_dynamics(model, sample_image.unsqueeze(0), task_idx=0, device=device)
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Neural Dynamics Visualization (Extended CTM)', fontsize=14, fontweight='bold')
    
    ax_img = axes3[0, 0]
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
    ax_img.set_title('Input Image')
    ax_img.axis('off')
    
    ax_activity = axes3[0, 1]
    activity_data = post_history[0].numpy()
    im = ax_activity.imshow(activity_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_activity.set_xlabel('Tick')
    ax_activity.set_ylabel('Neuron')
    ax_activity.set_title('Post-Activation Dynamics')
    plt.colorbar(im, ax=ax_activity, label='Activation')
    
    ax_traces = axes3[1, 0]
    n_neurons_to_plot = min(10, model.n_neurons)
    for i in range(n_neurons_to_plot):
        ax_traces.plot(activity_data[i], alpha=0.7, label=f'N{i}')
    ax_traces.set_xlabel('Tick')
    ax_traces.set_ylabel('Activation')
    ax_traces.set_title(f'Sample Neuron Traces (First {n_neurons_to_plot})')
    ax_traces.legend(loc='upper right', fontsize=8, ncol=2)
    ax_traces.grid(True, alpha=0.3)
    
    ax_sync = axes3[1, 1]
    sync_out = task_syncs['out'][-1, 0]  # Last tick, first sample
    sync_action = task_syncs['action'][-1, 0]
    x_out = np.arange(len(sync_out))
    x_action = np.arange(len(sync_action))
    ax_sync.bar(x_out, sync_out, alpha=0.7, label='sync_out', color='#3498db')
    ax_sync.bar(x_action + len(sync_out) + 2, sync_action, alpha=0.7, label='sync_action', color='#e74c3c')
    ax_sync.set_xlabel('Sync Pair Index')
    ax_sync.set_ylabel('Synchronization Value')
    ax_sync.set_title('Synchronization Values (Task 0)')
    ax_sync.legend()
    ax_sync.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_fig3_neural_dynamics.png', dpi=150, bbox_inches='tight')
    
    # Figure 4: Final Accuracy
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.suptitle('Final Model Performance (Extended CTM)', fontsize=14, fontweight='bold')
    
    x = np.arange(len(task_names))
    width = 0.35
    
    train_accs = [metrics.epoch_accuracies[name][-1] if metrics.epoch_accuracies[name] else 0 for name in task_names]
    test_accs = [test_accuracies.get(name, 0) for name in task_names]
    
    bars1 = ax4.bar(x - width/2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, test_accs, width, label='Test', color='#9b59b6', alpha=0.8)
    
    ax4.set_xlabel('Task')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.replace('_', '\n') for name in task_names], fontsize=9)
    ax4.legend()
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig4.savefig(f'{output_prefix}_fig4_final_accuracy.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print(f'\nPlots saved with prefix "{output_prefix}":')
    print(f'  - {output_prefix}_fig1_training_curves.png')
    print(f'  - {output_prefix}_fig2_batch_losses.png')
    print(f'  - {output_prefix}_fig3_neural_dynamics.png')
    print(f'  - {output_prefix}_fig4_final_accuracy.png')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: MultitaskExtendedCTM):
    """Print detailed model information."""
    total_params = count_parameters(model)
    print(f'\n=== MultitaskExtendedCTM Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Number of tasks: {model.n_tasks}')
    print(f'Sync pairs per task (output): {model.n_synch_out_per_task}')
    print(f'Sync pairs per task (action): {model.n_synch_action_per_task}')
    print(f'Attention heads: {model.n_attention_heads}')
    print(f'Max ticks: {model.max_ticks}')
    print(f'Memory length: {model.max_memory}')
    print(f'Task output sizes: {model.task_output_sizes}')
    
    print(f'\nShared components:')
    shared = ['image_encoder', 'kv_proj', 'synapse_model', 'neuron_level_models']
    for name in shared:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nTask-specific components:')
    task_specific = ['task_decay_params_action', 'task_decay_params_out', 
                     'task_query_projections', 'task_attention', 'task_output_projectors']
    for name in task_specific:
        if hasattr(model, name):
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {params:,} parameters')
    
    print(f'\nLearnable states:')
    print(f'  start_activated_state: {model.start_activated_state.numel():,} elements')
    print(f'  start_trace: {model.start_trace.numel():,} elements')
    print(f'================================================\n')


def train_combined(device: torch.device, selected_tasks: list[str], epochs: int, n_neurons: int):
    """Train MultitaskExtendedCTM on selected tasks."""
    print('\n' + '='*60)
    print(f'TRAINING EXTENDED CTM ON TASKS: {", ".join(selected_tasks)}')
    print('='*60)
    
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
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    
    if needs_cifar:
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    
    if needs_fashion:
        fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
        fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=mnist_transform)
    
    if needs_emnist:
        emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=mnist_transform)
        emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=mnist_transform)
    
    train_dataset = CombinedMultitaskDataset(mnist_train, cifar_train, fashion_train, emnist_train)
    test_dataset = CombinedMultitaskDataset(mnist_test, cifar_test, fashion_test, emnist_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
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
    
    model = MultitaskExtendedCTM(
        n_neurons=n_neurons,
        max_memory=10, 
        max_ticks=15, 
        n_representation_size=32,
        task_output_sizes=task_output_sizes,
        n_attention_heads=4,
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = train_combined_multitask_model(
        model, train_loader, optimizer, 
        epochs=epochs, device=device, task_names=task_names,
        task_index_mapping=task_index_mapping
    )
    
    test_accuracies = evaluate_combined_multitask_model(
        model, test_loader, device=device, task_names=task_names,
        task_index_mapping=task_index_mapping
    )
    
    output_prefix = '_'.join([t.split('_')[0][0] + t.split('_')[1][0] for t in selected_tasks]) + '_ext'
    
    torch.save(model.state_dict(), f'{output_prefix}_multitask_ctm.pth')
    print(f'Model saved to {output_prefix}_multitask_ctm.pth')
    
    sample_image, _, _ = test_dataset[0]
    
    print('\nGenerating training plots...')
    plot_training_results(
        metrics, test_accuracies, model, sample_image, device,
        output_prefix=output_prefix
    )
    
    return model, metrics, test_accuracies


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MultitaskExtendedCTM on selected tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
  mnist_digit      - MNIST digit classification (0-9)
  mnist_even_odd   - MNIST even/odd classification
  cifar_fine       - CIFAR-10 fine classification (10 classes)
  cifar_coarse     - CIFAR-10 coarse (animal vs vehicle)
  fashion_fine     - Fashion MNIST classification (10 classes)
  emnist_fine      - EMNIST balanced classification (47 classes)

Examples:
  py multitask_extended.py --tasks cifar_fine
  py multitask_extended.py --tasks mnist_digit cifar_fine
  py multitask_extended.py --tasks mnist_digit mnist_even_odd cifar_fine cifar_coarse
        """
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=list(ALL_TASKS.keys()),
        default=list(ALL_TASKS.keys()),
        help='Tasks to train on (default: all tasks)'
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
        default=32,
        help='Number of neurons in CTM (default: 32)'
    )
    
    args = parser.parse_args()
    
    if not args.tasks:
        print("Error: At least one task must be selected")
        return
    
    if len(args.tasks) != len(set(args.tasks)):
        print("Error: Duplicate tasks detected")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Selected tasks: {", ".join(args.tasks)}')
    print(f'Epochs: {args.epochs}')
    print(f'Neurons: {args.neurons}')
    
    train_combined(device, args.tasks, args.epochs, args.neurons)
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
