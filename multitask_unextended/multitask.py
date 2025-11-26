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


# Task configuration
ALL_TASKS = {
    'mnist_digit': {'output_size': 10, 'dataset': 'mnist', 'description': 'MNIST digit classification (0-9)'},
    'mnist_even_odd': {'output_size': 2, 'dataset': 'mnist', 'description': 'MNIST even/odd classification'},
    'cifar_fine': {'output_size': 10, 'dataset': 'cifar', 'description': 'CIFAR-10 fine classification (10 classes)'},
    'cifar_coarse': {'output_size': 2, 'dataset': 'cifar', 'description': 'CIFAR-10 coarse (animal vs vehicle)'},
}


class AdaptiveCNNEncoder(nn.Module):
    """
    Adaptive CNN encoder that handles variable input shapes.
    Uses adaptive pooling to produce fixed-size output regardless of input dimensions.
    Compact design with reduced parameters.
    """
    def __init__(self, n_representation_size: int):
        super(AdaptiveCNNEncoder, self).__init__()
        
        self.n_representation_size = n_representation_size
        
        # First conv block - handles 1 or 3 channel input
        self.conv1_1ch = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_3ch = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        
        # Shared conv blocks - reduced channels for efficiency
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
        
        # Adaptive pooling to smaller fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Final projection - much smaller FC layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_representation_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Select appropriate first conv based on input channels
        if x.shape[1] == 1:
            x = self.conv1_1ch(x)
        else:
            x = self.conv1_3ch(x)
        
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class MultitaskCTM(nn.Module):
    """
    Continuous Thought Machine with multitask support.
    
    Each task uses different neuron pairs from the synchronization matrix,
    allowing the same neural dynamics to support multiple outputs.
    Based on: https://arxiv.org/html/2505.05522v4
    """
    def __init__(
        self, 
        n_neurons: int, 
        max_memory: int, 
        max_ticks: int, 
        n_representation_size: int, 
        task_output_sizes: list[int]
    ):
        super(MultitaskCTM, self).__init__()

        self.initial_post = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))
        self.initial_pre = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_tasks = len(task_output_sizes)
        self.task_output_sizes = task_output_sizes
        
        # Calculate pairs per task - divide available pairs among tasks
        # Each task gets non-overlapping neuron pairs for distinct "views" into neural dynamics
        total_available_pairs = (n_neurons * (n_neurons - 1)) // 2
        self.n_pairs_per_task = min(total_available_pairs // self.n_tasks, n_neurons // 2)
        
        # Generate non-overlapping pairs for each task
        self._generate_task_pairs()
        
        # Learnable temporal decay factors r_ij for each task's neuron pairs
        self.task_decay_factors = nn.ParameterList([
            nn.Parameter(torch.ones(self.n_pairs_per_task) * 0.1)
            for _ in range(self.n_tasks)
        ])
        
        # Adaptive CNN encoder - handles any input shape
        # SHARED across all tasks
        self.image_encoder = AdaptiveCNNEncoder(n_representation_size)
        
        # The synapse model is a simple feedforward neural network
        # SHARED across all tasks - produces unified neural dynamics
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_neurons),
        )

        # Neuron-level models receive their whole pre activation history
        # SHARED across all tasks - each neuron has unique processing
        self.neuron_level_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_memory, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(n_neurons)
        ])

        # Attention mechanism dimensions
        self.attention_dim = 64
        
        # Key-Value projections from encoded input (4D -> attention_dim)
        self.key_projection = nn.Linear(n_representation_size, self.attention_dim)
        self.value_projection = nn.Linear(n_representation_size, self.attention_dim)
        
        # Task-specific query projections (each task's sync vector -> attention_dim)
        self.task_query_projections = nn.ModuleList([
            nn.Linear(self.n_pairs_per_task, self.attention_dim)
            for _ in range(self.n_tasks)
        ])
        
        # Task-specific synchronization readers
        # Each task reads its own synchronization subset to produce task-specific output
        self.task_sync_readers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.attention_dim + self.n_pairs_per_task, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            )
            for output_size in task_output_sizes
        ])
    
    def _generate_task_pairs(self):
        """Generate non-overlapping neuron pairs for each task."""
        # Generate all possible pairs
        all_pairs = []
        for i in range(self.n_neurons):
            for j in range(i + 1, self.n_neurons):
                all_pairs.append((i, j))
        
        # Shuffle to randomize assignment
        indices = torch.randperm(len(all_pairs))
        
        # Assign pairs to each task
        for task_idx in range(self.n_tasks):
            start_idx = task_idx * self.n_pairs_per_task
            end_idx = start_idx + self.n_pairs_per_task
            task_pair_indices = indices[start_idx:end_idx]
            task_pairs = torch.tensor([all_pairs[i] for i in task_pair_indices])
            # Register as buffer (not a parameter, but moves with device)
            self.register_buffer(f'pairs_task_{task_idx}', task_pairs)
    
    def get_task_pairs(self, task_idx: int) -> torch.Tensor:
        """Get neuron pairs for a specific task."""
        return getattr(self, f'pairs_task_{task_idx}')
    
    def _compute_task_synchronizations(
        self, 
        post_activations: torch.Tensor, 
        task_idx: int, 
        history_length: int
    ) -> torch.Tensor:
        """
        Compute synchronizations for a specific task using its neuron pairs.
        
        Args:
            post_activations: Neural post-activation history (batch_size, n_neurons, max_memory)
            task_idx: Which task's pairs to use
            history_length: How much history to consider
            
        Returns:
            Synchronization vector for the task (batch_size, n_pairs_per_task)
        """
        task_pairs = self.get_task_pairs(task_idx)
        task_decay = self.task_decay_factors[task_idx]
        
        synchronizations = []
        for pair_idx in range(self.n_pairs_per_task):
            pair = task_pairs[pair_idx]
            
            # Get post-activation histories for the pair (Z_i^t and Z_j^t)
            Z_i = post_activations[:, pair[0], -history_length:]  # (batch_size, history_length)
            Z_j = post_activations[:, pair[1], -history_length:]
            
            # Calculate temporal decay weights R_ij^t
            r_ij = task_decay[pair_idx]
            time_diffs = torch.arange(
                history_length - 1, -1, -1, 
                dtype=torch.float32, 
                device=r_ij.device
            )
            R_ij = torch.exp(-r_ij * time_diffs)  # Shape: (history_length,)
            
            # Calculate weighted dot product: S_ij^t = (Z_i^t)^T * diag(R_ij^t) * Z_j^t
            weighted_dot_product = torch.sum(Z_i * R_ij * Z_j, dim=-1)  # shape: (batch_size,)
            
            # Normalize by sqrt of sum of weights (as per paper Eq. 13)
            normalization_factor = torch.sqrt(torch.sum(R_ij))
            synchronization_value = weighted_dot_product / (normalization_factor + 1e-8)
            
            synchronizations.append(synchronization_value)
        
        return torch.stack(synchronizations, dim=1)  # Shape: (batch_size, n_pairs_per_task)
    
    def _compute_task_prediction(
        self,
        task_idx: int,
        sync_vector: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction for a specific task from its synchronizations.
        
        Args:
            task_idx: Which task to compute prediction for
            sync_vector: Synchronization vector (batch_size, n_pairs_per_task)
            keys: Attention keys from encoded input
            values: Attention values from encoded input
            
        Returns:
            Task prediction (batch_size, task_output_size)
        """
        # Create queries from this task's synchronizations
        queries = self.task_query_projections[task_idx](sync_vector)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        attention_scores = torch.sum(queries * keys, dim=-1, keepdim=True) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_features = attention_weights * values
        
        # Combine attended features with synchronizations
        combined_features = torch.cat([attended_features, sync_vector], dim=-1)
        
        # Read prediction from this task's sync reader
        return self.task_sync_readers[task_idx](combined_features)
    
    def forward(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        """
        Forward pass for a specific task.
        
        Args:
            x: Batch of images (batch_size, C, H, W) - handles any input shape
            task_idx: Which task to compute prediction for
            
        Returns:
            Task prediction (batch_size, task_output_size)
        """
        batch_size = x.shape[0]
        
        # Encode the image once at the beginning (SHARED) - CNN handles variable input
        encoded_image = self.image_encoder(x)
        
        # Attention mechanism: input attends to synchronizations
        keys = self.key_projection(encoded_image)
        values = self.value_projection(encoded_image)

        pre_activations = self.initial_pre.unsqueeze(0).expand(batch_size, -1, -1).clone()
        post_activations = self.initial_post.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for tick in range(self.max_ticks):
            # Get the last post-activation values for each neuron
            last_post_activations = post_activations[:, :, -1]
            
            # Concatenate encoded image with last post-activations
            synapse_input = torch.cat([encoded_image, last_post_activations], dim=1)
            
            # Synapse model produces new pre-activations (SHARED dynamics)
            new_pre_activations = self.synapse_model.forward(synapse_input)
            
            # Append new pre-activations to FIFO buffer
            pre_activations = torch.cat([pre_activations[:, :, 1:], new_pre_activations.unsqueeze(-1)], dim=-1)

            # Neuron-level models process their pre-activation history (SHARED)
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_pre_history = pre_activations[:, neuron_idx, :]
                post_activation = self.neuron_level_models[neuron_idx](neuron_pre_history)
                post_activations_list.append(post_activation)
            
            new_post_activations = torch.stack(post_activations_list, dim=1).squeeze(-1)
            
            # Append new post-activations to FIFO buffer
            post_activations = torch.cat([post_activations[:, :, 1:], new_post_activations.unsqueeze(-1)], dim=-1)
        
            # Calculate synchronizations for the SPECIFIED TASK only
            history_length = min(tick + 1, self.max_memory)
            sync_vector = self._compute_task_synchronizations(post_activations, task_idx, history_length)
            
            # Compute task-specific prediction
            prediction = self._compute_task_prediction(task_idx, sync_vector, keys, values)
            
            # Calculate confidence using softmax probabilities
            prediction_probs = F.softmax(prediction, dim=-1)
            max_confidence = torch.max(prediction_probs, dim=-1)[0]
            
            # Early stopping: if any sample has confidence > 0.8, or we're at the last tick
            if torch.any(max_confidence > 0.8) or tick == self.max_ticks - 1:
                return prediction
        
        return prediction  # Fallback (should not reach here)
    
    def forward_all_tasks(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass computing predictions for ALL tasks simultaneously.
        More efficient than calling forward() multiple times as neural dynamics are computed once.
        
        Args:
            x: Batch of images (batch_size, C, H, W) - handles any input shape
            
        Returns:
            List of predictions, one per task
        """
        batch_size = x.shape[0]
        
        # Encode the image once at the beginning (SHARED) - CNN handles variable input
        encoded_image = self.image_encoder(x)
        
        keys = self.key_projection(encoded_image)
        values = self.value_projection(encoded_image)

        pre_activations = self.initial_pre.unsqueeze(0).expand(batch_size, -1, -1).clone()
        post_activations = self.initial_post.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for tick in range(self.max_ticks):
            last_post_activations = post_activations[:, :, -1]
            synapse_input = torch.cat([encoded_image, last_post_activations], dim=1)
            new_pre_activations = self.synapse_model.forward(synapse_input)
            pre_activations = torch.cat([pre_activations[:, :, 1:], new_pre_activations.unsqueeze(-1)], dim=-1)

            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_pre_history = pre_activations[:, neuron_idx, :]
                post_activation = self.neuron_level_models[neuron_idx](neuron_pre_history)
                post_activations_list.append(post_activation)
            
            new_post_activations = torch.stack(post_activations_list, dim=1).squeeze(-1)
            post_activations = torch.cat([post_activations[:, :, 1:], new_post_activations.unsqueeze(-1)], dim=-1)
        
        # At the final tick, compute all task predictions
        history_length = min(self.max_ticks, self.max_memory)
        predictions = []
        for task_idx in range(self.n_tasks):
            sync_vector = self._compute_task_synchronizations(post_activations, task_idx, history_length)
            prediction = self._compute_task_prediction(task_idx, sync_vector, keys, values)
            predictions.append(prediction)
        
        return predictions


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


class MultitaskMNISTDataset(Dataset):
    """
    MNIST dataset with multiple task labels.
    Task 0: Digit classification (0-9)
    Task 1: Even/Odd classification (0=even, 1=odd)
    """
    def __init__(self, mnist_dataset: Dataset):
        self.mnist_dataset = mnist_dataset
    
    def __len__(self) -> int:
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, digit_label = self.mnist_dataset[idx]
        
        # Task 0: Original digit classification
        # Task 1: Even (0) or Odd (1)
        even_odd_label = digit_label % 2
        
        labels = {
            'digit': digit_label,
            'even_odd': even_odd_label
        }
        return image, labels


# CIFAR-10 class groupings for coarse classification
CIFAR10_COARSE_LABELS = {
    # Animals: bird, cat, deer, dog, frog, horse
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
    # Vehicles: airplane, automobile, ship, truck
    0: 1, 1: 1, 8: 1, 9: 1
}


class MultitaskCIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset with multiple task labels.
    Task 0: Fine classification (10 classes)
    Task 1: Coarse classification - Animal (0) vs Vehicle (1)
    """
    def __init__(self, cifar_dataset: Dataset):
        self.cifar_dataset = cifar_dataset
    
    def __len__(self) -> int:
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, fine_label = self.cifar_dataset[idx]
        
        # Task 0: Fine-grained classification (10 classes)
        # Task 1: Coarse classification - Animal (0) vs Vehicle (1)
        coarse_label = CIFAR10_COARSE_LABELS[fine_label]
        
        labels = {
            'fine': fine_label,
            'coarse': coarse_label
        }
        return image, labels


class CombinedMultitaskDataset(Dataset):
    """
    Combined dataset that mixes MNIST and CIFAR-10 for simultaneous training.
    
    Tasks:
    - Task 0: MNIST digit classification (10 classes)
    - Task 1: MNIST even/odd (2 classes)
    - Task 2: CIFAR-10 fine classification (10 classes)
    - Task 3: CIFAR-10 coarse classification - Animal vs Vehicle (2 classes)
    
    Uses -1 for non-applicable task labels (masked in loss computation).
    Supports None for either dataset to train on single dataset.
    """
    def __init__(self, mnist_dataset: Dataset | None, cifar_dataset: Dataset | None):
        self.mnist_dataset = mnist_dataset
        self.cifar_dataset = cifar_dataset
        self.mnist_len = len(mnist_dataset) if mnist_dataset is not None else 0
        self.cifar_len = len(cifar_dataset) if cifar_dataset is not None else 0
        self.total_len = self.mnist_len + self.cifar_len
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int], str]:
        if self.mnist_dataset is not None and idx < self.mnist_len:
            # MNIST sample
            image, digit_label = self.mnist_dataset[idx]
            even_odd_label = digit_label % 2
            
            labels = {
                'mnist_digit': digit_label,
                'mnist_even_odd': even_odd_label,
                'cifar_fine': -1,  # Not applicable
                'cifar_coarse': -1  # Not applicable
            }
            return image, labels, 'mnist'
        else:
            # CIFAR-10 sample
            cifar_idx = idx - self.mnist_len
            image, fine_label = self.cifar_dataset[cifar_idx]
            coarse_label = CIFAR10_COARSE_LABELS[fine_label]
            
            labels = {
                'mnist_digit': -1,  # Not applicable
                'mnist_even_odd': -1,  # Not applicable
                'cifar_fine': fine_label,
                'cifar_coarse': coarse_label
            }
            return image, labels, 'cifar'


def train_multitask_model(
    model: MultitaskCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    task_names: list[str]
) -> TrainingMetrics:
    """
    Training loop for MultitaskCTM with alternating task training.
    Each batch trains on one task, alternating between tasks.
    
    Returns:
        TrainingMetrics object with loss and accuracy history
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    metrics = TrainingMetrics(task_names=task_names)
    
    for epoch in range(epochs):
        task_losses = {name: 0.0 for name in task_names}
        task_correct = {name: 0 for name in task_names}
        task_total = {name: 0 for name in task_names}
        task_batch_count = {name: 0 for name in task_names}
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            
            # Alternate between tasks each batch
            task_idx = batch_idx % model.n_tasks
            task_name = task_names[task_idx]
            target = labels[task_name].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for specific task
            output = model(data, task_idx=task_idx)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Statistics
            task_losses[task_name] += loss.item()
            task_batch_count[task_name] += 1
            _, predicted = torch.max(output.data, 1)
            task_total[task_name] += target.size(0)
            task_correct[task_name] += (predicted == target).sum().item()
            
            # Track batch losses for detailed plotting
            metrics.batch_losses[task_name].append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Task: {task_name}, Loss: {loss.item():.4f}')
        
        # Store epoch-level metrics
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


def train_combined_multitask_model(
    model: MultitaskCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int]
) -> TrainingMetrics:
    """
    Training loop for MultitaskCTM on combined MNIST + CIFAR-10 dataset.
    Each sample is from either MNIST or CIFAR-10, with masked losses for non-applicable tasks.
    
    Args:
        task_index_mapping: Maps original task index (0-3) to new model task index
    
    Returns:
        TrainingMetrics object with loss and accuracy history
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    metrics = TrainingMetrics(task_names=task_names)
    
    # Map dataset source to original task names
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse']
    
    for epoch in range(epochs):
        task_losses = {name: 0.0 for name in task_names}
        task_correct = {name: 0 for name in task_names}
        task_total = {name: 0 for name in task_names}
        task_batch_count = {name: 0 for name in task_names}
        
        for batch_idx, (stacked_images, labels, sources, batch_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Process MNIST samples
            if stacked_images['mnist'] is not None:
                mnist_data = stacked_images['mnist'].to(device)
                mnist_batch_indices = batch_indices['mnist']
                
                # Check which MNIST tasks are selected
                for original_idx in [0, 1]:  # mnist_digit, mnist_even_odd
                    if original_idx in task_index_mapping:
                        model_task_idx = task_index_mapping[original_idx]
                        original_task_name = original_task_names[original_idx]
                        task_name = task_names[model_task_idx]
                        
                        target = torch.tensor([labels[original_task_name][i] for i in mnist_batch_indices]).to(device)
                        
                        if (target >= 0).any():
                            output = model(mnist_data, task_idx=model_task_idx)
                            loss = criterion(output, target)
                            total_loss = total_loss + loss
                            
                            task_losses[task_name] += loss.item()
                            task_batch_count[task_name] += 1
                            metrics.batch_losses[task_name].append(loss.item())
                            
                            valid_mask = target >= 0
                            if valid_mask.any():
                                _, predicted = torch.max(output[valid_mask].data, 1)
                                task_total[task_name] += valid_mask.sum().item()
                                task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
            
            # Process CIFAR samples
            if stacked_images['cifar'] is not None:
                cifar_data = stacked_images['cifar'].to(device)
                cifar_batch_indices = batch_indices['cifar']
                
                # Check which CIFAR tasks are selected
                for original_idx in [2, 3]:  # cifar_fine, cifar_coarse
                    if original_idx in task_index_mapping:
                        model_task_idx = task_index_mapping[original_idx]
                        original_task_name = original_task_names[original_idx]
                        task_name = task_names[model_task_idx]
                        
                        target = torch.tensor([labels[original_task_name][i] for i in cifar_batch_indices]).to(device)
                        
                        if (target >= 0).any():
                            output = model(cifar_data, task_idx=model_task_idx)
                            loss = criterion(output, target)
                            total_loss = total_loss + loss
                            
                            task_losses[task_name] += loss.item()
                            task_batch_count[task_name] += 1
                            metrics.batch_losses[task_name].append(loss.item())
                            
                            valid_mask = target >= 0
                            if valid_mask.any():
                                _, predicted = torch.max(output[valid_mask].data, 1)
                                task_total[task_name] += valid_mask.sum().item()
                                task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
            
            if total_loss.requires_grad and total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()
            
            if batch_idx % 25 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f}')
        
        # Store epoch-level metrics
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
    model: MultitaskCTM,
    test_loader: DataLoader,
    device: torch.device,
    task_names: list[str],
    task_index_mapping: dict[int, int]
) -> dict[str, float]:
    """
    Evaluate model on combined MNIST + CIFAR-10 test set.
    
    Args:
        task_index_mapping: Maps original task index (0-3) to new model task index
    
    Returns:
        Dictionary mapping task names to test accuracies
    """
    model.eval()
    
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    original_task_names = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse']
    
    with torch.no_grad():
        for stacked_images, labels, sources, batch_indices in test_loader:
            # Evaluate MNIST samples
            if stacked_images['mnist'] is not None:
                mnist_data = stacked_images['mnist'].to(device)
                mnist_batch_indices = batch_indices['mnist']
                
                for original_idx in [0, 1]:  # mnist_digit, mnist_even_odd
                    if original_idx in task_index_mapping:
                        model_task_idx = task_index_mapping[original_idx]
                        original_task_name = original_task_names[original_idx]
                        task_name = task_names[model_task_idx]
                        
                        target = torch.tensor([labels[original_task_name][i] for i in mnist_batch_indices]).to(device)
                        
                        if (target >= 0).any():
                            output = model(mnist_data, task_idx=model_task_idx)
                            valid_mask = target >= 0
                            _, predicted = torch.max(output[valid_mask].data, 1)
                            task_total[task_name] += valid_mask.sum().item()
                            task_correct[task_name] += (predicted == target[valid_mask]).sum().item()
            
            # Evaluate CIFAR samples
            if stacked_images['cifar'] is not None:
                cifar_data = stacked_images['cifar'].to(device)
                cifar_batch_indices = batch_indices['cifar']
                
                for original_idx in [2, 3]:  # cifar_fine, cifar_coarse
                    if original_idx in task_index_mapping:
                        model_task_idx = task_index_mapping[original_idx]
                        original_task_name = original_task_names[original_idx]
                        task_name = task_names[model_task_idx]
                        
                        target = torch.tensor([labels[original_task_name][i] for i in cifar_batch_indices]).to(device)
                        
                        if (target >= 0).any():
                            output = model(cifar_data, task_idx=model_task_idx)
                            valid_mask = target >= 0
                            _, predicted = torch.max(output[valid_mask].data, 1)
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


def evaluate_multitask_model(
    model: MultitaskCTM,
    test_loader: DataLoader,
    device: torch.device,
    task_names: list[str]
) -> dict[str, float]:
    """
    Evaluate model on all tasks.
    
    Returns:
        Dictionary mapping task names to test accuracies
    """
    model.eval()
    
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            
            # Get predictions for all tasks at once
            predictions = model.forward_all_tasks(data)
            
            for task_idx, task_name in enumerate(task_names):
                target = labels[task_name].to(device)
                _, predicted = torch.max(predictions[task_idx].data, 1)
                task_total[task_name] += target.size(0)
                task_correct[task_name] += (predicted == target).sum().item()
    
    test_accuracies = {}
    print('\n=== Test Results ===')
    for task_name in task_names:
        accuracy = 100. * task_correct[task_name] / task_total[task_name]
        test_accuracies[task_name] = accuracy
        print(f'  {task_name}: Accuracy={accuracy:.2f}%')
    print()
    
    return test_accuracies


def get_neural_dynamics(
    model: MultitaskCTM,
    x: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Extract neural dynamics and synchronizations for visualization.
    
    Returns:
        Tuple of (post_activation_history, task_synchronizations_dict)
    """
    model.eval()
    batch_size = x.shape[0]
    
    with torch.no_grad():
        # CNN encoder handles variable input shapes directly
        encoded_image = model.image_encoder(x)
        
        pre_activations = model.initial_pre.unsqueeze(0).expand(batch_size, -1, -1).clone()
        post_activations = model.initial_post.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Store post-activation history across all ticks
        post_history = []
        
        for tick in range(model.max_ticks):
            last_post_activations = post_activations[:, :, -1]
            synapse_input = torch.cat([encoded_image, last_post_activations], dim=1)
            new_pre_activations = model.synapse_model.forward(synapse_input)
            pre_activations = torch.cat([pre_activations[:, :, 1:], new_pre_activations.unsqueeze(-1)], dim=-1)

            post_activations_list = []
            for neuron_idx in range(model.n_neurons):
                neuron_pre_history = pre_activations[:, neuron_idx, :]
                post_activation = model.neuron_level_models[neuron_idx](neuron_pre_history)
                post_activations_list.append(post_activation)
            
            new_post_activations = torch.stack(post_activations_list, dim=1).squeeze(-1)
            post_activations = torch.cat([post_activations[:, :, 1:], new_post_activations.unsqueeze(-1)], dim=-1)
            
            post_history.append(new_post_activations.cpu())
        
        # Stack into (batch_size, n_neurons, max_ticks)
        post_history_tensor = torch.stack(post_history, dim=2)
        
        # Compute final synchronizations for each task
        history_length = min(model.max_ticks, model.max_memory)
        task_syncs = {}
        for task_idx in range(model.n_tasks):
            sync_vector = model._compute_task_synchronizations(post_activations, task_idx, history_length)
            task_syncs[task_idx] = sync_vector.cpu()
    
    return post_history_tensor, task_syncs


def plot_training_results(
    metrics: TrainingMetrics,
    test_accuracies: dict[str, float],
    model: MultitaskCTM,
    sample_image: torch.Tensor,
    device: torch.device,
    output_prefix: str
):
    """
    Plot comprehensive training results in multiple figures.
    
    Generates:
    - Figure 1: Training loss and accuracy curves per task
    - Figure 2: Batch-level loss curves
    - Figure 3: Neural dynamics visualization
    - Figure 4: Task synchronization comparison
    - Figure 5: Final accuracy comparison
    """
    task_names = metrics.task_names
    # Color palette for different task names
    color_palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(task_names)}
    
    # Figure 1: Training Loss and Accuracy per Epoch
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Training Progress per Task', fontsize=14, fontweight='bold')
    
    # Loss subplot
    ax_loss = axes1[0]
    for task_name in task_names:
        epochs = range(1, len(metrics.epoch_losses[task_name]) + 1)
        ax_loss.plot(epochs, metrics.epoch_losses[task_name], 
                     marker='o', label=f'{task_name}', color=colors[task_name], linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax_acc = axes1[1]
    for task_name in task_names:
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
    
    # Figure 2: Batch-level Loss (smoothed)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.suptitle('Batch-level Training Loss (Smoothed)', fontsize=14, fontweight='bold')
    
    window_size = 50
    for task_name in task_names:
        losses = metrics.batch_losses[task_name]
        if len(losses) > window_size:
            # Moving average smoothing
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(smoothed, label=f'{task_name}', color=colors[task_name], alpha=0.8)
        else:
            ax2.plot(losses, label=f'{task_name}', color=colors[task_name], alpha=0.8)
    
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_fig2_batch_losses.png', dpi=150, bbox_inches='tight')
    
    # Figure 3: Neural Dynamics Visualization
    post_history, task_syncs = get_neural_dynamics(model, sample_image.unsqueeze(0).to(device), device)
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Neural Dynamics Visualization', fontsize=14, fontweight='bold')
    
    # Sample image
    ax_img = axes3[0, 0]
    
    # Handle image plotting based on shape
    img_np = sample_image.cpu().numpy()
    
    if img_np.shape[0] == 3:  # RGB image (CIFAR) - (3, H, W) -> (H, W, 3)
        # Unnormalize for display: (img * std) + mean
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        ax_img.imshow(img_np)
    else:  # Grayscale (MNIST) - (1, H, W) -> (H, W)
        # Unnormalize
        img_np = img_np * 0.3081 + 0.1307
        img_np = np.clip(img_np, 0, 1)
        ax_img.imshow(img_np.squeeze(), cmap='gray')
    
    ax_img.set_title('Input Image')
    ax_img.axis('off')
    
    # Neural activity heatmap (neurons x ticks)
    ax_activity = axes3[0, 1]
    activity_data = post_history[0].numpy()  # First sample: (n_neurons, max_ticks)
    im = ax_activity.imshow(activity_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_activity.set_xlabel('Tick')
    ax_activity.set_ylabel('Neuron')
    ax_activity.set_title('Post-Activation Dynamics')
    plt.colorbar(im, ax=ax_activity, label='Activation')
    
    # Individual neuron traces
    ax_traces = axes3[1, 0]
    n_neurons_to_plot = min(10, model.n_neurons)
    for i in range(n_neurons_to_plot):
        ax_traces.plot(activity_data[i], alpha=0.7, label=f'N{i}')
    ax_traces.set_xlabel('Tick')
    ax_traces.set_ylabel('Activation')
    ax_traces.set_title(f'Sample Neuron Traces (First {n_neurons_to_plot})')
    ax_traces.legend(loc='upper right', fontsize=8, ncol=2)
    ax_traces.grid(True, alpha=0.3)
    
    # Synchronization comparison between tasks
    ax_sync = axes3[1, 1]
    x_pos = np.arange(model.n_pairs_per_task)
    width = 0.35
    
    for task_idx, task_name in enumerate(task_names):
        sync_values = task_syncs[task_idx][0].numpy()
        offset = (task_idx - 0.5) * width
        ax_sync.bar(x_pos + offset, sync_values, width, 
                    label=f'{task_name}', color=colors[task_name], alpha=0.8)
    
    ax_sync.set_xlabel('Neuron Pair Index')
    ax_sync.set_ylabel('Synchronization Value')
    ax_sync.set_title('Task-Specific Synchronizations')
    ax_sync.legend()
    ax_sync.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_fig3_neural_dynamics.png', dpi=150, bbox_inches='tight')
    
    # Figure 4: Task Pair Visualization
    n_tasks = len(task_names)
    fig4, axes4 = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5))
    fig4.suptitle('Task-Specific Neuron Pair Assignments', fontsize=14, fontweight='bold')
    
    # Handle single task case
    if n_tasks == 1:
        axes4 = [axes4]
    
    for task_idx, (ax, task_name) in enumerate(zip(axes4, task_names)):
        # Create connectivity matrix
        pairs = model.get_task_pairs(task_idx).cpu().numpy()
        conn_matrix = np.zeros((model.n_neurons, model.n_neurons))
        for i, j in pairs:
            conn_matrix[i, j] = 1
            conn_matrix[j, i] = 1
        
        im = ax.imshow(conn_matrix, cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
        ax.set_title(f'Task: {task_name} ({len(pairs)} pairs)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    fig4.savefig(f'{output_prefix}_fig4_task_pairs.png', dpi=150, bbox_inches='tight')
    
    # Figure 5: Final Accuracy Comparison
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    fig5.suptitle('Final Model Performance', fontsize=14, fontweight='bold')
    
    x = np.arange(len(task_names))
    width = 0.35
    
    train_accs = [metrics.epoch_accuracies[name][-1] for name in task_names]
    test_accs = [test_accuracies[name] for name in task_names]
    
    bars1 = ax5.bar(x - width/2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax5.bar(x + width/2, test_accs, width, label='Test', color='#9b59b6', alpha=0.8)
    
    ax5.set_xlabel('Task')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_xticks(x)
    ax5.set_xticklabels([name.replace('_', '/').title() for name in task_names])
    ax5.legend()
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig5.savefig(f'{output_prefix}_fig5_final_accuracy.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print(f'\nPlots saved with prefix "{output_prefix}":')
    print(f'  - {output_prefix}_fig1_training_curves.png')
    print(f'  - {output_prefix}_fig2_batch_losses.png')
    print(f'  - {output_prefix}_fig3_neural_dynamics.png')
    print(f'  - {output_prefix}_fig4_task_pairs.png')
    print(f'  - {output_prefix}_fig5_final_accuracy.png')


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: MultitaskCTM):
    """Print detailed information about the MultitaskCTM model."""
    total_params = count_parameters(model)
    print(f'\n=== MultitaskCTM Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Number of tasks: {model.n_tasks}')
    print(f'Pairs per task: {model.n_pairs_per_task}')
    print(f'Task output sizes: {model.task_output_sizes}')
    print(f'Image encoder: AdaptiveCNN (handles any input shape)')
    
    # Print parameter breakdown by component
    print(f'\nShared components:')
    shared_components = ['image_encoder', 'synapse_model', 'neuron_level_models', 
                         'key_projection', 'value_projection']
    for name in shared_components:
        if hasattr(model, name):
            module = getattr(model, name)
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {module_params:,} parameters')
    
    print(f'\nTask-specific components:')
    task_components = ['task_decay_factors', 'task_query_projections', 'task_sync_readers']
    for name in task_components:
        if hasattr(model, name):
            module = getattr(model, name)
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {module_params:,} parameters')
    
    # Print memory buffers (pre/post activations)
    pre_act_params = model.initial_pre.numel()
    post_act_params = model.initial_post.numel()
    print(f'\nLearnable initial states:')
    print(f'  initial_pre: {pre_act_params:,} elements')
    print(f'  initial_post: {post_act_params:,} elements')
    
    # Print neuron pairs per task
    print(f'\nNeuron pair assignments:')
    for task_idx in range(model.n_tasks):
        pairs = model.get_task_pairs(task_idx)
        print(f'  Task {task_idx}: {pairs.shape[0]} pairs')
    print(f'=========================================\n')


def custom_collate_fn(batch: list) -> tuple:
    """Custom collate function for combined dataset with mixed image sizes."""
    # Separate MNIST and CIFAR images since they have different shapes
    mnist_images = []
    cifar_images = []
    labels_dict = {key: [] for key in batch[0][1].keys()}
    sources = []
    batch_indices = {'mnist': [], 'cifar': []}
    
    for idx, (image, labels, source) in enumerate(batch):
        if source == 'mnist':
            mnist_images.append(image)
            batch_indices['mnist'].append(idx)
        else:
            cifar_images.append(image)
            batch_indices['cifar'].append(idx)
        
        for key, val in labels.items():
            labels_dict[key].append(val)
        sources.append(source)
    
    # Stack images by type
    stacked_images = {
        'mnist': torch.stack(mnist_images) if mnist_images else None,
        'cifar': torch.stack(cifar_images) if cifar_images else None
    }
    
    return stacked_images, labels_dict, sources, batch_indices


def train_combined(device: torch.device, selected_tasks: list[str], epochs: int, n_neurons: int):
    """
    Train MultitaskCTM on selected tasks.
    
    Args:
        device: Torch device
        selected_tasks: List of task names to train on (from ALL_TASKS)
        epochs: Number of training epochs
        n_neurons: Number of neurons in CTM
    """
    print('\n' + '='*60)
    print(f'TRAINING ON TASKS: {", ".join(selected_tasks)}')
    print('='*60)
    
    # Determine which datasets are needed
    needs_mnist = any(ALL_TASKS[t]['dataset'] == 'mnist' for t in selected_tasks)
    needs_cifar = any(ALL_TASKS[t]['dataset'] == 'cifar' for t in selected_tasks)
    
    # MNIST transforms
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # CIFAR-10 transforms
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load only needed datasets
    mnist_train = mnist_test = None
    cifar_train = cifar_test = None
    
    if needs_mnist:
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    
    if needs_cifar:
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    
    # Combined datasets
    train_dataset = CombinedMultitaskDataset(mnist_train, cifar_train)
    test_dataset = CombinedMultitaskDataset(mnist_test, cifar_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    
    # Build task configuration for selected tasks only
    # Map original task indices to new indices
    all_task_order = ['mnist_digit', 'mnist_even_odd', 'cifar_fine', 'cifar_coarse']
    task_names = selected_tasks
    task_output_sizes = [ALL_TASKS[t]['output_size'] for t in selected_tasks]
    
    # Create mapping from original task index to selected task index
    task_index_mapping = {}
    for new_idx, task_name in enumerate(selected_tasks):
        original_idx = all_task_order.index(task_name)
        task_index_mapping[original_idx] = new_idx
    
    print(f'Task configuration:')
    for i, task_name in enumerate(task_names):
        print(f'  Task {i}: {task_name} ({ALL_TASKS[task_name]["description"]})')
    
    model = MultitaskCTM(
        n_neurons=n_neurons,
        max_memory=10, 
        max_ticks=15, 
        n_representation_size=16,
        task_output_sizes=task_output_sizes
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
    
    # Create output prefix from task names
    output_prefix = '_'.join([t.split('_')[0][0] + t.split('_')[1][0] for t in selected_tasks])
    
    torch.save(model.state_dict(), f'{output_prefix}_multitask_ctm.pth')
    print(f'Model saved to {output_prefix}_multitask_ctm.pth')
    
    # Get sample image for visualization
    if needs_mnist:
        sample_image, _, _ = test_dataset[0]
    else:
        sample_image, _, _ = test_dataset[0]
    
    print('\nGenerating training plots...')
    plot_training_results(
        metrics, test_accuracies, model, sample_image, device,
        output_prefix=output_prefix
    )
    
    return model, metrics, test_accuracies


def train_on_mnist(device: torch.device):
    """Train MultitaskCTM on MNIST dataset only."""
    print('\n' + '='*60)
    print('TRAINING ON MNIST ONLY')
    print('='*60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    base_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    base_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_dataset = MultitaskMNISTDataset(base_train_dataset)
    test_dataset = MultitaskMNISTDataset(base_test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    task_names = ['digit', 'even_odd']
    task_output_sizes = [10, 2]
    
    model = MultitaskCTM(
        n_neurons=30, 
        max_memory=10, 
        max_ticks=15, 
        n_representation_size=8,
        task_output_sizes=task_output_sizes
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = train_multitask_model(
        model, train_loader, optimizer, 
        epochs=5, device=device, task_names=task_names
    )
    
    test_accuracies = evaluate_multitask_model(
        model, test_loader, device=device, task_names=task_names
    )
    
    torch.save(model.state_dict(), 'mnist_multitask_ctm.pth')
    print('Model saved to mnist_multitask_ctm.pth')
    
    sample_image, _ = test_dataset[0]
    
    print('\nGenerating MNIST plots...')
    plot_training_results(
        metrics, test_accuracies, model, sample_image, device, 
        output_prefix='mnist'
    )
    
    return model, metrics, test_accuracies


def train_on_cifar10(device: torch.device):
    """Train MultitaskCTM on CIFAR-10 dataset only."""
    print('\n' + '='*60)
    print('TRAINING ON CIFAR-10 ONLY')
    print('='*60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    base_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    base_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_dataset = MultitaskCIFAR10Dataset(base_train_dataset)
    test_dataset = MultitaskCIFAR10Dataset(base_test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    task_names = ['fine', 'coarse']
    task_output_sizes = [10, 2]
    
    model = MultitaskCTM(
        n_neurons=40,
        max_memory=10, 
        max_ticks=15, 
        n_representation_size=16,
        task_output_sizes=task_output_sizes
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = train_multitask_model(
        model, train_loader, optimizer, 
        epochs=5, device=device, task_names=task_names
    )
    
    test_accuracies = evaluate_multitask_model(
        model, test_loader, device=device, task_names=task_names
    )
    
    torch.save(model.state_dict(), 'cifar10_multitask_ctm.pth')
    print('Model saved to cifar10_multitask_ctm.pth')
    
    sample_image, _ = test_dataset[0]
    
    print('\nGenerating CIFAR-10 plots...')
    plot_training_results(
        metrics, test_accuracies, model, sample_image, device,
        output_prefix='cifar10'
    )
    
    return model, metrics, test_accuracies


def main():
    """Main training function with command-line arguments for task selection."""
    parser = argparse.ArgumentParser(
        description='Train MultitaskCTM on selected tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
  mnist_digit      - MNIST digit classification (0-9)
  mnist_even_odd   - MNIST even/odd classification
  cifar_fine       - CIFAR-10 fine classification (10 classes)
  cifar_coarse     - CIFAR-10 coarse (animal vs vehicle)

Examples:
  # Train on CIFAR fine only
  python multitask.py --tasks cifar_fine
  
  # Train on MNIST digit and CIFAR fine
  python multitask.py --tasks mnist_digit cifar_fine
  
  # Train on all tasks
  python multitask.py --tasks mnist_digit mnist_even_odd cifar_fine cifar_coarse
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
        default=20,
        help='Number of neurons in CTM (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Validate selected tasks
    if not args.tasks:
        print("Error: At least one task must be selected")
        return
    
    # Check for duplicates
    if len(args.tasks) != len(set(args.tasks)):
        print("Error: Duplicate tasks detected")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Selected tasks: {", ".join(args.tasks)}')
    print(f'Epochs: {args.epochs}')
    print(f'Neurons: {args.neurons}')
    
    # Train on selected tasks
    train_combined(device, args.tasks, args.epochs, args.neurons)
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
