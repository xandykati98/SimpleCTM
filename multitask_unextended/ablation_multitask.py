import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field

class SimplifiedCTM(nn.Module):
    def __init__(self, n_neurons: int, max_memory: int = 10, max_ticks: int = 10, n_representation_size: int = 4):
        super(SimplifiedCTM, self).__init__()

        self.input_shape: tuple = (1, 28, 28)
        self.initial_post = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))
        self.initial_pre = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_pairs = (n_neurons // 2) // 2

        self.pairs = torch.randint(0, n_neurons, (self.n_pairs, 2))
        
        # Learnable temporal decay factors r_ij for each neuron pair
        # Initialize with small positive values to start with some temporal weighting
        self.decay_factors = nn.Parameter(torch.ones(self.n_pairs) * 0.1)

        flattened_input_shape = self.input_shape[1] * self.input_shape[2]
        
        # Image encoder - reduces image to n_representation_size-item array, runs only once
        self.image_encoder = nn.Sequential(
            nn.Linear(flattened_input_shape, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, n_representation_size),
        )
        
        # the synapse model is a simple feedforward neural network
        # inputs for this are the encoded image (n_representation_size items) and the last post-activation values
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_neurons),
        )

        # they will receive their whole pre activation history
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
        
        # Query projection from synchronizations (n_pairs -> attention_dim)
        self.query_projection = nn.Linear(self.n_pairs, self.attention_dim)
        
        # Synchronization reader - now takes attended features + original synchronizations
        self.syncronization_reader = nn.Sequential(
            nn.Linear(self.attention_dim + self.n_pairs, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x: torch.Tensor):
        # x is a batch of images
        batch_size = x.shape[0]
        
        # Encode the image once at the beginning
        flattened_x = x.flatten(1)  # shape: (batch_size, 784)
        encoded_image = self.image_encoder(flattened_x)  # shape: (batch_size, n_representation_size)
        
        # Attention mechanism: input attends to synchronizations
        # Create keys and values from encoded input
        keys = self.key_projection(encoded_image)      # Shape: (batch_size, attention_dim)
        values = self.value_projection(encoded_image)  # Shape: (batch_size, attention_dim)

        pre_activations = self.initial_pre.unsqueeze(0).expand(batch_size, -1, -1).clone()
        post_activations = self.initial_post.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for tick in range(self.max_ticks):
            # Get the last post-activation values for each neuron
            last_post_activations = post_activations[:, :, -1]  # shape: (batch_size, n_neurons)
            
            # Concatenate encoded image with last post-activations
            synapse_input = torch.cat([encoded_image, last_post_activations], dim=1)  # shape: (batch_size, n_representation_size + n_neurons)
            
            new_pre_activations = self.synapse_model.forward(synapse_input)  # shape: (batch_size, n_neurons)
            
            # Append new pre-activations to FIFO buffer (shift left and add new values)
            # Create new buffer by concatenating shifted history with new values
            pre_activations = torch.cat([pre_activations[:, :, 1:], new_pre_activations.unsqueeze(-1)], dim=-1)

            # Neuron-level models will receive their whole pre activation history
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                # Get the pre-activation history for this neuron
                neuron_pre_history = pre_activations[:, neuron_idx, :]  # shape: (batch_size, max_memory)
                # Pass through the neuron-level model
                post_activation = self.neuron_level_models[neuron_idx](neuron_pre_history)  # shape: (batch_size, 1)
                post_activations_list.append(post_activation)
            
            # Stack all post-activations
            new_post_activations = torch.stack(post_activations_list, dim=1).squeeze(-1)  # shape: (batch_size, n_neurons)
            
            # Append new post-activations to FIFO buffer
            # Create new buffer by concatenating shifted history with new values
            post_activations = torch.cat([post_activations[:, :, 1:], new_post_activations.unsqueeze(-1)], dim=-1)
        
            # Calculate synchronizations with learnable temporal decay
            sincronizations = []
            # Use all available history (limited by max_memory)
            history_length = min(tick + 1, self.max_memory)
            
            for pair_idx, pair in enumerate(self.pairs):
                # Get post-activation histories for the pair (Z_i^t and Z_j^t)
                # Use the most recent history_length entries
                Z_i = post_activations[:, pair[0], -history_length:]  # (batch_size, history_length)
                Z_j = post_activations[:, pair[1], -history_length:]
                
                # Calculate temporal decay weights R_ij^t
                # R_ij^t = [exp(-r_ij*(history_length-1)), exp(-r_ij*(history_length-2)), ..., exp(0)]
                r_ij = self.decay_factors[pair_idx]
                time_diffs = torch.arange(history_length - 1, -1, -1, dtype=torch.float32, device=r_ij.device)
                R_ij = torch.exp(-r_ij * time_diffs)  # Shape: (history_length,)
                
                # Calculate weighted dot product: S_ij^t = (Z_i^t)^T * diag(R_ij^t) * Z_j^t
                # Sum across time dimension to get synchronization value per batch item
                weighted_dot_product = torch.sum(Z_i * R_ij * Z_j, dim=-1)  # shape: (batch_size,)
                
                # Normalize by sum of weights to prevent any pair from dominating
                normalization_factor = torch.sum(R_ij)
                synchronization_value = weighted_dot_product / (normalization_factor + 1e-8)  # shape: (batch_size,)
                
                sincronizations.append(synchronization_value)

            # Stack synchronizations into a vector for the reader
            sincronizations_vector = torch.stack(sincronizations, dim=1)  # Shape: (batch_size, n_pairs)
            
            
            # Create queries from synchronizations
            queries = self.query_projection(sincronizations_vector)  # Shape: (batch_size, attention_dim)
            
            # Compute attention scores: Q * K^T / sqrt(d_k)
            attention_scores = torch.sum(queries * keys, dim=-1, keepdim=True) / (self.attention_dim ** 0.5)  # Shape: (batch_size, 1)
            attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, 1)
            
            # Apply attention to values
            attended_features = attention_weights * values  # Shape: (batch_size, attention_dim)
            
            # Combine attended features with original synchronizations
            combined_features = torch.cat([attended_features, sincronizations_vector], dim=-1)  # Shape: (batch_size, attention_dim + n_pairs)
            
            sincronizations_read = self.syncronization_reader(combined_features)
            # get the prediction
            prediction = sincronizations_read
            
            # Calculate confidence using softmax probabilities
            prediction_probs = F.softmax(prediction, dim=-1)
            max_confidence = torch.max(prediction_probs, dim=-1)[0]  # Get max probability for each sample
            
            # Early stopping: if any sample has confidence > 0.8, or we're at the last tick
            if torch.any(max_confidence > 0.8) or tick == self.max_ticks - 1:
                return prediction


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics across epochs."""
    epoch_losses: list[float] = field(default_factory=list)
    epoch_accuracies: list[float] = field(default_factory=list)
    batch_losses: list[float] = field(default_factory=list)


def train_model(model: SimplifiedCTM, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, epochs: int, device: torch.device) -> TrainingMetrics:
    """Training loop for the SimplifiedCTM model."""
    model.train()
    metrics = TrainingMetrics()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Track batch losses
            metrics.batch_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / batch_count
        accuracy = 100. * correct / total
        metrics.epoch_losses.append(avg_loss)
        metrics.epoch_accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{epochs} completed - '
              f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return metrics


def evaluate_model(
    model: SimplifiedCTM,
    test_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return accuracy


def get_neural_dynamics(
    model: SimplifiedCTM,
    x: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract neural dynamics and synchronizations for visualization.
    
    Returns:
        Tuple of (post_activation_history, synchronizations)
    """
    model.eval()
    batch_size = x.shape[0]
    
    with torch.no_grad():
        flattened_x = x.flatten(1)
        encoded_image = model.image_encoder(flattened_x)
        
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
        
        # Compute final synchronizations
        history_length = min(model.max_ticks, model.max_memory)
        synchronizations = []
        for pair_idx, pair in enumerate(model.pairs):
            Z_i = post_activations[:, pair[0], -history_length:]
            Z_j = post_activations[:, pair[1], -history_length:]
            r_ij = model.decay_factors[pair_idx]
            time_diffs = torch.arange(history_length - 1, -1, -1, dtype=torch.float32, device=r_ij.device)
            R_ij = torch.exp(-r_ij * time_diffs)
            weighted_dot_product = torch.sum(Z_i * R_ij * Z_j, dim=-1)
            normalization_factor = torch.sum(R_ij)
            sync_value = weighted_dot_product / (normalization_factor + 1e-8)
            synchronizations.append(sync_value)
        
        sync_vector = torch.stack(synchronizations, dim=1).cpu()
    
    return post_history_tensor, sync_vector


def plot_training_results(
    metrics: TrainingMetrics,
    test_accuracy: float,
    model: SimplifiedCTM,
    sample_image: torch.Tensor,
    device: torch.device
):
    """
    Plot comprehensive training results in multiple figures.
    
    Generates:
    - Figure 1: Training loss and accuracy curves
    - Figure 2: Batch-level loss curves
    - Figure 3: Neural dynamics visualization
    - Figure 4: Final accuracy comparison
    """
    color_main = '#3498db'
    color_accent = '#e74c3c'
    
    # Figure 1: Training Loss and Accuracy per Epoch
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Training Progress', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(metrics.epoch_losses) + 1)
    
    # Loss subplot
    ax_loss = axes1[0]
    ax_loss.plot(epochs, metrics.epoch_losses, marker='o', color=color_main, linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax_acc = axes1[1]
    ax_acc.plot(epochs, metrics.epoch_accuracies, marker='s', color=color_main, linewidth=2)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Training Accuracy')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim([0, 105])
    
    plt.tight_layout()
    fig1.savefig('fig1_training_curves.png', dpi=150, bbox_inches='tight')
    
    # Figure 2: Batch-level Loss (smoothed)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.suptitle('Batch-level Training Loss (Smoothed)', fontsize=14, fontweight='bold')
    
    window_size = 50
    losses = metrics.batch_losses
    if len(losses) > window_size:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed, color=color_main, alpha=0.8)
    else:
        ax2.plot(losses, color=color_main, alpha=0.8)
    
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('fig2_batch_losses.png', dpi=150, bbox_inches='tight')
    
    # Figure 3: Neural Dynamics Visualization
    post_history, sync_vector = get_neural_dynamics(model, sample_image.unsqueeze(0).to(device), device)
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Neural Dynamics Visualization', fontsize=14, fontweight='bold')
    
    # Sample image
    ax_img = axes3[0, 0]
    ax_img.imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
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
    
    # Synchronization values
    ax_sync = axes3[1, 1]
    sync_values = sync_vector[0].numpy()
    ax_sync.bar(range(len(sync_values)), sync_values, color=color_main, alpha=0.8)
    ax_sync.set_xlabel('Neuron Pair Index')
    ax_sync.set_ylabel('Synchronization Value')
    ax_sync.set_title('Synchronization Values')
    ax_sync.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig3.savefig('fig3_neural_dynamics.png', dpi=150, bbox_inches='tight')
    
    # Figure 4: Final Accuracy Comparison
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    fig4.suptitle('Final Model Performance', fontsize=14, fontweight='bold')
    
    train_acc = metrics.epoch_accuracies[-1]
    
    x = np.arange(2)
    bars = ax4.bar(x, [train_acc, test_accuracy], color=[color_main, color_accent], alpha=0.8)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Train', 'Test'])
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    fig4.savefig('fig4_final_accuracy.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print('\nPlots saved:')
    print('  - fig1_training_curves.png')
    print('  - fig2_batch_losses.png')
    print('  - fig3_neural_dynamics.png')
    print('  - fig4_final_accuracy.png')


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module):
    """Print detailed information about the model."""
    total_params = count_parameters(model)
    print(f'\n=== Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    
    # Print parameter breakdown by component
    print(f'\nParameter breakdown:')
    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'  {name}: {module_params:,} parameters')
    
    # Print memory buffers (pre/post activations)
    pre_act_params = model.initial_pre.numel()
    post_act_params = model.initial_post.numel()
    print(f'  pre_activations buffer: {pre_act_params:,} elements')
    print(f'  post_activations buffer: {post_act_params:,} elements')
    print(f'=========================\n')


def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = SimplifiedCTM(n_neurons=30, max_memory=10, max_ticks=15, n_representation_size=8).to(device)
    
    # Print model information
    print_model_info(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    metrics = train_model(model, train_loader, optimizer, criterion, epochs=5, device=device)
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'ctm_model.pth')
    print('Model saved to ctm_model.pth')
    
    # Get a sample image for visualization
    sample_image, _ = test_dataset[0]
    
    # Plot training results
    print('\nGenerating plots...')
    plot_training_results(metrics, test_accuracy, model, sample_image, device)


if __name__ == '__main__':
    main()
