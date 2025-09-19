from datasets import load_dataset

ds = load_dataset("lordspline/arc-agi")

train_ds = ds["training"]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimplifiedCTM(nn.Module):
    def __init__(self, n_neurons: int, max_memory: int = 10, max_ticks: int = 10, n_representation_size: int = 4):
        super(SimplifiedCTM, self).__init__()

        self.input_shape = None  # Now handles variable input shapes
        self.initial_post = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))
        self.initial_pre = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_pairs = (n_neurons // 2)

        self.pairs = torch.randint(0, n_neurons, (self.n_pairs, 2))
        
        # Learnable temporal decay factors r_ij for each neuron pair
        # Initialize with small positive values to start with some temporal weighting
        self.decay_factors = nn.Parameter(torch.ones(self.n_pairs) * 0.1)

        # Image encoder - handles variable input shapes, reduces to n_representation_size
        # We'll use adaptive pooling to handle any input size
        self.image_encoder_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Always pool to 4x4
        self.image_encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 64),  # 128 channels * 4 * 4
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_representation_size),
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
        
        # Synchronization reader - reconstructs variable-sized images
        self.syncronization_reader_fc = nn.Sequential(
            nn.Linear(self.attention_dim + self.n_pairs, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Adaptive decoder that can reconstruct to any target shape
        self.adaptive_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fixed large decoder for any grid size up to 30x30 = 900 pixels
        # This prevents dynamic layer creation which never learns
        self.large_decoder = nn.Linear(64, 900)
        
    def reconstruct_to_shape(self, features: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Reconstruct image from features to specific target shape."""
        batch_size = features.shape[0]
        target_height, target_width = target_shape
        target_size = target_height * target_width
        
        if target_size > 900:
            raise ValueError(f"Target size {target_size} exceeds maximum supported size of 900 (30x30)")
        
        # Get features through adaptive decoder
        decoded_features = self.adaptive_decoder(features)  # (batch_size, 64)
        
        # Use fixed large decoder and slice what we need
        full_output = self.large_decoder(decoded_features)  # (batch_size, 900)
        output = full_output[:, :target_size]  # (batch_size, target_size)
        
        # Reshape to image format and add channel dimension
        output = output.view(batch_size, 1, target_height, target_width)
        
        # Apply sigmoid and scale to [0,9] range for ARC-AGI discrete values
        output = torch.sigmoid(output) * 9
        
        return output
    
    def forward(self, x: torch.Tensor, target_shape: tuple = None):
        # x is a batch of images with shape (batch_size, height, width) or (batch_size, 1, height, width)
        batch_size = x.shape[0]
        
        # Ensure x has channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, height, width)
        
        # Encode the image once at the beginning using new encoder
        conv_features = self.image_encoder_conv(x)  # (batch_size, 128, height, width)
        pooled_features = self.adaptive_pool(conv_features)  # (batch_size, 128, 4, 4)
        encoded_image = self.image_encoder_fc(pooled_features)  # (batch_size, n_representation_size)
        
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
            
            # Process through synchronization reader
            processed_features = self.syncronization_reader_fc(combined_features)  # (batch_size, 256)
            
            # Reconstruct to target shape if provided, otherwise use original input shape
            if target_shape is not None:
                prediction = self.reconstruct_to_shape(processed_features, target_shape)
            else:
                # Use original input shape as fallback
                input_height, input_width = x.shape[-2], x.shape[-1]
                prediction = self.reconstruct_to_shape(processed_features, (input_height, input_width))
            
        # Return final prediction after all ticks
        return prediction


def train_model_arcagi(model: SimplifiedCTM, optimizer: optim.Optimizer, 
                criterion: nn.Module, epochs: int, device: torch.device):
    """Training loop for the SimplifiedCTM model."""
    model.train()
    
    problem_set_count = 0
    for epoch in range(epochs):
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        problem_set_index = 0
        
        for problem_set in train_ds['train']:
            problem_index = 0
            for problem in problem_set:
                input_matrix = problem['input']
                target_matrix = problem['output']

                # Convert to tensors and ensure proper shape and dtype
                data = torch.tensor(input_matrix, dtype=torch.float32).to(device)
                target = torch.tensor(target_matrix, dtype=torch.float32).to(device)
                
                # Ensure data has batch dimension
                if len(data.shape) == 2:
                    data = data.unsqueeze(0)  # Add batch dimension
                if len(target.shape) == 2:
                    target = target.unsqueeze(0)  # Add batch dimension
                
                # Get target shape for reconstruction
                target_height, target_width = target.shape[-2], target.shape[-1]
                target_shape = (target_height, target_width)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with target shape
                output = model(data, target_shape)
                
                # Flatten target for loss calculation (remove channel dimension if present)
                if len(target.shape) == 4:  # (batch, channel, height, width)
                    target_flat = target.squeeze(1)  # Remove channel dimension
                else:
                    target_flat = target
                
                output_flat = output.squeeze(1)  # Remove channel dimension
                
                # Calculate loss (MSE for continuous values, then we can threshold for discrete)
                loss = criterion(output_flat, target_flat)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics - count correct pixels after rounding to integers
                total_loss += loss.item()
                predicted_discrete = torch.round(output_flat).clamp(0, 9)  # Round and clamp to [0,9]
                target_discrete = torch.round(target_flat).clamp(0, 9)
                
                correct_pixels += (predicted_discrete == target_discrete).sum().item()
                total_pixels += target_discrete.numel()
                
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {problem_index}/{len(problem_set)}, Problem Set: {problem_set_index}/{len(train_ds['train'])}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Pixel Accuracy: {100.*correct_pixels/total_pixels:.2f}%')
                problem_index+=1
            problem_set_index+=1
            problem_set_count+=1
            mlflow.log_metric("loss", loss.item(), step=problem_set_count)
            mlflow.log_metric("pixel_accuracy", 100.*correct_pixels/total_pixels, step=problem_set_count)
            
        avg_loss = total_loss / sum(len(problem_set) for problem_set in train_ds['train'])
        pixel_accuracy = 100. * correct_pixels / total_pixels
        print(f'Epoch {epoch+1}/{epochs} completed - '
            f'Average Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.2f}%')


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

import mlflow
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
    
    # Initialize model
    model = SimplifiedCTM(n_neurons=20, max_memory=20, max_ticks=15, n_representation_size=128).to(device)
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("arc-agi")
    mlflow.start_run()
    # Print model information
    print_model_info(model)
    
    # Analyze grid sizes and possible values in ARC-AGI training set
    input_shapes = []
    output_shapes = []
    all_values = set()

    for problem_set in train_ds['train']:
        for problem in problem_set:
            input_matrix = problem['input']
            target_matrix = problem['output']
            # Collect input and output grid shapes
            input_shapes.append((len(input_matrix), len(input_matrix[0]) if input_matrix else 0))
            output_shapes.append((len(target_matrix), len(target_matrix[0]) if target_matrix else 0))
            # Collect all unique values in input and output
            for row in input_matrix:
                all_values.update(row)
            for row in target_matrix:
                all_values.update(row)

    # Compute statistics for input and output grids
    def get_stats(shapes: list[tuple[int, int]]) -> tuple[float, int, int]:
        sizes = [h * w for h, w in shapes]
        if not sizes:
            raise ValueError("No grid sizes found for statistics.")
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        return avg_size, min_size, max_size

    input_avg, input_min, input_max = get_stats(input_shapes)
    output_avg, output_min, output_max = get_stats(output_shapes)

    print(f"Input grid sizes: avg={input_avg:.2f}, min={input_min}, max={input_max}")
    print(f"Output grid sizes: avg={output_avg:.2f}, min={output_min}, max={output_max}")
    print(f"Possible values in grids: {sorted(all_values)}")
    # Loss and optimizer - use MSE for pixel-level reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train the model
    train_model_arcagi(model, optimizer, criterion, epochs=10, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'ctm_model.pth')
    print('Model saved to ctm_model.pth')
    mlflow.end_run()


if __name__ == '__main__':
    main()

