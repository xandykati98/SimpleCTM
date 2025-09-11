import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimplifiedCTM(nn.Module):
    def __init__(self, n_neurons: int, max_memory: int = 10, max_ticks: int = 10,):
        super(SimplifiedCTM, self).__init__()

        self.input_shape: tuple = (1, 28, 28)
        self.post_activations = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))
        self.pre_activations = nn.Parameter(torch.normal(0, 1, (n_neurons, max_memory)))

        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_pairs = (n_neurons // 2) // 2

        self.pairs = torch.randint(0, n_neurons, (self.n_pairs, 2))
        
        # Learnable temporal decay factors r_ij for each neuron pair
        # Initialize with small positive values to start with some temporal weighting
        self.decay_factors = nn.Parameter(torch.ones(self.n_pairs) * 0.1)

        flattened_input_shape = self.input_shape[1] * self.input_shape[2]
        
        # Image encoder - reduces image to 4-item array, runs only once
        self.image_encoder = nn.Sequential(
            nn.Linear(flattened_input_shape, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        
        # the synapse model is a simple feedforward neural network
        # inputs for this are the encoded image (4 items) and the last post-activation values
        synapse_input_size = 4 + n_neurons
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
        self.key_projection = nn.Linear(4, self.attention_dim)
        self.value_projection = nn.Linear(4, self.attention_dim)
        
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
        encoded_image = self.image_encoder(flattened_x)  # shape: (batch_size, 4)
        
        # Attention mechanism: input attends to synchronizations
        # Create keys and values from encoded input
        keys = self.key_projection(encoded_image)      # Shape: (batch_size, attention_dim)
        values = self.value_projection(encoded_image)  # Shape: (batch_size, attention_dim)

        for tick in range(self.max_ticks):
            # Get the last post-activation values for each neuron
            last_post_activations = self.post_activations[:, -1]  # shape: (n_neurons,)
            
            # Concatenate encoded image with last post-activations
            # Expand last_post_activations to match batch size
            last_post_expanded = last_post_activations.unsqueeze(0).expand(batch_size, -1)  # shape: (batch_size, n_neurons)
            synapse_input = torch.cat([encoded_image, last_post_expanded], dim=1)  # shape: (batch_size, 4 + n_neurons)
            
            new_pre_activations = self.synapse_model.forward(synapse_input)  # shape: (batch_size, n_neurons)
            
            # Append new pre-activations to FIFO buffer (shift left and add new values)
            # Move existing values left and add new ones at the end
            self.pre_activations.data[:, :-1] = self.pre_activations.data[:, 1:]  # shift left
            self.pre_activations.data[:, -1] = new_pre_activations.mean(dim=0)  # average across batch and add to last column

            # Neuron-level models will receive their whole pre activation history
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                # Get the pre-activation history for this neuron
                neuron_pre_history = self.pre_activations[neuron_idx, :]  # shape: (max_memory,)
                # Pass through the neuron-level model
                post_activation = self.neuron_level_models[neuron_idx](neuron_pre_history)  # shape: (1,)
                post_activations_list.append(post_activation)
            
            # Stack all post-activations
            new_post_activations = torch.stack(post_activations_list, dim=0).squeeze(-1)  # shape: (n_neurons,)
            
            # Append new post-activations to FIFO buffer
            self.post_activations.data[:, :-1] = self.post_activations.data[:, 1:]  # shift left
            self.post_activations.data[:, -1] = new_post_activations  # add to last column
        
            # Calculate synchronizations with learnable temporal decay
            sincronizations = []
            # Use all available history (limited by max_memory)
            history_length = min(tick + 1, self.max_memory)
            
            for pair_idx, pair in enumerate(self.pairs):
                # Get post-activation histories for the pair (Z_i^t and Z_j^t)
                # Use the most recent history_length entries
                Z_i = self.post_activations[pair[0], -history_length:]  # Most recent entries
                Z_j = self.post_activations[pair[1], -history_length:]
                
                # Calculate temporal decay weights R_ij^t
                # R_ij^t = [exp(-r_ij*(history_length-1)), exp(-r_ij*(history_length-2)), ..., exp(0)]
                r_ij = self.decay_factors[pair_idx]
                time_diffs = torch.arange(history_length - 1, -1, -1, dtype=torch.float32, device=r_ij.device)
                R_ij = torch.exp(-r_ij * time_diffs)  # Shape: (history_length,)
                
                # Calculate weighted dot product: S_ij^t = (Z_i^t)^T * diag(R_ij^t) * Z_j^t
                weighted_dot_product = torch.sum(Z_i * R_ij * Z_j)
                
                # Normalize by sum of weights to prevent any pair from dominating
                normalization_factor = torch.sum(R_ij)
                synchronization_value = weighted_dot_product / (normalization_factor + 1e-8)  # Add epsilon for stability
                
                sincronizations.append(synchronization_value)

            # Stack synchronizations into a vector for the reader
            sincronizations_vector = torch.stack(sincronizations, dim=0)  # Shape: (n_pairs,)
            # Expand to match batch size
            sincronizations_batch = sincronizations_vector.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, n_pairs)
            
            
            # Create queries from synchronizations
            queries = self.query_projection(sincronizations_batch)  # Shape: (batch_size, attention_dim)
            
            # Compute attention scores: Q * K^T / sqrt(d_k)
            attention_scores = torch.sum(queries * keys, dim=-1, keepdim=True) / (self.attention_dim ** 0.5)  # Shape: (batch_size, 1)
            attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, 1)
            
            # Apply attention to values
            attended_features = attention_weights * values  # Shape: (batch_size, attention_dim)
            
            # Combine attended features with original synchronizations
            combined_features = torch.cat([attended_features, sincronizations_batch], dim=-1)  # Shape: (batch_size, attention_dim + n_pairs)
            
            sincronizations_read = self.syncronization_reader(combined_features)
            # get the prediction
            prediction = sincronizations_read
            
            # Calculate confidence using softmax probabilities
            prediction_probs = F.softmax(prediction, dim=-1)
            max_confidence = torch.max(prediction_probs, dim=-1)[0]  # Get max probability for each sample
            
            # Early stopping: if any sample has confidence > 0.8, or we're at the last tick
            if torch.any(max_confidence > 0.8) or tick == self.max_ticks - 1:
                return prediction


def train_model(model: SimplifiedCTM, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, epochs: int, device: torch.device):
    """Training loop for the SimplifiedCTM model."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
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
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} completed - '
              f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


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
    pre_act_params = model.pre_activations.numel()
    post_act_params = model.post_activations.numel()
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
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    # Initialize model
    model = SimplifiedCTM(n_neurons=30, max_memory=10, max_ticks=15).to(device)
    
    # Print model information
    print_model_info(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, optimizer, criterion, epochs=10, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'ctm_model.pth')
    print('Model saved to ctm_model.pth')


if __name__ == '__main__':
    main()
