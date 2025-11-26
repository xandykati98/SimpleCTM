import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import numpy as np


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


class SimplifiedCTM(nn.Module):
    """
    Simplified Continuous Thought Machine implementation.
    
    Based on: https://arxiv.org/abs/2505.05522
    Reference: https://github.com/SakanaAI/continuous-thought-machines
    
    Key components:
    1. Internal temporal axis (ticks) for thought unfolding
    2. Neuron-level models (NLMs) with private parameters per neuron
    3. Neural synchronization as representation for output and action
    4. Recursive synchronization computation (O(1) per tick)
    5. Separate synchronization pairs for output (S_out) and action (S_action)
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
        out_dims: int,
    ):
        super(SimplifiedCTM, self).__init__()

        self.input_shape: tuple = (1, 28, 28)
        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.n_neurons = n_neurons
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.out_dims = out_dims
        self.n_attention_heads = n_attention_heads
        
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

        flattened_input_shape = self.input_shape[1] * self.input_shape[2]
        
        # Image encoder - reduces image to n_representation_size-item array
        self.image_encoder = nn.Sequential(
            nn.Linear(flattened_input_shape, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_representation_size),
        )
        
        # Key-Value projection for attention (from encoded input)
        self.kv_proj = nn.Sequential(
            nn.Linear(n_representation_size, n_representation_size),
            nn.LayerNorm(n_representation_size)
        )
        
        # Query projection from synchronization_action
        self.q_proj = nn.Linear(n_synch_action, n_representation_size)
        
        # Multi-head attention (Section 3.4)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_representation_size,
            num_heads=n_attention_heads,
            dropout=0.0,
            batch_first=True
        )
        
        # Synapse model - combines attention output with current activated state
        # Input: attention_output (n_representation_size) + activated_state (n_neurons)
        synapse_input_size = n_representation_size + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),  # GLU as in official implementation
            nn.LayerNorm(n_neurons),
        )

        # Neuron-level models (NLMs) - Section 3.3
        # Each neuron has its own private MLP that processes its pre-activation history
        # Using GLU nonlinearity as in official implementation
        self.neuron_level_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_memory, 128 * 2),
                nn.GLU(),
                nn.Linear(128, 1),
            )
            for _ in range(n_neurons)
        ])

        # Synchronization neuron pairs - Section 3.4.1 (random-pairing strategy)
        # Separate pairs for output and action as per paper
        self._init_synchronization_pairs()
        
        # Learnable decay parameters for synchronization (Appendix H)
        # Separate decay params for action and output sync
        self.register_parameter(
            'decay_params_action',
            nn.Parameter(torch.zeros(n_synch_action), requires_grad=True)
        )
        self.register_parameter(
            'decay_params_out',
            nn.Parameter(torch.zeros(n_synch_out), requires_grad=True)
        )

        # Output projector - from synchronization_out to predictions
        self.output_projector = nn.Sequential(
            nn.Linear(n_synch_out, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, out_dims)
        )
    
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
        
        S_ij^t = α_ij^t / sqrt(β_ij^t)
        
        Where:
        - α_ij^(t+1) = e^(-r_ij) * α_ij^t + z_i^(t+1) * z_j^(t+1)
        - β_ij^(t+1) = e^(-r_ij) * β_ij^t + 1
        
        This enables O(1) computation per tick instead of O(t).
        """
        if synch_type == 'action':
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")
        
        # Get activations for paired neurons
        left = activated_state[:, neuron_indices_left]   # (batch_size, n_synch)
        right = activated_state[:, neuron_indices_right]  # (batch_size, n_synch)
        pairwise_product = left * right  # (batch_size, n_synch)
        
        # Recursive update (Equations 16-17 from Appendix H)
        if decay_alpha is None or decay_beta is None:
            # First tick: initialize
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            # Subsequent ticks: recursive update
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        # Compute synchronization with sqrt normalization (Equation 13)
        synchronization = decay_alpha / torch.sqrt(decay_beta + 1e-8)
        
        return synchronization, decay_alpha, decay_beta
    
    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute certainty as 1 - normalized_entropy.
        Returns tensor of shape (batch_size, 2) with [entropy, 1-entropy].
        """
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def forward(self, x: torch.Tensor, track: bool = False):
        """
        Forward pass through the CTM.
        
        Returns predictions and certainties for ALL ticks (Section 3.5),
        enabling loss computation across all internal iterations.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode the image once at the beginning
        flattened_x = x.flatten(1)  # (batch_size, 784)
        encoded_image = self.image_encoder(flattened_x)  # (batch_size, n_representation_size)
        
        # Prepare key-value features for attention
        kv = self.kv_proj(encoded_image).unsqueeze(1)  # (batch_size, 1, n_representation_size)
        
        # Initialize recurrent state from learnable parameters
        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone()  # (B, n_neurons, max_memory)
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1).clone()  # (B, n_neurons)
        
        # Prepare storage for outputs per iteration (Section 3.5 - loss across all ticks)
        predictions = torch.empty(batch_size, self.out_dims, self.max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.max_ticks, device=device, dtype=torch.float32)
        
        # Initialize synchronization recurrence values
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        
        # Clamp decay parameters to [0, 15] as in official implementation
        self.decay_params_action.data = torch.clamp(self.decay_params_action.data, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out.data, 0, 15)
        
        # Compute decay factors r = exp(-decay_params)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).expand(batch_size, -1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(batch_size, -1)
        
        # Initialize output sync at tick 0
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
        
        # Recurrent loop over internal ticks
        for tick in range(self.max_ticks):
            # Calculate synchronization for action (attention modulation)
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # Create query from action synchronization
            q = self.q_proj(sync_action).unsqueeze(1)  # (batch_size, 1, n_representation_size)
            
            # Multi-head attention: sync_action queries the encoded input
            attn_out, attn_weights = self.attention(q, kv, kv, need_weights=True)
            attn_out = attn_out.squeeze(1)  # (batch_size, n_representation_size)
            
            # Combine attention output with current activated state for synapse model
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            
            # Apply synapse model to get new pre-activations
            new_state = self.synapse_model(pre_synapse_input)  # (batch_size, n_neurons)
            
            # Update state trace (FIFO buffer of pre-activations)
            state_trace = torch.cat([state_trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply neuron-level models to get post-activations
            post_activations_list = []
            for neuron_idx in range(self.n_neurons):
                neuron_trace = state_trace[:, neuron_idx, :]  # (batch_size, max_memory)
                post_activation = self.neuron_level_models[neuron_idx](neuron_trace)  # (batch_size, 1)
                post_activations_list.append(post_activation)
            
            activated_state = torch.cat(post_activations_list, dim=-1)  # (batch_size, n_neurons)
            
            # Calculate synchronization for output predictions
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Get predictions and certainties for this tick
            current_prediction = self.output_projector(sync_out)
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
                np.array(attention_tracking)
            )
        
        return predictions, certainties


def train_model(
    model: SimplifiedCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
):
    """
    Training loop for the SimplifiedCTM model.
    Computes loss across ALL ticks as per Section 3.5 of the paper.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - returns predictions for ALL ticks
            predictions, certainties = model(data)
            # predictions shape: (batch_size, out_dims, max_ticks)
            
            # Compute loss across all ticks (Section 3.5)
            # This encourages the model to produce good predictions at every tick
            loss = 0.0
            for tick in range(model.max_ticks):
                tick_predictions = predictions[..., tick]  # (batch_size, out_dims)
                loss += criterion(tick_predictions, target)
            loss = loss / model.max_ticks  # Average loss across ticks
            
            loss.backward()
            optimizer.step()
            
            # Statistics using final tick prediction
            total_loss += loss.item()
            final_predictions = predictions[..., -1]
            _, predicted = torch.max(final_predictions, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                # Get average certainty at final tick
                avg_certainty = certainties[:, 1, -1].mean().item()  # 1-entropy
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%, '
                      f'Certainty: {avg_certainty:.3f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} completed - '
              f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: SimplifiedCTM):
    """Print detailed information about the model."""
    total_params = count_parameters(model)
    print(f'\n=== Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Memory length: {model.max_memory}')
    print(f'Max ticks: {model.max_ticks}')
    print(f'Sync pairs (output): {model.n_synch_out}')
    print(f'Sync pairs (action): {model.n_synch_action}')
    print(f'Attention heads: {model.n_attention_heads}')
    
    print(f'\nParameter breakdown:')
    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_params > 0:
                print(f'  {name}: {module_params:,} parameters')
    
    # Print learnable parameters
    print(f'  start_activated_state: {model.start_activated_state.numel():,} elements')
    print(f'  start_trace: {model.start_trace.numel():,} elements')
    print(f'  decay_params_action: {model.decay_params_action.numel():,} elements')
    print(f'  decay_params_out: {model.decay_params_out.numel():,} elements')
    print(f'=========================\n')


def main():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model with CTM architecture parameters
    model = SimplifiedCTM(
        n_neurons=64,           # Number of neurons (D in paper)
        max_memory=10,          # Memory length for NLMs (M in paper)
        max_ticks=15,           # Number of internal ticks (T in paper)
        n_representation_size=32,  # Input feature dimension
        n_synch_out=32,         # Sync pairs for output (D_out in paper)
        n_synch_action=16,      # Sync pairs for action (D_action in paper)
        n_attention_heads=4,    # Number of attention heads
        out_dims=10,            # Output dimension (MNIST classes)
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, optimizer, epochs=10, device=device)
    
    torch.save(model.state_dict(), 'ctm_extended_model.pth')
    print('Model saved to ctm_extended_model.pth')


if __name__ == '__main__':
    main()
