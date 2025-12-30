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
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        
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
        image_size: int,
        patch_size: int,
        in_channels: int,
        input_ndim: int = 2,
        dropout: float = 0.0,
        dropout_nlm: float = 0.0,
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
        self.image_size = image_size
        self.input_ndim = input_ndim

        if self.input_ndim == 2:
            self.n_patches = (image_size // patch_size) ** 2
        elif self.input_ndim == 1:
            self.n_patches = image_size // patch_size
            if self.n_patches < 1:
                raise ValueError(
                    f"Invalid 1D patching: image_size(input_length)={image_size}, patch_size={patch_size} -> n_patches={self.n_patches}"
                )
        else:
            raise ValueError(f"Invalid input_ndim: {input_ndim}. Expected 1 or 2.")
        
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

        if self.input_ndim == 2:
            self.patch_embed = PatchEmbedding(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=d_input,
            )
        else:
            self.patch_embed = PatchEmbedding1D(
                input_length=image_size,
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
        
        # Add positional information
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


def visualize_attention(model: SimplifiedCTM, image: torch.Tensor, device: torch.device):
    """
    Visualize how attention weights evolve across ticks.
    This shows the foveated attention behavior!
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
    
    grid_size = int(np.sqrt(model.n_patches))
    n_ticks_to_show = model.max_ticks
    tick_indices = np.arange(model.max_ticks)
    
    fig, axes = plt.subplots(3, n_ticks_to_show, figsize=(2 * n_ticks_to_show, 6))
    
    # Show original image
    img_np = image.squeeze().cpu().numpy()
    for i, tick_idx in enumerate(tick_indices):
        # Top row: attention heatmap overlaid on image
        axes[0, i].imshow(img_np, cmap='gray', alpha=0.5)
        attn_map = attention_avg[tick_idx].reshape(grid_size, grid_size)
        attn_resized = np.kron(attn_map, np.ones((model.patch_size, model.patch_size)))
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
    
    plt.suptitle('Foveated Attention and Predictions Over Ticks')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()
    print("Saved attention_visualization.png")


def visualize_nlm_activations(model: SimplifiedCTM, image: torch.Tensor, device: torch.device, epoch: int | None = None):
    """
    Visualize NLM activations per tick - simple grid of line plots for each neuron.
    
    Args:
        model: The CTM model
        image: Input image tensor
        device: Device to run on
        epoch: Optional epoch number for filename
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


def train_model(
    model: SimplifiedCTM, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    device: torch.device,
):
    """
    Training loop with adaptive compute loss (Section 3.5 & Appendix E.2).
    
    The loss function encourages the model to:
    1. Find the correct answer eventually (min_loss across ticks)
    2. Be confident about the correct answer (loss at most certain tick)
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
        print(f'Epoch {epoch+1}/{epochs} completed - '
              f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Visualize NLM activations at end of each epoch
        if sample_image is not None:
            print(f'Generating NLM activations visualization for epoch {epoch+1}...')
            visualize_nlm_activations(model, sample_image, device, epoch=epoch)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: SimplifiedCTM):
    total_params = count_parameters(model)
    print(f'\n=== Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Memory length: {model.max_memory}')
    print(f'Max ticks: {model.max_ticks}')
    if model.input_ndim == 2:
        print(f'Input ndim: 2')
        print(f'Image size: {model.image_size}x{model.image_size}')
        print(f'Number of patches: {model.n_patches} ({model.image_size//model.patch_size}x{model.image_size//model.patch_size})')
        print(f'Patch size: {model.patch_size}x{model.patch_size}')
    else:
        print(f'Input ndim: 1')
        print(f'Sequence length: {model.image_size}')
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model with spatial patches for foveated attention
    model = SimplifiedCTM(
        n_neurons=64,
        max_memory=10,
        max_ticks=15,
        d_input=64,              # Embedding dimension for patches
        n_synch_out=32,
        n_synch_action=16,
        n_attention_heads=4,
        out_dims=10,
        image_size=28,           # MNIST image size
        patch_size=7,            # 7x7 patches -> 4x4 = 16 patches
        in_channels=1,           # MNIST is grayscale
        dropout=0.0,             # Dropout rate (set to 0.1-0.2 for regularization)
        dropout_nlm=0.0,         # Separate dropout for NLMs (0.0 = use dropout)
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    train_model(model, train_loader, optimizer, epochs=3, device=device)
    
    # Visualize the foveated attention behavior
    print("\nVisualizing attention patterns...")
    test_image, _ = train_dataset[0]
    visualize_attention(model, test_image, device)
    
    torch.save(model.state_dict(), 'ctm_spatial_model.pth')
    print('Model saved to ctm_spatial_model.pth')


if __name__ == '__main__':
    main()

