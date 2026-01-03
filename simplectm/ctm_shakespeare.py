import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os
import wandb
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from PIL import Image

def compute_normalized_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized entropy from predictions.
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
    Batched Neuron-Level Model with per-neuron private parameters.
    Architecture: Linear(memory, 256) -> GLU -> Linear(128, 2) -> GLU -> 1
    """
    
    def __init__(self, n_neurons: int, max_memory: int, dropout: float = 0.0):
        super(BatchedNLM, self).__init__()
        self.n_neurons = n_neurons
        self.max_memory = max_memory
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Per-neuron weights for fc1: (n_neurons, max_memory, 256)
        self.fc1_weight = nn.Parameter(
            torch.randn(n_neurons, max_memory, 256) * math.sqrt(2.0 / max_memory)
        )
        self.fc1_bias = nn.Parameter(torch.zeros(n_neurons, 256))
        
        # Per-neuron weights for fc2: (n_neurons, 128, 2)
        self.fc2_weight = nn.Parameter(
            torch.randn(n_neurons, 128, 2) * math.sqrt(2.0 / 128)
        )
        self.fc2_bias = nn.Parameter(torch.zeros(n_neurons, 2))
        
        self.register_parameter('T', nn.Parameter(torch.ones(1)))
    
    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_trace: (batch, seq_len, neurons, memory) tensor
        Returns:
            (batch, seq_len, neurons) post-activations
        """
        B, S, N, M = state_trace.shape
        
        # Reshape for batched processing: (B*S, N, M)
        x = state_trace.reshape(B * S, N, M)
        x = self.dropout(x)
        
        # fc1: (B*S, N, M) @ (N, M, 256) -> (B*S, N, 256)
        x = torch.einsum('bnm,nmh->bnh', x, self.fc1_weight) + self.fc1_bias
        x = F.glu(x, dim=-1)  # -> (B*S, N, 128)
        
        # fc2: (B*S, N, 128) @ (N, 128, 2) -> (B*S, N, 2)
        x = torch.einsum('bnh,nho->bno', x, self.fc2_weight) + self.fc2_bias
        x = F.glu(x, dim=-1)  # -> (B*S, N, 1)
        
        x = (x.squeeze(-1) / self.T)  # -> (B*S, N)
        
        # Reshape back: (B, S, N)
        return x.reshape(B, S, N)


class TokenEmbedding(nn.Module):
    """Token + positional embedding for sequences."""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.0):
        super(TokenEmbedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len) token indices
        Returns:
            (B, seq_len, embed_dim) embeddings
        """
        seq_len = x.shape[1]
        token_emb = self.token_embed(x)
        pos_emb = self.pos_embed[:, :seq_len, :]
        return self.dropout(self.norm(token_emb + pos_emb))


class CausalCTM(nn.Module):
    """
    Causal Continuous Thought Machine for autoregressive language modeling.
    
    Key adaptations from image CTM:
    1. Token embeddings instead of patch embeddings
    2. Causal attention mask (each position only attends to itself and previous)
    3. Per-position state traces and NLM processing
    4. Per-position output predictions
    """
    
    def __init__(
        self, 
        vocab_size: int,
        max_seq_len: int,
        n_neurons: int, 
        max_memory: int,
        max_ticks: int,
        d_model: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        dropout: float = 0.0,
    ):
        super(CausalCTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_neurons = n_neurons
        self.max_memory = max_memory
        self.max_ticks = max_ticks
        self.d_model = d_model
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        
        # Token embedding
        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        
        # Learnable initial states per position (position-dependent for diverse queries)
        self.register_parameter(
            'start_activated_state', 
            nn.Parameter(torch.zeros(max_seq_len, n_neurons).uniform_(
                -math.sqrt(1/n_neurons), 
                math.sqrt(1/n_neurons)
            ))
        )
        self.register_parameter(
            'start_trace', 
            nn.Parameter(torch.zeros(max_seq_len, n_neurons, max_memory).uniform_(
                -math.sqrt(1/(n_neurons + max_memory)), 
                math.sqrt(1/(n_neurons + max_memory))
            ))
        )
        
        # Key-Value projection from token embeddings
        self.kv_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Query projection from synchronization_action (per position)
        self.q_proj = nn.Linear(n_synch_action, d_model)
        
        # Causal multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Synapse model: combines attention output with current activated state
        synapse_input_size = d_model + n_neurons
        self.synapse_model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(synapse_input_size, n_neurons * 2),
            nn.GLU(),
            nn.LayerNorm(n_neurons),
        )
        
        # Batched NLM
        self.batched_nlm = BatchedNLM(
            n_neurons=n_neurons, 
            max_memory=max_memory, 
            dropout=dropout
        )
        
        # Synchronization neuron pairs
        self._init_synchronization_pairs()
        
        # Learnable decay parameters
        self.register_parameter(
            'decay_params_action',
            nn.Parameter(torch.zeros(n_synch_action), requires_grad=True)
        )
        self.register_parameter(
            'decay_params_out',
            nn.Parameter(torch.zeros(n_synch_out), requires_grad=True)
        )
        
        # Output projector: sync -> vocab logits
        self.output_projector = nn.Linear(n_synch_out, vocab_size)
    
    def _init_synchronization_pairs(self):
        """Initialize neuron pairs for synchronization using random-pairing strategy."""
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
        decay_alpha: torch.Tensor | None,
        decay_beta: torch.Tensor | None,
        r: torch.Tensor,
        synch_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute synchronization for each position.
        
        Args:
            activated_state: (B, S, N) activated states per position
            decay_alpha, decay_beta: recurrence state (B, S, n_synch)
            r: decay rate (B, n_synch) expanded to match
            synch_type: 'action' or 'out'
        """
        if synch_type == 'action':
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")
        
        # (B, S, n_synch)
        left = activated_state[:, :, neuron_indices_left]
        right = activated_state[:, :, neuron_indices_right]
        pairwise_product = left * right
        
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            # r: (B, n_synch) -> (B, 1, n_synch) for broadcasting
            r_expanded = r.unsqueeze(1)
            decay_alpha = r_expanded * decay_alpha + pairwise_product
            decay_beta = r_expanded * decay_beta + 1
        
        synchronization = decay_alpha / torch.sqrt(decay_beta + 1e-8)
        
        return synchronization, decay_alpha, decay_beta
    
    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        """Compute certainty as 1 - normalized_entropy. (B, S) output."""
        normalized_entropy = compute_normalized_entropy(current_prediction)
        certainty = torch.stack((normalized_entropy, 1 - normalized_entropy), dim=-1)
        return certainty
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (True = masked, False = attend)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        track: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal attention.
        
        Args:
            x: (B, seq_len) input token indices
            track: if True, also return attention weights
        
        Returns:
            predictions: (B, seq_len, vocab_size, max_ticks) logits at each tick
            certainties: (B, seq_len, 2, max_ticks) uncertainty/certainty scores
            attention_weights: (max_ticks, B, n_heads, seq_len, seq_len) if track=True
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed tokens: (B, S, d_model)
        token_emb = self.token_embed(x)
        kv = self.kv_proj(token_emb)  # (B, S, d_model)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Initialize per-position states: (B, S, N) and (B, S, N, M)
        # Use position-dependent initial states for diverse queries at tick 1
        activated_state = self.start_activated_state[:seq_len].unsqueeze(0).expand(batch_size, -1, -1).clone()
        state_trace = self.start_trace[:seq_len].unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        
        # Prepare output storage
        predictions = torch.empty(batch_size, seq_len, self.vocab_size, self.max_ticks, device=device)
        certainties = torch.empty(batch_size, seq_len, 2, self.max_ticks, device=device)
        
        # Storage for attention weights if tracking
        if track:
            n_heads = self.attention.num_heads
            attention_weights_all = torch.empty(self.max_ticks, batch_size, n_heads, seq_len, seq_len, device=device)
        
        # Initialize synchronization recurrence
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
        
        # Recurrent loop over internal ticks
        for tick in range(self.max_ticks):
            # Compute action synchronization: (B, S, n_synch_action)
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronization(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )
            
            # Query from action synchronization: (B, S, d_model)
            q = self.q_proj(sync_action)
            
            # Causal attention: each position attends only to itself and previous
            attn_out, attn_weights = self.attention(
                q, kv, kv,
                attn_mask=causal_mask,
                need_weights=track,
                average_attn_weights=False,
            )  # (B, S, d_model), optionally (B, n_heads, S, S)
            
            if track:
                attention_weights_all[tick] = attn_weights
            
            # Combine attention output with current state: (B, S, d_model + N)
            pre_synapse_input = torch.cat([attn_out, activated_state], dim=-1)
            
            # Apply synapse model: (B, S, N)
            new_state = self.synapse_model(pre_synapse_input)
            
            # Update state trace (FIFO buffer): (B, S, N, M)
            state_trace = torch.cat([state_trace[:, :, :, 1:], new_state.unsqueeze(-1)], dim=-1)
            
            # Apply NLM: (B, S, N)
            activated_state = self.batched_nlm(state_trace)
            
            # Compute output synchronization: (B, S, n_synch_out)
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronization(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Get predictions: (B, S, vocab_size)
            current_prediction = self.output_projector(sync_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            predictions[..., tick] = current_prediction
            certainties[..., tick] = current_certainty
        
        if track:
            return predictions, certainties, attention_weights_all
        return predictions, certainties
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            prompt: (1, prompt_len) starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
        
        Returns:
            (1, prompt_len + max_new_tokens) generated sequence
        """
        self.eval()
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]
            
            # Forward pass
            predictions, certainties = self(context)
            
            # Use most certain tick for the last position
            # certainties: (B, S, 2, T) - index 1 is certainty
            last_pos_certainty = certainties[:, -1, 1, :]  # (B, T)
            most_certain_tick = last_pos_certainty.argmax(dim=-1)  # (B,)
            
            # Get logits at last position and most certain tick
            logits = predictions[0, -1, :, most_certain_tick[0]]  # (vocab_size,)
            
            # Apply temperature and sample
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated


def visualize_attention_evolution(
    model: CausalCTM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    dataset: 'ShakespeareDataset',
    device: torch.device,
    query_position: int = -1,
    top_k: int = 10,
    save_path: str | None = None,
):
    """
    Visualize attention evolution across ticks.
    
    Creates a grid showing:
    - Tokens arranged in a square with background colors representing attention weights
    - Bottom section showing top-k predicted tokens + the correct token
    
    Args:
        model: The CausalCTM model
        tokens: (1, seq_len) input token indices
        targets: (1, seq_len) target token indices
        dataset: ShakespeareDataset for decoding
        device: torch device
        query_position: which position to visualize attention from (-1 = last)
        top_k: number of top predictions to show
        save_path: if provided, save figure to this path
    
    Returns:
        BytesIO buffer containing the PNG image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    
    model.eval()
    with torch.no_grad():
        predictions, certainties, attention_weights = model(tokens.to(device), track=True)
    
    # attention_weights: (max_ticks, B, n_heads, S, S)
    # Average across heads: (max_ticks, S, S)
    attn = attention_weights[:, 0, :, :, :].mean(dim=1).cpu().numpy()  # (T, S, S)
    
    seq_len = tokens.shape[1]
    n_ticks = model.max_ticks
    
    # Decode tokens to characters
    if hasattr(dataset, 'idx_to_char'):
        token_chars = [dataset.idx_to_char[t.item()] for t in tokens[0]]
        target_chars = [dataset.idx_to_char[t.item()] for t in targets[0]]
    else:
        # Fallback to decode method if available (e.g. for BPE)
        token_chars = [dataset.decode(torch.tensor([t.item()])) for t in tokens[0]]
        target_chars = [dataset.decode(torch.tensor([t.item()])) for t in targets[0]]
    
    # Handle query position
    if query_position < 0:
        query_position = seq_len + query_position
    
    # Calculate grid dimensions for tokens
    grid_size = int(np.ceil(np.sqrt(seq_len)))
    
    # Create figure: one column per tick
    fig_width = 3 * n_ticks
    fig_height = 8
    fig, axes = plt.subplots(2, n_ticks, figsize=(fig_width, fig_height), 
                              gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
    
    # Custom colormap: white (low attention) to deep red (high attention)
    colors = ['#ffffff', '#fff5f5', '#fed7d7', '#feb2b2', '#fc8181', '#f56565', '#e53e3e', '#c53030', '#9b2c2c']
    cmap = LinearSegmentedColormap.from_list('attention', colors)
    
    for tick in range(n_ticks):
        ax_tokens = axes[0, tick]
        ax_preds = axes[1, tick]
        
        # Get attention from query_position to all positions at this tick
        attn_from_query = attn[tick, query_position, :]  # (S,)
        
        # Normalize for visualization
        attn_min, attn_max = attn_from_query.min(), attn_from_query.max()
        if attn_max > attn_min:
            attn_norm = (attn_from_query - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = np.zeros_like(attn_from_query)
        
        # Draw token grid
        ax_tokens.set_xlim(0, grid_size)
        ax_tokens.set_ylim(0, grid_size)
        ax_tokens.set_aspect('equal')
        ax_tokens.axis('off')
        
        for idx in range(seq_len):
            row = grid_size - 1 - (idx // grid_size)
            col = idx % grid_size
            
            # Background color based on attention
            bg_color = cmap(attn_norm[idx])
            
            # Highlight query position with border
            if idx == query_position:
                rect = mpatches.FancyBboxPatch(
                    (col + 0.05, row + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=bg_color,
                    edgecolor='blue',
                    linewidth=2
                )
            else:
                rect = mpatches.FancyBboxPatch(
                    (col + 0.05, row + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=bg_color,
                    edgecolor='#cccccc',
                    linewidth=0.5
                )
            ax_tokens.add_patch(rect)
            
            # Token character
            char = token_chars[idx]
            display_char = char if char.isprintable() and char != '\n' else '␣' if char == ' ' else '↵' if char == '\n' else '?'
            
            # Adjust font size for longer tokens
            font_size = 8
            if len(display_char) > 1:
                font_size = max(4, 8 - len(display_char))
            
            # Text color: dark for light backgrounds, light for dark backgrounds
            text_color = 'black' if attn_norm[idx] < 0.6 else 'white'
            ax_tokens.text(col + 0.5, row + 0.5, display_char, 
                          ha='center', va='center', fontsize=font_size, 
                          fontfamily='monospace', color=text_color, fontweight='bold')
        
        # Title with certainty
        certainty = certainties[0, query_position, 1, tick].item()
        ax_tokens.set_title(f'Tick {tick+1}\nCert: {certainty:.3f}', fontsize=10)
        
        # Bottom section: top-k predictions + correct
        ax_preds.axis('off')
        
        # Get predictions at query position for this tick
        logits = predictions[0, query_position, :, tick]  # (vocab_size,)
        probs = F.softmax(logits, dim=-1)
        
        # Top-k
        topk_probs, topk_indices = torch.topk(probs, top_k)
        
        # Correct token
        correct_idx = targets[0, query_position].item()
        correct_prob = probs[correct_idx].item()
        if hasattr(dataset, 'idx_to_char'):
            correct_char = dataset.idx_to_char[correct_idx]
        else:
            correct_char = dataset.decode(torch.tensor([correct_idx]))
        
        # Build display text
        pred_lines = []
        for i, (prob, idx) in enumerate(zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy())):
            if hasattr(dataset, 'idx_to_char'):
                char = dataset.idx_to_char[idx]
            else:
                char = dataset.decode(torch.tensor([idx]))
            
            display_char = char if char.isprintable() and char != '\n' else '␣' if char == ' ' else '↵' if char == '\n' else '?'
            # Truncate long tokens for display
            if len(display_char) > 10:
                display_char = display_char[:8] + '..'
                
            is_correct = (idx == correct_idx)
            marker = '✓' if is_correct else ' '
            pred_lines.append(f"{marker} '{display_char}' {prob:.3f}")
        
        # Add correct if not in top-k
        if correct_idx not in topk_indices.cpu().numpy():
            display_correct = correct_char if correct_char.isprintable() and correct_char != '\n' else '␣' if correct_char == ' ' else '↵' if correct_char == '\n' else '?'
            pred_lines.append(f"✓ '{display_correct}' {correct_prob:.3f}")
        
        # Display predictions
        pred_text = '\n'.join(pred_lines[:6])  # Show first 6 to fit
        ax_preds.text(0.5, 0.95, pred_text, ha='center', va='top', fontsize=7,
                     fontfamily='monospace', transform=ax_preds.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0, :].tolist(), orientation='horizontal', 
                        fraction=0.046, pad=0.08, aspect=50)
    cbar.set_label('Attention Weight (normalized)', fontsize=9)
    
    # Query position indicator
    query_char = token_chars[query_position]
    display_query = query_char if query_char.isprintable() and query_char != '\n' else '␣' if query_char == ' ' else '↵' if query_char == '\n' else '?'
    fig.suptitle(f"Attention Evolution: Query position {query_position} ('{display_query}')", 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to buffer for wandb
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(buf.getvalue())
        print(f"Saved attention visualization to {save_path}")
    
    plt.close()
    return buf


class ShakespeareDataset(Dataset):
    """Shakespeare dataset using custom BPE tokenization for efficiency."""
    
    def __init__(self, data_path: str, seq_len: int, split: str = 'train', vocab_size: int = 1024):
        self.seq_len = seq_len
        self.tokenizer_path = os.path.join(data_path, f'tokenizer_shakespeare_bytelevel_{vocab_size}.json')
        
        # Download or load Shakespeare text
        shakespeare_path = os.path.join(data_path, 'shakespeare.txt')
        if not os.path.exists(shakespeare_path):
            os.makedirs(data_path, exist_ok=True)
            import urllib.request
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, shakespeare_path)
        
        # Train or load tokenizer
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            print(f"Training new ByteLevel BPE tokenizer with vocab size {vocab_size}...")
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            
            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], 
                vocab_size=vocab_size,
                initial_alphabet=ByteLevel.alphabet()
            )
            self.tokenizer.train([shakespeare_path], trainer)
            self.tokenizer.save(self.tokenizer_path)
            print("Tokenizer trained and saved.")

        self.vocab_size = self.tokenizer.get_vocab_size()
        
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Encode text
        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded.ids, dtype=torch.long)
        
        # Split data
        n = len(self.data)
        train_data = self.data[:int(n * 0.9)]
        val_data = self.data[int(n * 0.9):]
        
        self.data = train_data if split == 'train' else val_data
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def decode(self, indices: torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self.tokenizer.decode(indices)
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(
    model: CausalCTM, 
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None
):
    total_params = count_parameters(model)
    
    # Calculate specific component parameters
    embedding_params = sum(p.numel() for p in model.token_embed.parameters() if p.requires_grad)
    output_params = sum(p.numel() for p in model.output_projector.parameters() if p.requires_grad)
    core_params = total_params - embedding_params - output_params
    
    print(f'\n=== Causal CTM Model Information ===')
    print(f'Total trainable parameters: {total_params:,}')
    print(f'  - Embedding params:     {embedding_params:,} ({embedding_params/total_params:.1%})')
    print(f'  - Output Head params:   {output_params:,} ({output_params/total_params:.1%})')
    print(f'  - Core Model params:    {core_params:,} ({core_params/total_params:.1%})')
    print(f'-----------------------------------------')
    print(f'Vocab size: {model.vocab_size}')
    print(f'Max sequence length: {model.max_seq_len}')
    print(f'Number of neurons: {model.n_neurons}')
    print(f'Memory length: {model.max_memory}')
    print(f'Max ticks: {model.max_ticks}')
    print(f'd_model: {model.d_model}')
    print(f'Sync pairs (output): {model.n_synch_out}')
    print(f'Sync pairs (action): {model.n_synch_action}')
    
    if batch_size is not None or epochs is not None or lr is not None:
        print(f'-----------------------------------------')
        print(f'Training Configuration:')
        if batch_size: print(f'  - Batch Size: {batch_size}')
        if epochs: print(f'  - Epochs: {epochs}')
        if lr: print(f'  - Learning Rate: {lr}')
        
    print(f'=========================================\n')


def train_model(
    model: CausalCTM,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
    dataset: ShakespeareDataset,
    val_loader: DataLoader | None = None,
    use_wandb: bool = False,
    checkpoint_dir: str | None = None,
):
    """Training loop with adaptive compute loss for language modeling."""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Calculate visualization interval (10 times per epoch)
    n_visualizations = 10
    viz_interval = max(1, len(train_loader) // n_visualizations)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        viz_count = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size, seq_len = x.shape
            
            optimizer.zero_grad()
            
            # predictions: (B, S, V, T), certainties: (B, S, 2, T)
            predictions, certainties = model(x)
            
            # Expand targets: (B, S) -> (B, S, T)
            targets_expanded = y.unsqueeze(-1).expand(-1, -1, model.max_ticks)
            
            # Compute loss at every tick: (B, S, V, T) vs (B, S, T) -> (B, S, T)
            # Reshape for cross entropy: (B*S*T, V) vs (B*S*T)
            pred_flat = predictions.permute(0, 1, 3, 2).reshape(-1, model.vocab_size)
            target_flat = targets_expanded.reshape(-1)
            loss_flat = criterion(pred_flat, target_flat)
            loss_all_ticks = loss_flat.reshape(batch_size, seq_len, model.max_ticks)
            
            # Minimum loss across ticks (encourage finding answer at some point)
            loss_min, _ = loss_all_ticks.min(dim=-1)
            
            # Loss at most certain tick
            # certainties[:, :, 1, :] is certainty (B, S, T)
            most_certain_indices = certainties[:, :, 1, :].argmax(dim=-1)  # (B, S)
            
            # Select loss at most certain tick
            batch_idx_range = torch.arange(batch_size, device=device)[:, None].expand(-1, seq_len)
            seq_idx_range = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1)
            loss_selected = loss_all_ticks[batch_idx_range, seq_idx_range, most_certain_indices]
            
            # Combined loss
            loss = (loss_min.mean() + loss_selected.mean()) / 2
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            
            # Batch-level statistics for wandb
            
            
            if batch_idx % 100 == 0:
                avg_certainty = certainties[:, :, 1, :].max(dim=-1).values.mean().item()
                if use_wandb:
                    batch_perplexity = math.exp(loss.item())
                    wandb.log({
                        "batch/loss": loss.item(),
                        "batch/certainty": avg_certainty,
                        "batch/perplexity": batch_perplexity,
                        "epoch": epoch,
                    })
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Certainty: {avg_certainty:.3f}')
            
            # Visualize attention 10 times per epoch
            if batch_idx % viz_interval == 0 and viz_count < n_visualizations:
                viz_count += 1
                model.eval()
                sample_idx = np.random.randint(len(dataset))
                sample_x, sample_y = dataset[sample_idx]
                save_path = None
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_path = os.path.join(checkpoint_dir, f'attention_epoch_{epoch+1}_viz_{viz_count}.png')
                
                fig_buffer = visualize_attention_evolution(
                    model=model,
                    tokens=sample_x.unsqueeze(0),
                    targets=sample_y.unsqueeze(0),
                    dataset=dataset,
                    device=device,
                    query_position=-1,
                    top_k=10,
                    save_path=save_path,
                )
                
                if use_wandb:
                    fig_buffer.seek(0)
                    pil_image = Image.open(fig_buffer)
                    wandb.log({f"attention_evolution": wandb.Image(pil_image)})
                
                model.train()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        # Validation
        val_loss = None
        val_ppl = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_tokens = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    predictions, certainties = model(x)
                    
                    # Use most certain tick
                    most_certain_indices = certainties[:, :, 1, :].argmax(dim=-1)
                    batch_size, seq_len = x.shape
                    batch_idx_range = torch.arange(batch_size, device=device)[:, None].expand(-1, seq_len)
                    seq_idx_range = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1)
                    final_pred = predictions[batch_idx_range, seq_idx_range, :, most_certain_indices]
                    
                    loss = F.cross_entropy(final_pred.reshape(-1, model.vocab_size), y.reshape(-1))
                    val_total_loss += loss.item() * batch_size * seq_len
                    val_total_tokens += batch_size * seq_len
            
            val_loss = val_total_loss / val_total_tokens
            val_ppl = math.exp(val_loss)
            model.train()
            
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {avg_loss:.4f}, Train PPL: {perplexity:.2f} | '
                  f'Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}')
        else:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, PPL: {perplexity:.2f}')
        
        if use_wandb:
            log_dict = {
                "train/loss": avg_loss,
                "train/perplexity": perplexity,
                "epoch": epoch,
            }
            if val_loss is not None:
                log_dict["val/loss"] = val_loss
                log_dict["val/perplexity"] = val_ppl
            wandb.log(log_dict)
        
        # Generate sample at end of each epoch
        model.eval()
        prompt = "ALEXANDER:\n\n"
        prompt_tokens = dataset.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
        sample_text = dataset.decode(generated[0])
        print(f"\n--- Epoch {epoch+1} Sample ---")
        print(sample_text)
        print("----------------------------\n")
        
        model.train()


def generate_sample(
    model: CausalCTM, 
    dataset: ShakespeareDataset, 
    device: torch.device, 
    prompt: str = "ROMEO:",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    """Generate text sample from the model."""
    model.eval()
    prompt_tokens = dataset.encode(prompt).unsqueeze(0).to(device)
    generated = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return dataset.decode(generated[0])


def main(data_path: str = './data', checkpoint_dir: str = '.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset parameters
    seq_len = 256        
    batch_size = 8
    target_vocab_size = 1024  # Custom BPE vocab size
    
    print("Loading Shakespeare dataset...")
    train_dataset = ShakespeareDataset(data_path=data_path, seq_len=seq_len, split='train', vocab_size=target_vocab_size)
    val_dataset = ShakespeareDataset(data_path=data_path, seq_len=seq_len, split='val', vocab_size=target_vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model hyperparameters
    n_neurons = 210
    max_memory = 8
    max_ticks = 16
    d_model = 128
    n_synch_out = 64
    n_synch_action = 32
    n_attention_heads = 4
    dropout = 0.1
    epochs = 5
    lr = 0.003
    
    # Initialize model
    model = CausalCTM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        n_neurons=n_neurons,
        max_memory=max_memory,
        max_ticks=max_ticks,
        d_model=d_model,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        n_attention_heads=n_attention_heads,
        dropout=dropout,
    ).to(device)
    
    print_model_info(model, batch_size=batch_size, epochs=epochs, lr=lr)
    
    # Create AdamW optimizer for all parameters
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Wandb config
    total_params = count_parameters(model)
    wandb_config = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": "AdamW",
        "n_neurons": n_neurons,
        "max_memory": max_memory,
        "max_ticks": max_ticks,
        "d_model": d_model,
        "n_synch_out": n_synch_out,
        "n_synch_action": n_synch_action,
        "n_attention_heads": n_attention_heads,
        "dropout": dropout,
        "total_params": total_params,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "device": str(device),
    }
    
    wandb.init(
        project="shakespeare",
        name=f"{optimizer.__class__.__name__}seq{seq_len}_ticks{max_ticks}/{max_memory}bs{batch_size}",
        config=wandb_config
    )
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        dataset=train_dataset,
        val_loader=val_loader,
        use_wandb=True,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Generate sample
    print("\n=== Generation Sample ===")
    sample = generate_sample(model, train_dataset, device, prompt="ROMEO:", max_new_tokens=300)
    print(sample)
    print("=========================\n")
    
    wandb.log({"generation_sample": wandb.Html(f"<pre>{sample}</pre>")})
    
    # Visualize attention evolution on a sample
    print("\nVisualizing attention evolution...")
    sample_idx = np.random.randint(len(train_dataset))
    sample_x, sample_y = train_dataset[sample_idx]
    fig_buffer = visualize_attention_evolution(
        model=model,
        tokens=sample_x.unsqueeze(0),
        targets=sample_y.unsqueeze(0),
        dataset=train_dataset,
        device=device,
        query_position=-1,
        top_k=10,
        save_path=os.path.join(checkpoint_dir, 'attention_evolution.png'),
    )
    fig_buffer.seek(0)
    pil_image = Image.open(fig_buffer)
    wandb.log({"attention_evolution": wandb.Image(pil_image)})
    
    # Save model
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, 'causal_ctm_shakespeare.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    wandb.finish()


if __name__ == '__main__':
    main()

