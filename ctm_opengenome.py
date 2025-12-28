import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from typing import Dict, List
from datasets import load_dataset, concatenate_datasets, DownloadConfig
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from ctm import SimplifiedCTM, print_model_info

# Enable HuggingFace datasets progress bars
os.environ["HF_DATASETS_VERBOSE"] = "1"

# DNA encoding
DNA_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
IDX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}

# Target families for classification
FAMILIES = [
    "Autographiviridae", "Kyanoviridae", "Microviridae", "Demerecviridae",
    "Inoviridae", "Schitoviridae", "Zobellviridae", "Herelleviridae",
    "Straboviridae", "Ackermannviridae"
]
FAMILY_TO_IDX = {family: idx for idx, family in enumerate(FAMILIES)}
IDX_TO_FAMILY = {idx: family for family, idx in FAMILY_TO_IDX.items()}


def parse_family(text: str) -> str:
    """Extract family from taxonomy string. Format: |taxonomy|sequence"""
    if not text.startswith("|"):
        return ""
    parts = text.split("|")
    if len(parts) < 2:
        return ""
    for item in parts[1].split(";"):
        if item.startswith("f__"):
            return item[3:]
    return ""


def extract_sequence(text: str) -> str:
    """Extract DNA sequence from text. Format: |taxonomy|sequence"""
    parts = text.split("|")
    return parts[2] if len(parts) >= 3 else ""


def encode_sequence(sequence: str, max_len: int) -> torch.Tensor:
    """Encode DNA sequence as nucleotide indices (0-4). Pads shorter sequences with N."""
    sequence = sequence.upper()
    encoded = np.full(max_len, DNA_TO_IDX['N'], dtype=np.int64)
    for i, char in enumerate(sequence):
        encoded[i] = DNA_TO_IDX.get(char, DNA_TO_IDX['N'])
    return torch.from_numpy(encoded)


class OpenGenomeDataset(torch.utils.data.Dataset):
    """Dataset for OpenGenome stage2 with on-the-fly encoding."""
    
    def __init__(self, hf_dataset, max_len: int):
        self.data = hf_dataset
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        sequence = extract_sequence(item["text"])
        encoded = encode_sequence(sequence, self.max_len)
        label = FAMILY_TO_IDX[item["family"]]
        return encoded, label


class DNAEmbedding(nn.Module):
    """Embed DNA nucleotide indices to dense vectors."""
    
    def __init__(self, num_nucleotides: int, embedding_dim: int):
        super(DNAEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nucleotides, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (B, L, embedding_dim)
        return embedded.transpose(1, 2)  # (B, embedding_dim, L)


class OpenGenomeCTM(nn.Module):
    """CTM for DNA sequence classification."""
    
    def __init__(
        self,
        seq_length: int,
        num_classes: int,
        n_neurons: int,
        max_memory: int,
        max_ticks: int,
        d_input: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        patch_size: int,
        nucleotide_embedding_dim: int,
        dropout: float,
        dropout_nlm: float,
    ):
        super(OpenGenomeCTM, self).__init__()
        self.dna_embedding = DNAEmbedding(5, nucleotide_embedding_dim)
        self.ctm = SimplifiedCTM(
            n_neurons=n_neurons,
            max_memory=max_memory,
            max_ticks=max_ticks,
            d_input=d_input,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            n_attention_heads=n_attention_heads,
            out_dims=num_classes,
            image_size=seq_length,
            patch_size=patch_size,
            in_channels=nucleotide_embedding_dim,
            input_ndim=1,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
        )
    
    def forward(self, x: torch.Tensor, track: bool = False):
        embedded = self.dna_embedding(x)
        return self.ctm(embedded, track=track)


def visualize_dna_attention(
    model: OpenGenomeCTM,
    sequence: torch.Tensor,
    label: int,
    device: torch.device,
    epoch: int,
    patch_size: int,
    seq_length: int,
    grid_width: int,
) -> None:
    """
    Visualize attention over DNA sequence as a 2D grid of colored nucleotides.
    
    Each cell shows the nucleotide character (A/C/G/T/N) with color intensity
    representing attention weight. Shows multiple ticks side by side.
    """
    model.eval()
    with torch.no_grad():
        predictions, certainties, synch, pre_act, post_act, attention = model(
            sequence.unsqueeze(0).to(device),
            track=True
        )
    
    # Get sequence as characters
    seq_indices = sequence.cpu().numpy()
    seq_chars = [IDX_TO_DNA[idx] for idx in seq_indices]
    
    # attention shape: (n_ticks, batch, n_heads, 1, n_patches)
    # Convert to numpy early to avoid tensor/numpy confusion
    if hasattr(attention, 'cpu'):
        attention = attention.cpu().numpy()
    attention = attention[:, 0, :, 0, :]  # (n_ticks, n_heads, n_patches)
    attention_avg = attention.mean(axis=1)  # (n_ticks, n_patches)
    
    n_ticks = model.ctm.max_ticks
    n_patches = attention_avg.shape[1]
    
    # Calculate class probabilities
    probs = F.softmax(predictions, dim=1)
    
    # Get certainties to find the most certain tick
    certainty_scores = certainties[0, 1, :].cpu().numpy()
    most_certain_tick_idx = certainty_scores.argmax()
    
    # Calculate grid dimensions for sequence display
    grid_height = seq_length // grid_width
    
    # Nucleotide colors (base colors before attention modulation)
    nuc_colors = {'A': '#2ecc71', 'C': '#3498db', 'G': '#f39c12', 'T': '#e74c3c', 'N': '#95a5a6'}
    
    # Create figure: one column per tick
    fig, axes = plt.subplots(2, n_ticks, figsize=(3 * n_ticks, 8))
    
    for tick_idx in range(n_ticks):
        # Get attention for this tick and expand to sequence level
        attn_patches = attention_avg[tick_idx]  # (n_patches,) - already numpy
        
        # Expand patch attention to nucleotide level
        attn_seq = np.repeat(attn_patches, patch_size)[:seq_length]
        
        # Normalize attention for visualization
        attn_min, attn_max = attn_seq.min(), attn_seq.max()
        if attn_max > attn_min:
            attn_norm = (attn_seq - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = np.ones_like(attn_seq) * 0.5
        
        # Create RGB image for the grid
        img = np.zeros((grid_height, grid_width, 3))
        
        for i, (char, attn_val) in enumerate(zip(seq_chars, attn_norm)):
            row = i // grid_width
            col = i % grid_width
            if row >= grid_height:
                break
            
            # Get base color and modulate by attention
            base_color = mcolors.to_rgb(nuc_colors[char])
            # Mix with black based on attention (higher attention = brighter)
            brightness = 0.2 + 0.8 * attn_val
            img[row, col] = [c * brightness for c in base_color]
        
        # Top row: attention-colored sequence grid
        ax_top = axes[0, tick_idx] if n_ticks > 1 else axes[0]
        ax_top.imshow(img, aspect='auto', interpolation='nearest')
        ax_top.set_title(f'Tick {tick_idx}', fontsize=10)
        ax_top.axis('off')
        
        # Add border for most certain tick
        if tick_idx == most_certain_tick_idx:
            for spine in ax_top.spines.values():
                spine.set_visible(True)
                spine.set_color('blue')
                spine.set_linewidth(3)
        
        # Bottom row: class probabilities
        ax_bot = axes[1, tick_idx] if n_ticks > 1 else axes[1]
        tick_probs = probs[0, :, tick_idx].cpu().numpy()
        bars = ax_bot.bar(range(len(FAMILIES)), tick_probs, color='steelblue')
        ax_bot.set_ylim(0, 1)
        
        pred_idx = tick_probs.argmax()
        pred_name = FAMILIES[pred_idx][:12] + '...' if len(FAMILIES[pred_idx]) > 12 else FAMILIES[pred_idx]
        ax_bot.set_title(f'{pred_name}', fontsize=8)
        
        # Highlight correct class bar
        if pred_idx == label:
            bars[pred_idx].set_color('green')
        else:
            bars[pred_idx].set_color('red')
            bars[label].set_color('green')
            bars[label].set_alpha(0.5)
        
        ax_bot.set_xticks([])
        if tick_idx > 0:
            ax_bot.set_yticks([])
        
        # Show correctness
        is_correct = (pred_idx == label)
        color = 'green' if is_correct else 'red'
        text = "✓" if is_correct else "✗"
        ax_bot.set_xlabel(text, color=color, fontweight='bold', fontsize=12)
    
    # Add legend for nucleotides
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=nuc_colors[n], label=n) for n in 'ACGTN']
    fig.legend(handles=legend_elements, loc='upper right', title='Nucleotides', fontsize=8)
    
    true_family = FAMILIES[label]
    plt.suptitle(f'Epoch {epoch} - DNA Attention Visualization\nTrue: {true_family}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'dna_attention_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved dna_attention_epoch_{epoch}.png")


def load_opengenome_stage2(max_seq_length: int, use_cache: bool) -> tuple:
    """Load LongSafari/open-genome stage2 train/validation/test splits, filtered to target families and length."""
    from datasets import Dataset
    
    # Set cache directory to D: drive
    cache_dir = "D:/huggingface/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Path for the processed/filtered dataset cache
    filtered_cache_dir = os.path.join(cache_dir, f"opengenome_filtered_maxlen{max_seq_length}")
    train_cache = os.path.join(filtered_cache_dir, "train")
    val_cache = os.path.join(filtered_cache_dir, "validation")
    test_cache = os.path.join(filtered_cache_dir, "test")
    
    # Check if filtered cache exists and use_cache is True
    if use_cache and os.path.exists(train_cache):
        print(f"Loading cached filtered dataset from: {filtered_cache_dir}")
        train_ds = Dataset.load_from_disk(train_cache)
        val_ds = Dataset.load_from_disk(val_cache)
        test_ds = Dataset.load_from_disk(test_cache)
        print(f"Loaded from cache: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_ds, val_ds, test_ds
    
    print("Loading LongSafari/open-genome stage2...")
    print("This may take a while for the first download (17GB)...")
    print(f"Will filter sequences longer than {max_seq_length} bp")
    
    # Enable verbose logging and progress bars
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Cache directory: {cache_dir}")
    
    # Configure download with progress bars
    download_config = DownloadConfig(
        num_proc=1,  # Single process to avoid conflicts
        max_retries=3,
        cache_dir=cache_dir,
    )
    
    # Check if dataset might already be cached
    dataset_cache_path = os.path.join(cache_dir, "LongSafari___open-genome", "stage2")
    if os.path.exists(dataset_cache_path):
        print(f"Found existing HF cache at: {dataset_cache_path}")
        print("If stuck, the dataset might be verifying checksums...")
    else:
        print("No cache found - downloading from HuggingFace (this may take a while)...")
    
    print("Starting dataset download/loading (this may take several minutes)...")
    print("If this appears stuck, check your network connection and disk space.")
    
    try:
        dataset = load_dataset(
            "LongSafari/open-genome", 
            "stage2",
            cache_dir=cache_dir,
            download_config=download_config,
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This might be a network issue or disk space problem.")
        raise
    
    family_set = set(FAMILIES)
    
    def add_family_and_length(example: Dict) -> Dict:
        family = parse_family(example["text"])
        sequence = extract_sequence(example["text"])
        seq_len = len(sequence)
        keep_family = family in family_set
        keep_length = seq_len <= max_seq_length
        return {
            "family": family,
            "seq_length": seq_len,
            "keep": keep_family and keep_length
        }
    
    # Process all splits with progress indication
    print("Processing train split...")
    train_ds = dataset["train"].map(add_family_and_length, desc="Adding family/length").filter(lambda x: x["keep"], desc="Filtering").remove_columns(["keep", "seq_length"])
    print("Processing validation split...")
    val_ds = dataset["validation"].map(add_family_and_length, desc="Adding family/length").filter(lambda x: x["keep"], desc="Filtering").remove_columns(["keep", "seq_length"])
    print("Processing test split...")
    test_ds = dataset["test"].map(add_family_and_length, desc="Adding family/length").filter(lambda x: x["keep"], desc="Filtering").remove_columns(["keep", "seq_length"])
    
    print(f"Filtered: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    # Save filtered dataset to cache
    print(f"Saving filtered dataset to cache: {filtered_cache_dir}")
    os.makedirs(filtered_cache_dir, exist_ok=True)
    train_ds.save_to_disk(train_cache)
    val_ds.save_to_disk(val_cache)
    test_ds.save_to_disk(test_cache)
    print("Filtered dataset cached for future runs!")
    
    return train_ds, val_ds, test_ds


def sanity_check_dataset(dataset, dataset_name: str) -> bool:
    """
    Verify all target families are represented in the dataset.
    Returns True if all classes are present, raises error otherwise.
    """
    print(f"\nSanity check for {dataset_name}:")
    
    if len(dataset) == 0:
        print(f"  WARNING: {dataset_name} is empty!")
        return False
    
    counts = Counter(item["family"] for item in dataset)
    missing_classes: List[str] = []
    
    for family in FAMILIES:
        count = counts.get(family, 0)
        if count == 0:
            missing_classes.append(family)
            print(f"  ✗ {family}: 0 samples (MISSING!)")
        else:
            print(f"  ✓ {family}: {count} samples")
    
    if missing_classes:
        raise ValueError(
            f"SANITY CHECK FAILED for {dataset_name}!\n"
            f"Missing classes: {missing_classes}\n"
            f"Consider increasing max_seq_length or using different target families."
        )
    
    print(f"  All {len(FAMILIES)} classes present in {dataset_name}")
    return True


def compute_class_weights(dataset, num_classes: int, power: float) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    counts = Counter(item["family"] for item in dataset)
    total = sum(counts.values())
    
    weights = torch.ones(num_classes)
    for family, idx in FAMILY_TO_IDX.items():
        if family in counts:
            weights[idx] = (total / (num_classes * counts[family])) ** power
    return weights


def train_epoch(
    model: OpenGenomeCTM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train for one epoch, returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        
        predictions, certainties = model(sequences)
        B, C, T = predictions.shape
        
        # CTM adaptive compute loss
        labels_expanded = labels.unsqueeze(-1).expand(-1, T)
        loss_all = F.cross_entropy(
            predictions.permute(0, 2, 1).reshape(-1, C),
            labels_expanded.reshape(-1),
            reduction='none'
        ).reshape(B, T)
        
        loss_min, _ = loss_all.min(dim=1)
        most_certain_idx = certainties[:, 1, :].argmax(dim=1)
        batch_indices = torch.arange(B, device=device)
        loss_selected = loss_all[batch_indices, most_certain_idx]
        
        loss = (loss_min.mean() + loss_selected.mean()) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        final_preds = predictions[batch_indices, :, most_certain_idx]
        _, predicted = torch.max(final_preds, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(
    model: OpenGenomeCTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Evaluate model, returns (loss, accuracy, predictions, labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    # Handle empty loader
    if len(loader) == 0:
        return 0.0, 0.0, all_preds, all_labels
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            predictions, certainties = model(sequences)
            B = predictions.shape[0]
            
            most_certain_idx = certainties[:, 1, :].argmax(dim=1)
            batch_indices = torch.arange(B, device=device)
            final_preds = predictions[batch_indices, :, most_certain_idx]
            
            loss = F.cross_entropy(final_preds, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(final_preds, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total == 0:
        return 0.0, 0.0, all_preds, all_labels
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels


def main():
    # Config
    seq_length = 8192
    patch_size = 32
    batch_size = 16
    epochs = 10
    lr = 0.001
    use_cached_dataset = True  # Set to False to force re-processing
    
    # CTM config
    n_neurons = 256
    max_memory = 12
    max_ticks = 8
    d_input = 128
    n_synch_out = 64
    n_synch_action = 32
    n_attention_heads = 4
    nucleotide_embedding_dim = 16
    dropout = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data (filter by family AND sequence length)
    train_hf, val_hf, test_hf = load_opengenome_stage2(
        max_seq_length=seq_length,
        use_cache=use_cached_dataset,
    )
    
    # If validation/test sets are empty, split training data
    if len(val_hf) == 0 or len(test_hf) == 0:
        print("\nWarning: Validation/test sets are empty after filtering.")
        print("Splitting training data into train/val/test (80/10/10)...")
        train_hf_shuffled = train_hf.shuffle(seed=42)
        total_size = len(train_hf_shuffled)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_hf = train_hf_shuffled.select(range(train_size))
        val_hf = train_hf_shuffled.select(range(train_size, train_size + val_size))
        test_hf = train_hf_shuffled.select(range(train_size + val_size, total_size))
        
        print(f"Split: train={len(train_hf)}, val={len(val_hf)}, test={len(test_hf)}")
    
    # Sanity check: verify all classes are present in final datasets
    sanity_check_dataset(train_hf, "train")
    if len(val_hf) > 0:
        sanity_check_dataset(val_hf, "validation")
    if len(test_hf) > 0:
        sanity_check_dataset(test_hf, "test")
    
    # Create dataloaders
    train_ds = OpenGenomeDataset(train_hf, seq_length)
    val_ds = OpenGenomeDataset(val_hf, seq_length)
    test_ds = OpenGenomeDataset(test_hf, seq_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    num_classes = len(FAMILIES)
    model = OpenGenomeCTM(
        seq_length=seq_length,
        num_classes=num_classes,
        n_neurons=n_neurons,
        max_memory=max_memory,
        max_ticks=max_ticks,
        d_input=d_input,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        n_attention_heads=n_attention_heads,
        patch_size=patch_size,
        nucleotide_embedding_dim=nucleotide_embedding_dim,
        dropout=dropout,
        dropout_nlm=dropout,
    ).to(device)
    
    print_model_info(model.ctm)
    
    # Class weights & optimizer
    alpha_weights = compute_class_weights(train_hf, num_classes, power=0.5).to(device)
    print(f"\nClass weights: {alpha_weights.cpu().numpy()}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    vis_grid_width = 128  # Grid width for visualization (seq_length / grid_width = grid_height)
    
    # Pre-select random samples for visualization (one per epoch, reproducible)
    np.random.seed(42)
    vis_indices = np.random.choice(len(train_ds), size=epochs, replace=False)
    
    for epoch in range(epochs):
        start = time.time()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        if len(val_loader) > 0:
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
            f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if len(val_labels) > 0 else 0.0
            elapsed = time.time() - start
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={f1:.4f}")
        else:
            elapsed = time.time() - start
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Skipped (empty)")
        print(f"  Time: {elapsed:.1f}s")
        
        # Visualize attention with a different sample each epoch
        vis_sequence, vis_label = train_ds[vis_indices[epoch]]
        visualize_dna_attention(
            model=model,
            sequence=vis_sequence,
            label=vis_label,
            device=device,
            epoch=epoch + 1,
            patch_size=patch_size,
            seq_length=seq_length,
            grid_width=vis_grid_width,
        )
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    if len(test_loader) > 0:
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
        print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        if len(test_labels) > 0:
            target_names = [IDX_TO_FAMILY[i] for i in range(num_classes)]
            print("\nClassification Report:")
            print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(test_labels, test_preds))
        else:
            print("No test samples available for evaluation.")
    else:
        print("Test set is empty - skipping evaluation.")
    
    # Save
    torch.save(model.state_dict(), "opengenome_ctm_model.pth")
    print("\nModel saved to opengenome_ctm_model.pth")


if __name__ == "__main__":
    main()
