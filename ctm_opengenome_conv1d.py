"""
Conv1D baseline for OpenGenome viral family classification.
Reuses the cached filtered dataset from ctm_opengenome.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from collections import Counter
from typing import List
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import wandb

# DNA encoding
DNA_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

# Target families for classification
FAMILIES = [
    "Autographiviridae", "Kyanoviridae", "Microviridae", "Demerecviridae",
    "Inoviridae", "Schitoviridae", "Zobellviridae", "Herelleviridae",
    "Straboviridae", "Ackermannviridae"
]
FAMILY_TO_IDX = {family: idx for idx, family in enumerate(FAMILIES)}
IDX_TO_FAMILY = {idx: family for family, idx in FAMILY_TO_IDX.items()}


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
    """Dataset for OpenGenome with on-the-fly encoding."""
    
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


class Conv1DGenomeClassifier(nn.Module):
    """Simple Conv1D model for DNA sequence classification."""
    
    def __init__(
        self,
        seq_length: int,
        num_classes: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
    ):
        super(Conv1DGenomeClassifier, self).__init__()
        
        # Nucleotide embedding
        self.embedding = nn.Embedding(5, embedding_dim)
        
        # Multiple conv layers with different kernel sizes (captures different motif lengths)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Batch norm for each conv
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # Global pooling will reduce to (batch, num_filters * len(kernel_sizes))
        self.fc_input_dim = num_filters * len(kernel_sizes)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length) - nucleotide indices
        
        # Embed: (batch, seq_length, embedding_dim)
        x = self.embedding(x)
        
        # Transpose for conv1d: (batch, embedding_dim, seq_length)
        x = x.transpose(1, 2)
        
        # Apply each conv + relu + global max pool
        conv_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            c = conv(x)  # (batch, num_filters, seq_length)
            c = bn(c)
            c = F.relu(c)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (batch, num_filters)
            conv_outputs.append(c)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * n_kernels)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_cached_dataset(max_seq_length: int, data_path: str = "D:/huggingface/datasets") -> tuple:
    """Load the cached filtered dataset."""
    cache_dir = data_path
    filtered_cache_dir = os.path.join(cache_dir, f"opengenome_filtered_maxlen{max_seq_length}")
    train_cache = os.path.join(filtered_cache_dir, "train")
    val_cache = os.path.join(filtered_cache_dir, "validation")
    test_cache = os.path.join(filtered_cache_dir, "test")
    
    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"Cached dataset not found at {filtered_cache_dir}\n"
            f"Run ctm_opengenome.py first to create the cached dataset."
        )
    
    print(f"Loading cached dataset from: {filtered_cache_dir}")
    train_ds = Dataset.load_from_disk(train_cache)
    val_ds = Dataset.load_from_disk(val_cache)
    test_ds = Dataset.load_from_disk(test_cache)
    print(f"Loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    return train_ds, val_ds, test_ds


def compute_class_weights(dataset, num_classes: int, power: float) -> torch.Tensor:
    """Compute inverse frequency class weights with stronger scaling."""
    counts = Counter(item["family"] for item in dataset)
    total = sum(counts.values())
    
    weights = torch.ones(num_classes)
    for family, idx in FAMILY_TO_IDX.items():
        if family in counts:
            weights[idx] = (total / (num_classes * counts[family])) ** power
    return weights


def train_epoch(
    model: Conv1DGenomeClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Train for one epoch, returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(
    model: Conv1DGenomeClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model, returns (loss, accuracy, predictions, labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    if len(loader) == 0:
        return 0.0, 0.0, all_preds, all_labels
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total == 0:
        return 0.0, 0.0, all_preds, all_labels
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels


def main(data_path: str = "D:/huggingface/datasets"):
    # Config
    seq_length = 8192
    batch_size = 32  # Can be larger since Conv1D is simpler
    epochs = 15
    lr = 0.001
    
    # Model config
    embedding_dim = 32
    num_filters = 128
    kernel_sizes = [3, 7, 15, 31, 63]  # Different motif lengths: 3bp to 63bp
    dropout = 0.3
    class_weight_power = 0.7  # Stronger class weighting
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load cached data
    train_hf, val_hf, test_hf = load_cached_dataset(max_seq_length=seq_length, data_path=data_path)
    
    # Split training data if val/test empty
    if len(val_hf) == 0 or len(test_hf) == 0:
        print("\nSplitting training data into train/val/test (80/10/10)...")
        train_hf_shuffled = train_hf.shuffle(seed=42)
        total_size = len(train_hf_shuffled)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_hf = train_hf_shuffled.select(range(train_size))
        val_hf = train_hf_shuffled.select(range(train_size, train_size + val_size))
        test_hf = train_hf_shuffled.select(range(train_size + val_size, total_size))
        print(f"Split: train={len(train_hf)}, val={len(val_hf)}, test={len(test_hf)}")
    
    # Print class distribution
    print("\nClass distribution (train):")
    counts = Counter(item["family"] for item in train_hf)
    for family in FAMILIES:
        print(f"  {family}: {counts.get(family, 0)}")
    
    # Create dataloaders
    train_ds = OpenGenomeDataset(train_hf, seq_length)
    val_ds = OpenGenomeDataset(val_hf, seq_length)
    test_ds = OpenGenomeDataset(test_hf, seq_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    num_classes = len(FAMILIES)
    model = Conv1DGenomeClassifier(
        seq_length=seq_length,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Conv1D Model ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Num filters: {num_filters}")
    print(f"Kernel sizes: {kernel_sizes}")
    print(f"Dropout: {dropout}")
    print(f"====================\n")
    
    # Class weights for imbalanced data (stronger weighting)
    class_weights = compute_class_weights(train_hf, num_classes, power=class_weight_power).to(device)
    print(f"Class weights (power={class_weight_power}): {class_weights.cpu().numpy().round(2)}")
    
    # Initialize wandb
    wandb.init(
        project="opengenome",
        name=f"conv1d_seq{seq_length}_ep{epochs}",
        config={
            "seq_length": seq_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "embedding_dim": embedding_dim,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
            "class_weight_power": class_weight_power,
            "num_classes": num_classes,
            "total_params": total_params,
            "train_size": len(train_hf),
            "val_size": len(val_hf),
            "test_size": len(test_hf),
            "device": str(device),
        }
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        start = time.time()
        
        print(f"\nEpoch {epoch+1}/{epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if len(val_loader) > 0:
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            elapsed = time.time() - start
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={f1:.4f}")
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1_macro": f1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": elapsed,
            })
            
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), "conv1d_best_model.pth")
                wandb.run.summary["best_val_f1"] = f1
                wandb.run.summary["best_epoch"] = epoch + 1
                print(f"  New best model saved! (F1={f1:.4f})")
        else:
            elapsed = time.time() - start
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            
            # Log metrics to wandb (no validation)
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": elapsed,
            })
        
        print(f"  Time: {elapsed:.1f}s")
        scheduler.step()
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("conv1d_best_model.pth"))
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    if len(test_loader) > 0:
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        # Log test metrics
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/f1_macro": test_f1,
        })
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_accuracy"] = test_acc
        wandb.run.summary["test_f1_macro"] = test_f1
        
        if len(test_labels) > 0:
            target_names = [IDX_TO_FAMILY[i] for i in range(num_classes)]
            print("\nClassification Report:")
            report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0, output_dict=True)
            print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
            
            # Log per-class metrics
            for family in target_names:
                if family in report:
                    wandb.log({
                        f"test/{family}_precision": report[family]['precision'],
                        f"test/{family}_recall": report[family]['recall'],
                        f"test/{family}_f1": report[family]['f1-score'],
                    })
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(test_labels, test_preds)
            print(cm)
            
            # Log confusion matrix using wandb.plot
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=test_labels,
                    preds=test_preds,
                    class_names=target_names,
                )
            })
    else:
        print("Test set is empty - skipping evaluation.")
    
    # Save final model
    torch.save(model.state_dict(), "conv1d_genome_model.pth")
    print("\nFinal model saved to conv1d_genome_model.pth")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

