import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Data
from torch_geometric.transforms import SamplePoints, NormalizeScale
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from typing import Tuple
from sklearn.metrics import classification_report, confusion_matrix
from ctm import SimplifiedCTM, print_model_info, save_checkpoint, load_checkpoint
import wandb


class ModelNetDataset(torch.utils.data.Dataset):
    """Dataset wrapper for ModelNet point clouds with fixed-size sampling."""
    
    def __init__(self, pyg_dataset, num_points: int):
        self.pyg_dataset = pyg_dataset
        self.num_points = num_points
    
    def __len__(self) -> int:
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.pyg_dataset[idx]
        
        # Extract point cloud: (N, 3) where N can vary
        points = data.pos  # (N, 3)
        
        # Sample or pad to fixed number of points
        if points.shape[0] > self.num_points:
            # Randomly sample num_points
            indices = torch.randperm(points.shape[0])[:self.num_points]
            points = points[indices]
        elif points.shape[0] < self.num_points:
            # Pad with last point (or zeros)
            padding = points[-1:].repeat(self.num_points - points.shape[0], 1)
            points = torch.cat([points, padding], dim=0)
        
        # Convert to (3, num_points) format for CTM: (channels, length)
        # This treats xyz coordinates as 3 channels
        points = points.transpose(0, 1)  # (3, num_points)
        
        label = int(data.y)
        
        return points, label


class ModelNetCTM(nn.Module):
    """CTM for 3D point cloud classification using Perceiver embedding."""
    
    def __init__(
        self,
        num_points: int,
        num_classes: int,
        n_neurons: int,
        max_memory: int,
        max_ticks: int,
        d_input: int,
        n_synch_out: int,
        n_synch_action: int,
        n_attention_heads: int,
        dropout: float,
        dropout_nlm: float,
    ):
        super(ModelNetCTM, self).__init__()
        
        # CTM with Perceiver embedding
        # Input shape: (3, num_points) -> flattened to (3 * num_points) bytes
        # Using Perceiver to convert point cloud to byte-level tokens
        self.ctm = SimplifiedCTM(
            n_neurons=n_neurons,
            max_memory=max_memory,
            max_ticks=max_ticks,
            d_input=d_input,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            n_attention_heads=n_attention_heads,
            out_dims=num_classes,
            image_size=num_points,  # Used as sequence length for 1D
            patch_size=1,  # Not used with Perceiver, but required
            in_channels=3,  # xyz coordinates
            input_ndim=1,  # Treat as 1D sequence
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            use_perceiver=True,  # Enable Perceiver embedding
        )
    
    def forward(self, x: torch.Tensor, track: bool = False):
        """
        Forward pass.
        
        Args:
            x: (B, 3, num_points) point cloud tensor
            track: If True, return intermediate activations
        Returns:
            predictions, certainties (and tracking info if track=True)
        """
        return self.ctm(x, track=track)


def visualize_point_cloud_attention(
    model: ModelNetCTM,
    points: torch.Tensor,
    label: int,
    class_names: list,
    device: torch.device,
    epoch: int,
    num_points: int,
) -> None:
    """
    Visualize attention over point cloud.
    
    Shows attention weights as color intensity on 3D point cloud.
    """
    model.eval()
    with torch.no_grad():
        predictions, certainties, synch, pre_act, post_act, attention, nlm_activations = model(
            points.unsqueeze(0).to(device),
            track=True
        )
    
    # Convert attention to numpy
    if hasattr(attention, 'cpu'):
        attention = attention.cpu().numpy()
    
    # attention shape: (n_ticks, batch, n_heads, 1, n_patches)
    attention = attention[:, 0, :, 0, :]  # (n_ticks, n_heads, n_patches)
    attention_avg = attention.mean(axis=1)  # (n_ticks, n_patches)
    
    n_ticks = model.ctm.max_ticks
    
    # Get point coordinates: (3, num_points) -> (num_points, 3)
    points_np = points.transpose(0, 1).cpu().numpy()
    
    # Calculate class probabilities
    probs = F.softmax(predictions, dim=1)
    
    # Get certainties to find the most certain tick
    certainty_scores = certainties[0, 1, :].cpu().numpy()
    most_certain_tick_idx = certainty_scores.argmax()
    
    # Create figure: 2 rows (point cloud visualization, class probabilities)
    fig = plt.figure(figsize=(4 * n_ticks, 8))
    
    for tick_idx in range(n_ticks):
        # Get attention for this tick
        attn_flat = attention_avg[tick_idx]  # (n_patches,) - already numpy
        
        # For Perceiver, n_patches = 3 * num_points (flattened xyz)
        # Reshape to (3, num_points) and take mean across channels
        if len(attn_flat) == 3 * num_points:
            attn_reshaped = attn_flat.reshape(3, num_points).mean(axis=0)  # (num_points,)
        else:
            # Fallback: assume attention matches points
            attn_reshaped = attn_flat[:num_points]
        
        # Normalize attention for visualization
        attn_min, attn_max = attn_reshaped.min(), attn_reshaped.max()
        if attn_max > attn_min:
            attn_norm = (attn_reshaped - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = np.ones_like(attn_reshaped) * 0.5
        
        # Top row: 3D point cloud visualization
        ax_top = fig.add_subplot(2, n_ticks, tick_idx + 1, projection='3d')
        
        # Color points by attention
        scatter = ax_top.scatter(
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            c=attn_norm,
            cmap='hot',
            s=10,
            alpha=0.6,
        )
        ax_top.set_title(f'Tick {tick_idx}', fontsize=10)
        ax_top.set_xlabel('X')
        ax_top.set_ylabel('Y')
        ax_top.set_zlabel('Z')
        
        # Add border for most certain tick
        if tick_idx == most_certain_tick_idx:
            for spine in ax_top.spines.values():
                spine.set_visible(True)
                spine.set_color('blue')
                spine.set_linewidth(3)
        
        # Bottom row: class probabilities
        ax_bot = fig.add_subplot(2, n_ticks, n_ticks + tick_idx + 1)
        tick_probs = probs[0, :, tick_idx].cpu().numpy()
        bars = ax_bot.bar(range(len(class_names)), tick_probs, color='steelblue')
        ax_bot.set_ylim(0, 1)
        
        pred_idx = tick_probs.argmax()
        pred_name = class_names[pred_idx][:15] + '...' if len(class_names[pred_idx]) > 15 else class_names[pred_idx]
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
    
    true_class = class_names[label]
    plt.suptitle(f'Epoch {epoch} - Point Cloud Attention Visualization\nTrue: {true_class}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'modelnet_attention_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved modelnet_attention_epoch_{epoch}.png")


def train_epoch(
    model: ModelNetCTM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch, returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (points, labels) in enumerate(loader):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        
        predictions, certainties = model(points)
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
    model: ModelNetCTM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, list, list]:
    """Evaluate model, returns (loss, accuracy, predictions, labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: list = []
    all_labels: list = []
    
    if len(loader) == 0:
        return 0.0, 0.0, all_preds, all_labels
    
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            predictions, certainties = model(points)
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


def main(data_path: str = "./data", checkpoint_dir: str = "./checkpoints", resume_from: str | None = None):
    # Config
    num_points = 1024  # Fixed number of points per cloud
    batch_size = 16
    epochs = 10
    lr = 0.001
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # CTM config
    n_neurons = 128
    max_memory = 8
    max_ticks = 8
    d_input = 128
    n_synch_out = 64
    n_synch_action = 64
    n_attention_heads = 8
    dropout = 0.1
    dropout_nlm = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load ModelNet dataset
    print("Loading ModelNet10 dataset...")
    print("This may take a while for the first download...")
    
    # Transforms: sample fixed number of points and normalize
    transform = SamplePoints(num=num_points)
    normalize = NormalizeScale()
    
    train_dataset_pyg = ModelNet(
        root=data_path,
        name='10',
        train=True,
        transform=transform,
        pre_transform=normalize,
    )
    test_dataset_pyg = ModelNet(
        root=data_path,
        name='10',
        train=False,
        transform=transform,
        pre_transform=normalize,
    )
    
    print(f"Train samples: {len(train_dataset_pyg)}")
    print(f"Test samples: {len(test_dataset_pyg)}")
    
    # Get class names - extract unique labels from dataset
    # ModelNet10 has 10 classes, but we'll extract them dynamically
    all_labels = []
    for data in train_dataset_pyg:
        all_labels.append(int(data.y))
    unique_labels = sorted(set(all_labels))
    num_classes = len(unique_labels)
    
    # ModelNet10 class names (in order)
    modelnet10_classes = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet'
    ]
    
    # Use provided class names if available, otherwise use generic names
    if num_classes == 10 and len(modelnet10_classes) == 10:
        class_names = modelnet10_classes
    else:
        # Fallback: use generic class names
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Create dataset wrappers
    train_dataset = ModelNetDataset(train_dataset_pyg, num_points)
    test_dataset = ModelNetDataset(test_dataset_pyg, num_points)
    
    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model
    model = ModelNetCTM(
        num_points=num_points,
        num_classes=num_classes,
        n_neurons=n_neurons,
        max_memory=max_memory,
        max_ticks=max_ticks,
        d_input=d_input,
        n_synch_out=n_synch_out,
        n_synch_action=n_synch_action,
        n_attention_heads=n_attention_heads,
        dropout=dropout,
        dropout_nlm=dropout_nlm,
    ).to(device)
    
    print_model_info(model.ctm)
    
    # Compute total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Initialize wandb
    wandb.init(
        project="modelnet",
        name=f"ctm_perceiver_points{num_points}_ep{epochs}",
        config={
            "num_points": num_points,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "n_neurons": n_neurons,
            "max_memory": max_memory,
            "max_ticks": max_ticks,
            "d_input": d_input,
            "n_synch_out": n_synch_out,
            "n_synch_action": n_synch_action,
            "n_attention_heads": n_attention_heads,
            "dropout": dropout,
            "num_classes": num_classes,
            "total_params": total_params,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "device": str(device),
            "use_perceiver": True,
        }
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint_info = load_checkpoint(
            model=model.ctm,
            optimizer=optimizer,
            checkpoint_path=resume_from,
            device=device,
            strict=False,
            load_task_specific=True,
        )
        start_epoch = checkpoint_info['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Pre-select random samples for visualization (one per epoch, reproducible)
    np.random.seed(42)
    vis_indices = np.random.choice(len(train_dataset), size=min(epochs, len(train_dataset)), replace=False)
    
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, epochs):
        start = time.time()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": elapsed,
        })
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
        
        # Save checkpoint after each epoch
        # Exclude output_projector for multi-task pretraining (task-specific component)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(
            model=model.ctm,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            metrics={
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            },
            exclude_components=['output_projector'],  # Task-specific, exclude for pretraining
        )
        
        # Also save best model separately
        if val_acc == best_val_acc:
            best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
            save_checkpoint(
                model=model.ctm,
                optimizer=optimizer,
                epoch=epoch,
                checkpoint_path=best_checkpoint_path,
                metrics={
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                },
                exclude_components=['output_projector'],
            )
        
        # Visualize attention with a different sample each epoch
        if epoch < len(vis_indices):
            vis_points, vis_label = train_dataset[vis_indices[epoch]]
            visualize_point_cloud_attention(
                model=model,
                points=vis_points,
                label=vis_label,
                class_names=class_names,
                device=device,
                epoch=epoch + 1,
                num_points=num_points,
            )
            
            # Log attention visualization image to wandb
            vis_image_path = f'modelnet_attention_epoch_{epoch + 1}.png'
            if os.path.exists(vis_image_path):
                wandb.log({
                    "attention_visualization": wandb.Image(vis_image_path),
                })
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    
    # Log test metrics
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc,
    })
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_accuracy"] = test_acc
    
    if len(test_labels) > 0:
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        print(cm)
        
        # Log confusion matrix using wandb.plot
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_labels,
                preds=test_preds,
                class_names=class_names,
            )
        })
    
    # Save
    torch.save(model.state_dict(), "modelnet_ctm_perceiver_model.pth")
    print("\nModel saved to modelnet_ctm_perceiver_model.pth")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

