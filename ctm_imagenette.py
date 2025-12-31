import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import numpy as np
from ctm import SimplifiedCTM, train_model, print_model_info

def visualize_attention(
    model: SimplifiedCTM, 
    image: torch.Tensor, 
    device: torch.device,
    target: int = None,
    class_names: list = None
):
    """
    Visualize how attention weights evolve across ticks.
    This shows the foveated attention behavior!
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        predictions, certainties, synch, pre_act, post_act, attention = model(
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
    
    # Prepare image for visualization (denormalize if RGB, else grayscale)
    if image.shape[0] == 3:  # RGB
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_show = image.to(device) * std + mean
        img_np = img_show.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        cmap = None
    else:
        img_np = image.squeeze().cpu().numpy()
        cmap = 'gray'

    # Get certainties to find the most certain tick
    certainty_scores = certainties[0, 1, :].cpu().numpy()
    most_certain_tick_idx = certainty_scores.argmax()

    for i, tick_idx in enumerate(tick_indices):
        # Top row: attention heatmap overlaid on image
        axes[0, i].imshow(img_np, cmap=cmap, alpha=0.5)
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
        
        pred_idx = tick_probs.argmax()
        pred_text = str(pred_idx)
        if class_names and pred_idx < len(class_names):
            # Handle tuple class names (common in Imagenette)
            name = class_names[pred_idx]
            if isinstance(name, tuple):
                name = name[0]  # Take the first element (common name)
            
            name = str(name)
            # Truncate long class names
            pred_text = name[:15] + '...' if len(name) > 15 else name
            
        axes[2, i].set_title(f'{pred_text}', fontsize=9)
        
        # Underline the most certain tick
        if i == most_certain_tick_idx:
            axes[2, i].annotate('', xy=(0.05, 1.15), xytext=(0.95, 1.15), xycoords='axes fraction',
                                arrowprops=dict(arrowstyle='-', color='blue', lw=2))
        
        axes[2, i].set_xticks([])
        if i > 0:
            axes[2, i].set_yticks([])
            
        # Correctness at every tick
        if target is not None:
            is_correct = (pred_idx == target)
            color = 'green' if is_correct else 'red'
            text = "Correct" if is_correct else "Incorrect"
            axes[2, i].set_xlabel(text, color=color, fontweight='bold', fontsize=8)
    
    plt.suptitle('Foveated Attention and Predictions Over Ticks')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    # plt.show()
    print("Saved attention_visualization.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Imagenette setup
    input_size = 128
    patch_size = 16
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("Loading Imagenette dataset...")
    # Use Imagenette
    train_dataset = datasets.Imagenette(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model with spatial patches for foveated attention
    model = SimplifiedCTM(
        n_neurons=128,           # Increased for Imagenette
        max_memory=20,
        max_ticks=15,
        d_input=128,             # Increased embedding dimension
        n_synch_out=32,
        n_synch_action=32,
        n_attention_heads=4,
        out_dims=10,             # Imagenette has 10 classes
        input_size=input_size,
        patch_size=patch_size,
        in_channels=3,           # RGB
        dropout=0.1,             # Added dropout
        dropout_nlm=0.1,
    ).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    train_model(model, train_loader, optimizer, epochs=1, device=device)
    
    # Visualize the foveated attention behavior
    print("\nVisualizing attention patterns...")
    test_image, test_label = train_dataset[0]
    
    # Get class names if available, otherwise use indices
    class_names = getattr(train_dataset, 'classes', None)
    if class_names is None:
        # Fallback for Imagenette if classes not found
        class_names = [
            'tench', 'English springer', 'cassette player', 'chain saw', 'church', 
            'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
        ]
        
    visualize_attention(model, test_image, device, target=test_label, class_names=class_names)
    
    torch.save(model.state_dict(), 'ctm_imagenette_model.pth')
    print('Model saved to ctm_imagenette_model.pth')


if __name__ == '__main__':
    main()

