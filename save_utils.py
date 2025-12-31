import torch
import torch.optim as optim
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ctm import SimplifiedCTM


def save_model_components(
    model: 'SimplifiedCTM',
    checkpoint_path: str,
    exclude_components: list[str] | None = None,
) -> None:
    """
    Save model components selectively for multi-task pretraining.
    
    Args:
        model: The SimplifiedCTM model to save
        checkpoint_path: Path to save the checkpoint
        exclude_components: List of component names to exclude (e.g., ['output_projector'] for task-specific components)
    
    Component names that can be excluded:
    - 'output_projector': Task-specific output layer (different number of classes per task)
    - 'start_activated_state', 'start_trace': Initial states (can be task-specific)
    - Any other parameter/buffer name
    """
    if exclude_components is None:
        exclude_components = []
    
    state_dict = model.state_dict()
    
    # Filter out excluded components
    filtered_state_dict = {
        key: value for key, value in state_dict.items()
        if not any(excluded in key for excluded in exclude_components)
    }
    
    # Extract d_input and in_channels from patch_embed
    d_input = None
    in_channels = None
    if hasattr(model.patch_embed, 'embed_dim'):
        d_input = model.patch_embed.embed_dim
    elif hasattr(model.patch_embed, 'proj'):
        if hasattr(model.patch_embed.proj, 'out_channels'):
            d_input = model.patch_embed.proj.out_channels
        if hasattr(model.patch_embed.proj, 'in_channels'):
            in_channels = model.patch_embed.proj.in_channels
    
    checkpoint = {
        'model_state_dict': filtered_state_dict,
        'model_config': {
            'n_neurons': model.n_neurons,
            'max_memory': model.max_memory,
            'max_ticks': model.max_ticks,
            'd_input': d_input,
            'n_synch_out': model.n_synch_out,
            'n_synch_action': model.n_synch_action,
            'n_attention_heads': model.n_attention_heads,
            'image_size': model.image_size,
            'patch_size': model.patch_size,
            'in_channels': in_channels,
            'input_ndim': model.input_ndim,
            'use_perceiver': model.use_perceiver,
        },
        'excluded_components': exclude_components,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved model components to {checkpoint_path}")
    if exclude_components:
        print(f"  Excluded components: {exclude_components}")


def load_model_components(
    model: 'SimplifiedCTM',
    checkpoint_path: str,
    strict: bool = False,
    load_task_specific: bool = False,
) -> dict:
    """
    Load model components from checkpoint.
    
    Args:
        model: The SimplifiedCTM model to load into
        checkpoint_path: Path to the checkpoint file
        strict: If True, all keys must match exactly. If False, missing keys are ignored.
        load_task_specific: If True, also load task-specific components (like output_projector)
                          even if they were excluded during save
    
    Returns:
        Dictionary with loaded information (epoch, metrics, etc. if present)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        excluded = checkpoint.get('excluded_components', [])
        
        if excluded and not load_task_specific:
            print(f"Note: Checkpoint excluded components: {excluded}")
            print("  Set load_task_specific=True to load them anyway")
        
        # If load_task_specific is False, filter out task-specific components
        if not load_task_specific and excluded:
            state_dict = {
                key: value for key, value in state_dict.items()
                if not any(excluded in key for excluded in excluded)
            }
    else:
        # Legacy format: direct state_dict
        state_dict = checkpoint
    
    # Load state dict with error handling
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys (not loaded): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    print(f"Loaded model components from {checkpoint_path}")
    
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(
    model: 'SimplifiedCTM',
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_path: str,
    metrics: dict | None = None,
    exclude_components: list[str] | None = None,
) -> None:
    """
    Save a complete checkpoint including model, optimizer, epoch, and metrics.
    
    Args:
        model: The SimplifiedCTM model
        optimizer: The optimizer
        epoch: Current epoch number
        checkpoint_path: Path to save the checkpoint
        metrics: Optional dictionary of metrics to save (e.g., {'train_loss': 0.5, 'val_acc': 0.9})
        exclude_components: List of component names to exclude (for multi-task pretraining)
    """
    if exclude_components is None:
        exclude_components = []
    
    # Save model components
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        key: value for key, value in model_state_dict.items()
        if not any(excluded in key for excluded in exclude_components)
    }
    
    # Extract d_input and in_channels from patch_embed
    d_input = None
    in_channels = None
    if hasattr(model.patch_embed, 'embed_dim'):
        d_input = model.patch_embed.embed_dim
    elif hasattr(model.patch_embed, 'proj'):
        if hasattr(model.patch_embed.proj, 'out_channels'):
            d_input = model.patch_embed.proj.out_channels
        if hasattr(model.patch_embed.proj, 'in_channels'):
            in_channels = model.patch_embed.proj.in_channels
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': filtered_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'n_neurons': model.n_neurons,
            'max_memory': model.max_memory,
            'max_ticks': model.max_ticks,
            'd_input': d_input,
            'n_synch_out': model.n_synch_out,
            'n_synch_action': model.n_synch_action,
            'n_attention_heads': model.n_attention_heads,
            'image_size': model.image_size,
            'patch_size': model.patch_size,
            'in_channels': in_channels,
            'input_ndim': model.input_ndim,
            'use_perceiver': model.use_perceiver,
        },
        'excluded_components': exclude_components,
    }
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_path}")
    if exclude_components:
        print(f"  Excluded components: {exclude_components}")


def load_checkpoint(
    model: 'SimplifiedCTM',
    optimizer: optim.Optimizer | None,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
    load_task_specific: bool = False,
) -> dict:
    """
    Load a complete checkpoint including model, optimizer, epoch, and metrics.
    
    Args:
        model: The SimplifiedCTM model to load into
        optimizer: The optimizer to load state into (can be None to skip optimizer loading)
        checkpoint_path: Path to the checkpoint file
        device: Device to load tensors to
        strict: If True, all model keys must match exactly
        load_task_specific: If True, also load task-specific components
    
    Returns:
        Dictionary with epoch, metrics, and other checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    state_dict = checkpoint['model_state_dict']
    excluded = checkpoint.get('excluded_components', [])
    
    if excluded and not load_task_specific:
        state_dict = {
            key: value for key, value in state_dict.items()
            if not any(excluded in key for excluded in excluded)
        }
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}")
    if metrics:
        print(f"  Metrics: {metrics}")
    
    return {
        'epoch': epoch,
        'metrics': metrics,
        'model_config': checkpoint.get('model_config', {}),
    }


def create_model_from_checkpoint(
    checkpoint_path: str,
    out_dims: int,
    device: torch.device,
    dropout: float = 0.0,
    dropout_nlm: float = 0.0,
) -> tuple['SimplifiedCTM', dict]:
    """
    Create a new SimplifiedCTM model from a checkpoint config, useful for multi-task pretraining.
    
    This function loads the model configuration from a checkpoint and creates a new model
    with potentially different output dimensions (for a new task). The pretrained weights
    are then loaded, excluding task-specific components.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        out_dims: Number of output classes for the new task (can differ from original)
        device: Device to create the model on
        dropout: Dropout rate for the new model
        dropout_nlm: Dropout rate for NLMs in the new model
    
    Returns:
        Tuple of (model, checkpoint_info_dict)
    
    Example:
        # Pretrain on task 1 (10 classes)
        model1 = SimplifiedCTM(..., out_dims=10)
        # ... train and save checkpoint excluding output_projector ...
        
        # Transfer to task 2 (20 classes) using pretrained weights
        model2, info = create_model_from_checkpoint('checkpoint.pth', out_dims=20, device=device)
        # model2 now has pretrained weights but new output_projector for 20 classes
    """
    from ctm import SimplifiedCTM
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config. Use a checkpoint saved with save_checkpoint()")
    
    config = checkpoint['model_config']
    
    # Create model with new output dimensions
    model = SimplifiedCTM(
        n_neurons=config['n_neurons'],
        max_memory=config['max_memory'],
        max_ticks=config['max_ticks'],
        d_input=config['d_input'],
        n_synch_out=config['n_synch_out'],
        n_synch_action=config['n_synch_action'],
        n_attention_heads=config['n_attention_heads'],
        out_dims=out_dims,  # New task-specific output dimension
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=config['in_channels'],
        input_ndim=config['input_ndim'],
        dropout=dropout,
        dropout_nlm=dropout_nlm,
        use_perceiver=config['use_perceiver'],
    ).to(device)
    
    # Load pretrained weights (excluding task-specific components)
    load_model_components(model, checkpoint_path, strict=False, load_task_specific=False)
    
    return model, checkpoint

