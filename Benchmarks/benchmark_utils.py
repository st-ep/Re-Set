import torch

def relative_L2(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Mean relative L2 error  ‖pred-tgt‖₂ / ‖tgt‖₂  over the batch.
    Both tensors have shape (B , N)
    """
    err   = torch.norm(pred - tgt, dim=1)
    denom = torch.norm(tgt,          dim=1)
    # Add a small epsilon to denom to prevent division by zero if tgt is all zeros
    return (err / (denom + 1e-8)).mean()

def normalize_coordinates(coords: torch.Tensor, original_range: list[float]) -> torch.Tensor:
    """Normalizes coordinates from original_range to [-1, 1]."""
    min_val, max_val = original_range
    return 2 * (coords - min_val) / (max_val - min_val) - 1

def normalize_values(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardizes values using given mean and std."""
    return (values - mean) / (std + 1e-8) # Add epsilon for stability

def denormalize_values(norm_values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """De-standardizes values using given mean and std."""
    return norm_values * (std + 1e-8) + mean # Add epsilon for stability 