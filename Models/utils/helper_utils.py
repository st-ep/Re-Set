import torch
# Import generate_batch, assuming Data is a sibling directory to Models or accessible in PYTHONPATH
# Adjust the import path if your project structure is different.
# If Data.data_utils is in the same parent directory as Models:
import sys
import os
# Add project root to sys.path if helper_utils.py might be run in a context where it can't find Data
# This is a bit defensive, usually the main script (Derivative.py) handles path setup.
# current_helper_path = os.path.abspath(__file__)
# project_root_from_helper = os.path.dirname(os.path.dirname(os.path.dirname(current_helper_path))) # Models -> utils -> helper_utils.py
# if project_root_from_helper not in sys.path:
#    sys.path.append(project_root_from_helper)
from Data.data_utils import generate_batch

def calculate_l2_relative_error(y_pred, y_true):
    """
    Calculates the mean L2 relative error.
    Args:
        y_pred (torch.Tensor): Predicted values, shape (batch_size, num_points).
        y_true (torch.Tensor): True values, shape (batch_size, num_points).
    Returns:
        torch.Tensor: Mean L2 relative error.
    """
    error_norm = torch.norm(y_pred - y_true, dim=1)
    true_norm = torch.norm(y_true, dim=1)
    # Add a small epsilon to prevent division by zero if true_norm is zero for some samples
    relative_error = error_norm / (true_norm + 1e-8) 
    return relative_error.mean()

def prepare_setonet_inputs(sensor_x_global, batch_f_values, batch_x_eval, current_batch_size, global_sensor_size):
    """
    Prepares the input tensors for SetONet.
    Args:
        sensor_x_global (torch.Tensor): Global sensor locations, shape (global_sensor_size,).
        batch_f_values (torch.Tensor): Batch of function values at sensor locations, shape (batch_size, global_sensor_size).
        batch_x_eval (torch.Tensor): Batch of trunk evaluation points, shape (n_trunk_points, 1) or (batch_size, n_trunk_points, 1).
        current_batch_size (int): The current batch size.
        global_sensor_size (int): The total number of sensor points.
    Returns:
        tuple: (xs_setonet, us_setonet, ys_setonet)
            xs_setonet (torch.Tensor): Sensor locations for SetONet, shape (batch_size, global_sensor_size, 1).
            us_setonet (torch.Tensor): Sensor values for SetONet, shape (batch_size, global_sensor_size, 1).
            ys_setonet (torch.Tensor): Trunk input locations for SetONet, shape (batch_size, n_trunk_points, 1).
    """
    # xs: sensor locations (batch_size, n_sensors, input_size_src=1)
    xs_setonet = sensor_x_global.view(1, global_sensor_size, 1).expand(current_batch_size, -1, -1)
    # us: sensor values (batch_size, n_sensors, output_size_src=1)
    us_setonet = batch_f_values.unsqueeze(-1)
    
    # ys: trunk input locations (batch_size, n_points, input_size_tgt=1)
    # batch_x_eval from generate_batch is [T, 1]. It needs to be [B, T, 1] for SetONet.
    if batch_x_eval.dim() == 2: # Expected [T, 1]
        ys_setonet = batch_x_eval.unsqueeze(0).expand(current_batch_size, -1, -1)
    elif batch_x_eval.dim() == 3: # Already [B, T, 1]
        ys_setonet = batch_x_eval
    else:
        raise ValueError(f"batch_x_eval has unexpected dimensions: {batch_x_eval.shape}")
        
    return xs_setonet, us_setonet, ys_setonet

def normalize_coordinates(coords: torch.Tensor, original_range: list[float]) -> torch.Tensor:
    """Normalizes coordinates from original_range to [-1, 1]."""
    min_val, max_val = float(original_range[0]), float(original_range[1])
    # Add epsilon to prevent division by zero if min_val == max_val
    return 2 * (coords - min_val) / (max_val - min_val + 1e-8) - 1

def normalize_values(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardizes values using given mean and std."""
    return (values - mean) / (std + 1e-8) # Add epsilon for stability

def denormalize_values(norm_values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """De-standardizes values using given mean and std."""
    return norm_values * (std + 1e-8) + mean 

# --- Helper Functions for Derivative Benchmark (moved from Benchmarks/Derivative.py) ---

def prepare_normalized_batch_for_setonet(
    batch_f_values, batch_x_eval, sensor_x_norm_global, 
    input_range_local, current_batch_size_local, sensor_size_local
):
    """
    Normalizes a batch of data (coordinates and function values) and prepares SetONet inputs.
    Assumes batch_f_values and batch_x_eval are already on the correct device.
    sensor_x_norm_global is also assumed to be on the correct device.
    """
    # 1. Normalize trunk evaluation points (coordinates)
    batch_x_eval_norm = normalize_coordinates(batch_x_eval.clone(), input_range_local)

    # 2. Normalize sensor values (input function u)
    mean_f_batch = batch_f_values.mean(dim=1, keepdim=True) 
    std_f_batch = batch_f_values.std(dim=1, keepdim=True)   
    batch_f_values_norm = normalize_values(batch_f_values, mean_f_batch, std_f_batch)

    xs, us, ys = prepare_setonet_inputs(
        sensor_x_norm_global, batch_f_values_norm, batch_x_eval_norm, 
        current_batch_size_local, sensor_size_local
    )
    return xs, us, ys, mean_f_batch, std_f_batch

def evaluate_setonet_on_test_set(
    model_to_eval, sensor_x_orig_eval, sensor_x_norm_global_eval, 
    input_range_eval, scale_eval, device_eval, 
    n_test_samples, n_trunk_points_eval, sensor_size_eval
):
    """Evaluates the SetONet model on a newly generated test set."""
    model_to_eval.eval()
    with torch.no_grad():
        f_values_test, x_eval_test, y_true_test = generate_batch(
            batch_size=n_test_samples, 
            n_trunk_points=n_trunk_points_eval, 
            sensor_x=sensor_x_orig_eval, # Original scale for data generation
            scale=scale_eval, 
            input_range=input_range_eval,
            device=device_eval
        )
        current_test_batch_size = f_values_test.shape[0]

        # Prepare normalized inputs for the model using the local helper
        xs_setonet_test, us_setonet_test, ys_setonet_test, _, _ = \
            prepare_normalized_batch_for_setonet( # Call the function within this module
                f_values_test, x_eval_test, sensor_x_norm_global_eval,
                input_range_eval, current_test_batch_size, sensor_size_eval
            )
        
        # Normalize test targets G(u)(y) - for denormalizing predictions
        mean_y_test = y_true_test.mean(dim=1, keepdim=True)
        std_y_test = y_true_test.std(dim=1, keepdim=True)
        
        y_pred_setonet_norm_raw = model_to_eval(xs_setonet_test, us_setonet_test, ys_setonet_test)
        
        # Denormalize test predictions using target's (G(u)(y)) statistics
        y_pred_setonet_denorm_raw = denormalize_values(
            y_pred_setonet_norm_raw, mean_y_test.unsqueeze(-1), std_y_test.unsqueeze(-1)
        )
        y_pred_setonet_denorm_squeezed = y_pred_setonet_denorm_raw.squeeze(-1)
        
        rel_error = calculate_l2_relative_error(y_pred_setonet_denorm_squeezed, y_true_test)
    return rel_error 