import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os 
from datetime import datetime 
from tqdm import tqdm 

# Add the project root directory to sys.path to allow finding Models, Plotting, Data
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.SetONet import SetONet
import argparse # Import argparse
from Plotting.plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions
from Data.data_utils import generate_batch # Import the new function
from Models.utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs # Import helpers
from Models.utils.helper_utils import normalize_coordinates, normalize_values, denormalize_values # Import normalization utils
# Import the newly moved helper functions
from Models.utils.helper_utils import prepare_normalized_batch_for_setonet, evaluate_setonet_on_test_set 

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train DeepONet and SetONet for a derivative task.")

# SetONet arguments
parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], help='Aggregation type for SetONet')
parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
parser.add_argument('--son_epochs', type=int, default=20000, help='Number of epochs for SetONet')
parser.add_argument('--pos_encoding_type', type=str, default='skip', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[100000, 50000, 150000, 200000, 250000], help="List of steps (iterations) for LR decay milestones.")
parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2], help="List of multiplicative factors (gammas) for LR decay at each step.")
args = parser.parse_args()

# project_root is already defined and points to Re-Set
logs_base_in_project = os.path.join(project_root, "logs")
model_folder_name = "SetONet" # Hardcode as only SetONet is run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")

# Fix random seed
torch.manual_seed(0)
np.random.seed(0)

# some hyperparmeters for the problem
input_range = [-1,1]
scale = 0.1
sensor_size = 1000

# Sensor points (fixed for branch input)
sensor_x_original = torch.linspace(input_range[0], input_range[1], sensor_size).to(device)  # e.g., 20 equidistant points
# Normalize global sensor_x once
sensor_x_norm = normalize_coordinates(sensor_x_original.clone(), input_range)

# Initialize model variables to None
# model = None # This was for DeepONet
setonet_model = None

# Define loss function globally
loss_fn = nn.MSELoss()

# --- SetONet Implementation ---
print("\n--- Initializing and Training SetONet ---")
# Define SetONet model
setonet_p_dim = args.son_p_dim # Use parsed argument
setonet_model = SetONet(
    input_size_src=1,       # Dimensionality of sensor location x_i (1D)
    output_size_src=1,      # Dimensionality of sensor value u(x_i) (scalar)
    input_size_tgt=1,       # Dimensionality of trunk input y (1D)
    output_size_tgt=1,      # Dimensionality of final output G(u)(y) (scalar)
    p=setonet_p_dim,
    phi_hidden_size=args.son_phi_hidden,
    rho_hidden_size=args.son_rho_hidden,
    trunk_hidden_size=args.son_trunk_hidden,
    n_trunk_layers=args.son_n_trunk_layers,
    activation_fn=nn.Tanh,
    use_deeponet_bias=True,
    phi_output_size=args.son_phi_output_size,
    pos_encoding_type=args.pos_encoding_type,
    aggregation_type=args.son_aggregation,
    initial_lr=args.son_lr, # Pass initial_lr
    lr_schedule_steps=args.lr_schedule_steps, # Pass LR schedule steps
    lr_schedule_gammas=args.lr_schedule_gammas # Pass LR schedule gammas
).to(device)

# Training loop for SetONet
optimizer_son = optim.Adam(setonet_model.parameters(), lr=args.son_lr)
# loss_fn is defined globally

print("\nTraining SetONet...")
# Wrap the epoch range with tqdm for a progress bar
epoch_pbar = tqdm(range(args.son_epochs), desc="Training SetONet")
for epoch in epoch_pbar: 
    setonet_model.train()
    # Generate data
    # f_values: [B, S], x_eval: [T, 1], y_target: [B, T]
    batch_f_values, batch_x_eval, batch_y_target = generate_batch(
        batch_size=64, 
        n_trunk_points=1000, 
        sensor_x=sensor_x_original, # Use original sensor_x for data generation
        scale=scale, 
        input_range=input_range,
        device=device
    )
    
    current_batch_size = batch_f_values.shape[0]
    # num_trunk_points = batch_x_eval.shape[0] # Not directly used after this

    # Prepare normalized inputs for SetONet using the helper function
    xs_setonet, us_setonet, ys_setonet, mean_f_batch, std_f_batch = \
        prepare_normalized_batch_for_setonet( # Updated to call imported function
            batch_f_values, batch_x_eval, sensor_x_norm,
            input_range, current_batch_size, sensor_size
        )
    # mean_f_batch, std_f_batch are for the input function u, not used for target G(u) normalization here.

    # Normalize target derivative values G(u)(y)
    mean_y_batch = batch_y_target.mean(dim=1, keepdim=True) # Shape [B, 1]
    std_y_batch = batch_y_target.std(dim=1, keepdim=True)   # Shape [B, 1]
    batch_y_target_norm = normalize_values(batch_y_target, mean_y_batch, std_y_batch) # Shape [B, T]
    pred_setonet_norm = setonet_model(xs_setonet, us_setonet, ys_setonet) # Output: [B, T, 1], normalized scale of target
    
    # Target for loss is normalized and needs to be [B, T, 1]
    target_setonet_norm_for_loss = batch_y_target_norm.unsqueeze(-1) 
    
    loss_setonet = loss_fn(pred_setonet_norm, target_setonet_norm_for_loss)

    # For metrics, denormalize prediction using target's (G(u)(y)) statistics
    # pred_setonet_norm is [B,T,1]. mean_y_batch/std_y_batch are [B,1] -> unsqueeze to [B,1,1] for broadcasting
    pred_setonet_denorm_for_metric = denormalize_values(pred_setonet_norm, mean_y_batch.unsqueeze(-1), std_y_batch.unsqueeze(-1))
    
    pred_setonet_denorm_squeezed = pred_setonet_denorm_for_metric.squeeze(-1) # [B,T]
    # batch_y_target is the original scale target [B,T]
    rel_l2_error_batch = calculate_l2_relative_error(pred_setonet_denorm_squeezed, batch_y_target)

    optimizer_son.zero_grad()
    loss_setonet.backward()
    optimizer_son.step()

    # Update the progress bar's postfix with the current loss and L2 relative error
    epoch_pbar.set_postfix(MSE=f"{loss_setonet.item():.4e}", RelL2=f"{rel_l2_error_batch.item():.4e}")

# Evaluate SetONet model using the helper function
n_test_samples_eval = 1000
n_trunk_points_eval = 100
rel_error_setonet = evaluate_setonet_on_test_set( # Updated to call imported function
    setonet_model, sensor_x_original, sensor_x_norm, 
    input_range, scale, device,
    n_test_samples_eval, n_trunk_points_eval, sensor_size
)
print(f"SetONet: Average L2 Relative Error over {n_test_samples_eval} test examples: {rel_error_setonet:.6f}")

# Save SetONet model
setonet_model_path = os.path.join(log_dir, "setonet_model.pth")
torch.save(setonet_model.state_dict(), setonet_model_path)
print(f"SetONet model saved to {setonet_model_path}")

# --- Plotting ---
print("\n--- Generating Plots ---")
x_dense_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)
x_basis_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)

if setonet_model: # Check if SetONet model was trained
    plot_derivative_comparison(
        deeponet_model=None, # DeepONet model is removed
        setonet_model=setonet_model,
        sensor_x=sensor_x_original, # Pass original sensor locations for plotting context
        x_dense=x_dense_plot,       # x_dense_plot is also in original scale
        input_range=input_range,    # Pass input_range for normalization within plotting
        scale=scale,
        log_dir=log_dir,
        num_samples_to_plot=3 # Generate plots for 3 samples
    )
else:
    print("No models were specified to run, skipping plotting.")

print("\nScript finished.")