import torch
import torch.nn as nn
import numpy as np
import sys
import os
from datetime import datetime

# Add the project root directory to sys.path to allow finding Models, Plotting, Data
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from Pipelines.derivative_pipelines import run_deeponet_pipeline, run_setonet_pipeline
from Plotting.plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train DeepONet and SetONet for a derivative task.")
# General arguments
parser.add_argument('--model_to_run', type=str, default='setonet', choices=['deeponet', 'setonet'],
                    help='Which model to run: "deeponet" or "setonet".')

# DeepONet arguments
parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate for DeepONet')
parser.add_argument('--don_epochs', type=int, default=500, help='Number of epochs for DeepONet')
parser.add_argument('--don_branch_hidden', nargs='+', type=int, default=[256, 256], help='Hidden layer sizes for DeepONet branch net')
parser.add_argument('--don_trunk_hidden', nargs='+', type=int, default=[256, 256], help='Hidden layer sizes for DeepONet trunk net')
parser.add_argument('--don_output_dim', type=int, default=32, help='Output dimension for DeepONet branch and trunk nets')
parser.add_argument('--don_lr_schedule_steps', type=int, nargs='+', default=[30000, 100000, 150000, 200000, 250000],
                    help='Epoch milestones for DeepONet LR decay.')
parser.add_argument('--don_lr_schedule_gammas', type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2],
                    help='Multiplicative factors for DeepONet LR decay at each milestone.')

# SetONet arguments
parser.add_argument('--son_p_dim', type=int, default=16, help='Latent dimension p for SetONet')
parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
parser.add_argument('--son_n_trunk_layers', type=int, default=6, help='Number of layers in SetONet trunk network')
parser.add_argument('--son_phi_output_size', type=int, default=16, help='Output size of SetONet phi network before aggregation')
parser.add_argument('--son_aggregation', type=str, default="mean", choices=["mean", "attention"], help='Aggregation type for SetONet')
parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
parser.add_argument('--son_epochs', type=int, default=200000, help='Number of epochs for SetONet')
parser.add_argument('--pos_encoding_type', type=str, default='skip', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
parser.add_argument('--son_concat_sensor_derivative_to_branch', action='store_true',
                    help='For SetONet, concatenate derivative values at sensor locations to the branch input.')
parser.add_argument('--son_ortho_check_interval', type=int, default=500,
                    help='Epoch interval to check SetONet trunk orthogonality. 0 means only at the end. <0 means never.')
parser.add_argument('--son_lr_schedule_steps', type=int, nargs='+', default=[30000, 600000, 1500000, 2000000, 2500000],
                    help='Epoch milestones for SetONet LR decay.')
parser.add_argument('--son_lr_schedule_gammas', type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2],
                    help='Multiplicative factors for SetONet LR decay at each milestone.')

args = parser.parse_args()

# Validate LR schedule arguments
if len(args.don_lr_schedule_steps) != len(args.don_lr_schedule_gammas):
    raise ValueError("DeepONet --don_lr_schedule_steps and --don_lr_schedule_gammas must have the same number of elements.")
if len(args.son_lr_schedule_steps) != len(args.son_lr_schedule_gammas):
    raise ValueError("SetONet --son_lr_schedule_steps and --son_lr_schedule_gammas must have the same number of elements.")

# Create log directory based on model name and timestamp
logs_base_in_project = os.path.join(project_root, "logs")

model_folder_name = ""
if args.model_to_run == 'deeponet':
    model_folder_name = "DeepONet"
elif args.model_to_run == 'setonet':
    model_folder_name = "SetONet"
else:
    model_folder_name = "UnknownModel"

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Construct the final log directory
log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
os.makedirs(log_dir, exist_ok=True)
print(f"Logging to: {log_dir}")

# Fix random seed
torch.manual_seed(0)
np.random.seed(0)

# Some hyperparameters for the problem
input_range = [-1, 1]
scale = 1
sensor_size = 1000

# Sensor points (fixed for branch input)
sensor_x = torch.linspace(input_range[0], input_range[1], sensor_size).to(device)

# Define x_basis_plot
x_basis_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)

# Define loss function
loss_fn = nn.MSELoss()

# Initialize model variables to None
model = None
setonet_model = None

# Run the appropriate pipeline based on the selected model
if args.model_to_run == 'deeponet':
    model = run_deeponet_pipeline(
        args=args,
        device=device,
        sensor_x=sensor_x,
        loss_fn=loss_fn,
        log_dir=log_dir,
        input_range=input_range,
        scale=scale
    )
elif args.model_to_run == 'setonet':
    setonet_model = run_setonet_pipeline(
        args=args,
        device=device,
        sensor_x=sensor_x,
        x_basis_plot=x_basis_plot,
        loss_fn=loss_fn,
        log_dir=log_dir,
        input_range=input_range,
        scale=scale
    )

# --- Plotting ---
print("\n--- Generating Plots ---")
x_dense_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)

if model or setonet_model:
    plot_derivative_comparison(
        deeponet_model=model,
        setonet_model=setonet_model,
        sensor_x=sensor_x,
        x_dense=x_dense_plot,
        input_range=input_range,
        scale=scale,
        log_dir=log_dir
    )

    plot_trunk_basis_functions(
        deeponet_model=model,
        setonet_model=setonet_model,
        x_basis=x_basis_plot,
        setonet_p_dim=args.son_p_dim if setonet_model else None,
        log_dir=log_dir
    )
else:
    print("No models were specified to run, skipping plotting.")

print("\nScript finished.")