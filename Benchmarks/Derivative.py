import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys # Add sys import
import os # Add os import
from datetime import datetime # Import datetime

# Add the project root directory to sys.path to allow finding Models, Plotting, Data
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.SetONet import SetONet
import argparse # Import argparse
# from plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions
from Plotting.plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions
from Data.data_utils import generate_batch # Import the new function
from Models.deeponet_model import MLP, DeepONet # Import new classes

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train DeepONet and SetONet for a derivative task.")
# General arguments
parser.add_argument('--model_to_run', type=str, default='deeponet', choices=['deeponet', 'setonet'],
                    help='Which model to run: "deeponet" or "setonet".')

# DeepONet arguments (can be expanded)
parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate for DeepONet')
parser.add_argument('--don_epochs', type=int, default=500, help='Number of epochs for DeepONet')
parser.add_argument('--don_branch_hidden', nargs='+', type=int, default=[64, 64], help='Hidden layer sizes for DeepONet branch net')
parser.add_argument('--don_trunk_hidden', nargs='+', type=int, default=[64, 64], help='Hidden layer sizes for DeepONet trunk net')
parser.add_argument('--don_output_dim', type=int, default=20, help='Output dimension for DeepONet branch and trunk nets')

# SetONet arguments
parser.add_argument('--son_p_dim', type=int, default=20, help='Latent dimension p for SetONet')
parser.add_argument('--son_phi_hidden', type=int, default=64, help='Hidden size for SetONet phi network')
parser.add_argument('--son_rho_hidden', type=int, default=64, help='Hidden size for SetONet rho network')
parser.add_argument('--son_trunk_hidden', type=int, default=64, help='Hidden size for SetONet trunk network')
parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
parser.add_argument('--son_phi_output_size', type=int, default=64, help='Output size of SetONet phi network before aggregation')
parser.add_argument('--son_aggregation', type=str, default="mean", choices=["mean", "sum", "max"], help='Aggregation type for SetONet')
parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
parser.add_argument('--son_epochs', type=int, default=500, help='Number of epochs for SetONet')
parser.add_argument('--pos_encoding_type', type=str, default='skip', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')

args = parser.parse_args()

# Create log directory based on model name and timestamp
# Desired structure: Re-Set/logs/{model_name}/{date-time}

# project_root is already defined and points to Re-Set
logs_base_in_project = os.path.join(project_root, "logs")

model_folder_name = ""
if args.model_to_run == 'deeponet':
    model_folder_name = "DeepONet"
elif args.model_to_run == 'setonet':
    model_folder_name = "SetONet"
else: # Should not be reached
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

# some hyperparmeters for the problem
input_range = [-1,1]
scale = 1
sensor_size = 1000

# Sensor points (fixed for branch input)
sensor_x = torch.linspace(input_range[0], input_range[1], sensor_size).to(device)  # e.g., 20 equidistant points

# Initialize model variables to None
model = None
setonet_model = None

# Define loss function globally
loss_fn = nn.MSELoss()

# --- DeepONet Implementation ---
if args.model_to_run == 'deeponet':
    print("\n--- Initializing and Training DeepONet ---")
    # Define DeepONet
    branch_net = MLP(input_dim=sensor_size, hidden_dims=args.don_branch_hidden, output_dim=args.don_output_dim, activation=nn.Tanh)
    trunk_net = MLP(input_dim=1, hidden_dims=args.don_trunk_hidden, output_dim=args.don_output_dim, activation=nn.Tanh)
    model = DeepONet(branch_net=branch_net, trunk_net=trunk_net).to(device)

    # Training loop
    optimizer = optim.Adam(model.parameters(), lr=args.don_lr)
    # loss_fn is defined globally

    print("\nTraining DeepONet...")
    for epoch in range(args.don_epochs):
        model.train()
        branch_input, trunk_input, target = generate_batch(
            batch_size=64, 
            n_trunk_points=40, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )
        pred = model(branch_input, trunk_input)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    model.eval()
    # l2 relative error over large batch
    with torch.no_grad():
        n_test = 1000
        branch_input, trunk_input, y_true = generate_batch(
            batch_size=n_test, 
            n_trunk_points=100, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )
        y_pred = model(branch_input, trunk_input)
        error = torch.norm(y_pred - y_true, dim=1)
        denom = torch.norm(y_true, dim=1)
        rel_error = (error / denom).mean()
        print(f"\nDeepONet: Average L2 Relative Error over {n_test} test examples: {rel_error:.6f}")

    # Save DeepONet model
    deeponet_model_path = os.path.join(log_dir, "deeponet_model.pth")
    torch.save(model.state_dict(), deeponet_model_path)
    print(f"DeepONet model saved to {deeponet_model_path}")

# --- SetONet Implementation ---
if args.model_to_run == 'setonet':
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
        aggregation_type=args.son_aggregation
    ).to(device)

    # Training loop for SetONet
    optimizer_son = optim.Adam(setonet_model.parameters(), lr=args.son_lr)
    # loss_fn is defined globally

    print("\nTraining SetONet...")
    for epoch in range(args.son_epochs): # Using the same number of epochs for comparison
        setonet_model.train()
        # Generate data
        # f_values: [B, S], x_eval: [T, 1], y_target: [B, T]
        batch_f_values, batch_x_eval, batch_y_target = generate_batch(
            batch_size=64, 
            n_trunk_points=40, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )
        
        current_batch_size = batch_f_values.shape[0]
        num_trunk_points = batch_x_eval.shape[0]

        # Prepare inputs for SetONet
        # xs: sensor locations (batch_size, n_sensors, input_size_src=1)
        xs_setonet = sensor_x.view(1, sensor_size, 1).expand(current_batch_size, -1, -1)
        # us: sensor values (batch_size, n_sensors, output_size_src=1)
        us_setonet = batch_f_values.unsqueeze(-1)
        # ys: trunk input locations (batch_size, n_points, input_size_tgt=1)
        ys_setonet = batch_x_eval.unsqueeze(0).expand(current_batch_size, -1, -1)

        pred_setonet = setonet_model(xs_setonet, us_setonet, ys_setonet) # Output: [B, T, 1]
        
        # Target needs to be [B, T, 1]
        target_setonet = batch_y_target.unsqueeze(-1)
        
        loss_setonet = loss_fn(pred_setonet, target_setonet)

        optimizer_son.zero_grad()
        loss_setonet.backward()
        optimizer_son.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, SetONet Loss: {loss_setonet.item():.6f}")

    setonet_model.eval()
    # l2 relative error over large batch for SetONet
    with torch.no_grad():
        n_test = 1000
        f_values_test, x_eval_test, y_true_test = generate_batch(
            batch_size=n_test, 
            n_trunk_points=100, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )

        current_test_batch_size = f_values_test.shape[0]

        xs_setonet_test = sensor_x.view(1, sensor_size, 1).expand(current_test_batch_size, -1, -1)
        us_setonet_test = f_values_test.unsqueeze(-1)
        ys_setonet_test = x_eval_test.unsqueeze(0).expand(current_test_batch_size, -1, -1)
        
        y_pred_setonet = setonet_model(xs_setonet_test, us_setonet_test, ys_setonet_test) # [N_test, N_test_points, 1]
        y_true_setonet_reshaped = y_true_test.unsqueeze(-1) # [N_test, N_test_points, 1]

        # Ensure dimensions are compatible for norm calculation, typically (Batch, Values)
        # Original y_pred_don is [N_test, N_test_points]
        # y_pred_setonet is [N_test, N_test_points, 1], squeeze it.
        error_setonet = torch.norm(y_pred_setonet.squeeze(-1) - y_true_test, dim=1)
        denom_setonet = torch.norm(y_true_test, dim=1)
        rel_error_setonet = (error_setonet / denom_setonet).mean()
        print(f"SetONet: Average L2 Relative Error over {n_test} test examples: {rel_error_setonet:.6f}")

    # Save SetONet model
    setonet_model_path = os.path.join(log_dir, "setonet_model.pth")
    torch.save(setonet_model.state_dict(), setonet_model_path)
    print(f"SetONet model saved to {setonet_model_path}")

# --- Plotting ---
print("\n--- Generating Plots ---")
x_dense_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)
x_basis_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)

# model and setonet_model are already correctly set to the trained model or None
# based on args.model_to_run

if model or setonet_model: # Check if any model was trained
    plot_derivative_comparison(
        deeponet_model=model, # Will be None if SetONet was run
        setonet_model=setonet_model, # Will be None if DeepONet was run
        sensor_x=sensor_x,
        x_dense=x_dense_plot,
        input_range=input_range,
        scale=scale,
        log_dir=log_dir
    )

    plot_trunk_basis_functions(
        deeponet_model=model, # Will be None if SetONet was run
        setonet_model=setonet_model, # Will be None if DeepONet was run
        x_basis=x_basis_plot,
        setonet_p_dim=args.son_p_dim if setonet_model else None,
        log_dir=log_dir
    )
else:
    print("No models were specified to run, skipping plotting.")

print("\nScript finished.")