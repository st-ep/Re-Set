import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from SetONet import DeepOSet
import os
# from plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions
from utilits.plotting_utils import plot_derivative_comparison, plot_trunk_basis_functions
from utilits.data_utils import generate_batch # Import the new function
from utilits.deeponet_model import MLP, DeepONet # Import new classes

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create log directory
log_dir = "logs/derivative"
os.makedirs(log_dir, exist_ok=True)

# Fix random seed
torch.manual_seed(0)
np.random.seed(0)

# some hyperparmeters for the problem
input_range = [-1,1]
scale = 1
sensor_size = 1000

# Sensor points (fixed for branch input)
sensor_x = torch.linspace(input_range[0], input_range[1], sensor_size).to(device)  # e.g., 20 equidistant points

# Define DeepONet
branch_net = MLP(input_dim=sensor_size, hidden_dims=[64, 64], output_dim=20, activation=nn.Tanh)
trunk_net = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=20, activation=nn.Tanh)
model = DeepONet(branch_net=branch_net, trunk_net=trunk_net).to(device)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

for epoch in range(50000):
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
    print(f"\nAverage L2 Relative Error over {n_test} test examples: {rel_error:.6f}")

# Save DeepONet model
deeponet_model_path = os.path.join(log_dir, "deeponet_model.pth")
torch.save(model.state_dict(), deeponet_model_path)
print(f"DeepONet model saved to {deeponet_model_path}")

# --- SetONet Implementation ---
# Define SetONet model
setonet_p_dim = 20 # Latent dimension, similar to DeepONet's output_dim
setonet_model = DeepOSet(
    input_size_src=1,       # Dimensionality of sensor location x_i (1D)
    output_size_src=1,      # Dimensionality of sensor value u(x_i) (scalar)
    input_size_tgt=1,       # Dimensionality of trunk input y (1D)
    output_size_tgt=1,      # Dimensionality of final output G(u)(y) (scalar)
    p=setonet_p_dim,
    phi_hidden_size=64,
    rho_hidden_size=64,
    trunk_hidden_size=64,
    n_trunk_layers=4,       # e.g., input, hidden, hidden, output
    activation_fn=nn.Tanh,
    use_deeponet_bias=True,
    phi_output_size=64,     # Output dim of phi before aggregation
    pos_encoding_type='skip', # No explicit positional encoding for sensor locations beyond their values
    aggregation_type="mean"
).to(device)

# Training loop for SetONet
optimizer_son = optim.Adam(setonet_model.parameters(), lr=5e-4)
# loss_fn is already defined

print("\nTraining SetONet...")
for epoch in range(50000): # Using the same number of epochs for comparison
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

# Call the plotting function for derivative comparison
x_dense_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)
plot_derivative_comparison(
    deeponet_model=model, # This is the trained DeepONet
    setonet_model=setonet_model,
    sensor_x=sensor_x,
    x_dense=x_dense_plot,
    input_range=input_range,
    scale=scale,
    log_dir=log_dir,
    device=device
)

# Call the plotting function for trunk basis functions
x_basis_plot = torch.linspace(input_range[0], input_range[1], 200).view(-1, 1).to(device)
plot_trunk_basis_functions(
    deeponet_model=model, # This is the trained DeepONet
    setonet_model=setonet_model,
    x_basis=x_basis_plot,
    setonet_p_dim=setonet_p_dim,
    log_dir=log_dir,
    device=device
)