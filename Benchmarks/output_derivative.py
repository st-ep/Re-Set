"""
Benchmark: Encoding the output derivative function with SetONet
----------------------------------------------------------------
Instead of learning the full operator  u ↦ F(u)(y)  we focus on
encoding *only* the output space, i.e. we reconstruct the derivative
f′(x) from a handful of sample points.

branch input     : {(x_i , f′(x_i))}_i      ➜ coefficients  (B , p , 1)
trunk  input     : {y_j}_j                  ➜ basis values  (B , n_y , p , 1)
prediction       :  Σ_p  coeff_p · basis_p  ≈ f′(y)

The data generator is shared with the other pipelines (`generate_batch`).
"""

import torch
import torch.nn as nn
from tqdm import trange
import os
from datetime import datetime
import matplotlib.pyplot as plt            # trunk–basis plot
from torch.optim.lr_scheduler import MultiStepLR # Import the scheduler

# Project imports ------------------------------------------------------------
from Models.SetONet import SetONet
from Data.data_utils import generate_batch
from Models.utils.orthogonality_utils import (
    calculate_setonet_trunk_orthogonality,
    plot_setonet_trunk_orthogonality,
)
from Plotting.plotting_utils import plot_derivative_comparison           

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def relative_L2(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Mean relative L2 error  ‖pred-tgt‖₂ / ‖tgt‖₂  over the batch.
    Both tensors have shape (B , N)
    """
    err   = torch.norm(pred - tgt, dim=1)
    denom = torch.norm(tgt,          dim=1)
    return (err / denom).mean()

def normalize_coordinates(coords: torch.Tensor, original_range: list[float]) -> torch.Tensor:
    """Normalizes coordinates from original_range to [-1, 1]."""
    min_val, max_val = original_range
    return 2 * (coords - min_val) / (max_val - min_val) - 1

def normalize_values(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardizes values using given mean and std."""
    return (values - mean) / std

def denormalize_values(norm_values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """De-standardizes values using given mean and std."""
    return norm_values * std + mean

# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def main() -> None:
    # ----------------- configuration ---------------------------------------
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    input_range       = [-1.0, 1.0]   # domain of x
    scale             = 0.1          # coefficient scale for cubic polynomials
    n_sensor_points   = 200           # samples used as branch input
    n_trunk_points    = 100           # evaluation points per function
    latent_p          = 16            # SetONet latent dimension
    batch_size        = 256
    n_epochs          = 50_000        
    print_interval    = 500
    lr                = 1e-3
    # Learning rate scheduler parameters
    lr_schedule_steps = [400000, 750000, 1000000] # Epoch milestones for LR decay
    lr_schedule_gamma = 0.5                  # Multiplicative factor for LR decay at each milestone
    l1_lambda         = 2e-5                 # Strength of L1 regularization

    # ----------------- logging directory -----------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir      = os.path.join(project_root, "logs", "output_derivative", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # ----------------- fixed sensor grid -----------------------------------
    sensor_x = torch.linspace(
        input_range[0], input_range[1], n_sensor_points, device=device
    )                                   # (n_sensor_points,)

    # Normalized sensor grid for model input
    sensor_x_normalized = normalize_coordinates(sensor_x, input_range)
    sensor_x_norm_expanded = sensor_x_normalized.view(1, -1, 1) # (1, n_sensor_points, 1)

    # ----------------- model ------------------------------------------------
    model = SetONet(
        input_size_src   = 1,
        output_size_src  = 1,
        input_size_tgt   = 1,
        output_size_tgt  = 1,
        p                = latent_p,
        phi_hidden_size  = 256,
        rho_hidden_size  = 256,
        trunk_hidden_size= 256,
        n_trunk_layers   = 4,          # ← default in Derivative.py
        activation_fn    = nn.ReLU,
        use_deeponet_bias= True,
        phi_output_size  = 16,         # ← default in Derivative.py
        aggregation_type = "attention",
        pos_encoding_type= "skip",      # no extra positional encoding
        concat_sensor_derivative_to_branch_input = False,
        initial_lr       = lr
    ).to(device)

    opt     = torch.optim.Adam(model.parameters(), lr=lr) # Removed weight_decay
    loss_fn = nn.MSELoss()
    scheduler = MultiStepLR(opt, milestones=lr_schedule_steps, gamma=lr_schedule_gamma)

    # ----------------- training loop ---------------------------------------
    ortho_check_interval = 500
    ortho_epochs, ortho_scores = [], []

    bar = trange(n_epochs, desc="Training   ")
    for epoch in bar:
        # Generate synthetic batch of cubic polynomials and their derivatives. These are in the original scale and domain
        _, x_eval_orig, y_true_orig, df_dx_sensors_orig = generate_batch(
            batch_size       = batch_size,
            n_trunk_points   = n_trunk_points,
            sensor_x         = sensor_x, # Pass original sensor_x for data generation
            scale            = scale,
            input_range      = input_range,
            device           = device
        )

        # -------- Normalize data for SetONet -------------------------------
        # 1. Normalize coordinates. Sensor_x is already normalized to sensor_x_normalized
        x_eval_norm = normalize_coordinates(x_eval_orig, input_range)

        # 2. Standardize function values (y_true_orig and df_dx_sensors_orig)
        # Calculate mean and std from y_true_orig (batch-wise)
        # y_true_orig shape: (B, n_trunk_points)
        # df_dx_sensors_orig shape: (B, n_sensor_points)
        y_true_mean = y_true_orig.mean(dim=1, keepdim=True)
        y_true_std  = y_true_orig.std(dim=1, keepdim=True) + 1e-8 # for numerical stability
        y_true_norm = normalize_values(y_true_orig, y_true_mean, y_true_std)
        
        df_dx_sensors_norm = normalize_values(df_dx_sensors_orig,
                                              y_true_mean, 
                                              y_true_std)  


        # -------- SetONet inputs (normalized) ------------------------------------------
        xs_norm = sensor_x_norm_expanded.expand(batch_size, -1, -1) # (B , n_sensor , 1)
        us_norm = df_dx_sensors_norm.unsqueeze(-1)                  # (B , n_sensor , 1)
        
        # Reshape and expand ys_norm for the batch
        # Assuming x_eval_norm is (n_trunk_points, 1) 
        # We want ys_norm to be (B, n_trunk_points, 1)
        ys_norm = x_eval_norm.unsqueeze(0).expand(batch_size, -1, -1) # (B , n_trunk , 1)

        # Forward / loss / update
        pred_norm = model(xs_norm, us_norm, ys_norm)                # (B , n_trunk , 1)
        target_norm = y_true_norm.unsqueeze(-1)                     # (B , n_trunk , 1)

        mse_loss = loss_fn(pred_norm, target_norm)

        # Add L1 regularization penalty
        l1_penalty = 0.0
        if l1_lambda > 0: # Only apply if l1_lambda is positive
            for param in model.parameters():
                l1_penalty += torch.abs(param).sum()
        
        loss = mse_loss + l1_lambda * l1_penalty

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step() # Step the scheduler after each optimizer step

        # Logging (de-normalize prediction for meaningful RelL2)
        if epoch % print_interval == 0:
            pred_denorm = denormalize_values(pred_norm.squeeze(-1), y_true_mean, y_true_std)
            rel_err = relative_L2(pred_denorm, y_true_orig).item()
            bar.set_postfix({"MSE (norm)": f"{loss.item():.2e}",
                             "RelL2 (orig)": f"{rel_err:.2e}"})

        # --------- trunk orthogonality monitoring -------------------------
        if epoch % ortho_check_interval == 0 or epoch == n_epochs - 1:
            # Trunk orthogonality should be checked on the normalized domain it sees
            ortho = calculate_setonet_trunk_orthogonality(
                model,
                sensor_x_normalized.unsqueeze(-1), # Use normalized sensor grid
                latent_p,
                device
            )
            if ortho is not None:
                ortho_epochs.append(epoch)
                ortho_scores.append(ortho)

    # ----------------- simple test -----------------------------------------
    model.eval()
    with torch.no_grad():
        n_test = 1000
        _, x_eval_orig_test, y_true_orig_test, df_dx_sensors_orig_test = generate_batch(
            batch_size       = n_test,
            n_trunk_points   = n_trunk_points,
            sensor_x         = sensor_x, # Original sensor_x
            scale            = scale,
            input_range      = input_range,
            device           = device
        )

        # Normalize test data
        x_eval_norm_test = normalize_coordinates(x_eval_orig_test, input_range)
        
        y_true_test_mean = y_true_orig_test.mean(dim=1, keepdim=True)
        y_true_test_std  = y_true_orig_test.std(dim=1, keepdim=True) + 1e-8
        
        df_dx_sensors_norm_test = normalize_values(df_dx_sensors_orig_test, y_true_test_mean, y_true_test_std)

        xs_norm_test = sensor_x_norm_expanded.expand(n_test, -1, -1)
        us_norm_test = df_dx_sensors_norm_test.unsqueeze(-1)
        
        # Reshape and expand ys_norm_test for the batch
        # Assuming x_eval_norm_test is (n_trunk_points, 1).
        # We want ys_norm_test to be (n_test, n_trunk_points, 1)
        ys_norm_test = x_eval_norm_test.unsqueeze(0).expand(n_test, -1, -1)

        pred_norm_test = model(xs_norm_test, us_norm_test, ys_norm_test).squeeze(-1) # (B, N)
        
        # De-normalize for error calculation
        pred_denorm_test = denormalize_values(pred_norm_test, y_true_test_mean, y_true_test_std)
        
        rel_err_test = relative_L2(pred_denorm_test, y_true_orig_test).item()
        print(f"\nTest  Relative L2 Error (original scale): {rel_err_test:.4e}")

    # ----------------- save model ------------------------------------------
    model_path = os.path.join(log_dir, "setonet_output_encoder.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ----------------- plots ------------------------------------------------
    # 1) Orthogonality evolution
    plot_setonet_trunk_orthogonality(ortho_epochs, ortho_scores, log_dir)

    # 2) Plot learned trunk basis functions. Trunk functions are learned over the normalized domain [-1, 1]
    with torch.no_grad():
        x_basis_normalized = torch.linspace(-1.0, 1.0, 400).to(device).view(1, -1, 1)
        basis   = model.trunk(x_basis_normalized).squeeze(0)      # shape (N , p)

    plt.figure(figsize=(10, 6))
    for i in range(latent_p):
        plt.plot(
            x_basis_normalized.squeeze(0).cpu().numpy(), # Plot against normalized x
            basis[:, i].cpu().numpy()
        )
    plt.title("SetONet Trunk Basis Functions (on Normalized Domain)")
    plt.xlabel("y (normalized to [-1, 1])")
    plt.ylabel("ψ_i(y_norm)")
    plt.grid(True)
    basis_plot_path = os.path.join(log_dir, "trunk_basis_functions.png")
    plt.savefig(basis_plot_path)
    plt.close()
    print(f"Trunk basis plot saved to {basis_plot_path}")

    # 3) Prediction plot  ----------------------------------------------------
    x_dense_plot_orig = torch.linspace(input_range[0], input_range[1], 200, device=device).view(-1, 1)
    plot_derivative_comparison(
        deeponet_model=None,
        setonet_model=model,
        sensor_x       = sensor_x.cpu(),          
        x_dense        = x_dense_plot_orig.cpu(), 
        input_range    = input_range,            
        scale          = scale,
        log_dir        = log_dir,
    )

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
