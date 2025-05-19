"""
Plotting utilities for 1D SetONet benchmarks, standardized for f(x) approximation view.
"""
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.gridspec as gridspec

from Data.data_utils import generate_batch, generate_batch_sin, generate_batch_cool_basis
from Models.SetONet import SetONet # For type hinting
from Models.utils.orthogonality_utils import plot_setonet_trunk_orthogonality
from Benchmarks.benchmark_utils import normalize_coordinates, normalize_values, denormalize_values
# plot_derivative_comparison is no longer imported as its functionality for individual plots
# will be handled by _plot_single_sample_approximation.

# --- Helper function to plot Gramian Matrix ---
def plot_trunk_gramian_heatmap(
    gramian_matrix: torch.Tensor, # Should be a CPU tensor
    log_dir: str,
    epoch: int,
    p_dim: int,
    task_type: str
):
    """
    Plots and saves a heatmap of the trunk basis Gramian matrix.
    """
    if gramian_matrix is None or gramian_matrix.shape[0] != p_dim or gramian_matrix.shape[1] != p_dim:
        print(f"Warning: Invalid Gramian matrix for epoch {epoch}. Skipping heatmap plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(gramian_matrix.numpy(), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Inner product value')
    plt.title(f"Trunk Basis Gramian Matrix (Epoch {epoch})\nTask: {task_type.replace('_', ' ').title()}, p={p_dim}")
    plt.xlabel("Basis function index")
    plt.ylabel("Basis function index")
    
    # Add text annotations for values if p_dim is small enough
    if p_dim <= 10: # Arbitrary threshold for readability
        for i in range(p_dim):
            for j in range(p_dim):
                text_color = "white" if gramian_matrix[i, j] < 0.5 else "black"
                plt.text(j, i, f"{gramian_matrix[i, j]:.2f}",
                         ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    plot_filename = f"trunk_gramian_matrix_epoch_{epoch}_{task_type}.png"
    plot_path = os.path.join(log_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        print(f"Trunk Gramian matrix heatmap saved to {plot_path}")
    except Exception as e:
        print(f"Error saving trunk Gramian matrix heatmap: {e}")
    plt.close()


# --- Standardized Single Sample Plot Helper ---
def _plot_single_sample_approximation(
    x_dense_orig: torch.Tensor,  # Original scale dense points for plotting (CPU, 1D)
    true_values: torch.Tensor,   # True target values at x_dense_orig (CPU, 1D)
    predicted_values: np.ndarray, # Predicted target values by SetONet (CPU, 1D, numpy)
    log_dir: str,
    plot_idx: int,
    task_type_for_filename: str # To differentiate filenames if needed, e.g. "input_function"
):
    """
    Plots a single sample comparison (true vs. predicted) with standardized labels.
    Saves the plot to the specified log directory.
    """
    plt.figure(figsize=(10, 6)) # Consistent figure size for sample plots
    plt.plot(x_dense_orig.cpu().numpy(), true_values.cpu().numpy(), 'k-', label="True $f(x)$", linewidth=2)
    plt.plot(x_dense_orig.cpu().numpy(), predicted_values, 'r--', label="SetONet Predicted $f(x)$", linewidth=2)
    
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$f(x)$", fontsize=14)
    plt.title(f"Approximation Comparison (Sample {plot_idx})", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Use task_type_for_filename to ensure unique names if both tasks run in same dir (though unlikely with current setup)
    plot_path = os.path.join(log_dir, f"approximation_sample_{task_type_for_filename}_{plot_idx}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Approximation plot for sample {plot_idx} ({task_type_for_filename}) saved to {plot_path}")


# --- Unified Combined Overview Plot ---
def plot_combined_overview_1d(
    model: SetONet,
    device: torch.device,
    task_type: str, # "input_function" or "output_derivative"
    latent_p: int,
    sensor_x_orig: torch.Tensor, # Original scale sensor grid (CPU tensor)
    input_range: list[float],
    # Data for the specific sample 0 to plot:
    x_dense_orig_plot_sample0: torch.Tensor,    # (n_points_plot, 1) or (n_points_plot,), CPU tensor
    true_values_plot_sample0: torch.Tensor,   # (n_points_plot,), CPU tensor (f(x) or df/dx)
    true_sens_values_branch_sample0: torch.Tensor, # (n_sensors,), CPU tensor (f(x_s) or df/dx_s)
    n_trunk_points_plot_basis: int = 200,
    font_size: int = 20
):
    """
    Generates a combined plot:
    1. Function/Derivative reconstruction for the provided sample 0.
    2. Trunk basis functions.
    """
    model.eval()
    model.to(device)
    print(f"\n--- Generating Combined Overview Plot (Task: {task_type}, using Sample 0 data) ---")

    fig = plt.figure(figsize=(16, 7)) # Consistent figsize
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    # --- Subplot 1: Reconstruction (Function or Derivative) ---
    sensor_x_norm = normalize_coordinates(sensor_x_orig.cpu(), input_range).unsqueeze(0).unsqueeze(-1).to(device)

    # Ensure x_dense_orig_plot_sample0 is [N,1] for normalization if 1D
    if x_dense_orig_plot_sample0.ndim == 1:
        x_dense_orig_plot_sample0_shaped = x_dense_orig_plot_sample0.unsqueeze(-1)
    else:
        x_dense_orig_plot_sample0_shaped = x_dense_orig_plot_sample0


    target_mean = true_values_plot_sample0.mean().item()
    target_std = true_values_plot_sample0.std().item() + 1e-8

    sens_values_norm = normalize_values(
        true_sens_values_branch_sample0.unsqueeze(0).to(device),
        torch.tensor([[target_mean]], device=device),
        torch.tensor([[target_std]], device=device)
    ).unsqueeze(-1)

    x_dense_norm_plot = normalize_coordinates(x_dense_orig_plot_sample0_shaped.cpu(), input_range).unsqueeze(0).to(device)

    pred_values_denorm_sample0 = None
    with torch.no_grad():
        pred_norm = model(sensor_x_norm, sens_values_norm, x_dense_norm_plot)
        pred_values_denorm_sample0 = denormalize_values(
            pred_norm.squeeze(-1),
            torch.tensor([[target_mean]], device=device),
            torch.tensor([[target_std]], device=device)
        ).cpu().numpy().flatten()

    # Standardized labels and colors for the combined plot's reconstruction
    true_label = "True $f(x)$"
    pred_label = "SetONet Predicted $f(x)$"
    y_axis_label = "$f(x)$"
    true_color = 'k'
    pred_color = 'r'
    true_linestyle = '-'
    pred_linestyle = '--'

    axs[0].plot(
        x_dense_orig_plot_sample0.cpu().numpy(), # Ensure x is 1D for plotting
        true_values_plot_sample0.cpu().numpy(),
        color=true_color, linestyle=true_linestyle, label=true_label, linewidth=2
    )
    if pred_values_denorm_sample0 is not None:
        axs[0].plot(
            x_dense_orig_plot_sample0.cpu().numpy(), # Ensure x is 1D for plotting
            pred_values_denorm_sample0,
            color=pred_color, linestyle=pred_linestyle, label=pred_label, linewidth=2
        )
    axs[0].set_xlabel("$x$", fontsize=font_size)
    axs[0].set_ylabel(y_axis_label, fontsize=font_size)
    axs[0].legend(fontsize=font_size-2)
    axs[0].grid(True)
    axs[0].tick_params(axis='both', which='major', labelsize=font_size-2)

    # --- Subplot 2: Trunk Basis Functions (Common for both tasks) ---
    with torch.no_grad():
        x_basis_normalized = torch.linspace(-1.0, 1.0, n_trunk_points_plot_basis, device=device).view(1, -1, 1)
        basis = model.trunk(x_basis_normalized).squeeze(0)

    if basis.shape[1] == latent_p:
        for i in range(latent_p):
            axs[1].plot(
                x_basis_normalized.squeeze().cpu().numpy(),
                basis[:, i].cpu().numpy()
            )
        axs[1].set_xlabel("y", fontsize=font_size) # Consistent label
        axs[1].set_ylabel("t_i(y)", fontsize=font_size) # Consistent label
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=font_size-2)
    else:
        warning_text = f"Warning: Trunk output dim {basis.shape[1]} != latent_p {latent_p}.\nSkipping basis plot."
        axs[1].text(0.5, 0.5, warning_text, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        print(f"{warning_text.replace('n', ' ')} for combined plot.")

    return fig, axs


# --- Unified Plot Generation Function ---
def generate_plots_1d(
    model: SetONet,
    device: torch.device,
    log_dir: str,
    task_type: str, # "input_function", "output_derivative", "input_function_sin", or "input_function_cool_basis"
    ortho_epochs: list,
    ortho_scores: list,
    latent_p: int,
    sensor_x_orig: torch.Tensor, # Original scale sensor points (CPU)
    input_range: list[float],
    scale: float, # Used by both data generators
    n_trunk_points_plot: int, # For input_function sample plots & basis plots
    n_samples_to_plot: int,
    # Specific to output_derivative, but can have defaults
    n_dense_points_derivative_plot: int = 200,
    # Parameters for combined plot appearance
    font_size_combined: int = 20,
    margin_left: float = 0.09,
    margin_right: float = 0.99,
    margin_bottom: float = 0.1,
    margin_top: float = 0.98
):
    """
    Generates and saves all relevant plots for the specified 1D task.
    """
    print(f"\n--- Generating Plots for Task: {task_type} ---")
    model.eval()
    model.to(device)

    # 1. Plot trunk basis function orthogonality development
    ortho_title = f"SetONet Trunk Orthogonality ({task_type.replace('_', ' ').title()})"
    if ortho_epochs and ortho_scores:
        plot_setonet_trunk_orthogonality(
            ortho_epochs,
            ortho_scores,
            log_dir,
            title=ortho_title
        )

    # To store data for sample 0 for the combined plot
    data_for_combined_plot = {}
    
    # Define x_dense for individual sample plots (consistent resolution)
    # For "input_function", x_dense comes from generate_batch.
    # For "output_derivative", we define it here.
    x_dense_individual_plots_cpu = torch.linspace(
        input_range[0], input_range[1],
        n_dense_points_derivative_plot if task_type == "output_derivative" else n_trunk_points_plot
    ).cpu().view(-1, 1) # Ensure [N,1] for normalization, then squeeze for plotting

    print(f"\n--- Generating {n_samples_to_plot} Individual Sample Plots (Standardized View) ---")

    coeffs_sample0 = None # For derivative task sample 0
    if task_type == "output_derivative":
        a0 = torch.randn(1).item() * scale
        b0 = torch.randn(1).item() * scale
        c0_coeff = torch.randn(1).item() * scale
        d0_coeff = torch.randn(1).item() * scale # Renamed from d0
        coeffs_sample0 = (a0, b0, c0_coeff, d0_coeff)

        # Prepare data for combined plot (Sample 0 of derivative task)
        true_df_sens_values_branch_sample0 = (3*a0*sensor_x_orig.cpu()**2 + 2*b0*sensor_x_orig.cpu() + c0_coeff).squeeze()
        true_df_values_plot_sample0 = (3*a0*x_dense_individual_plots_cpu.squeeze()**2 + 2*b0*x_dense_individual_plots_cpu.squeeze() + c0_coeff)
        
        data_for_combined_plot['x_dense_orig_plot_sample0'] = x_dense_individual_plots_cpu.squeeze()
        data_for_combined_plot['true_values_plot_sample0'] = true_df_values_plot_sample0
        data_for_combined_plot['true_sens_values_branch_sample0'] = true_df_sens_values_branch_sample0

    for i in range(n_samples_to_plot):
        current_x_dense_cpu = None
        true_target_values_cpu = None # This will be f(x) or df/dx for the current sample
        sensor_target_values_cpu = None # Branch input values for the current sample

        if task_type == "input_function":
            f_sens_orig_sample, x_eval_orig_sample, f_eval_orig_sample, _, _ = generate_batch(
                batch_size=1, n_trunk_points=n_trunk_points_plot, # n_trunk_points_plot for x_dense resolution
                sensor_x=sensor_x_orig.to(device), scale=scale,
                input_range=input_range, device=device
            )
            current_x_dense_cpu = x_eval_orig_sample.squeeze(0).cpu() # Should be [N,1] or [N,]
            true_target_values_cpu = f_eval_orig_sample.squeeze(0).cpu()
            sensor_target_values_cpu = f_sens_orig_sample.squeeze(0).cpu()
            
            if i == 0: # Store data for combined plot (Sample 0 of input_function task)
                data_for_combined_plot['x_dense_orig_plot_sample0'] = current_x_dense_cpu.squeeze()
                data_for_combined_plot['true_values_plot_sample0'] = true_target_values_cpu
                data_for_combined_plot['true_sens_values_branch_sample0'] = sensor_target_values_cpu
        
        elif task_type == "input_function_sin":
            f_sens_orig_sample, x_eval_orig_sample, f_eval_orig_sample = generate_batch_sin(
                batch_size=1, n_trunk_points=n_trunk_points_plot,
                sensor_x=sensor_x_orig.to(device), scale=scale,
                input_range=input_range, device=device
            )
            current_x_dense_cpu = x_eval_orig_sample.squeeze(0).cpu()
            true_target_values_cpu = f_eval_orig_sample.squeeze(0).cpu()
            sensor_target_values_cpu = f_sens_orig_sample.squeeze(0).cpu()

            if i == 0: # Store data for combined plot (Sample 0 of input_function_sin task)
                data_for_combined_plot['x_dense_orig_plot_sample0'] = current_x_dense_cpu.squeeze()
                data_for_combined_plot['true_values_plot_sample0'] = true_target_values_cpu
                data_for_combined_plot['true_sens_values_branch_sample0'] = sensor_target_values_cpu

        elif task_type == "input_function_cool_basis":
            f_sens_orig_sample, x_eval_orig_sample, f_eval_orig_sample = generate_batch_cool_basis(
                batch_size=1, n_trunk_points=n_trunk_points_plot,
                sensor_x=sensor_x_orig.to(device), scale=scale,
                input_range=input_range, device=device
            )
            current_x_dense_cpu = x_eval_orig_sample.squeeze(0).cpu()
            true_target_values_cpu = f_eval_orig_sample.squeeze(0).cpu()
            sensor_target_values_cpu = f_sens_orig_sample.squeeze(0).cpu()

            if i == 0: # Store data for combined plot (Sample 0 of input_function_cool_basis task)
                data_for_combined_plot['x_dense_orig_plot_sample0'] = current_x_dense_cpu.squeeze()
                data_for_combined_plot['true_values_plot_sample0'] = true_target_values_cpu
                data_for_combined_plot['true_sens_values_branch_sample0'] = sensor_target_values_cpu

        elif task_type == "output_derivative":
            current_x_dense_cpu = x_dense_individual_plots_cpu # Use predefined dense x
            
            current_coeffs_poly = coeffs_sample0 if i == 0 else (
                torch.randn(1).item() * scale,
                torch.randn(1).item() * scale,
                torch.randn(1).item() * scale,
                torch.randn(1).item() * scale # d for the cubic, not used for derivative directly
            )
            a, b, c_poly, _ = current_coeffs_poly # c_poly to avoid clash
            
            true_target_values_cpu = (3*a*current_x_dense_cpu.squeeze()**2 + 2*b*current_x_dense_cpu.squeeze() + c_poly)
            sensor_target_values_cpu = (3*a*sensor_x_orig.cpu()**2 + 2*b*sensor_x_orig.cpu() + c_poly).squeeze()
            # Note: data_for_combined_plot for derivative task already populated if i==0

        # Ensure current_x_dense_cpu is shaped [N,1] for normalization functions
        if current_x_dense_cpu.ndim == 1:
            current_x_dense_shaped_cpu = current_x_dense_cpu.unsqueeze(-1)
        else:
            current_x_dense_shaped_cpu = current_x_dense_cpu

        # Normalization stats from the current sample's true target values
        target_mean_sample = true_target_values_cpu.mean().item()
        target_std_sample = true_target_values_cpu.std().item() + 1e-8

        # Prepare inputs for the model
        sensor_x_norm_model = normalize_coordinates(sensor_x_orig.cpu(), input_range).unsqueeze(0).unsqueeze(-1).to(device)
        sensor_values_norm_model = normalize_values(
            sensor_target_values_cpu.unsqueeze(0).to(device),
            torch.tensor([[target_mean_sample]], device=device),
            torch.tensor([[target_std_sample]], device=device)
        ).unsqueeze(-1)
        x_dense_norm_model = normalize_coordinates(current_x_dense_shaped_cpu, input_range).unsqueeze(0).to(device)

        predicted_values_denorm_sample = None
        with torch.no_grad():
            pred_norm_sample = model(sensor_x_norm_model, sensor_values_norm_model, x_dense_norm_model)
            predicted_values_denorm_sample = denormalize_values(
                pred_norm_sample.squeeze(-1),
                torch.tensor([[target_mean_sample]], device=device),
                torch.tensor([[target_std_sample]], device=device)
            ).cpu().numpy().flatten()
            
        _plot_single_sample_approximation(
            x_dense_orig=current_x_dense_cpu.squeeze(), # Pass 1D tensor for plotting
            true_values=true_target_values_cpu,
            predicted_values=predicted_values_denorm_sample,
            log_dir=log_dir,
            plot_idx=i,
            task_type_for_filename=task_type
        )

    # 3. Plot standalone trunk basis functions
    basis_plot_filename = f"{task_type}_trunk_basis_functions.png"
    basis_plot_title = f"SetONet Trunk Basis Functions (p={latent_p}, Normalized Domain, Task: {task_type.replace('_', ' ').title()})"
    
    with torch.no_grad():
        x_basis_norm_standalone = torch.linspace(-1.0, 1.0, n_trunk_points_plot, device=device).view(1, -1, 1)
        basis_standalone = model.trunk(x_basis_norm_standalone).squeeze(0)

    if basis_standalone.shape[1] == latent_p:
        plt.figure(figsize=(10, 6))
        for i in range(latent_p):
            plt.plot(x_basis_norm_standalone.squeeze().cpu().numpy(), basis_standalone[:, i].cpu().numpy())
        plt.title(basis_plot_title)
        plt.xlabel("y (normalized to [-1, 1])")
        plt.ylabel("ψ_i(y_norm)") # Standard label for basis functions
        plt.grid(True)
        basis_plot_path = os.path.join(log_dir, basis_plot_filename)
        plt.savefig(basis_plot_path)
        plt.close()
        print(f"Standalone trunk basis plot saved to {basis_plot_path}")
    else:
        print(f"Warning: Standalone trunk output dim {basis_standalone.shape[1]} != latent_p {latent_p}. Skipping plot.")

    # 4. Generate the combined overview plot using the stored Sample 0 data
    if data_for_combined_plot:
        fig_combined, _ = plot_combined_overview_1d(
            model=model, device=device, task_type=task_type, latent_p=latent_p,
            sensor_x_orig=sensor_x_orig.cpu(), input_range=input_range,
            x_dense_orig_plot_sample0=data_for_combined_plot['x_dense_orig_plot_sample0'],
            true_values_plot_sample0=data_for_combined_plot['true_values_plot_sample0'],
            true_sens_values_branch_sample0=data_for_combined_plot['true_sens_values_branch_sample0'],
            n_trunk_points_plot_basis=n_trunk_points_plot, # Use consistent number of points
            font_size=font_size_combined
        )
        fig_combined.tight_layout(pad=2.0)
        fig_combined.subplots_adjust(
            left=margin_left, right=margin_right, bottom=margin_bottom, top=margin_top
        )
        combined_plot_path = os.path.join(log_dir, f"combined_{task_type}_overview_plot.png")
        fig_combined.savefig(combined_plot_path)
        plt.close(fig_combined)
        print(f"Combined overview plot saved to {combined_plot_path}")
    else:
        print(f"Warning: Data for sample 0 not available for combined {task_type} plot. Skipping.")

    print(f"All plots for task {task_type} generated and saved to {log_dir}") 