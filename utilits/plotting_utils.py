import torch
import matplotlib.pyplot as plt
import os

def plot_derivative_comparison(deeponet_model, setonet_model, sensor_x, x_dense, input_range, scale, log_dir, device):
    """
    Plots the comparison of DeepONet and SetONet in mapping cubic polynomials to their derivatives.
    Saves the plot to the specified log directory.
    Models and input tensors (sensor_x, x_dense) are expected to be on the 'device'.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("DeepONet vs SetONet: Mapping Cubic Polynomial to Its Derivative", fontsize=16)

    for i in range(3):
        # Random cubic coefficients
        # Note: if input_range is not [-1, 1] or scale is not 1, this random generation might need adjustment
        # to match the original intent if 'scale' was meant to interact with a fixed rand range.
        # For now, assuming torch.randn gives values generally around 0, scaled by 'scale'.
        a = torch.randn(1).item() * scale
        b = torch.randn(1).item() * scale
        c = torch.randn(1).item() * scale
        d = torch.randn(1).item() * scale

        # Compute input (function values at sensor points)
        f_sensor = a * sensor_x**3 + b * sensor_x**2 + c * sensor_x + d
        f_dense_cpu = a * x_dense.squeeze().cpu()**3 + b * x_dense.squeeze().cpu()**2 + c * x_dense.squeeze().cpu() + d # For plotting
        df_true_cpu = 3*a*x_dense.squeeze().cpu()**2 + 2*b*x_dense.squeeze().cpu() + c # For plotting

        # Prepare DeepONet input for plotting
        branch_input_don_plot = f_sensor.unsqueeze(0)  # Already on device as f_sensor is

        # Prepare SetONet inputs for plotting
        xs_son_plot = sensor_x.view(1, sensor_x.shape[0], 1)
        us_son_plot = f_sensor.view(1, sensor_x.shape[0], 1) # f_sensor is on device
        ys_son_plot = x_dense.view(1, x_dense.shape[0], 1)

        with torch.no_grad():
            # Models are on device, inputs (branch_input_don_plot, x_dense, etc.) are on device
            df_pred_don = deeponet_model(branch_input_don_plot, x_dense).squeeze()
            df_pred_son = setonet_model(xs_son_plot, us_son_plot, ys_son_plot).squeeze().detach()

        # Plot function input (cubic polynomial)
        axs[i, 0].plot(x_dense.squeeze().cpu(), f_dense_cpu, label="Input f(x)") # Plot CPU tensors
        axs[i, 0].scatter(sensor_x.cpu(), f_sensor.cpu(), color="red", label="Sensor values") # Plot CPU tensors
        axs[i, 0].set_title(f"[{i+1}] Input Cubic Function")
        axs[i, 0].set_xlabel("x")
        axs[i, 0].set_ylabel("f(x)")
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # Plot true vs predicted derivative
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_true_cpu, 'k--', label="True f'(x)")
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_pred_don.cpu(), 'b-', label="DeepONet f'(x)")
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_pred_son.cpu(), 'g-.', label="SetONet f'(x)")
        axs[i, 1].set_title(f"[{i+1}] Derivative Prediction")
        axs[i, 1].set_xlabel("x")
        axs[i, 1].set_ylabel("f'(x)")
        axs[i, 1].legend()
        axs[i, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    derivative_plot_path = os.path.join(log_dir, "derivative_comparison_plot.png")
    plt.savefig(derivative_plot_path)
    print(f"Derivative comparison plot saved to {derivative_plot_path}")
    plt.show()

def plot_trunk_basis_functions(deeponet_model, setonet_model, x_basis, setonet_p_dim, log_dir, device):
    """
    Visualizes and saves the learned trunk net basis functions for DeepONet and SetONet.
    Models and x_basis are expected to be on 'device'.
    """
    plt.figure(figsize=(12, 6))

    # DeepONet Trunk Basis
    plt.subplot(1, 2, 1)
    with torch.no_grad():
        # model and x_basis are on device
        trunk_out_don = deeponet_model.trunk_net(x_basis) 
        trunk_out_don_np = trunk_out_don.cpu().numpy() # Move to CPU for numpy/plotting
    for i in range(trunk_out_don_np.shape[1]):
        plt.plot(x_basis.squeeze().cpu(), trunk_out_don_np[:, i], label=f"Basis {i+1}", alpha=0.6) # x_basis to CPU for plotting
    plt.title("DeepONet Trunk Net Basis Functions")
    plt.xlabel("x")
    plt.ylabel("Basis Function Value")
    plt.grid(True)

    # SetONet Trunk Basis
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        # x_basis is on device
        ys_basis_son = x_basis.unsqueeze(0) 
        # model is on device
        trunk_out_son_raw = setonet_model.forward_trunk(ys_basis_son)
        trunk_out_son = trunk_out_son_raw.squeeze(0).squeeze(-1).cpu().numpy() # Move to CPU for numpy/plotting

    # The number of basis functions for SetONet is setonet_p_dim
    for i in range(trunk_out_son.shape[1]): # trunk_out_son.shape[1] should be setonet_p_dim
        plt.plot(x_basis.squeeze().cpu(), trunk_out_son[:, i], label=f"Basis {i+1}", alpha=0.6) # x_basis to CPU for plotting
    plt.title("SetONet Trunk Net Basis Functions")
    plt.xlabel("x")
    plt.ylabel("Basis Function Value")
    plt.grid(True)

    plt.suptitle("Comparison of Learned Trunk Net Basis Functions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    trunk_basis_plot_path = os.path.join(log_dir, "trunk_basis_functions_plot.png")
    plt.savefig(trunk_basis_plot_path)
    print(f"Trunk basis functions plot saved to {trunk_basis_plot_path}")
    plt.show() 