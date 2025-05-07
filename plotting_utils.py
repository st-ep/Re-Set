import torch
import matplotlib.pyplot as plt
import os

def plot_derivative_comparison(deeponet_model, setonet_model, sensor_x, x_dense, input_range, scale, log_dir):
    """
    Plots the comparison of DeepONet and SetONet in mapping cubic polynomials to their derivatives.
    Saves the plot to the specified log directory.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("DeepONet vs SetONet: Mapping Cubic Polynomial to Its Derivative", fontsize=16)

    for i in range(3):
        # Random cubic coefficients
        a = torch.randn(1).item() * scale
        b = torch.randn(1).item() * scale
        c = torch.randn(1).item() * scale
        d = torch.randn(1).item() * scale

        # Compute input (function values at sensor points)
        f_sensor = a * sensor_x**3 + b * sensor_x**2 + c * sensor_x + d  # [S]
        f_dense = a * x_dense.squeeze()**3 + b * x_dense.squeeze()**2 + c * x_dense.squeeze() + d  # [200]
        df_true = 3*a*x_dense.squeeze()**2 + 2*b*x_dense.squeeze() + c  # [200]

        # Prepare DeepONet input for plotting
        branch_input_don_plot = f_sensor.unsqueeze(0)  # [1, S]

        # Prepare SetONet inputs for plotting
        xs_son_plot = sensor_x.view(1, sensor_x.shape[0], 1)
        us_son_plot = f_sensor.view(1, sensor_x.shape[0], 1)
        ys_son_plot = x_dense.view(1, x_dense.shape[0], 1)

        with torch.no_grad():
            df_pred_don = deeponet_model(branch_input_don_plot, x_dense).squeeze().cpu()
            df_pred_son = setonet_model(xs_son_plot, us_son_plot, ys_son_plot).squeeze().detach().cpu()

        # Plot function input (cubic polynomial)
        axs[i, 0].plot(x_dense.squeeze().cpu(), f_dense.cpu(), label="Input f(x)")
        axs[i, 0].scatter(sensor_x.cpu(), f_sensor.cpu(), color="red", label="Sensor values")
        axs[i, 0].set_title(f"[{i+1}] Input Cubic Function")
        axs[i, 0].set_xlabel("x")
        axs[i, 0].set_ylabel("f(x)")
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # Plot true vs predicted derivative
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_true.cpu(), 'k--', label="True f'(x)")
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_pred_don, 'b-', label="DeepONet f'(x)")
        axs[i, 1].plot(x_dense.squeeze().cpu(), df_pred_son, 'g-.', label="SetONet f'(x)")
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

def plot_trunk_basis_functions(deeponet_model, setonet_model, x_basis, setonet_p_dim, log_dir):
    """
    Visualizes and saves the learned trunk net basis functions for DeepONet and SetONet.
    """
    plt.figure(figsize=(12, 6))

    # DeepONet Trunk Basis
    plt.subplot(1, 2, 1)
    with torch.no_grad():
        trunk_out_don = deeponet_model.trunk_net(x_basis)
        trunk_out_don_np = trunk_out_don.cpu().numpy()
    for i in range(trunk_out_don_np.shape[1]):
        plt.plot(x_basis.squeeze().cpu(), trunk_out_don_np[:, i], label=f"Basis {i+1}", alpha=0.6)
    plt.title("DeepONet Trunk Net Basis Functions")
    plt.xlabel("x")
    plt.ylabel("Basis Function Value")
    plt.grid(True)

    # SetONet Trunk Basis
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        ys_basis_son = x_basis.unsqueeze(0)
        trunk_out_son_raw = setonet_model.forward_trunk(ys_basis_son)
        trunk_out_son = trunk_out_son_raw.squeeze(0).squeeze(-1).cpu().numpy()
    for i in range(trunk_out_son.shape[1]): # Should be setonet_p_dim
        plt.plot(x_basis.squeeze().cpu(), trunk_out_son[:, i], label=f"Basis {i+1}", alpha=0.6)
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