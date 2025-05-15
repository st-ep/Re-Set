import torch
import matplotlib.pyplot as plt
import os

def plot_derivative_comparison(deeponet_model, setonet_model, sensor_x, x_dense, input_range, scale, log_dir):
    """
    Plots the results for DeepONet or SetONet in mapping a single cubic polynomial
    to its derivative, showing input and output side-by-side.
    Saves the plot to the specified log directory.
    """
    model_name = ""

    if deeponet_model:
        model_name = "DeepONet"
    elif setonet_model:
        model_name = "SetONet"
    else:
        print("No model provided to plot_derivative_comparison. Skipping plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), squeeze=False) # 1 row, 2 columns

    # Generate one random cubic coefficient example
    a = torch.randn(1).item() * scale
    b = torch.randn(1).item() * scale
    c = torch.randn(1).item() * scale
    d = torch.randn(1).item() * scale

    # Compute input (function values at sensor points and dense points)
    # Ensure sensor_x and x_dense are on the CPU for numpy operations if they come from GPU
    sensor_x_cpu = sensor_x.cpu()
    x_dense_cpu = x_dense.cpu().squeeze() # Squeeze for 1D array if it's [N,1]

    f_sensor = a * sensor_x_cpu**3 + b * sensor_x_cpu**2 + c * sensor_x_cpu + d
    f_dense = a * x_dense_cpu**3 + b * x_dense_cpu**2 + c * x_dense_cpu + d
    df_true = 3*a*x_dense_cpu**2 + 2*b*x_dense_cpu + c

    # Get model prediction
    df_pred = None # Initialize df_pred
    pred_label = "Predicted df/dx"
    pred_color = 'gray' # Default color

    if deeponet_model:
        # Ensure inputs to model are on the correct device
        model_device = next(deeponet_model.trunk_net.parameters()).device # Get device from model parameters
        branch_input_don_plot = f_sensor.to(model_device).unsqueeze(0)
        x_dense_model_input = x_dense.to(model_device)
        with torch.no_grad():
            df_pred = deeponet_model(branch_input_don_plot, x_dense_model_input).squeeze().cpu()
        pred_label = "Predicted df/dx (DeepONet)"
        pred_color = 'blue'
    elif setonet_model:
        # Ensure inputs to model are on the correct device
        model_device = next(setonet_model.parameters()).device
        xs_son_plot = sensor_x.to(model_device).view(1, sensor_x.shape[0], 1)
        us_son_plot = f_sensor.to(model_device).view(1, sensor_x.shape[0], 1)
        ys_son_plot = x_dense.to(model_device).view(1, x_dense.shape[0], 1) # x_dense is already [N,1]
        
        us_derivs_son_plot_arg = None
        # Check the flag on the model instance and if the attribute exists
        if hasattr(setonet_model, 'concat_sensor_derivative_to_branch_input') and \
           setonet_model.concat_sensor_derivative_to_branch_input:
            # Calculate derivative of the test function at sensor_x locations
            # Coefficients a, b, c were scalars. Convert them to tensors on the model_device.
            # sensor_x should also be on model_device for this calculation.
            a_coeff_dev = torch.tensor(a, device=model_device).view(1, 1)
            b_coeff_dev = torch.tensor(b, device=model_device).view(1, 1)
            c_coeff_dev = torch.tensor(c, device=model_device).view(1, 1)
            
            # Use sensor_x directly (it's already on device or will be moved by xs_son_plot)
            # For clarity, ensure sensor_x_for_deriv is on model_device
            sensor_x_for_deriv = sensor_x.to(model_device)

            # df/dx = 3ax^2 + 2bx + c
            # sensor_x_for_deriv has shape [num_sensors]
            # a_coeff_dev, b_coeff_dev, c_coeff_dev have shape [1, 1] for broadcasting
            df_u_test_at_sensors = 3 * a_coeff_dev * sensor_x_for_deriv**2 + \
                                   2 * b_coeff_dev * sensor_x_for_deriv + \
                                   c_coeff_dev 
            # df_u_test_at_sensors will have shape [1, num_sensors] after broadcasting
            us_derivs_son_plot_arg = df_u_test_at_sensors.view(1, -1, 1) # Reshape to [1, num_sensors, 1]

        with torch.no_grad():
            # Pass us_derivs_son_plot_arg to the model call
            df_pred = setonet_model(xs_son_plot, us_son_plot, ys_son_plot, us_derivs=us_derivs_son_plot_arg).squeeze().detach().cpu()
        pred_label = "Predicted df/dx (SetONet)"
        pred_color = 'orange'
    
    # Subplot 1: Input Cubic Function
    ax_input = axs[0, 0]
    ax_input.plot(x_dense_cpu, f_dense, label="Input f(x)", linestyle='-')
    # Plot only every 100th sensor point for visualization
    sensor_x_viz = sensor_x_cpu[::100]
    f_sensor_viz = f_sensor[::100]
    ax_input.scatter(sensor_x_viz, f_sensor_viz, color="red", label="Sensor values (subset)", s=25, alpha=0.9, zorder=5)
    ax_input.set_xlabel("x")
    ax_input.set_ylabel("f(x)")
    ax_input.legend()
    ax_input.grid(True, linestyle='--', alpha=0.7)

    # Subplot 2: Derivative Prediction
    ax_deriv = axs[0, 1]
    ax_deriv.plot(x_dense_cpu, df_true, label="True df/dx", color='green', linestyle='-')
    if df_pred is not None:
        ax_deriv.plot(x_dense_cpu, df_pred, label=pred_label, color=pred_color, linestyle='--')
    ax_deriv.set_xlabel("x")
    ax_deriv.set_ylabel("df/dx")
    ax_deriv.legend()
    ax_deriv.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout to make space for suptitle
    plot_filename = os.path.join(log_dir, f"{model_name.lower()}_derivative_example_plot.png")
    plt.savefig(plot_filename)
    print(f"{model_name} derivative example plot saved to {plot_filename}")
    plt.close(fig) # Close the figure to free memory

def plot_trunk_basis_functions(deeponet_model, setonet_model, x_basis, setonet_p_dim, log_dir):
    """
    Plots the trunk basis functions for DeepONet or SetONet.
    All basis functions for a model are plotted on a single subplot.
    Saves the plots to the specified log directory.
    """
    if deeponet_model is None and setonet_model is None:
        print("No models provided to plot_trunk_basis_functions. Skipping plots.")
        return

    x_basis_cpu = x_basis.cpu().numpy().squeeze()

    if deeponet_model:
        try:
            with torch.no_grad():
                basis_deeponet = deeponet_model.trunk_net(x_basis).cpu().numpy()
            
            p_deeponet = basis_deeponet.shape[1]
            
            fig_don, ax_don = plt.subplots(figsize=(10, 6)) # Single subplot
            # fig_don.suptitle("DeepONet Trunk Basis Functions", fontsize=16) # Removed title
            
            for i in range(p_deeponet):
                ax_don.plot(x_basis_cpu, basis_deeponet[:, i], label=f"Basis {i+1}") # Label kept for potential future use, but legend removed
            
            ax_don.set_xlabel("y")
            ax_don.set_ylabel("Trunk Output Value")
            # ax_don.legend(loc='best') # Removed legend
            ax_don.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Keep tight_layout, adjust if suptitle was the only reason for rect
            plot_filename_don = os.path.join(log_dir, "deeponet_trunk_basis_plot.png")
            plt.savefig(plot_filename_don)
            print(f"DeepONet trunk basis plot saved to {plot_filename_don}")
            plt.close(fig_don)
        except Exception as e:
            print(f"Could not plot DeepONet trunk basis functions: {e}")


    if setonet_model and setonet_p_dim is not None and setonet_p_dim > 0:
        try:
            with torch.no_grad():
                basis_setonet = setonet_model.trunk(x_basis).cpu().numpy()

            if basis_setonet.shape[1] == setonet_p_dim * setonet_model.output_size_tgt:
                if setonet_model.output_size_tgt > 1:
                    print(f"Warning: SetONet output_size_tgt is {setonet_model.output_size_tgt}. Plotting assumes it's effectively 1 for basis visualization.")
                
                fig_son, ax_son = plt.subplots(figsize=(10, 6)) # Single subplot
                # fig_son.suptitle(f"SetONet Trunk Basis Functions (p={setonet_p_dim})", fontsize=16) # Removed title

                for i in range(setonet_p_dim):
                    # Assuming output_size_tgt is 1, so we take the i-th column directly
                    ax_son.plot(x_basis_cpu, basis_setonet[:, i], label=f"Basis {i+1}") # Label kept, legend removed
                
                ax_son.set_xlabel("y")
                ax_son.set_ylabel("Trunk Output Value")
                # ax_son.legend(loc='best') # Removed legend
                ax_son.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Keep tight_layout
                plot_filename_son = os.path.join(log_dir, "setonet_trunk_basis_plot.png")
                plt.savefig(plot_filename_son)
                print(f"SetONet trunk basis plot saved to {plot_filename_son}")
                plt.close(fig_son)
            else:
                print(f"Could not plot SetONet trunk basis functions: Trunk output shape {basis_setonet.shape} not compatible with p={setonet_p_dim} and assumed output_size_tgt=1.")
        except Exception as e:
            print(f"Could not plot SetONet trunk basis functions: {e}")
    elif setonet_model and (setonet_p_dim is None or setonet_p_dim <= 0):
        print("SetONet model provided but setonet_p_dim is invalid. Skipping SetONet trunk basis plot.") 