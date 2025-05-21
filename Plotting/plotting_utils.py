import torch
import matplotlib.pyplot as plt
import os
# Assuming normalization functions are accessible, e.g. if Plotting is a sibling to Models
# Adjust path if necessary, or ensure helper_utils is in PYTHONPATH
from Models.utils.helper_utils import normalize_coordinates, normalize_values, denormalize_values

def plot_derivative_comparison(deeponet_model, setonet_model, sensor_x, x_dense, input_range, scale, log_dir, num_samples_to_plot=1):
    """
    Plots the results for DeepONet or SetONet in mapping cubic polynomials
    to their derivatives, showing input and output side-by-side.
    Saves plots for a specified number of samples to the log directory.
    """
    model_to_use = None
    model_name_str = ""

    if deeponet_model:
        model_to_use = deeponet_model
        model_name_str = "DeepONet"
    elif setonet_model:
        model_to_use = setonet_model
        model_name_str = "SetONet"
    else:
        print("No model provided to plot_derivative_comparison. Skipping plot.")
        return

    # Ensure sensor_x and x_dense are on the CPU for numpy operations and plotting
    sensor_x_cpu = sensor_x.cpu() # Original scale sensor_x
    x_dense_cpu = x_dense.cpu().squeeze() # Original scale x_dense
    
    # Ensure model is on the same device as inputs for prediction, or move inputs to model's device
    # Assuming model_to_use is already on a device, and inputs should be moved to it.
    # For this function, let's assume sensor_x and x_dense are passed on the correct device for the model.
    # The script calling this should handle device placement.
    # However, for plotting, data needs to be on CPU.

    for i in range(num_samples_to_plot):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), squeeze=False) # 1 row, 2 columns

        # Generate one random cubic coefficient example for this sample
        a = torch.randn(1).item() * scale
        b = torch.randn(1).item() * scale
        c = torch.randn(1).item() * scale
        d = torch.randn(1).item() * scale

        # Compute input (function values at sensor points and dense points)
        f_sensor_vals = a * sensor_x_cpu**3 + b * sensor_x_cpu**2 + c * sensor_x_cpu + d
        f_dense_vals = a * x_dense_cpu**3 + b * x_dense_cpu**2 + c * x_dense_cpu + d
        df_true_vals = 3*a*x_dense_cpu**2 + 2*b*x_dense_cpu + c

        # Prepare inputs for the model (ensure they are on the model's device)
        # f_sensor_vals needs to be [1, num_sensors] for DeepONet-like branch or [1, num_sensors, 1] for SetONet u_s
        # x_dense needs to be [num_dense_points, 1] for DeepONet-like trunk or [1, num_dense_points, 1] for SetONet y_s
        
        df_pred_vals_denorm_for_plot = None # Stores final denormalized prediction for plotting
        pred_label = "Predicted df/dx"
        pred_color = 'gray'

        with torch.no_grad():
            model_to_use.eval()
            if model_name_str == "SetONet":
                # Device for model operations
                model_device = next(model_to_use.parameters()).device

                # Normalize coordinates (sensor_x and x_dense are original scale from args)
                current_sensor_x_norm_model_dev = normalize_coordinates(sensor_x.clone().to(model_device), input_range).view(1, -1, 1)
                current_x_dense_norm_model_dev = normalize_coordinates(x_dense.clone().to(model_device), input_range).view(1, -1, 1)

                # Normalize sensor values (f_sensor_vals for input u) for this sample
                f_sensor_vals_torch = torch.tensor(f_sensor_vals, device=model_device) 
                mean_f_sample = f_sensor_vals_torch.mean() 
                std_f_sample = f_sensor_vals_torch.std()   
                current_f_sensor_norm_model_dev = normalize_values(f_sensor_vals_torch, mean_f_sample, std_f_sample).view(1, -1, 1)
                
                # Model predicts in normalized target scale
                pred_output_norm_target_scale = model_to_use(current_sensor_x_norm_model_dev, current_f_sensor_norm_model_dev, current_x_dense_norm_model_dev)
                
                # Denormalize prediction using statistics of the true derivative (df_true_vals) for this sample
                df_true_vals_torch = torch.tensor(df_true_vals, device=model_device)
                mean_df_true_sample = df_true_vals_torch.mean()
                std_df_true_sample = df_true_vals_torch.std()
                pred_output_denorm = denormalize_values(pred_output_norm_target_scale, mean_df_true_sample, std_df_true_sample)
                
                df_pred_vals_denorm_for_plot = pred_output_denorm.squeeze().cpu().numpy()
                pred_label = f"{model_name_str} df/dx"
                pred_color = 'red'
            # Add elif for DeepONet if it's ever re-introduced and has different input prep
            # elif model_name_str == "DeepONet":
            #     # Example for DeepONet (adjust as per your DeepONet's expected input shapes)
            #     branch_input_don = torch.tensor(f_sensor_vals, device=sensor_x.device).unsqueeze(0) # [1, num_sensors]
            #     trunk_input_don = x_dense # [num_dense_points, 1] (already on device)
            #     pred_output = model_to_use(branch_input_don, trunk_input_don)
            #     df_pred_vals = pred_output.squeeze().cpu().numpy()
            #     pred_label = f"{model_name_str} df/dx"
            #     pred_color = 'blue'


        # Plot 1: Input function u(x)
        axs[0, 0].plot(x_dense_cpu.numpy(), f_dense_vals.numpy(), label='True u(x) (dense)', color='blue', linestyle='-')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('u(x)')
        axs[0, 0].set_title(f'Sample {i+1}: Input Function')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2: Output derivative G(u)(y) - use denormalized predictions
        axs[0, 1].plot(x_dense_cpu.numpy(), df_true_vals.numpy(), label='True df/dx', color='green', linestyle='-')
        if df_pred_vals_denorm_for_plot is not None:
            axs[0, 1].plot(x_dense_cpu.numpy(), df_pred_vals_denorm_for_plot, label=pred_label, color=pred_color, linestyle='--')
        axs[0, 1].set_xlabel('y (evaluation points)')
        axs[0, 1].set_ylabel('G(u)(y) or df/dx')
        axs[0, 1].set_title(f'Sample {i+1}: Output Derivative ({model_name_str})')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        fig.suptitle(f'{model_name_str}: Derivative Prediction - Sample {i+1}', fontsize=16)
        
        plot_filename = f"derivative_comparison_{model_name_str.lower()}_sample_{i+1}.png"
        save_path = os.path.join(log_dir, plot_filename)
        plt.savefig(save_path)
        print(f"Saved derivative comparison plot for sample {i+1} to {save_path}")
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