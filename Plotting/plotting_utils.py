import torch
import matplotlib.pyplot as plt
import os

def plot_derivative_comparison(deeponet_model, setonet_model, sensor_x, x_dense, input_range, scale, log_dir):
    """
    Plots the results for DeepONet or SetONet in mapping a single cubic polynomial
    to its derivative, showing input and output side-by-side.
    Saves the plot to the specified log directory.
    Handles normalization internally for SetONet if input_range is provided.
    """
    model_name = ""

    if deeponet_model:
        model_name = "DeepONet"
    elif setonet_model:
        model_name = "SetONet"
    else:
        print("No model provided to plot_derivative_comparison. Skipping plot.")
        return

    fig, ax_deriv = plt.subplots(figsize=(8, 5))                  # single subplot

    # Generate one random cubic coefficient example
    a = torch.randn(1).item() * scale
    b = torch.randn(1).item() * scale
    c = torch.randn(1).item() * scale
    d = torch.randn(1).item() * scale

    # Compute input f(x)  and its derivative (on original scale)
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

        # Original sensor locations and derivative values for this plot example
        sensor_x_plot_orig_cpu = sensor_x.cpu() # Ensure it's CPU for consistency before moving
        df_sensor_plot_orig_cpu = (3 * a * sensor_x_plot_orig_cpu**2 + 2 * b * sensor_x_plot_orig_cpu + c)
        
        # Original dense x locations for trunk input
        x_dense_plot_orig_cpu = x_dense.cpu() # Ensure it's CPU

        # --- Normalize for SetONet input ---
        # 1. Normalize coordinates to [-1, 1]
        min_val, max_val = input_range
        sensor_x_plot_norm = 2 * (sensor_x_plot_orig_cpu.to(model_device) - min_val) / (max_val - min_val) - 1
        x_dense_plot_norm  = 2 * (x_dense_plot_orig_cpu.to(model_device) - min_val) / (max_val - min_val) - 1

        # 2. Standardize function values using statistics from df_true (the target for this plot)
        # df_true is already computed and is on CPU. Move to model_device for calculations if needed,
        # or keep stats on CPU if df_sensor_plot_orig_cpu is also on CPU before normalization.
        # For consistency, let's calculate stats from df_true (which is on CPU)
        # and apply to df_sensor_plot_orig_cpu (also on CPU) before moving to device.
        
        target_mean_plot = df_true.mean(dim=0, keepdim=True) # df_true is 1D
        target_std_plot  = df_true.std(dim=0, keepdim=True) + 1e-8
        
        # Normalize the sensor derivative values using target's statistics
        df_sensor_plot_norm_vals = (df_sensor_plot_orig_cpu - target_mean_plot) / target_std_plot
        
        xs_son_plot = sensor_x_plot_norm.view(1, -1, 1) # Already on model_device
        us_son_plot = df_sensor_plot_norm_vals.to(model_device).view(1, -1, 1)
        ys_son_plot = x_dense_plot_norm.view(1, -1, 1) # Already on model_device
        
        us_derivs_son_plot_arg = None # Assuming concat_sensor_derivative_to_branch_input is false

        with torch.no_grad():
            # Model expects inputs on its device
            pred_norm = setonet_model(xs_son_plot, us_son_plot, ys_son_plot, us_derivs=us_derivs_son_plot_arg).squeeze(0).squeeze(-1) 
        
        # De-normalize the prediction using target's statistics
        # pred_norm is on model_device, target_mean_plot/target_std_plot are on CPU. Move for calculation.
        df_pred = pred_norm * target_std_plot.to(model_device) + target_mean_plot.to(model_device)
        df_pred = df_pred.detach().cpu()
        # --- End Normalization Handling ---
        
        pred_label = "Predicted df/dx (SetONet)"
        pred_color = 'orange'
    
    # ------------------ Derivative plot --------------------------------
    ax_deriv.plot(x_dense_cpu, df_true, label="True df/dx", color='green', linestyle='-')
    if df_pred is not None:
        ax_deriv.plot(x_dense_cpu, df_pred, label=pred_label, color=pred_color, linestyle='--')
    ax_deriv.set_xlabel("x")
    ax_deriv.set_ylabel("df/dx")
    ax_deriv.legend()
    ax_deriv.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
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