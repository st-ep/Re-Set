"""
Example: learning the derivative (output space) of cubic polynomials with
FunctionEncoder + SetONet representation encoder.

Run with:
    python Examples/learn_output_space_with_function_encoder.py --epochs 1000 --device cuda
"""

import argparse
import torch
from Data.data_utils import generate_batch
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from FunctionEncoder.Dataset.BaseDataset import BaseDataset   # just for type / inheritance
import matplotlib.pyplot as plt # Added for plotting
import os # Added for path joining
from datetime import datetime # Added for timestamped log directories
# --- Added imports for callbacks ---
from FunctionEncoder.Callbacks.OrthonormalityCallback import OrthonormalityCallback
from FunctionEncoder.Callbacks.ListCallback import ListCallback
from FunctionEncoder.Callbacks.TensorboardCallback import TensorboardCallback

class CubicDerivativeDataset(BaseDataset):
    """
    Samples random cubic polynomials f(x)=ax³+bx²+cx+d.

    example_xs  : sensor locations  (1000 evenly-spaced points)
    example_ys  : f(x)              at the sensor locations
    query_xs    : trunk locations   (n_trunk_points evenly-spaced points)
    query_ys    : f'(x)             evaluated at trunk locations
    """
    def __init__(self,
                 sensor_x: torch.Tensor,
                 input_range,
                 scale: float = 1.0,
                 n_trunk_points: int = 200,
                 batch_size: int = 32):
        # Tell BaseDataset about the data format
        #
        # • input_size  = (1,)  -> each sensor / trunk point is 1-D (x coordinate)
        # • output_size = (1,)  -> target value (derivative) is scalar
        # • data_type   = "deterministic"
        #
        # Also pass n_functions, n_examples, and n_queries, which BaseDataset expects.
        num_sensors_val = sensor_x.numel()
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         data_type="deterministic",
                         n_functions=batch_size,
                         n_examples=num_sensors_val,
                         n_queries=n_trunk_points)
        self.sensor_x = sensor_x                     # (num_sensors,)
        self.input_range = input_range
        self.scale = scale
        self.n_trunk_points = n_trunk_points
        self.batch_size = batch_size
        self.num_sensors = num_sensors_val # Store it based on the pre-calculated value
        self.device = sensor_x.device

    # Required by FunctionEncoder.train_model
    def sample(self, *_, **__):
        """
        Returns:
            example_xs   : (B, num_sensors, 1)
            example_ys   : (B, num_sensors, 1)
            query_xs     : (B, n_trunk_points, 1)
            query_ys     : (B, n_trunk_points, 1)
            _dummy       : any extra object (we return None)
        """
        f_vals, x_eval, y_tgt, _ = generate_batch(
            batch_size=self.batch_size,
            n_trunk_points=self.n_trunk_points,
            sensor_x=self.sensor_x,       # already on correct device
            scale=self.scale,
            input_range=self.input_range,
            device=self.device
        )

        # Shapes to match FunctionEncoder expectations
        example_xs = self.sensor_x.expand(self.batch_size, -1).unsqueeze(-1)         # (B, S, 1)
        example_ys = f_vals.unsqueeze(-1)                                            # (B, S, 1)
        query_xs   = x_eval.unsqueeze(0).expand(self.batch_size, -1, -1)             # (B, T, 1)
        query_ys   = y_tgt.unsqueeze(-1)                                             # (B, T, 1)

        return example_xs, example_ys, query_xs, query_ys, None

    # Minimal no-op checker
    def check_dataset(self): 
        pass

@torch.no_grad()
def evaluate(fe: FunctionEncoder, dataset: CubicDerivativeDataset):
    """
    Draw one fresh batch and compute MSE on the derivative prediction.
    """
    ex_xs, ex_ys, q_xs, q_ys, _ = dataset.sample()
    reps, _ = fe.compute_representation(ex_xs, ex_ys)
    y_hat = fe.predict(q_xs, reps)
    mse = torch.mean((y_hat - q_ys) ** 2).item()
    return mse


# ----------------------------------------------------------------------------
# Plotting Utility for FunctionEncoder Prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def plot_fe_prediction_example(fe_model: FunctionEncoder,
                               sensor_x_coords: torch.Tensor,
                               x_dense_eval_coords: torch.Tensor,
                               scale: float,
                               plot_save_path: str = "fe_derivative_prediction.png"):
    """
    Plots a single example of function to derivative mapping using the FunctionEncoder.
    - Generates a random cubic polynomial.
    - Shows the input function and sensor points.
    - Shows the true derivative and the model's predicted derivative.
    """
    fe_model.eval()
    device = sensor_x_coords.device

    # 1. Generate one random cubic coefficient example
    a = (torch.rand(1, device=device) * 2 - 1) * scale
    b = (torch.rand(1, device=device) * 2 - 1) * scale
    c = (torch.rand(1, device=device) * 2 - 1) * scale
    d = (torch.rand(1, device=device) * 2 - 1) * scale

    # 2. Compute true function values
    # f(x) = ax^3 + bx^2 + cx + d
    f_values_at_sensors = a * sensor_x_coords**3 + b * sensor_x_coords**2 + c * sensor_x_coords + d
    true_f_values_dense = a * x_dense_eval_coords**3 + b * x_dense_eval_coords**2 + c * x_dense_eval_coords + d

    # 3. Compute true derivative values
    # f'(x) = 3ax^2 + 2bx + c
    true_df_dx_values_dense = 3 * a * x_dense_eval_coords**2 + 2 * b * x_dense_eval_coords + c

    # 4. Prepare inputs for FunctionEncoder
    # Reshape for batch_size = 1
    # example_xs: (1, num_sensors, 1) - sensor locations
    # example_ys: (1, num_sensors, 1) - function values at sensor locations
    # query_xs:   (1, num_dense_points, 1) - locations to predict derivative
    example_xs_plot = sensor_x_coords.unsqueeze(0).unsqueeze(-1)
    example_ys_plot = f_values_at_sensors.unsqueeze(0).unsqueeze(-1)
    query_xs_plot = x_dense_eval_coords.unsqueeze(0).unsqueeze(-1)


    # 5. Get prediction from FunctionEncoder
    representations, _ = fe_model.compute_representation(example_xs_plot, example_ys_plot)
    predicted_df_dx_dense_tensor = fe_model.predict(query_xs_plot, representations) # Shape (1, num_dense_points, 1)
    predicted_df_dx_dense = predicted_df_dx_dense_tensor.squeeze().cpu().numpy()

    # Prepare data for plotting (move to CPU, convert to numpy)
    sensor_x_cpu = sensor_x_coords.cpu().numpy()
    f_values_at_sensors_cpu = f_values_at_sensors.cpu().numpy()
    x_dense_eval_cpu = x_dense_eval_coords.cpu().numpy()
    true_f_values_dense_cpu = true_f_values_dense.cpu().numpy()
    true_df_dx_values_dense_cpu = true_df_dx_values_dense.cpu().numpy()


    # 6. Plot - Modified for single panel (derivative only)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) # Changed from 1, 2 to 1, 1

    # Panel 2 (now just ax): Derivative Prediction
    ax.plot(x_dense_eval_cpu, true_df_dx_values_dense_cpu, label="True df/dx", color='green', linestyle='-')
    ax.plot(x_dense_eval_cpu, predicted_df_dx_dense, label="Predicted df/dx (FunctionEncoder)", color='orange', linestyle='--')
    ax.set_xlabel("x")
    ax.set_ylabel("df/dx")
    ax.set_title("Derivative Prediction using FunctionEncoder") # Updated title
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Prediction plot saved to {plot_save_path}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    device = torch.device(args.device)
    torch.manual_seed(0)

    # --- Log directory ---
    # All outputs (plots, tensorboard logs) for a run will go into a timestamped subdirectory
    base_log_dir = "logs_cubic_derivative_example"
    run_log_dir = base_log_dir
    if args.plot:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_log_dir = os.path.join(base_log_dir, current_time)
        os.makedirs(run_log_dir, exist_ok=True)
    elif not os.path.exists(base_log_dir) and not args.plot:
        # Create base log dir even if not plotting, if other non-plot artifacts might be saved later.
        # For now, only plotting uses it.
        pass


    # --- Fixed sensor grid --------------------------------------------------
    sensor_size = 1000
    sensor_x = torch.linspace(args.input_range[0],
                              args.input_range[1],
                              sensor_size,
                              device=device)

    # --- Dataset ------------------------------------------------------------
    dataset = CubicDerivativeDataset(
        sensor_x=sensor_x,
        input_range=args.input_range,
        scale=args.scale,
        n_trunk_points=args.n_trunk_points,
        batch_size=args.batch
    )

    # --- FunctionEncoder set-up --------------------------------------------
    fe = FunctionEncoder(
        input_size=(1,),                       # x-coordinate
        output_size=(1,),                      # derivative value
        data_type="deterministic",
        n_basis=args.n_basis,                  # size of learned basis
        model_type="MLP",                      # basis functions φ_k(x)
        model_kwargs=dict(                   # kwargs forwarded to MLP
            hidden_size=128,                   # Size of each hidden layer
            n_layers=4                         # Total layers: 1 input, 2 hidden, 1 output
        ),
        representation_mode="encoder_network",
        encoder_type="SetONet",
        encoder_kwargs=dict(                   # kwargs forwarded to SetONet
            phi_hidden_size=256,
            rho_hidden_size=256,
            phi_output_size=128,
            p=args.n_basis,
            aggregation="mean"
        ),
        use_residuals_method=False,            # not needed here
        optimizer_kwargs={"lr": args.lr}
    ).to(device)

    print(f"\nTotal trainable parameters: "
          f"{sum(p.numel() for p in fe.parameters()):,}")

    # --- Callbacks ---
    ortho_cb_instance = None
    callbacks_to_use = []
    if args.plot:
        print(f"TensorBoard logs and plots will be saved to: {run_log_dir}")
        tb_cb = TensorboardCallback(logdir=run_log_dir)
        ortho_cb_instance = OrthonormalityCallback(
            input_range=args.input_range,
            tensorboard_writer=tb_cb.tensorboard,
            log_freq=100 # Log orthonormality every 100 epochs
        )
        callbacks_to_use.extend([tb_cb, ortho_cb_instance])

    # --- Train --------------------------------------------------------------
    if callbacks_to_use:
        callback_manager = ListCallback(callbacks_to_use)
        fe.train_model(
            dataset=dataset,
            epochs=args.epochs,
            progress_bar=True,
            callback=callback_manager
        )
    else:
        fe.train_model(
            dataset=dataset,
            epochs=args.epochs,
            progress_bar=True
        )

    # --- Quick evaluation ---------------------------------------------------
    test_mse = evaluate(fe, dataset)
    print(f"\nTest MSE on derivative prediction: {test_mse:.4e}")

    # --- Plotting -----------------------------------------------------------
    if args.plot:
        print("\n--- Generating Plots ---")

        # 1. Prediction Example Plot (adapted from original)
        x_dense_plot = torch.linspace(args.input_range[0], args.input_range[1], 400, device=device)
        prediction_plot_filename = os.path.join(run_log_dir, "function_encoder_derivative_prediction.png")
        plot_fe_prediction_example(
            fe_model=fe,
            sensor_x_coords=sensor_x,
            x_dense_eval_coords=x_dense_plot,
            scale=args.scale,
            plot_save_path=prediction_plot_filename
        )

        # 2. Orthonormality Plot
        if ortho_cb_instance:
            ortho_epochs, ortho_errors = ortho_cb_instance.get_history()
            if ortho_epochs:
                fig_ortho, ax_ortho = plt.subplots(1, 1, figsize=(10, 6))
                ax_ortho.plot(ortho_epochs, ortho_errors, marker='o', linestyle='-')
                ax_ortho.set_xlabel("Epoch")
                ax_ortho.set_ylabel("Orthonormality Error (Frobenius Norm)")
                ax_ortho.set_title("Basis Orthonormality Error during Training")
                ax_ortho.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                ortho_plot_path = os.path.join(run_log_dir, "orthonormality_plot.png")
                plt.savefig(ortho_plot_path)
                print(f"Orthonormality plot saved to {ortho_plot_path}")
                plt.close(fig_ortho)
            else:
                print("No orthonormality history recorded, skipping orthonormality plot.")
        else:
            # This should ideally not be reached if args.plot is True,
            # as ortho_cb_instance would have been initialized.
            print("OrthonormalityCallback was not used, skipping orthonormality plot.")

        # 3. Basis Functions Plot
        fig_basis, ax_basis = plt.subplots(1, 1, figsize=(12, 8))
        # Create a dense grid for plotting basis functions
        x_dense_basis_plot = torch.linspace(args.input_range[0],
                                            args.input_range[1],
                                            500, device=device).unsqueeze(-1) # Shape (N_points, 1)

        with torch.no_grad():
            # Get basis function values: Shape (N_points, 1, N_basis)
            basis_functions_values = fe.forward_basis_functions(x_dense_basis_plot)

        num_basis_to_plot = basis_functions_values.shape[2]
        for i in range(num_basis_to_plot):
            ax_basis.plot(x_dense_basis_plot.cpu().numpy().flatten(),
                          basis_functions_values[:, 0, i].cpu().numpy(),
                          label=f"Basis {i+1}" if num_basis_to_plot <= 10 else None) # Avoid cluttering legend

        ax_basis.set_xlabel("x")
        ax_basis.set_ylabel("Basis function value φ(x)")
        ax_basis.set_title(f"Learned Basis Functions (n_basis={args.n_basis})")
        if num_basis_to_plot <= 10: # Show legend only if not too many basis functions
            ax_basis.legend()
        ax_basis.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        basis_plot_filename = os.path.join(run_log_dir, "basis_functions_plot.png")
        plt.savefig(basis_plot_filename)
        print(f"Basis functions plot saved to {basis_plot_filename}")
        plt.close(fig_basis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' or 'cuda'")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_trunk_points", type=int, default=200)
    parser.add_argument("--n_basis", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--input_range", type=float, nargs=2,
                        default=[-1.0, 1.0])
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args) 