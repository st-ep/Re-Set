"""
Combined SetONet 1D Benchmark Runner
------------------------------------
This script consolidates multiple 1D benchmark tasks for SetONet.
It allows selecting a specific task via command-line arguments.

Available tasks:
1.  output_derivative: Reconstruct the derivative f'(x) from samples (x_i, f'(x_i)).
    Original: Benchmarks/output_derivative.py

2.  input_function: Reconstruct the function f(x) from samples (x_i, f(x_i)).
    Original: Benchmarks/input_derivative.py (task_type="input_function")

3.  input_function_sin: Learn f(x) = a*sin(x) + b*x^2 + c*x + d from samples (x_i, f(x_i)).
    Original: Benchmarks/function_with_sin.py

4.  input_function_cool_basis: Learn f(x) = a*g1(x) + b*g2(x) + c*g3(x) + d*g4(x)
    from samples (x_i, f(x_i)) using "cool-looking" basis functions.
    Original: Benchmarks/function_cool_basis.py
"""

import argparse
from Benchmarks.setonet_1d_benchmark import run_setonet_benchmark

# --------------------------------------------------------------------------- #
# Configuration Getter
# --------------------------------------------------------------------------- #
def get_config(task_type: str) -> dict:
    """
    Returns the configuration dictionary for the specified task_type.
    Hyperparameters are preserved from the original benchmark scripts.
    """
    if task_type == "output_derivative":
        cfg = dict(
            task_type   = "output_derivative",
            # Experiment set-up from output_derivative.py
            input_range = [-1.0, 1.0],
            scale       = 0.1,
            n_sensor_points = 200,
            n_trunk_points  = 100,
            latent_p        = 16,
            n_samples_to_plot = 3,
            batch_size      = 256,
            n_epochs        = 50_000,
            print_interval  = 500,
            # Optimisation from output_derivative.py
            lr                = 1e-3,
            lr_schedule_steps = [400000, 750000, 1000000],
            lr_schedule_gamma = 0.5,
            l1_lambda         = 2e-5,
            # Model hyper-params from output_derivative.py
            phi_hidden_size   = 256,
            rho_hidden_size   = 256,
            trunk_hidden_size = 256,
            n_trunk_layers    = 4,
            phi_output_size   = 16, # Should match latent_p
            aggregation_type  = "attention",
            attention_n_tokens= 1,
            pos_encoding_type = "skip",
            use_deeponet_bias = True,
            concat_sensor_derivative_to_branch_input = False,
        )
    elif task_type == "input_function":
        cfg = dict(
            task_type   = "input_function",
            # Experiment set-up from input_derivative.py
            input_range = [-1.0, 1.0],
            scale       = 0.1,
            n_sensor_points = 200,
            n_trunk_points  = 100,
            latent_p        = 16,
            n_samples_to_plot = 3,
            batch_size      = 256,
            n_epochs        = 50_000,
            print_interval  = 500,
            # Optimisation from input_derivative.py
            lr                = 1e-3,
            lr_schedule_steps = [400000, 750000, 100000], # Note: 100000 is kept as is.
            lr_schedule_gamma = 0.5,
            l1_lambda         = 2e-5,
            # Model hyper-params from input_derivative.py
            phi_hidden_size   = 256,
            rho_hidden_size   = 256,
            trunk_hidden_size = 256,
            n_trunk_layers    = 4,
            phi_output_size   = 16, # Should match latent_p
            aggregation_type  = "attention",
            attention_n_tokens= 4,
            pos_encoding_type = "skip",
            use_deeponet_bias = True,
            concat_sensor_derivative_to_branch_input = False,
        )
    elif task_type == "input_function_sin":
        cfg = dict(
            task_type   = "input_function_sin",
            # Experiment set-up from function_with_sin.py
            input_range = [-3.0, 3.0],
            scale       = 0.5,
            n_sensor_points = 100,
            n_trunk_points  = 100,
            latent_p        = 32,
            n_samples_to_plot = 3,
            batch_size      = 256,
            n_epochs        = 50_000,
            print_interval  = 500,
            # Optimisation from function_with_sin.py
            lr                = 1e-3,
            lr_schedule_steps = [450000, 800000],
            lr_schedule_gamma = 0.5,
            l1_lambda         = 2e-5,
            # Model hyper-params from function_with_sin.py
            phi_hidden_size   = 256,
            rho_hidden_size   = 256,
            trunk_hidden_size = 256,
            n_trunk_layers    = 4,
            phi_output_size   = 32,    # Should match latent_p
            aggregation_type  = "attention",
            attention_n_tokens= 1,
            pos_encoding_type = "skip",
            use_deeponet_bias = True,
            concat_sensor_derivative_to_branch_input = False,
        )
    elif task_type == "input_function_cool_basis":
        cfg = dict(
            task_type   = "input_function_cool_basis",
            # Experiment set-up from function_cool_basis.py
            input_range = [-1.5, 1.5],
            scale       = 0.3,
            n_sensor_points = 200,
            n_trunk_points  = 100,
            latent_p        = 16, # As in original file
            n_samples_to_plot = 4,
            batch_size      = 128,
            n_epochs        = 80_000,
            print_interval  = 500,
            # Optimisation from function_cool_basis.py
            lr                = 1e-3,
            lr_schedule_steps = [500000, 1000000],
            lr_schedule_gamma = 0.5,
            l1_lambda         = 3.0e-5,
            # Model hyper-params from function_cool_basis.py
            phi_hidden_size   = 256,
            rho_hidden_size   = 256,
            trunk_hidden_size = 256,
            n_trunk_layers    = 4,
            phi_output_size   = 32,    # As in original file (note: mismatch with latent_p)
            aggregation_type  = "attention",
            attention_n_tokens= 4,
            pos_encoding_type = "skip",
            use_deeponet_bias = True,
            concat_sensor_derivative_to_branch_input = False,
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be one of "
                         "'output_derivative', 'input_function', "
                         "'input_function_sin', 'input_function_cool_basis'.")
    return cfg

# --------------------------------------------------------------------------- #
# Main experiment runner
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run SetONet 1D benchmarks.")
    parser.add_argument(
        "--task_type",
        type=str,
        default="output_derivative",
        choices=[
            "output_derivative",
            "input_function",
            "input_function_sin",
            "input_function_cool_basis"
        ],
        help="The type of benchmark task to run (default: output_derivative)."
    )
    args = parser.parse_args()

    print(f"Running benchmark for task_type: {args.task_type}")
    cfg = get_config(args.task_type)
    run_setonet_benchmark(cfg)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main() 