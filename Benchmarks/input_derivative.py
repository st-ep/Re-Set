"""
Benchmark: Encoding the input function with SetONet
----------------------------------------------------
Instead of learning the derivative, we focus on encoding the input
function space, i.e., we reconstruct the function f(x) from a
handful of sample points.

branch input     : {(x_i , f(x_i))}_i       ➜ coefficients  (B , p , 1)
trunk  input     : {y_j}_j                  ➜ basis values  (B , n_y , p , 1)
prediction       :  Σ_p  coeff_p · basis_p  ≈ f(y)

The data generator is shared with the other pipelines (`generate_batch`).
"""

from Benchmarks.setonet_1d_benchmark import run_setonet_benchmark

def main() -> None:
    cfg = dict(
        task_type   = "input_function",

        input_range = [-1.0, 1.0],
        scale       = 0.1,
        n_sensor_points = 200,
        n_trunk_points  = 100,
        latent_p        = 16,
        n_samples_to_plot = 3,
        batch_size      = 256,
        n_epochs        = 50_000,
        print_interval  = 500,

        lr                = 1e-3,
        lr_schedule_steps = [400000, 750000, 100000],
        lr_schedule_gamma = 0.5,
        l1_lambda         = 2e-5,

        phi_hidden_size   = 256,
        rho_hidden_size   = 256,
        trunk_hidden_size = 256,
        n_trunk_layers    = 4,
        phi_output_size   = 16,
        aggregation_type  = "attention",
        attention_n_tokens= 4,          
        pos_encoding_type = "skip",
        use_deeponet_bias = True,
        concat_sensor_derivative_to_branch_input = False,
    )

    run_setonet_benchmark(cfg)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
