"""
Generic 1-D SetONet benchmark runner.
Handles   – data generation
          – normalisation
          – training / testing loops
          – logging & plots

Two tasks are supported at the moment:
    • "output_derivative"  – learn f′ from samples of f′
    • "input_function"     – learn  f  from samples of  f
"""

from __future__ import annotations
import os, json
from datetime import datetime
from types import SimpleNamespace
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# ─── Project imports ──────────────────────────────────────────────────────────
from Models.SetONet import SetONet
from Data.data_utils import generate_batch, generate_batch_sin, generate_batch_cool_basis
from Benchmarks.benchmark_utils import (                 # already contains helpers
    relative_L2, normalize_coordinates,
    normalize_values, denormalize_values,
)
from Models.utils.orthogonality_utils import (
    calculate_setonet_trunk_orthogonality,
)
# Import the new unified plotting function
from Plotting.plots_1d import generate_plots_1d, plot_trunk_gramian_heatmap

# -----------------------------------------------------------------------------


def _prepare_setonet_inputs(
    sensor_x_norm_expanded: torch.Tensor,
    branch_norm:           torch.Tensor,
    x_eval_norm:           torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shapes everything so it can be fed straight into SetONet.
        sensor_x_norm_expanded : (1, n_sensor, 1)  – normalised sensor grid
        branch_norm            : (B, n_sensor)     – normalised branch values
        x_eval_norm            : (n_trunk, 1)      – normalised evaluation grid
    Returns
        xs_norm : (B, n_sensor, 1)
        us_norm : (B, n_sensor, 1)
        ys_norm : (B, n_trunk , 1)
    """
    bsz = branch_norm.size(0)
    xs_norm = sensor_x_norm_expanded.expand(bsz, -1, -1)
    us_norm = branch_norm.unsqueeze(-1)
    ys_norm = x_eval_norm.unsqueeze(0).expand(bsz, -1, -1)
    return xs_norm, us_norm, ys_norm


# ──────────────────────────────────────────────────────────────────────────────
def run_setonet_benchmark(cfg_dict: Dict) -> None:
    """
    Main entry-point used by each benchmark script.
    `cfg_dict` is a *flat* dictionary – the individual benchmark scripts keep
    full control over the values they pass.
    """

    # Allow attribute as well as key access --------------------------------------------------
    cfg = SimpleNamespace(**cfg_dict)

    # ─── Environment / logging dirs ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir      = os.path.join(project_root, "logs", cfg.task_type, timestamp)
    tb_dir       = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    print(f"Logging   ➜  {log_dir}")
    print(f"TensorBoard ➜  {tb_dir}")

    writer = SummaryWriter(tb_dir)

    # ─── Fixed sensor grid (shared by all batches) ───────────────────────────
    sensor_x = torch.linspace(
        cfg.input_range[0], cfg.input_range[1], cfg.n_sensor_points, device=device
    )                                                            # (n_sensor,)
    sensor_x_norm = normalize_coordinates(sensor_x, cfg.input_range)
    sensor_x_norm_expanded = sensor_x_norm.view(1, -1, 1)        # (1, n_sensor, 1)

    # ─── Model ----------------------------------------------------------------
    model = SetONet(
        input_size_src   = 1,
        output_size_src  = 1,
        input_size_tgt   = 1,
        output_size_tgt  = 1,
        p                = cfg.latent_p,
        phi_hidden_size  = cfg.phi_hidden_size,
        rho_hidden_size  = cfg.rho_hidden_size,
        trunk_hidden_size= cfg.trunk_hidden_size,
        n_trunk_layers   = cfg.n_trunk_layers,
        activation_fn    = nn.ReLU,
        use_deeponet_bias= cfg.use_deeponet_bias,
        phi_output_size  = cfg.phi_output_size,
        aggregation_type = cfg.aggregation_type,
        attention_n_tokens = cfg.attention_n_tokens,
        pos_encoding_type  = cfg.pos_encoding_type,
        concat_sensor_derivative_to_branch_input = (
            cfg.concat_sensor_derivative_to_branch_input
        ),
        initial_lr       = cfg.lr,
    ).to(device)

    opt        = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler  = MultiStepLR(opt, milestones=cfg.lr_schedule_steps,
                             gamma=cfg.lr_schedule_gamma)
    loss_fn    = nn.MSELoss()

    # ─── Save config to JSON ---------------------------------------------------
    full_cfg_for_json = cfg_dict.copy()
    full_cfg_for_json["model_architecture_details"] = model.get_params()
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as fp:
        json.dump(full_cfg_for_json, fp, indent=4)

    # ─── Training loop ---------------------------------------------------------
    ortho_epochs, ortho_scores = [], []
    bar = trange(cfg.n_epochs, desc="Training")

    for epoch in bar:
        # 1) ── Data generation (same generator, different slicing) ------------
        if cfg.task_type == "input_function_sin":
            # generate_batch_sin returns: f_values_at_sensors, x_eval, f_values_at_x_eval
            branch_orig, x_eval_orig, y_true_orig = generate_batch_sin(
                batch_size     = cfg.batch_size,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )
        elif cfg.task_type == "input_function_cool_basis":
            # generate_batch_cool_basis returns: f_values_at_sensors, x_eval, f_values_at_x_eval
            branch_orig, x_eval_orig, y_true_orig = generate_batch_cool_basis(
                batch_size     = cfg.batch_size,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )
        else: # Original tasks using generate_batch
            batch = generate_batch(
                batch_size     = cfg.batch_size,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )

            if cfg.task_type == "output_derivative":
                # (_, x_eval, _, f′(x_eval), f′(sensor_x))
                _, x_eval_orig, _, y_true_orig, branch_orig = batch
            elif cfg.task_type == "input_function":
                # (f(sensor_x), x_eval, f(x_eval), _, _)
                branch_orig, x_eval_orig, y_true_orig, _, _ = batch
            else:  # future-proof: raise an error for unsupported tasks
                raise ValueError(f"Unknown task_type for data generation: {cfg.task_type}")

        # 2) ── Normalisation ---------------------------------------------------
        x_eval_norm   = normalize_coordinates(x_eval_orig, cfg.input_range)

        target_mean   = y_true_orig.mean(dim=1, keepdim=True)
        target_std    = y_true_orig.std(dim=1, keepdim=True) + 1e-8

        target_norm   = normalize_values(y_true_orig, target_mean, target_std)
        branch_norm   = normalize_values(branch_orig, target_mean, target_std)

        # 3) ── Model forward / loss -------------------------------------------
        xs_norm, us_norm, ys_norm = _prepare_setonet_inputs(
            sensor_x_norm_expanded, branch_norm, x_eval_norm
        )

        pred_norm   = model(xs_norm, us_norm, ys_norm)               # (B, N, 1)
        mse_loss    = loss_fn(pred_norm, target_norm.unsqueeze(-1))

        # L1 reg
        l1_penalty = sum(torch.abs(p).sum() for p in model.parameters())
        loss       = mse_loss + cfg.l1_lambda * l1_penalty

        # Optimisation step
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # 4) ── Book-keeping ----------------------------------------------------
        if epoch % cfg.print_interval == 0:
            pred_denorm = denormalize_values(
                pred_norm.squeeze(-1), target_mean, target_std
            )
            rel_err = relative_L2(pred_denorm, y_true_orig).item()

            bar.set_postfix(
                {"MSE(n)": f"{mse_loss.item():.2e}",
                 "RelL2":   f"{rel_err:.2e}"}
            )

            writer.add_scalar("Loss/Train_MSE_Normalized", mse_loss.item(), epoch)
            writer.add_scalar("Loss/Train_Total",          loss.item(),      epoch)
            writer.add_scalar("Error/Train_RelL2_Orig",    rel_err,          epoch)
            writer.add_scalar("LearningRate",              scheduler.get_last_lr()[0], epoch)

        # ── Trunk orthogonality ------------------------------------------------
        if epoch % 500 == 0 or epoch == cfg.n_epochs - 1:
            ortho, gramian_matrix = calculate_setonet_trunk_orthogonality(
                model, sensor_x_norm.unsqueeze(-1), cfg.latent_p, device
            )
            if ortho is not None:
                ortho_epochs.append(epoch)
                ortho_scores.append(ortho)
                writer.add_scalar("Metrics/Trunk_Orthogonality", ortho, epoch)
            
            # Plot heatmap only at 10000 epochs and at the end
            if gramian_matrix is not None and (epoch == 10000 or epoch == cfg.n_epochs - 1):
                plot_trunk_gramian_heatmap(
                    gramian_matrix=gramian_matrix, # Already on CPU from calculate_...
                    log_dir=log_dir,
                    epoch=epoch,
                    p_dim=cfg.latent_p,
                    task_type=cfg.task_type
                )

    # ─── Simple test set -------------------------------------------------------
    model.eval()
    with torch.no_grad():
        n_test = 1000
        if cfg.task_type == "input_function_sin":
            # generate_batch_sin returns: f_values_at_sensors, x_eval, f_values_at_x_eval
            branch_orig, x_eval_orig, y_true_orig = generate_batch_sin(
                batch_size     = n_test,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )
        elif cfg.task_type == "input_function_cool_basis":
            # generate_batch_cool_basis returns: f_values_at_sensors, x_eval, f_values_at_x_eval
            branch_orig, x_eval_orig, y_true_orig = generate_batch_cool_basis(
                batch_size     = n_test,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )
        else: # Original tasks using generate_batch
            batch  = generate_batch(
                batch_size     = n_test,
                n_trunk_points = cfg.n_trunk_points,
                sensor_x       = sensor_x,
                scale          = cfg.scale,
                input_range    = cfg.input_range,
                device         = device,
            )

            if cfg.task_type == "output_derivative":
                _, x_eval_orig, _, y_true_orig, branch_orig = batch
            elif cfg.task_type == "input_function": 
                branch_orig, x_eval_orig, y_true_orig, _, _ = batch
            else:
                # This case should ideally be caught by the check in the training loop,
                # but as a safeguard for the test section:
                raise ValueError(f"Unknown task_type for test data generation: {cfg.task_type}")

        x_eval_norm   = normalize_coordinates(x_eval_orig, cfg.input_range)
        target_mean   = y_true_orig.mean(dim=1, keepdim=True)
        target_std    = y_true_orig.std(dim=1, keepdim=True) + 1e-8
        branch_norm   = normalize_values(branch_orig, target_mean, target_std)

        xs_norm, us_norm, ys_norm = _prepare_setonet_inputs(
            sensor_x_norm_expanded, branch_norm, x_eval_norm
        )
        pred_norm = model(xs_norm, us_norm, ys_norm).squeeze(-1)

        pred_denorm = denormalize_values(pred_norm, target_mean, target_std)
        rel_err_test = relative_L2(pred_denorm, y_true_orig).item()
        print(f"\nTest Relative L2 error (orig scale): {rel_err_test:.4e}")
        writer.add_scalar("Error/Test_RelL2_Orig", rel_err_test, cfg.n_epochs)

    # ─── Save model ------------------------------------------------------------
    ckpt_path = os.path.join(log_dir, f"setonet_{cfg.task_type}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved ➜ {ckpt_path}")

    # ─── Plots -----------------------------------------------------------------
    # Common parameters for plotting that might come from cfg
    n_trunk_points_plot = cfg.n_trunk_points # Used for basis plots and input_function sample plots
    font_size_combined = getattr(cfg, "font_size_combined_plot", 20)
    margin_left = getattr(cfg, "margin_left_combined_plot", 0.09)
    margin_right = getattr(cfg, "margin_right_combined_plot", 0.99)
    margin_bottom = getattr(cfg, "margin_bottom_combined_plot", 0.1)
    margin_top = getattr(cfg, "margin_top_combined_plot", 0.98)
    n_dense_points_derivative_plot = getattr(cfg, "n_dense_points_derivative_plot", 200)


    generate_plots_1d(
        model=model,
        device=device,
        log_dir=log_dir,
        task_type=cfg.task_type,
        ortho_epochs=ortho_epochs,
        ortho_scores=ortho_scores,
        latent_p=cfg.latent_p,
        sensor_x_orig=sensor_x.cpu(), # Ensure sensor_x is on CPU for plotting functions
        input_range=cfg.input_range,
        scale=cfg.scale,
        n_trunk_points_plot=n_trunk_points_plot,
        n_samples_to_plot=cfg.n_samples_to_plot,
        n_dense_points_derivative_plot=n_dense_points_derivative_plot,
        font_size_combined=font_size_combined,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_top=margin_top
    )

    writer.close() 