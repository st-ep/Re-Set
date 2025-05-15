import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np # For seed setting if we move it here, or ensure it's set before calling

# Assuming these utilities are accessible or passed as arguments if needed
from Models.deeponet_model import MLP, DeepONet
from Models.SetONet import SetONet
from Data.data_utils import generate_batch
from Models.utils.orthogonality_utils import calculate_setonet_trunk_orthogonality, plot_setonet_trunk_orthogonality

def run_deeponet_pipeline(args, device, sensor_x, loss_fn, log_dir, input_range, scale):
    print("\n--- Initializing and Training DeepONet ---")
    # Define DeepONet
    branch_net = MLP(input_dim=sensor_x.shape[0], hidden_dims=args.don_branch_hidden, output_dim=args.don_output_dim, activation=torch.nn.Tanh)
    trunk_net = MLP(input_dim=1, hidden_dims=args.don_trunk_hidden, output_dim=args.don_output_dim, activation=torch.nn.Tanh)
    model = DeepONet(branch_net=branch_net, trunk_net=trunk_net).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.don_lr)

    # LR Scheduler for DeepONet
    if args.don_lr_schedule_steps and args.don_lr_schedule_gammas:
        don_lambda_func = lambda epoch: np.prod([gamma for i, gamma in enumerate(args.don_lr_schedule_gammas) if epoch >= args.don_lr_schedule_steps[i]]) \
                                      if args.don_lr_schedule_steps else 1.0
        scheduler_don = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=don_lambda_func)
        print(f"Using LambdaLR scheduler for DeepONet with steps {args.don_lr_schedule_steps} and gammas {args.don_lr_schedule_gammas}")
    else:
        scheduler_don = None

    print("\nTraining DeepONet...")
    for epoch in range(args.don_epochs):
        model.train()
        # For DeepONet, generate_batch is expected to return 3 items: branch_input, trunk_input, target
        # If generate_batch is modified to return 4 for SetONet, ensure it's handled or a specific version is called.
        data_from_generator = generate_batch(
            batch_size=64, 
            n_trunk_points=40, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )
        branch_input, trunk_input, target = data_from_generator[:3] # Take first 3

        pred = model(branch_input, trunk_input)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler_don:
            scheduler_don.step()

        if epoch % 500 == 0: # Using a common print interval
            print(f"Epoch {epoch}, DeepONet Loss: {loss.item():.6f}")
            if scheduler_don:
                print(f"Current DeepONet LR: {scheduler_don.get_last_lr()[0]:.2e}")

    model.eval()
    with torch.no_grad():
        n_test = 1000
        data_from_generator_test = generate_batch(
            batch_size=n_test, 
            n_trunk_points=100, 
            sensor_x=sensor_x, 
            scale=scale, 
            input_range=input_range,
            device=device
        )
        branch_input_test, trunk_input_test, y_true_test = data_from_generator_test[:3]

        y_pred_test = model(branch_input_test, trunk_input_test)
        error = torch.norm(y_pred_test - y_true_test, dim=1)
        denom = torch.norm(y_true_test, dim=1)
        rel_error = (error / denom).mean()
        print(f"\nDeepONet: Average L2 Relative Error over {n_test} test examples: {rel_error:.6f}")

    deeponet_model_path = os.path.join(log_dir, "deeponet_model.pth")
    torch.save(model.state_dict(), deeponet_model_path)
    print(f"DeepONet model saved to {deeponet_model_path}")
    return model

def run_setonet_pipeline(args, device, sensor_x, x_basis_plot, loss_fn, log_dir, input_range, scale):
    print("\n--- Initializing and Training SetONet ---")
    setonet_model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=args.son_p_dim,
        phi_hidden_size=args.son_phi_hidden,
        rho_hidden_size=args.son_rho_hidden,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        activation_fn=torch.nn.Tanh,
        use_deeponet_bias=True,
        phi_output_size=args.son_phi_output_size,
        pos_encoding_type=args.pos_encoding_type,
        aggregation_type=args.son_aggregation,
        concat_sensor_derivative_to_branch_input=args.son_concat_sensor_derivative_to_branch,
        initial_lr=args.son_lr,
        lr_schedule_steps=None,
        lr_schedule_gammas=None
    ).to(device)

    optimizer_son = optim.Adam(setonet_model.parameters(), lr=args.son_lr)

    # LR Scheduler for SetONet
    if args.son_lr_schedule_steps and args.son_lr_schedule_gammas:
        son_lambda_func = lambda epoch: np.prod([gamma for i, gamma in enumerate(args.son_lr_schedule_gammas) if epoch >= args.son_lr_schedule_steps[i]]) \
                                       if args.son_lr_schedule_steps else 1.0
        scheduler_son = torch.optim.lr_scheduler.LambdaLR(optimizer_son, lr_lambda=son_lambda_func)
        print(f"Using LambdaLR scheduler for SetONet with steps {args.son_lr_schedule_steps} and gammas {args.son_lr_schedule_gammas}")
    else:
        scheduler_son = None

    tb_log_dir_setonet = os.path.join(log_dir, 'tensorboard_setonet')
    os.makedirs(tb_log_dir_setonet, exist_ok=True)
    writer_setonet = SummaryWriter(log_dir=tb_log_dir_setonet)

    print("\nTraining SetONet...")
    setonet_orthogonality_scores = []
    setonet_orthogonality_epochs = []
    x_basis_plot_ortho = x_basis_plot.to(device) # x_basis_plot is passed in

    for epoch in range(args.son_epochs):
        setonet_model.train()
        data_from_generator = generate_batch(
            batch_size=64,
            n_trunk_points=40,
            sensor_x=sensor_x,
            scale=scale,
            input_range=input_range,
            device=device
        )

        batch_df_dx_sensors = None
        if len(data_from_generator) == 4:
            batch_f_values, batch_x_eval, batch_y_target, batch_df_dx_sensors = data_from_generator
        elif len(data_from_generator) == 3:
            batch_f_values, batch_x_eval, batch_y_target = data_from_generator
            if args.son_concat_sensor_derivative_to_branch:
                raise ValueError("generate_batch needs to return 4 items for SetONet derivative concatenation.")
        else:
            raise ValueError(f"generate_batch returned {len(data_from_generator)} items.")

        current_batch_size = batch_f_values.shape[0]
        
        xs_setonet = sensor_x.view(1, sensor_x.shape[0], 1).expand(current_batch_size, -1, -1)
        us_setonet = batch_f_values.unsqueeze(-1)
        ys_setonet = batch_x_eval.unsqueeze(0).expand(current_batch_size, -1, -1)
        us_derivs_setonet_arg = batch_df_dx_sensors.unsqueeze(-1) if args.son_concat_sensor_derivative_to_branch and batch_df_dx_sensors is not None else None
        
        pred_setonet = setonet_model(xs_setonet, us_setonet, ys_setonet, us_derivs=us_derivs_setonet_arg)
        target_setonet = batch_y_target.unsqueeze(-1)
        loss_setonet = loss_fn(pred_setonet, target_setonet)

        optimizer_son.zero_grad()
        loss_setonet.backward()
        optimizer_son.step()

        if scheduler_son:
            scheduler_son.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, SetONet Loss: {loss_setonet.item():.6f}")
            if scheduler_son:
                writer_setonet.add_scalar('LR/SetONet', scheduler_son.get_last_lr()[0], epoch)
                print(f"Current SetONet LR: {scheduler_son.get_last_lr()[0]:.2e}")
            else:
                writer_setonet.add_scalar('LR/SetONet', args.son_lr, epoch)
            
            # Calculate and log MSE for current batch
            mse = loss_setonet.item()
            writer_setonet.add_scalar('Metrics/MSE', mse, epoch)
            
            # Calculate and log relative L2 error for current batch
            with torch.no_grad():
                error = torch.norm(pred_setonet.squeeze(-1) - target_setonet.squeeze(-1), dim=1)
                denom = torch.norm(target_setonet.squeeze(-1), dim=1)
                rel_error = (error / denom).mean().item()
                writer_setonet.add_scalar('Metrics/Relative_L2_Error', rel_error, epoch)

        # Evaluate on a validation set every eval_interval epochs
        if epoch % 500 == 0 or epoch == args.son_epochs - 1:
            setonet_model.eval()
            with torch.no_grad():
                val_data = generate_batch(
                    batch_size=128,
                    n_trunk_points=100,
                    sensor_x=sensor_x,
                    scale=scale,
                    input_range=input_range,
                    device=device
                )
                
                if len(val_data) == 4:
                    val_f_values, val_x_eval, val_y_true, val_df_dx_sensors = val_data
                    val_derivs_arg = val_df_dx_sensors.unsqueeze(-1) if args.son_concat_sensor_derivative_to_branch else None
                else:
                    val_f_values, val_x_eval, val_y_true = val_data
                    val_derivs_arg = None
                
                val_batch_size = val_f_values.shape[0]
                val_xs = sensor_x.view(1, sensor_x.shape[0], 1).expand(val_batch_size, -1, -1)
                val_us = val_f_values.unsqueeze(-1)
                val_ys = val_x_eval.unsqueeze(0).expand(val_batch_size, -1, -1)
                
                val_pred = setonet_model(val_xs, val_us, val_ys, us_derivs=val_derivs_arg)
                
                # Calculate validation MSE
                val_target = val_y_true.unsqueeze(-1)
                val_mse = loss_fn(val_pred, val_target).item()
                writer_setonet.add_scalar('Validation/MSE', val_mse, epoch)
                
                # Calculate validation relative L2 error
                val_error = torch.norm(val_pred.squeeze(-1) - val_y_true, dim=1)
                val_denom = torch.norm(val_y_true, dim=1)
                val_rel_error = (val_error / val_denom).mean().item()
                writer_setonet.add_scalar('Validation/Relative_L2_Error', val_rel_error, epoch)
            
            setonet_model.train()

        # Check SetONet trunk orthogonality at specified intervals
        if args.son_ortho_check_interval > 0 and (epoch % args.son_ortho_check_interval == 0 or epoch == args.son_epochs - 1):
            current_ortho_score = calculate_setonet_trunk_orthogonality(setonet_model, x_basis_plot_ortho, args.son_p_dim, device)
            if current_ortho_score is not None:
                setonet_orthogonality_scores.append(current_ortho_score)
                setonet_orthogonality_epochs.append(epoch)
                writer_setonet.add_scalar('Metrics/SetONet_Trunk_Orthogonality_Error', current_ortho_score, epoch)
    
    if args.son_ortho_check_interval == 0 or \
       (args.son_ortho_check_interval > 0 and (args.son_epochs - 1) not in setonet_orthogonality_epochs and args.son_epochs > 0):
        final_ortho_score = calculate_setonet_trunk_orthogonality(setonet_model, x_basis_plot_ortho, args.son_p_dim, device)
        if final_ortho_score is not None:
            epoch_to_log_final_ortho = args.son_epochs - 1 if args.son_epochs > 0 else 0
            if not setonet_orthogonality_epochs or setonet_orthogonality_epochs[-1] != epoch_to_log_final_ortho:
                setonet_orthogonality_scores.append(final_ortho_score)
                setonet_orthogonality_epochs.append(epoch_to_log_final_ortho)
            elif not setonet_orthogonality_scores : # Ensure it's added if list is empty (e.g. interval=0)
                 setonet_orthogonality_scores.append(final_ortho_score)
                 setonet_orthogonality_epochs.append(epoch_to_log_final_ortho)
            writer_setonet.add_scalar('Metrics/SetONet_Trunk_Orthogonality_Error', final_ortho_score, epoch_to_log_final_ortho)

    setonet_model.eval()
    with torch.no_grad():
        n_test = 1000
        test_data_from_generator = generate_batch(
            batch_size=n_test, n_trunk_points=100, sensor_x=sensor_x, scale=scale, input_range=input_range, device=device
        )
        df_dx_sensors_test_eval = None
        if len(test_data_from_generator) == 4:
            f_values_test, x_eval_test, y_true_test, df_dx_sensors_test_eval = test_data_from_generator
        elif len(test_data_from_generator) == 3:
            f_values_test, x_eval_test, y_true_test = test_data_from_generator
            if args.son_concat_sensor_derivative_to_branch:
                raise ValueError("generate_batch needs 4 items for SetONet derivative eval.")
        else:
            raise ValueError(f"generate_batch returned {len(test_data_from_generator)} items for eval.")

        current_test_batch_size = f_values_test.shape[0]
        xs_setonet_test = sensor_x.view(1, sensor_x.shape[0], 1).expand(current_test_batch_size, -1, -1)
        us_setonet_test = f_values_test.unsqueeze(-1)
        ys_setonet_test = x_eval_test.unsqueeze(0).expand(current_test_batch_size, -1, -1)
        us_derivs_setonet_test_arg = df_dx_sensors_test_eval.unsqueeze(-1) if args.son_concat_sensor_derivative_to_branch and df_dx_sensors_test_eval is not None else None

        y_pred_setonet = setonet_model(xs_setonet_test, us_setonet_test, ys_setonet_test, us_derivs=us_derivs_setonet_test_arg)
        y_true_setonet_reshaped = y_true_test.unsqueeze(-1)
        error_setonet = torch.norm(y_pred_setonet.squeeze(-1) - y_true_test, dim=1)
        denom_setonet = torch.norm(y_true_test, dim=1)
        rel_error_setonet = (error_setonet / denom_setonet).mean()
        print(f"SetONet: Average L2 Relative Error over {n_test} test examples: {rel_error_setonet:.6f}")

    setonet_model_path = os.path.join(log_dir, "setonet_model.pth")
    torch.save(setonet_model.state_dict(), setonet_model_path)
    print(f"SetONet model saved to {setonet_model_path}")

    plot_setonet_trunk_orthogonality(setonet_orthogonality_epochs, setonet_orthogonality_scores, log_dir)
    writer_setonet.close()
    return setonet_model 