import torch

def sample_trunk_points(n_points, input_range, device):
    """
    Generates linearly spaced points within the specified input_range on the given device.
    """
    return torch.linspace(input_range[0], input_range[1], n_points, device=device).view(-1, 1)

def generate_batch(batch_size, n_trunk_points, sensor_x, scale, input_range, device):
    """
    Generates a batch of cubic polynomials and their derivatives on the specified device.
    
    Args:
        batch_size (int): Number of samples in the batch.
        n_trunk_points (int): Number of points to evaluate the derivative at (trunk points).
        sensor_x (torch.Tensor): Fixed sensor locations for evaluating the function (should be on device).
        scale (float): Scaling factor for polynomial coefficients.
        input_range (list or tuple): The [min, max] range for trunk points.
        device (torch.device): The device to create tensors on.
        
    Returns:
        f_values (torch.Tensor): Function values at sensor_x locations. Shape: [batch_size, num_sensors]
        x_eval (torch.Tensor): Trunk evaluation points. Shape: [n_trunk_points, 1]
        y_target (torch.Tensor): True derivative values at x_eval points. Shape: [batch_size, n_trunk_points]
        df_dx_sensors (torch.Tensor): True derivative values at sensor_x locations. Shape: [batch_size, num_sensors]
    """
    a = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    d = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    # a, b, c, d = a * scale, b * scale, c * scale, d * scale # Scale is now applied directly

    # Function values at sensor locations
    # sensor_x is already on the correct device (passed in)
    f_values = a * sensor_x**3 + b * sensor_x**2 + c * sensor_x + d

    # Trunk input points (evaluation x for derivative)
    x_eval = sample_trunk_points(n_trunk_points, input_range, device)

    # True derivative values at x_eval points
    y_target = 3 * a * x_eval.T**2 + 2 * b * x_eval.T + c
    
    # True derivative values at sensor_x locations
    # f'(x) = 3ax^2 + 2bx + c
    # sensor_x has shape [num_sensors], a, b, c have shape [batch_size, 1]
    # We want df_dx_sensors to have shape [batch_size, num_sensors]
    df_dx_sensors = 3 * a * sensor_x**2 + 2 * b * sensor_x + c
    
    return f_values, x_eval, y_target, df_dx_sensors 