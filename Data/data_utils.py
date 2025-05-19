import torch

def sample_trunk_points(n_points, input_range, device):
    """
    Generates linearly spaced points within the specified input_range on the given device.
    """
    return torch.linspace(input_range[0], input_range[1], n_points, device=device).view(-1, 1)

def generate_batch(batch_size, n_trunk_points, sensor_x, scale, input_range, device):
    """
    Generates a batch of cubic polynomials, their function values, and their derivatives 
    on the specified device.
    
    Args:
        batch_size (int): Number of samples in the batch.
        n_trunk_points (int): Number of points to evaluate the function/derivative at (trunk points).
        sensor_x (torch.Tensor): Fixed sensor locations for evaluating the function (should be on device).
        scale (float): Scaling factor for polynomial coefficients.
        input_range (list or tuple): The [min, max] range for trunk points.
        device (torch.device): The device to create tensors on.
        
    Returns:
        f_values_at_sensors (torch.Tensor): Function values f(x) at sensor_x locations. Shape: [batch_size, num_sensors]
        x_eval (torch.Tensor): Trunk evaluation points. Shape: [n_trunk_points, 1]
        f_values_at_x_eval (torch.Tensor): True function values f(x) at x_eval points. Shape: [batch_size, n_trunk_points]
        df_dx_at_x_eval (torch.Tensor): True derivative values f'(x) at x_eval points. Shape: [batch_size, n_trunk_points]
        df_dx_at_sensors (torch.Tensor): True derivative values f'(x) at sensor_x locations. Shape: [batch_size, num_sensors]
    """
    a = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    d = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    # a, b, c, d = a * scale, b * scale, c * scale, d * scale # Scale is now applied directly

    # Function values at sensor locations
    # sensor_x is already on the correct device (passed in)
    f_values_at_sensors = a * sensor_x**3 + b * sensor_x**2 + c * sensor_x + d

    # Trunk input points (evaluation x for derivative or function)
    x_eval = sample_trunk_points(n_trunk_points, input_range, device) # Shape: [n_trunk_points, 1]

    # True function values at x_eval points
    # x_eval.T has shape [1, n_trunk_points]. a,b,c,d have shape [batch_size, 1]
    # Result f_values_at_x_eval should be [batch_size, n_trunk_points]
    f_values_at_x_eval = a * x_eval.T**3 + b * x_eval.T**2 + c * x_eval.T + d
    
    # True derivative values at x_eval points
    # y_target = 3 * a * x_eval.T**2 + 2 * b * x_eval.T + c # Old y_target
    df_dx_at_x_eval = 3 * a * x_eval.T**2 + 2 * b * x_eval.T + c
    
    # True derivative values at sensor_x locations
    # f'(x) = 3ax^2 + 2bx + c
    # sensor_x has shape [num_sensors], a, b, c have shape [batch_size, 1]
    # We want df_dx_sensors to have shape [batch_size, num_sensors]
    df_dx_at_sensors = 3 * a * sensor_x**2 + 2 * b * sensor_x + c
    
    return f_values_at_sensors, x_eval, f_values_at_x_eval, df_dx_at_x_eval, df_dx_at_sensors 

def generate_batch_sin(batch_size, n_trunk_points, sensor_x, scale, input_range, device):
    """
    Generates a batch of functions a*sin(x) + b*x^2 + c*x + d and their function values
    on the specified device.
    
    Args:
        batch_size (int): Number of samples in the batch.
        n_trunk_points (int): Number of points to evaluate the function at (trunk points).
        sensor_x (torch.Tensor): Fixed sensor locations for evaluating the function (should be on device).
        scale (float): Scaling factor for polynomial coefficients.
        input_range (list or tuple): The [min, max] range for trunk points.
        device (torch.device): The device to create tensors on.
        
    Returns:
        f_values_at_sensors (torch.Tensor): Function values f(x) at sensor_x locations. Shape: [batch_size, num_sensors]
        x_eval (torch.Tensor): Trunk evaluation points. Shape: [n_trunk_points, 1]
        f_values_at_x_eval (torch.Tensor): True function values f(x) at x_eval points. Shape: [batch_size, n_trunk_points]
    """
    a = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    d = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale

    # Function values at sensor locations
    # sensor_x is already on the correct device (passed in)
    # f(x) = a*sin(x) + b*x^2 + c*x + d
    f_values_at_sensors = a * torch.sin(sensor_x) + b * sensor_x**2 + c * sensor_x + d

    # Trunk input points (evaluation x for function)
    x_eval = sample_trunk_points(n_trunk_points, input_range, device) # Shape: [n_trunk_points, 1]

    # True function values at x_eval points
    # x_eval.T has shape [1, n_trunk_points]. a,b,c,d have shape [batch_size, 1]
    # Result f_values_at_x_eval should be [batch_size, n_trunk_points]
    f_values_at_x_eval = a * torch.sin(x_eval.T) + b * x_eval.T**2 + c * x_eval.T + d
    
    return f_values_at_sensors, x_eval, f_values_at_x_eval 

# --- Constants for "cool_basis" function (3 basis functions) ---

# g1 (formerly _g2): Difference of two sharp exponential decays (Laplace-like)
_G2_AMP1 = 1.0       # Amplitude of first exponential
_G2_DECAY1 = -5.0    # Decay rate of first exponential
_G2_CENTER1 = 0.4    # Center of first exponential
_G2_AMP2 = -0.8      # Amplitude of second exponential (negative for a dip)
_G2_DECAY2 = -4.5    # Decay rate of second exponential
_G2_CENTER2 = -0.5   # Center of second exponential

# g3: "Flat top" bump using tanh differences, with a high-frequency ripple
_G3_TANH_SCALE = 5.0   # Scale for tanh transitions
_G3_BUMP_CENTER1 = -0.8 # Start of the bump region (approx)
_G3_BUMP_CENTER2 = 0.2  # End of the bump region (approx)
_G3_BUMP_AMPLITUDE = 0.5 # Overall amplitude of the bump
_G3_RIPPLE_AMP = 0.15   # Amplitude of the ripple
_G3_RIPPLE_FREQ_BASE = 12.0 # Base frequency of ripple
_G3_RIPPLE_CHIRP_RATE = 1.0 # Chirp rate for ripple frequency

# g4: Growing oscillations with a non-symmetric envelope
_G4_ENV_SIG_SCALE = 2.0   # Sigmoid scale for envelope
_G4_ENV_SIG_SHIFT = -1.0  # Sigmoid shift for envelope
_G4_ENV_LINEAR_A = 0.5    # Linear term for envelope (slope)
_G4_ENV_LINEAR_B = 0.3    # Linear term for envelope (offset)
_G4_OSC_BASE_FREQ = 2.5 * torch.pi # Base oscillation frequency
_G4_OSC_FM_AMP = 0.1      # Amplitude of frequency modulation
_G4_OSC_FM_FREQ = 3.0     # Frequency of the modulating cosine for FM

# --- Helper basis functions for generate_batch_cool_basis (3 basis functions) ---

def _g2(x: torch.Tensor) -> torch.Tensor:
    """Basis function g2: Difference of two sharp exponential decays."""
    exp1 = _G2_AMP1 * torch.exp(_G2_DECAY1 * torch.abs(x - _G2_CENTER1))
    exp2 = _G2_AMP2 * torch.exp(_G2_DECAY2 * torch.abs(x - _G2_CENTER2))
    return exp1 + exp2

def _g3(x: torch.Tensor) -> torch.Tensor:
    """Basis function 3: 'Flat top' bump with high-frequency ripple."""
    bump_profile = torch.tanh(_G3_TANH_SCALE * (x - _G3_BUMP_CENTER1)) - torch.tanh(_G3_TANH_SCALE * (x - _G3_BUMP_CENTER2))
    ripple = _G3_RIPPLE_AMP * torch.cos(_G3_RIPPLE_FREQ_BASE * x + _G3_RIPPLE_CHIRP_RATE * x**2)
    return _G3_BUMP_AMPLITUDE * 0.5 * bump_profile * (1 + ripple) # 0.5 because tanh diff goes from -2 to 2

def _g4(x: torch.Tensor) -> torch.Tensor:
    """Basis function 4: Growing oscillations with a non-symmetric envelope."""
    envelope_sigmoid_part = torch.sigmoid(_G4_ENV_SIG_SCALE * x + _G4_ENV_SIG_SHIFT)
    envelope_linear_part = (_G4_ENV_LINEAR_A * x + _G4_ENV_LINEAR_B)
    envelope = envelope_sigmoid_part * envelope_linear_part
    
    frequency_modulation = _G4_OSC_FM_AMP * torch.cos(_G4_OSC_FM_FREQ * x)
    oscillation = torch.sin(_G4_OSC_BASE_FREQ * x + frequency_modulation)
    return envelope * oscillation


def generate_batch_cool_basis(batch_size, n_trunk_points, sensor_x, scale, input_range, device):
    """
    Generates a batch of functions f(x) = a*g2(x) + b*g3(x) + c*g4(x)
    using a set of predefined "cool-looking" basis functions.
    
    Args:
        batch_size (int): Number of samples in the batch.
        n_trunk_points (int): Number of points to evaluate the function at (trunk points).
        sensor_x (torch.Tensor): Fixed sensor locations for evaluating the function (should be on device).
        scale (float): Scaling factor for the random coefficients a, b, c.
        input_range (list or tuple): The [min, max] range for trunk points.
        device (torch.device): The device to create tensors on.
        
    Returns:
        f_values_at_sensors (torch.Tensor): Function values f(x) at sensor_x locations. Shape: [batch_size, num_sensors]
        x_eval (torch.Tensor): Trunk evaluation points. Shape: [n_trunk_points, 1]
        f_values_at_x_eval (torch.Tensor): True function values f(x) at x_eval points. Shape: [batch_size, n_trunk_points]
    """
    # Generate random coefficients for the linear combination of basis functions
    a = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale

    # Ensure sensor_x and x_eval are on the correct device for basis function calculations
    # sensor_x is passed in already on device. x_eval will be created on device.

    # Function values at sensor locations
    # f(x) = a*g2(x) + b*g3(x) + c*g4(x)
    f_values_at_sensors = (a * _g2(sensor_x) +
                           b * _g3(sensor_x) +
                           c * _g4(sensor_x))

    # Trunk input points (evaluation x for function)
    x_eval = sample_trunk_points(n_trunk_points, input_range, device) # Shape: [n_trunk_points, 1]

    # True function values at x_eval points
    # x_eval.T has shape [1, n_trunk_points]. a,b,c have shape [batch_size, 1]
    # Result f_values_at_x_eval should be [batch_size, n_trunk_points]
    x_eval_transposed = x_eval.T
    f_values_at_x_eval = (a * _g2(x_eval_transposed) +
                          b * _g3(x_eval_transposed) +
                          c * _g4(x_eval_transposed))
    
    return f_values_at_sensors, x_eval, f_values_at_x_eval 