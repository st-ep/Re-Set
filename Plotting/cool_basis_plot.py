import torch
import matplotlib.pyplot as plt
import os

# Attempt to import from the project structure.
# This assumes the script is run from the root of the project or the Plotting directory
# and the Python path is set up to find 'Data'.
try:
    from Data.data_utils import _g2, _g3, _g4
except ImportError:
    # Fallback if direct import fails (e.g. running script directly from Plotting folder without project context)
    # This requires data_utils.py to be in a discoverable path or re-defining them here.
    # For simplicity in this standalone script, if the import fails, we'll re-define them.
    # This is not ideal for a project but makes the plotting script more robust if run in isolation.
    print("Warning: Could not import basis functions from Data.data_utils. Re-defining them locally for plotting.")
    print("For project integration, ensure your PYTHONPATH is set correctly or run from the project root.")

    # --- Constants for "cool_basis" function (NEW SET - Fallback, 3 functions) ---
    _G2_AMP1 = 1.0
    _G2_DECAY1 = -5.0
    _G2_CENTER1 = 0.4
    _G2_AMP2 = -0.8
    _G2_DECAY2 = -4.5
    _G2_CENTER2 = -0.5

    _G3_TANH_SCALE = 5.0
    _G3_BUMP_CENTER1 = -0.8
    _G3_BUMP_CENTER2 = 0.2
    _G3_BUMP_AMPLITUDE = 0.5
    _G3_RIPPLE_AMP = 0.15
    _G3_RIPPLE_FREQ_BASE = 12.0
    _G3_RIPPLE_CHIRP_RATE = 1.0

    _G4_ENV_SIG_SCALE = 2.0
    _G4_ENV_SIG_SHIFT = -1.0
    _G4_ENV_LINEAR_A = 0.5
    _G4_ENV_LINEAR_B = 0.3
    _G4_OSC_BASE_FREQ = 2.5 * torch.pi
    _G4_OSC_FM_AMP = 0.1
    _G4_OSC_FM_FREQ = 3.0
    # --- End of Fallback Constants ---

    def _g2(x: torch.Tensor) -> torch.Tensor:
        """Basis function 2: Difference of two sharp exponential decays."""
        exp1 = _G2_AMP1 * torch.exp(_G2_DECAY1 * torch.abs(x - _G2_CENTER1))
        exp2 = _G2_AMP2 * torch.exp(_G2_DECAY2 * torch.abs(x - _G2_CENTER2))
        return exp1 + exp2

    def _g3(x: torch.Tensor) -> torch.Tensor:
        """Basis function 3: 'Flat top' bump with high-frequency ripple."""
        bump_profile = torch.tanh(_G3_TANH_SCALE * (x - _G3_BUMP_CENTER1)) - torch.tanh(_G3_TANH_SCALE * (x - _G3_BUMP_CENTER2))
        ripple = _G3_RIPPLE_AMP * torch.cos(_G3_RIPPLE_FREQ_BASE * x + _G3_RIPPLE_CHIRP_RATE * x**2)
        return _G3_BUMP_AMPLITUDE * 0.5 * bump_profile * (1 + ripple)

    def _g4(x: torch.Tensor) -> torch.Tensor:
        """Basis function 4: Growing oscillations with a non-symmetric envelope."""
        envelope_sigmoid_part = torch.sigmoid(_G4_ENV_SIG_SCALE * x + _G4_ENV_SIG_SHIFT)
        envelope_linear_part = (_G4_ENV_LINEAR_A * x + _G4_ENV_LINEAR_B)
        envelope = envelope_sigmoid_part * envelope_linear_part
        
        frequency_modulation = _G4_OSC_FM_AMP * torch.cos(_G4_OSC_FM_FREQ * x)
        oscillation = torch.sin(_G4_OSC_BASE_FREQ * x + frequency_modulation)
        return envelope * oscillation

def plot_cool_basis_functions():
    """
    Generates and saves plots of the 'cool basis' functions.
    """
    # Define the x range for plotting
    x_min, x_max = -1.5, 1.5
    num_points = 400
    x_values = torch.linspace(x_min, x_max, num_points)

    # Calculate basis function values
    g2_values = _g2(x_values)
    g3_values = _g3(x_values)
    g4_values = _g4(x_values)

    basis_functions = [
        {"name": "$g_2(x)$: Diff. of Exp. Decays", "values": g2_values, "color": "green"},
        {"name": "$g_3(x)$: Flat Bump + Ripple", "values": g3_values, "color": "red"},
        {"name": "$g_4(x)$: Growing Osc. w/ Envelope", "values": g4_values, "color": "purple"},
    ]

    # Create output directory if it doesn't exist
    output_dir = "Plots/basis_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Plot each basis function on a separate subplot
    fig_separate, axs = plt.subplots(len(basis_functions), 1, figsize=(10, 4 * len(basis_functions)), sharex=True)
    fig_separate.suptitle("Individual 'Cool Basis' Functions $g_i(x)$", fontsize=16)

    for i, bf in enumerate(basis_functions):
        axs[i].plot(x_values.numpy(), bf["values"].numpy(), label=bf["name"], color=bf["color"], linewidth=2)
        axs[i].set_ylabel("$g_i(x)$", fontsize=12)
        axs[i].set_title(bf["name"], fontsize=14)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].tick_params(axis='both', which='major', labelsize=10)

    axs[-1].set_xlabel("$x$", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for suptitle
    separate_plot_path = os.path.join(output_dir, "cool_basis_functions_separate.png")
    plt.savefig(separate_plot_path)
    plt.close(fig_separate)
    print(f"Saved separate basis function plots to: {separate_plot_path}")

    # Plot all basis functions on a single plot
    plt.figure(figsize=(12, 7))
    for bf in basis_functions:
        plt.plot(x_values.numpy(), bf["values"].numpy(), label=bf["name"], linewidth=2, color=bf["color"])
    
    plt.title("Combined 'Cool Basis' Functions $g_i(x)$", fontsize=16)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5) # Add x-axis line
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    combined_plot_path = os.path.join(output_dir, "cool_basis_functions_combined.png")
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Saved combined basis function plot to: {combined_plot_path}")

if __name__ == "__main__":
    plot_cool_basis_functions()
