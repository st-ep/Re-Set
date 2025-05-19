import torch
import matplotlib.pyplot as plt
import os

def calculate_setonet_trunk_orthogonality(model, x_basis, p_dim, device):
    """
    Calculates an orthogonality metric for the SetONet trunk basis functions.
    Metric: Frobenius norm of (Gramian(normalized_basis) - Identity). Lower is better.
    Returns the orthogonality error and the Gramian matrix.
    """
    if not hasattr(model, 'trunk') or p_dim <= 0:
        return None, None

    model.eval() # Ensure model is in eval mode for this calculation
    with torch.no_grad():
        # x_basis is [N_points, 1]. Trunk expects [Batch, N_points, 1]
        trunk_input = x_basis.unsqueeze(0).to(device) # Shape [1, N_points, 1]
        
        # Output is [Batch, N_points, p * output_size_tgt]
        # Assuming output_size_tgt = 1 for this problem, so [1, N_points, p]
        basis_on_x = model.trunk(trunk_input)
        basis_on_x = basis_on_x.squeeze(0) # Shape [N_points, p]

        if basis_on_x.shape[1] != p_dim:
            print(f"Warning: Trunk output dimension {basis_on_x.shape[1]} does not match p_dim {p_dim}. Skipping orthogonality check.")
            return None, None

        # Normalize each basis function (column)
        normalized_basis = torch.zeros_like(basis_on_x)
        for i in range(p_dim):
            col = basis_on_x[:, i]
            norm = torch.norm(col, p=2)
            if norm > 1e-8: # Avoid division by zero
                normalized_basis[:, i] = col / norm
            else:
                normalized_basis[:, i] = col # Keep as is if norm is ~0
        
        # Compute Gramian matrix: G_ij = <psi_i_norm, psi_j_norm>
        # gramian = torch.matmul(normalized_basis.T, normalized_basis) * (x_basis[1] - x_basis[0]) # Optional: scale by dx for integral approx.
        gramian = torch.matmul(normalized_basis.T, normalized_basis)

        # Orthogonality metric: Frobenius norm of (Gramian - Identity)
        identity = torch.eye(p_dim, device=device)
        ortho_error = torch.norm(gramian - identity, p='fro')
        
        return ortho_error.item(), gramian.cpu()

def plot_setonet_trunk_orthogonality(
    epochs: list,
    scores: list,
    log_dir: str,
    title: str = "SetONet Trunk Orthogonality"
) -> None:
    """
    Plots the evolution of the SetONet trunk orthogonality score.
    """
    if not epochs or not scores:
        print("No orthogonality data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, scores, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Orthogonality Score (Frobenius Norm)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "setonet_trunk_orthogonality.png")
    try:
        plt.savefig(plot_path)
        print(f"Trunk orthogonality plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving trunk orthogonality plot: {e}")
    plt.close() 