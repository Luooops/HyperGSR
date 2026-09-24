"""Incidence structures and geometry operations for HyperGSR."""

import torch


def build_upper_triu_edges(n_t: int):
    """Upper-tri edge list (i<j) and size M, order matches STP-GSR flatten."""
    ij = []
    for i in range(n_t):
        for j in range(i + 1, n_t):
            ij.append((i, j))
    edges_ij = torch.tensor(ij, dtype=torch.long)
    return edges_ij, len(ij)


def build_incidence_sparse(n_t: int, device=None):
    """Sparse incidence B: [M, n_t], each row has two 1's at endpoints (i,j)."""
    edges_ij, M = build_upper_triu_edges(n_t)
    if device is None:
        device = edges_ij.device
    rows = torch.arange(M, device=device).repeat_interleave(2)      # [2M]
    cols = edges_ij.reshape(-1).to(device)                          # [2M] -> i1,j1,i2,j2,...
    indices = torch.stack([rows, cols], dim=0)                      # [2, 2M]
    values = torch.ones(2 * M, device=device)
    B = torch.sparse_coo_tensor(indices, values, size=(M, n_t), device=device).coalesce()
    return B, edges_ij


def build_incidence_indices(n_t: int, device=None):
    """Bipartite edge indices for PyG convs: edge-nodes <-> hyperedges (ROIs)."""
    edges_ij, M = build_upper_triu_edges(n_t)
    if device is None:
        device = edges_ij.device
    e_ids = torch.arange(M, device=device).repeat_interleave(2)     # [2M] : 0,0,1,1,2,2,...
    r_ids = edges_ij.reshape(-1).to(device)                         # [2M] : i1,j1,i2,j2,...
    e2h = torch.stack([e_ids, r_ids], dim=0)                        # [2, 2M]
    h2e = torch.stack([r_ids, e_ids], dim=0)                        # [2, 2M]
    return e2h, h2e, M, edges_ij


def _min_max_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Global min-max normalize to [0,1] (STP-GSR style)."""
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def zscore_coords(coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global z-score per axis (same for all samples since coords are shared).
    coords: [n_t, 3]
    """
    mu = coords.mean(dim=0, keepdim=True)
    std = coords.std(dim=0, keepdim=True) + eps
    return (coords - mu) / std


def pairwise_edge_distance(edges_ij: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Euclidean distance for each upper-tri edge (i,j). Returns [M,1]."""
    i = edges_ij[:, 0]
    j = edges_ij[:, 1]
    d = torch.linalg.norm(coords[j] - coords[i], dim=-1, keepdim=True)  # [M,1]
    return d


def normalize_scalar_vector(x: torch.Tensor, mode: str = "zscore", eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a 1-D scalar vector globally (since coords are shared).
    mode = 'zscore' | 'minmax' | 'log1p_minmax'
    """
    if mode == "minmax":
        return _min_max_normalize(x, eps)
    if mode == "log1p_minmax":
        x = torch.log1p(x)
        return _min_max_normalize(x, eps)
    # default: global z-score
    mu = x.mean()
    std = x.std(unbiased=False) + eps
    return (x - mu) / std


