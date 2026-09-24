"""Full HyperGSR model; legacy helper imports remain available here."""

import torch
import torch.nn as nn

from src.dual_graph_utils import create_dual_graph_feature_matrix
from src.hyper_graph_utils import (
    build_upper_triu_edges, build_incidence_sparse, build_incidence_indices,
    _min_max_normalize, zscore_coords, pairwise_edge_distance, normalize_scalar_vector,
)
from .hyper_layers import TargetEdgeInitializer, TwoStepBipartiteLayer, HyperDualLearner


class HyperGSR(nn.Module):
    """
    Hyper-dual GSR with geometric priors (coords shared across subjects):
      - Use ROI xyz once -> cache:
          * standardized ROI features (optional, for attention backends),
          * all edge distances & normalized variants,
          * incidence structures.
      - Distance -> small MLP -> concat to edge-node features.
      - Optional: use distance as edge_attr for TransformerConv ('trans' + edge_dim=1).
    """
    def __init__(self, config=None, *, n_source_nodes=None,
                 n_target_nodes=None, model_config=None):
        """Use explicit graph sizes/model_config; config is the legacy adapter."""
        super().__init__()
        if config is not None:
            n_source_nodes = config.dataset.n_source_nodes
            n_target_nodes = config.dataset.n_target_nodes
            model_config = config.model

        # ---- A) initializer ----
        self.target_edge_initializer = TargetEdgeInitializer(
            n_source_nodes,
            n_target_nodes,
            num_heads=model_config.target_edge_initializer.num_heads,
            edge_dim=model_config.target_edge_initializer.edge_dim,
            dropout=model_config.target_edge_initializer.dropout,
            beta=model_config.target_edge_initializer.beta,
        )

        # ---- Geometry / hyperedge feature flags ----
        hd_conf = getattr(model_config, "hyper_dual_learner", None)

        # One switch controls both distance features and ROI-coordinate features.
        self.use_geo_priors = getattr(hd_conf, "use_geo_priors", False)
        self.edge_geo_dim      = getattr(hd_conf, "edge_geo_dim", 8)
        self.dist_norm_mode    = getattr(hd_conf, "dist_norm", "zscore")  # 'zscore'|'minmax'|'log1p_minmax'

        hidden   = getattr(hd_conf, "hidden_dim", 32)
        dropout  = getattr(hd_conf, "dropout", 0.0)
        mode     = getattr(hd_conf, "mode", "spmm")
        heads    = getattr(hd_conf, "heads", 4)
        use_h_emb= (getattr(hd_conf, "use_hyper_emb", True) and mode in ['sage','gat','trans'] and not self.use_geo_priors)
        edge_dim = getattr(hd_conf, "edge_dim", 0)    # set 1 for transformer edge_attr
        use_shrink_output = getattr(hd_conf, "use_shrink_output", True)
        shrink_threshold = getattr(hd_conf, "shrink_threshold", 0.01)

        # Edge-distance -> feature MLP
        if self.use_geo_priors:
            self.edge_geo_mlp = nn.Sequential(
                nn.Linear(1, self.edge_geo_dim),
                nn.ReLU(),
                nn.Linear(self.edge_geo_dim, self.edge_geo_dim),
            )
            extra_in = self.edge_geo_dim
        else:
            self.edge_geo_mlp = None
            extra_in = 0

        # ROI xyz -> hyperedge feature MLP (used only in 'sage'/'gat'/'trans')
        if self.use_geo_priors:
            self.roi_mlp = nn.Sequential(
                nn.Linear(3, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.roi_mlp = None

        in_dim = 1 + extra_in  # base dual feature is 1-D from initializer

        self.hyper_dual = HyperDualLearner(
            n_target_nodes=n_target_nodes,
            in_dim=in_dim,
            hidden_dim=hidden,
            dropout=dropout,
            mode=mode,
            heads=heads,
            use_hyper_emb=use_h_emb,
            edge_dim=edge_dim,
            use_shrink_output=use_shrink_output,
            shrink_threshold=shrink_threshold,
            use_geo_priors=self.use_geo_priors,
        )

        # cache for geometry based on shared coords (filled on first forward)
        edges_ij, _ = build_upper_triu_edges(n_target_nodes)
        self.register_buffer("edges_ij", edges_ij)  # [M,2]

        self._roi_coords_std = None   # [n_t,3], standardized & cached
        self._dist_norm      = None   # [M,1], normalized & cached
        self.mode = mode
        self.edge_dim = edge_dim

    # -- helper to set & cache shared coordinates once --
    def _ensure_geometry_cache(self, roi_coords: torch.Tensor, device):
        if self._roi_coords_std is not None:
            # already cached (shared across subjects)
            return

        if roi_coords is None:
            raise ValueError("HyperGSR: roi_coords must be provided once (shared across subjects).")

        roi_coords = roi_coords.detach().to(device).float()        # [n_t,3]
        self._roi_coords_std = zscore_coords(roi_coords)           # global standardization

        # distances on CPU then move to device
        d = pairwise_edge_distance(self.edges_ij.to(device), self._roi_coords_std)  # [M,1]
        self._dist_norm = normalize_scalar_vector(d, mode=self.dist_norm_mode)      # [M,1], cached

    def forward(self, source_pyg, target_mat, roi_coords: torch.Tensor = None):
        """
        Args:
            source_pyg: PyG LR graph
            target_mat: [n_t, n_t] HR ground-truth adjacency
            roi_coords: [n_t, 3] shared ROI xyz (pass once; it will be cached)
        Returns:
            dual_pred_x: [M,1] predicted upper-tri weights in [0,1]
            dual_target_x: [M,1] ground-truth upper-tri vector
        """
        device = source_pyg.x.device

        # Cache geometry (coords & distances) once (shared across all samples)
        self._ensure_geometry_cache(roi_coords, device)

        # A) LR -> HR hot-start as dual-node base feature
        x_dual = self.target_edge_initializer(source_pyg).to(device)   # [M,1]

        # B) concat edge distance feature (cached)
        x_hyper = None
        e2h_attr = None
        h2e_attr = None

        if self.use_geo_priors and (self._dist_norm is not None):
            edge_feat = self.edge_geo_mlp(self._dist_norm)             # [M, edge_geo_dim]
            x_dual = torch.cat([x_dual, edge_feat], dim=-1)            # [M, 1+edge_geo_dim]

            # optional: use distance as edge_attr for transformer attention
            if (self.mode == "trans") and (self.edge_dim > 0):
                d_rep = self._dist_norm.repeat_interleave(2, dim=0)    # [2M,1] for e2h & h2e
                e2h_attr = d_rep
                h2e_attr = d_rep

        # C) hyperedge (ROI) features from coords (attention backends only)
        if (self.roi_mlp is not None) and (self._roi_coords_std is not None):
            x_hyper = self.roi_mlp(self._roi_coords_std.to(device))    # [n_t, hidden]

        # D) hyper-dual learner
        dual_pred_x = self.hyper_dual(x_dual, x_hyper, e2h_attr, h2e_attr)  # [M,1], min-max to [0,1]

        # E) supervision vector from HR adjacency
        dual_target_x = create_dual_graph_feature_matrix(target_mat).to(device)     # [M,1]
        return dual_pred_x, dual_target_x
