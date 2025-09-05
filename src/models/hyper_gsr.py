import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    TransformerConv,
    GraphNorm,
    SAGEConv,
    GATConv,
    TransformerConv as PYGTransformerConv,
)

from src.dual_graph_utils import create_dual_graph_feature_matrix


# --------------------------- utilities ---------------------------

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


# ----------------------- Part A: initializer -----------------------

class TargetEdgeInitializer(nn.Module):
    """
    Same as STP-GSR:
      1) TransformerConv on LR graph -> node embeddings
      2) X^T X hot-start on HR
      3) take upper-tri entries as dual-node features
    """
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1,
                 dropout=0.2, beta=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0
        self.conv1 = TransformerConv(
            in_channels=n_source_nodes,
            out_channels=n_target_nodes // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=beta,
        )
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        xt = x.T @ x                                  # [n_t, n_t]
        ut_mask = torch.triu(torch.ones_like(xt), diagonal=1).bool()
        x_dual = torch.masked_select(xt, ut_mask).view(-1, 1)  # [M,1]
        return x_dual


# -------- Part B: two-step bipartite layer with switchable backends --------

class TwoStepBipartiteLayer(nn.Module):
    """
    Node(=edge-nodes) -> Hyperedge(=ROIs) -> Node two-step message passing.

    mode:
      - 'spmm' : H = D_h^{-1} B^T Lin(X);  X' = D_e^{-1} B Lin(H)
      - 'sage' : two SAGEConv passes on the bipartite graph
      - 'gat'  : two GATConv  passes on the bipartite graph
      - 'trans': two TransformerConv passes (multi-head, supports edge_attr if edge_dim>0)
    """
    def __init__(self, n_t: int, hidden_dim: int = 32,
                 mode: str = 'spmm', heads: int = 4, dropout: float = 0.0,
                 use_hyper_emb: bool = True, edge_dim: int = 0,
                 use_edge_distance: bool = False, use_hyper_coords: bool = False):
        super().__init__()
        assert mode in ['spmm', 'sage', 'gat', 'trans']
        self.mode = mode
        self.n_t = n_t
        self.hidden = hidden_dim
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.heads = heads
        self.edge_dim = edge_dim
        self.use_edge_distance = use_edge_distance
        self.use_hyper_coords = use_hyper_coords

        # cache structures (on CPU buffers, moved on demand)
        B_cpu, edges_ij = build_incidence_sparse(n_t, device=torch.device('cpu'))
        self.register_buffer("_B_cpu", B_cpu)
        self._B = None

        e2h_cpu, h2e_cpu, M, edges_ij2 = build_incidence_indices(n_t, device=torch.device('cpu'))
        self.register_buffer("e2h_cpu", e2h_cpu)
        self.register_buffer("h2e_cpu", h2e_cpu)
        self._e2h = None
        self._h2e = None

        assert torch.equal(edges_ij, edges_ij2)
        self.register_buffer("edges_ij", edges_ij)
        self.M = M

        if self.mode == 'spmm':
            self.lin_in  = nn.Linear(hidden_dim, hidden_dim)
            self.lin_out = nn.Linear(hidden_dim, hidden_dim)

        self.hyper_init = nn.Embedding(n_t, hidden_dim) if (use_hyper_emb and self.mode in ['sage','gat','trans']) else None

        if self.mode == 'sage':
            self.u2h = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
            self.h2u = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        elif self.mode == 'gat':
            self.u2h = GATConv((hidden_dim, hidden_dim), hidden_dim,
                               heads=heads, concat=False, dropout=dropout, add_self_loops=False)
            self.h2u = GATConv((hidden_dim, hidden_dim), hidden_dim,
                               heads=heads, concat=False, dropout=dropout, add_self_loops=False)
        elif self.mode == 'trans':
            if edge_dim > 0:
                self.u2h = PYGTransformerConv((hidden_dim, hidden_dim), hidden_dim // heads,
                                              heads=heads, beta=False, dropout=dropout, edge_dim=edge_dim)
                self.h2u = PYGTransformerConv((hidden_dim, hidden_dim), hidden_dim // heads,
                                              heads=heads, beta=False, dropout=dropout, edge_dim=edge_dim)
                # --- [新增] 当未显式提供 edge_attr 时的可学习 fallback ---
                # 纯离散：ROI 身份、端点角色(LEFT/RIGHT)、边桶打散
                if not self.use_edge_distance and not self.use_hyper_coords:
                    self._edgeattr_roi_dim   = 8
                    self._edgeattr_role_dim  = 2
                    self._edgeattr_bucket_dim= 8
                    self.roi_id_emb   = nn.Embedding(n_t, self._edgeattr_roi_dim)
                    self.role_emb     = nn.Embedding(2,   self._edgeattr_role_dim)      # 0: LEFT(i) / 1: RIGHT(j)
                    self.bucket_emb   = nn.Embedding(64,  self._edgeattr_bucket_dim)     # 64 个桶，不依赖坐标
                    in_dim = self._edgeattr_roi_dim + self._edgeattr_role_dim + self._edgeattr_bucket_dim
                    self.edge_attr_proj = nn.Sequential(
                        nn.Linear(in_dim, edge_dim),
                        nn.LayerNorm(edge_dim)
                    )
            else:
                # fallback to GAT if no edge_attr is needed
                self.u2h = GATConv((hidden_dim, hidden_dim), hidden_dim,
                                   heads=heads, concat=False, dropout=dropout, add_self_loops=False)
                self.h2u = GATConv((hidden_dim, hidden_dim), hidden_dim,
                                   heads=heads, concat=False, dropout=dropout, add_self_loops=False)

    def _get_B(self, device):
        if (self._B is None) or (self._B.device != device):
            self._B = self._B_cpu.to(device)
        return self._B

    def _get_edge_index(self, device):
        if (self._e2h is None) or (self._e2h.device != device):
            self._e2h = self.e2h_cpu.to(device)
            self._h2e = self.h2e_cpu.to(device)
        return self._e2h, self._h2e

    def _build_learned_edge_attrs(self, device):
        """
        当外部未提供 edge_attr 时，基于离散身份自动生成 (不使用任何坐标)。
        返回: attr_e2h, attr_h2e，形状均为 [2M, edge_dim]
        """
        e2h, h2e = self._get_edge_index(device)  # [2, 2M]
        # ----- e2h: edge_id -> roi_id -----
        e_ids = e2h[0].long()    # [2M]
        r_ids = e2h[1].long()    # [2M]

        # 端点角色：ROI 是否是该边的左端点 i；否则为 RIGHT(j)
        left_endpoint = self.edges_ij[e_ids, 0]          # [2M]
        role_idx = (r_ids != left_endpoint).long()       # 0: LEFT(i), 1: RIGHT(j)

        # 边桶：用 e_id 做模运算，纯打散
        bucket_idx = (e_ids % self.bucket_emb.num_embeddings).long()

        feat_e2h = torch.cat([
            self.roi_id_emb(r_ids),                      # ROI 身份
            self.role_emb(role_idx),                     # 端点角色
            self.bucket_emb(bucket_idx)                  # 边桶
        ], dim=-1)
        attr_e2h = self.edge_attr_proj(feat_e2h)         # → [2M, edge_dim]

        # ----- h2e: roi_id -> edge_id -----
        r_ids2 = h2e[0].long()
        e_ids2 = h2e[1].long()
        left2  = self.edges_ij[e_ids2, 0]
        role2  = (r_ids2 != left2).long()
        bucket2= (e_ids2 % self.bucket_emb.num_embeddings).long()

        feat_h2e = torch.cat([
            self.roi_id_emb(r_ids2),
            self.role_emb(role2),
            self.bucket_emb(bucket2)
        ], dim=-1)
        attr_h2e = self.edge_attr_proj(feat_h2e)         # → [2M, edge_dim]

        return attr_e2h, attr_h2e


    @torch.no_grad()
    def _degrees(self, B: torch.Tensor):
        deg_e = torch.sparse.sum(B, dim=1).to_dense().clamp_min_(1.0)
        deg_h = torch.sparse.sum(B, dim=0).to_dense().clamp_min_(1.0)
        return deg_e, deg_h

    def forward(self, X_e: torch.Tensor, X_h: torch.Tensor = None,
                e2h_edge_attr: torch.Tensor = None, h2e_edge_attr: torch.Tensor = None):
        device = X_e.device

        if self.mode == 'spmm':
            B = self._get_B(device)
            deg_e, deg_h = self._degrees(B)
            Z  = self.lin_in(X_e)                               # [M, hidden]
            H  = torch.sparse.mm(B.transpose(0, 1), Z)          # [n_t, hidden]
            H  = H / deg_h.unsqueeze(-1)
            H  = self.drop(H)
            Z2 = torch.sparse.mm(B, H)                          # [M, hidden]
            Z2 = Z2 / deg_e.unsqueeze(-1)
            X_out = self.lin_out(Z2)                            # [M, hidden]
            return X_out

        e2h, h2e = self._get_edge_index(device)

        if X_h is None:
            if self.hyper_init is not None:
                H = self.hyper_init.weight                      # [n_t, hidden]
            else:
                H = torch.zeros(self.n_t, self.hidden, device=device)
        else:
            H = X_h

        if self.mode == 'sage':
            H = self.u2h((X_e, H), e2h)
            H = self.drop(H)
            X_out = self.h2u((H, X_e), h2e)
            return X_out

        if self.mode == 'gat':
            H = self.u2h((X_e, H), e2h)
            H = self.drop(H)
            X_out = self.h2u((H, X_e), h2e)
            return X_out

        # 'trans'
        if self.edge_dim > 0:
            # 若未显式提供 edge_attr，则自动生成可学习的离散 edge_attr
            if (e2h_edge_attr is None) or (h2e_edge_attr is None):
                e2h_edge_attr, h2e_edge_attr = self._build_learned_edge_attrs(device)
            H = self.u2h((X_e, H), e2h, edge_attr=e2h_edge_attr)
            H = self.drop(H)
            X_out = self.h2u((H, X_e), h2e, edge_attr=h2e_edge_attr)
        else:
            H = self.u2h((X_e, H), e2h)
            H = self.drop(H)
            X_out = self.h2u((H, X_e), h2e)
        return X_out


# --------------- Part C: Hyper-dual learner (pre -> layer -> head) ---------------

class HyperDualLearner(nn.Module):
    """
    pre (Linear) -> TwoStepBipartiteLayer -> readout -> min-max
    """
    def __init__(self, n_target_nodes: int, in_dim: int,
        hidden_dim: int = 32, dropout: float = 0.0,
        mode: str = 'spmm', heads: int = 4,
        use_hyper_emb: bool = True, edge_dim: int = 0,
        use_edge_distance: bool = False, use_hyper_coords: bool = False,
        use_shrink_output: bool = False, shrink_threshold: float = 0.01):
        super().__init__()
        self.n_t = n_target_nodes
        self.pre = nn.Linear(in_dim, hidden_dim)
        self.layer = TwoStepBipartiteLayer(
            n_t=n_target_nodes,
            hidden_dim=hidden_dim,
            mode=mode,
            heads=heads,
            dropout=dropout,
            use_hyper_emb=use_hyper_emb,
            edge_dim=edge_dim,
            use_edge_distance=use_edge_distance,
            use_hyper_coords=use_hyper_coords,
        )
        self.readout = nn.Linear(hidden_dim, 1)
        self.use_shrink_output = use_shrink_output

        if use_shrink_output:
            self.shrink = nn.Parameter(torch.tensor(shrink_threshold))

    def forward(self, x_dual: torch.Tensor, x_hyper: torch.Tensor = None,
                e2h_edge_attr: torch.Tensor = None, h2e_edge_attr: torch.Tensor = None):
        x = self.pre(x_dual)                                  # [M, hidden]
        x = self.layer(x, x_hyper, e2h_edge_attr, h2e_edge_attr)
        
        if self.use_shrink_output:
            y_raw = self.readout(x)               # [M, 1]
            lam   = self.shrink.abs()
            y_shr = torch.sign(y_raw) * F.relu(torch.abs(y_raw) - lam)
            y_shr = y_shr.clamp_min(0)
            return _min_max_normalize(y_shr)
        else:
            return _min_max_normalize(self.readout(x))



# ------------------------ Part D: Full model ------------------------

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
    def __init__(self, config):
        super().__init__()
        n_source_nodes = config.dataset.n_source_nodes
        n_target_nodes = config.dataset.n_target_nodes

        # ---- A) initializer ----
        self.target_edge_initializer = TargetEdgeInitializer(
            n_source_nodes,
            n_target_nodes,
            num_heads=config.model.target_edge_initializer.num_heads,
            edge_dim=config.model.target_edge_initializer.edge_dim,
            dropout=config.model.target_edge_initializer.dropout,
            beta=config.model.target_edge_initializer.beta,
        )

        # ---- Geometry / hyperedge feature flags ----
        hd_conf = getattr(config.model, "hyper_dual_learner", None)

        # Edge geometry (distance) -> feature
        self.use_edge_distance = getattr(hd_conf, "use_edge_distance", True)
        self.edge_geo_dim      = getattr(hd_conf, "edge_geo_dim", 8)
        self.dist_norm_mode    = getattr(hd_conf, "dist_norm", "zscore")  # 'zscore'|'minmax'|'log1p_minmax'

        # ROI coords -> hyperedge features (for attention backends)
        self.use_hyper_coords  = getattr(hd_conf, "use_hyper_coords", True)

        hidden   = getattr(hd_conf, "hidden_dim", 32)
        dropout  = getattr(hd_conf, "dropout", 0.0)
        mode     = getattr(hd_conf, "mode", "spmm")
        heads    = getattr(hd_conf, "heads", 4)
        use_h_emb= (getattr(hd_conf, "use_hyper_emb", True) and mode in ['sage','gat','trans'] and not self.use_hyper_coords)
        edge_dim = getattr(hd_conf, "edge_dim", 0)    # set 1 for transformer edge_attr
        use_shrink_output = getattr(hd_conf, "use_shrink_output", True)
        shrink_threshold = getattr(hd_conf, "shrink_threshold", 0.01)

        # Edge-distance -> feature MLP
        if self.use_edge_distance:
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
        if self.use_hyper_coords:
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
            use_edge_distance=self.use_edge_distance,
            use_hyper_coords=self.use_hyper_coords,
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

        if self.use_edge_distance and (self._dist_norm is not None):
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