"""HyperGSR initialization, bipartite message passing and readout layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm, SAGEConv, GATConv
from torch_geometric.nn import TransformerConv as PYGTransformerConv

from src.hyper_graph_utils import (
    build_incidence_sparse, build_incidence_indices, _min_max_normalize,
)


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
                 use_geo_priors: bool = False):
        super().__init__()
        assert mode in ['spmm', 'sage', 'gat', 'trans']
        self.mode = mode
        self.n_t = n_t
        self.hidden = hidden_dim
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.heads = heads
        self.edge_dim = edge_dim
        self.use_geo_priors = use_geo_priors

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
                if not self.use_geo_priors:
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
        use_geo_priors: bool = False,
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
            use_geo_priors=use_geo_priors,
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



