from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryInteriorSparseRNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_surface: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        coords: torch.Tensor,
        node_types: torch.Tensor,
        input_node_ids: torch.Tensor,
        pred_readout_ids: torch.Tensor,
        ctrl_readout_ids: torch.Tensor,
        input_dim: int = 12,
        obs_dim: int = 10,
        memory_num_classes: int = 8,
        control_num_classes: int = 5,
        ctrl_readout_strategy: str = "legacy_fixed_node_readout",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_surface = num_surface
        self.ctrl_readout_strategy = ctrl_readout_strategy
        self.register_buffer("coords", coords)
        self.register_buffer("node_types", node_types)
        self.register_buffer("edge_index", edge_index)
        self.edge_weight = nn.Parameter(edge_weight.clone())
        self.bias = nn.Parameter(torch.zeros(num_nodes))
        self.W_in = nn.Parameter(torch.randn(input_dim, input_node_ids.numel()) * 0.05)
        self.pred_head = nn.Linear(pred_readout_ids.numel(), obs_dim)
        self.mem_head = nn.Linear(num_nodes, memory_num_classes)
        # Control head strategy selection
        if ctrl_readout_strategy == "distributed_projection":
            # Control head reads from ALL hidden nodes (learned projection)
            self.ctrl_head = nn.Linear(num_nodes, control_num_classes)
            self._ctrl_readout_all = True
        else:
            # Legacy: control head reads from fixed subset
            self.ctrl_head = nn.Linear(ctrl_readout_ids.numel(), control_num_classes)
            self._ctrl_readout_all = False
        self.register_buffer("input_node_ids", input_node_ids)
        self.register_buffer("pred_readout_ids", pred_readout_ids)
        self.register_buffer("ctrl_readout_ids", ctrl_readout_ids)
        self.last_hidden: torch.Tensor | None = None
        self.activity_buffer: torch.Tensor | None = None
        self.active_node_mask: torch.Tensor | None = None

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.last_hidden = torch.zeros(batch_size, self.num_nodes, device=device)

    def get_active_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.edge_index, self.edge_weight

    def set_active_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        self.edge_index = edge_index
        with torch.no_grad():
            self.edge_weight = nn.Parameter(edge_weight.clone())

    def update_activity_buffer(self, hidden_traj: torch.Tensor) -> None:
        flattened = hidden_traj.reshape(-1, hidden_traj.shape[-1]).detach()
        self.activity_buffer = flattened if self.activity_buffer is None else torch.cat([self.activity_buffer, flattened], dim=0)[-4096:]

    def compute_readout_pools(self, hidden_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_pool = hidden_t[:, self.pred_readout_ids]
        mem_pool = hidden_t
        if self._ctrl_readout_all:
            ctrl_pool = hidden_t  # distributed projection: use all nodes
        else:
            ctrl_pool = hidden_t[:, self.ctrl_readout_ids]
        return pred_pool, mem_pool, ctrl_pool

    def ctrl_readout_from_hidden(self, hidden_traj: torch.Tensor) -> torch.Tensor:
        """Recompute control logits from hidden trajectory.

        Strategy-aware: handles both legacy and distributed projection.
        """
        if self._ctrl_readout_all:
            return self.ctrl_head(hidden_traj)
        else:
            return self.ctrl_head(hidden_traj[:, :, self.ctrl_readout_ids])

    def _message_pass(self, hidden_t: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_t.shape[0]
        messages = torch.zeros(batch_size, self.num_nodes, device=hidden_t.device)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        weighted_hidden = hidden_t[:, src] * self.edge_weight.unsqueeze(0)
        messages.index_add_(1, dst, weighted_hidden)
        return messages

    def forward(self, obs: torch.Tensor, mem_query: torch.Tensor, danger_flag: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = obs.shape
        device = obs.device
        self.reset_state(batch_size, device)
        assert self.last_hidden is not None
        hidden_t = self.last_hidden
        pred_logits = []
        mem_logits = []
        ctrl_logits = []
        hidden_traj = []

        for t in range(seq_len):
            input_t = torch.cat([obs[:, t], mem_query[:, t], danger_flag[:, t]], dim=-1)
            rec_message = self._message_pass(hidden_t)
            input_message = torch.zeros(batch_size, self.num_nodes, device=device)
            input_message[:, self.input_node_ids] = input_t @ self.W_in
            preact = rec_message + input_message + self.bias.unsqueeze(0)
            hidden_t = torch.tanh(preact)
            if self.active_node_mask is not None:
                hidden_t = hidden_t.masked_fill(~self.active_node_mask.view(1, -1), 0.0)
            pred_pool, mem_pool, ctrl_pool = self.compute_readout_pools(hidden_t)
            pred_logits.append(self.pred_head(pred_pool))
            mem_logits.append(self.mem_head(mem_pool))
            ctrl_logits.append(self.ctrl_head(ctrl_pool))
            hidden_traj.append(hidden_t)

        hidden_stack = torch.stack(hidden_traj, dim=1)
        self.update_activity_buffer(hidden_stack)
        return {
            "pred_logits": torch.stack(pred_logits, dim=1),
            "mem_logits": torch.stack(mem_logits, dim=1),
            "ctrl_logits": torch.stack(ctrl_logits, dim=1),
            "hidden_traj": hidden_stack,
        }


# ===========================================================================
# Localized Sparse Adapter 3D Sparse Model
# Route B Phase-0: sparse adapter bottleneck readout
# ===========================================================================


class LocalizedSparseAdapter3DSparse(nn.Module):
    """3D sparse graph network with localized sparse adapter readout.

    Architecture:
      - N graph nodes with 3D spatial positions and message passing.
      - No fixed ctrl_readout_ids.
      - No dense all-node projection to control head.
      - K virtual adapter channels, each reading from top-k graph nodes.
      - Control head reads from flattened adapter output (not graph nodes).
      - Adapter virtual units are NOT region candidates.
    """

    def __init__(
        self,
        num_nodes: int,
        num_surface: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        coords: torch.Tensor,
        node_types: torch.Tensor,
        input_node_ids: torch.Tensor,
        pred_readout_ids: torch.Tensor,
        input_dim: int = 12,
        obs_dim: int = 10,
        memory_num_classes: int = 8,
        control_num_classes: int = 5,
        num_adapter_channels: int = 4,
        adapter_topk_nodes: int = 4,
        ctrl_readout_strategy: str = "localized_sparse_adapter_readout",
        adapter_sparsity_regularizer: bool = True,
        adapter_entropy_regularizer: bool = True,
        adapter_l1_weight: float = 1e-4,
        adapter_entropy_weight: float = 1e-4,
        adapter_temperature: float = 1.0,
        adapter_convergence_regularizer: bool = False,
        adapter_channel_diversity_weight: float = 0.01,
        adapter_entropy_floor_weight: float = 0.005,
        adapter_min_union_nodes: int = 4,
        adapter_regularizer_diagnostic_only: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_surface = num_surface
        self.ctrl_readout_strategy = ctrl_readout_strategy
        self.num_adapter_channels = num_adapter_channels
        self.adapter_topk_nodes = adapter_topk_nodes
        self.adapter_sparsity_regularizer = adapter_sparsity_regularizer
        self.adapter_entropy_regularizer = adapter_entropy_regularizer
        self.adapter_l1_weight = adapter_l1_weight
        self.adapter_entropy_weight = adapter_entropy_weight
        self.adapter_temperature = adapter_temperature
        self.adapter_convergence_regularizer = adapter_convergence_regularizer
        self.adapter_channel_diversity_weight = adapter_channel_diversity_weight
        self.adapter_entropy_floor_weight = adapter_entropy_floor_weight
        self.adapter_min_union_nodes = adapter_min_union_nodes
        self.adapter_regularizer_diagnostic_only = adapter_regularizer_diagnostic_only

        self.register_buffer("coords", coords)
        self.register_buffer("node_types", node_types)
        self.register_buffer("edge_index", edge_index)
        self.edge_weight = nn.Parameter(edge_weight.clone())
        self.bias = nn.Parameter(torch.zeros(num_nodes))
        self.W_in = nn.Parameter(torch.randn(input_dim, input_node_ids.numel()) * 0.05)
        self.pred_head = nn.Linear(pred_readout_ids.numel(), obs_dim)
        self.mem_head = nn.Linear(num_nodes, memory_num_classes)

        # Adapter: K virtual channels, each with learned logits over N nodes
        # adapter_logits[k, i] = raw weight of channel k reading from node i
        self.adapter_logits = nn.Parameter(
            torch.randn(num_adapter_channels, num_nodes) * 0.02
        )

        # Control head reads from flattened adapter output (K * 1 = K scalars per node)
        # Since graph nodes are scalar, each adapter channel produces 1 value per time step
        # adapter_z shape: [batch, K]
        # We flatten to [batch, K] and project to control classes
        self.ctrl_head = nn.Linear(num_adapter_channels, control_num_classes)

        # Register buffers for interface tracking
        self.register_buffer("input_node_ids", input_node_ids)
        self.register_buffer("pred_readout_ids", pred_readout_ids)
        # No ctrl_readout_ids - the adapter is the readout mechanism

        # Adapter node mask: 1 for allowed, 0 for excluded (e.g., input and output nodes)
        # Exclude input nodes (0-7) so adapter can't shortcut through inputs.
        # Exclude output/pred_readout nodes (8-15) to prevent interface contamination.
        # Without this, the adapter can read from output interface nodes, creating a
        # shortcut pathway that invalidates oracle calibration (see Phase-3 seed44).
        self._adapter_exclude_input_nodes = True
        self._adapter_exclude_output_nodes = True
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        if self._adapter_exclude_input_nodes:
            node_mask[input_node_ids.long()] = False
        if self._adapter_exclude_output_nodes:
            node_mask[pred_readout_ids.long()] = False
        self.register_buffer("_adapter_node_mask", node_mask)

        # Buffers for adapter statistics (updated each forward pass)
        self.register_buffer("_adapter_weights_buffer", torch.zeros(num_adapter_channels, num_nodes))
        self.register_buffer("_adapter_entropy_buffer", torch.zeros(num_adapter_channels))
        self.register_buffer("_adapter_effective_count_buffer", torch.zeros(num_adapter_channels, dtype=torch.long))
        self.register_buffer("_adapter_topk_indices_buffer", torch.zeros(num_adapter_channels, adapter_topk_nodes, dtype=torch.long))

        self.last_hidden: torch.Tensor | None = None
        self.activity_buffer: torch.Tensor | None = None
        self.active_node_mask: torch.Tensor | None = None

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.last_hidden = torch.zeros(batch_size, self.num_nodes, device=device)

    def get_active_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.edge_index, self.edge_weight

    def set_active_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        self.edge_index = edge_index
        with torch.no_grad():
            self.edge_weight = nn.Parameter(edge_weight.clone())

    def update_activity_buffer(self, hidden_traj: torch.Tensor) -> None:
        flattened = hidden_traj.reshape(-1, hidden_traj.shape[-1]).detach()
        self.activity_buffer = flattened if self.activity_buffer is None else torch.cat([self.activity_buffer, flattened], dim=0)[-4096:]

    def compute_readout_pools(self, hidden_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_pool = hidden_t[:, self.pred_readout_ids]
        mem_pool = hidden_t
        # Control pool: adapter output
        ctrl_pool = self._compute_adapter_output(hidden_t)
        return pred_pool, mem_pool, ctrl_pool

    def _compute_adapter_output(self, hidden_t: torch.Tensor) -> torch.Tensor:
        """Compute sparse adapter readout from hidden states.

        Returns adapter_z: [batch, num_adapter_channels]
        """
        batch_size = hidden_t.shape[0]
        device = hidden_t.device
        K = self.num_adapter_channels
        topk = self.adapter_topk_nodes

        # Apply node mask: set logits of excluded nodes to -inf
        masked_logits = self.adapter_logits.clone()
        if hasattr(self, "_adapter_node_mask"):
            masked_logits[:, ~self._adapter_node_mask] = float("-inf")

        # adapter_logits: [K, N] -> softmax over nodes per channel
        adapter_weights = torch.softmax(
            masked_logits / self.adapter_temperature, dim=-1
        )  # [K, N]

        # Hard top-k: zero out all but top-k nodes per channel
        topk_indices = torch.topk(adapter_weights, k=topk, dim=-1).indices  # [K, topk]
        topk_mask = torch.zeros_like(adapter_weights)  # [K, N]
        topk_mask.scatter_(1, topk_indices, 1.0)

        # Soft top-k weights (preserve relative weights within selected nodes)
        adapter_weights_topk = adapter_weights * topk_mask
        # Re-normalize per channel
        norm = adapter_weights_topk.sum(dim=-1, keepdim=True)  # [K, 1]
        norm = norm.clamp(min=1e-8)
        adapter_weights_topk = adapter_weights_topk / norm

        # Compute adapter_z: [batch, K]
        # Each channel = weighted sum of graph node hidden values
        # hidden_t: [batch, N] (scalar per node)
        adapter_z = torch.einsum("kn,bn->bk", adapter_weights_topk, hidden_t)  # [batch, K]

        # Update statistics buffers
        self._adapter_weights_buffer = adapter_weights_topk.detach()
        self._adapter_topk_indices_buffer = topk_indices.detach()
        self._adapter_entropy_buffer = self._compute_channel_entropy(adapter_weights_topk)
        self._adapter_effective_count_buffer = self._compute_effective_node_count(adapter_weights_topk)

        return adapter_z

    def _compute_channel_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy per adapter channel. [K]"""
        p = weights.clamp(min=1e-12)
        entropy = -(p * torch.log(p)).sum(dim=-1)  # [K]
        return entropy

    def _compute_effective_node_count(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute effective number of nodes per channel (nonzero weight count). [K]"""
        return (weights > 1e-6).sum(dim=-1)  # [K]

    def compute_adapter_convergence_loss(self) -> dict[str, torch.Tensor]:
        """Compute convergence regularizer loss terms.

        Uses pre-topk soft weights only — no outcome leakage.
        Returns dict of differentiable loss terms for logging and training.
        """
        # Pre-topk softmax weights (before hard top-k masking)
        masked_logits = self.adapter_logits.clone()
        if hasattr(self, "_adapter_node_mask"):
            masked_logits[:, ~self._adapter_node_mask] = float("-inf")
        w = torch.softmax(masked_logits / self.adapter_temperature, dim=-1)  # [K, N]

        K = self.num_adapter_channels

        # 1. Channel diversity penalty: penalize overlap between channel weight distributions
        # For K channels, compute pairwise overlap = sum(min(w_i, w_j)) over nodes
        overlap = torch.zeros((), device=w.device)
        pair_count = 0
        for i in range(K):
            for j in range(i + 1, K):
                pairwise_overlap = torch.sum(torch.min(w[i], w[j]))
                overlap = overlap + pairwise_overlap
                pair_count += 1
        if pair_count > 0:
            overlap = overlap / pair_count

        # 2. Entropy floor penalty: prevent single-node collapse
        # Compute per-channel entropy on pre-topk weights
        p = w.clamp(min=1e-12)
        channel_entropy = -(p * torch.log(p)).sum(dim=-1)  # [K]
        # Target entropy: uniform over topk_nodes
        entropy_target = torch.log(
            torch.tensor(self.adapter_topk_nodes, dtype=torch.float, device=w.device)
        )
        entropy_shortfall = torch.clamp(entropy_target - channel_entropy, min=0.0).mean()

        # 3. Union size proxy (diagnostic only — not used in loss)
        # Effective union = sum over nodes of max weight across channels
        max_across_channels = torch.max(w, dim=0).values  # [N]
        effective_union = torch.sum(max_across_channels)

        return {
            "channel_overlap": overlap,
            "entropy_shortfall": entropy_shortfall,
            "effective_union_proxy": effective_union,
        }

    def ctrl_readout_from_hidden(self, hidden_traj: torch.Tensor) -> torch.Tensor:
        """Recompute control logits from hidden trajectory via adapter.

        hidden_traj: [B, T, N] or [B, N]
        Returns ctrl_logits: [B, T, C] or [B, C]
        """
        # Apply node mask
        masked_logits = self.adapter_logits.clone()
        if hasattr(self, "_adapter_node_mask"):
            masked_logits[:, ~self._adapter_node_mask] = float("-inf")
        adapter_weights = torch.softmax(masked_logits / self.adapter_temperature, dim=-1)

        if hidden_traj.dim() == 3:
            B, T, N = hidden_traj.shape
            flat = hidden_traj.reshape(-1, N)  # [B*T, N]
            # Apply top-k mask
            topk_indices = torch.topk(adapter_weights, k=self.adapter_topk_nodes, dim=-1).indices
            topk_mask = torch.zeros_like(adapter_weights)
            topk_mask.scatter_(1, topk_indices, 1.0)
            adapter_weights_topk = adapter_weights * topk_mask
            norm = adapter_weights_topk.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            adapter_weights_topk = adapter_weights_topk / norm
            adapter_z = torch.einsum("kn,bn->bk", adapter_weights_topk, flat)
            return self.ctrl_head(adapter_z).reshape(B, T, -1)  # [B, T, C]
        else:
            adapter_z = torch.einsum("kn,bn->bk", adapter_weights, hidden_traj)
            return self.ctrl_head(adapter_z)

    def recompute_from_hidden(self, hidden_traj: torch.Tensor) -> dict[str, torch.Tensor]:
        """Recompute all outputs from hidden trajectory for patching experiments."""
        device = hidden_traj.device
        if hidden_traj.dim() == 2:
            hidden_traj = hidden_traj.unsqueeze(1)
        B, T, N = hidden_traj.shape

        pred_pool = hidden_traj[:, :, self.pred_readout_ids]  # [B, T, 8]
        pred_logits = self.pred_head(pred_pool)

        mem_logits = self.mem_head(hidden_traj)  # [B, T, mem_classes]

        ctrl_logits = self.ctrl_readout_from_hidden(hidden_traj)

        return {
            "pred_logits": pred_logits,
            "mem_logits": mem_logits,
            "ctrl_logits": ctrl_logits,
            "hidden_traj": hidden_traj,
        }

    def _message_pass(self, hidden_t: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_t.shape[0]
        messages = torch.zeros(batch_size, self.num_nodes, device=hidden_t.device)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        weighted_hidden = hidden_t[:, src] * self.edge_weight.unsqueeze(0)
        messages.index_add_(1, dst, weighted_hidden)
        return messages

    def forward(self, obs: torch.Tensor, mem_query: torch.Tensor, danger_flag: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = obs.shape
        device = obs.device
        self.reset_state(batch_size, device)
        assert self.last_hidden is not None
        hidden_t = self.last_hidden
        pred_logits = []
        mem_logits = []
        ctrl_logits = []
        hidden_traj = []

        for t in range(seq_len):
            input_t = torch.cat([obs[:, t], mem_query[:, t], danger_flag[:, t]], dim=-1)
            rec_message = self._message_pass(hidden_t)
            input_message = torch.zeros(batch_size, self.num_nodes, device=device)
            input_message[:, self.input_node_ids] = input_t @ self.W_in
            preact = rec_message + input_message + self.bias.unsqueeze(0)
            hidden_t = torch.tanh(preact)
            if self.active_node_mask is not None:
                hidden_t = hidden_t.masked_fill(~self.active_node_mask.view(1, -1), 0.0)
            pred_pool, mem_pool, ctrl_pool = self.compute_readout_pools(hidden_t)
            pred_logits.append(self.pred_head(pred_pool))
            mem_logits.append(self.mem_head(mem_pool))
            # ctrl_head reads adapter_z, not node pool directly
            ctrl_logits.append(self.ctrl_head(ctrl_pool))
            hidden_traj.append(hidden_t)

        hidden_stack = torch.stack(hidden_traj, dim=1)
        self.update_activity_buffer(hidden_stack)
        return {
            "pred_logits": torch.stack(pred_logits, dim=1),
            "mem_logits": torch.stack(mem_logits, dim=1),
            "ctrl_logits": torch.stack(ctrl_logits, dim=1),
            "hidden_traj": hidden_stack,
        }
