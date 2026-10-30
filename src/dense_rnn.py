from __future__ import annotations

import torch
import torch.nn as nn


class DenseSequenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        obs_dim: int,
        memory_num_classes: int,
        control_num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, obs_dim)
        self.mem_head = nn.Linear(hidden_dim, memory_num_classes)
        self.ctrl_head = nn.Linear(hidden_dim, control_num_classes)
        self.active_node_mask: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor, mem_query: torch.Tensor, danger_flag: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = obs.shape
        device = obs.device
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        pred_logits = []
        mem_logits = []
        ctrl_logits = []
        hidden_traj = []

        for t in range(seq_len):
            input_t = torch.cat([obs[:, t], mem_query[:, t], danger_flag[:, t]], dim=-1)
            hidden = self.cell(input_t, hidden)
            if self.active_node_mask is not None:
                hidden = hidden.masked_fill(~self.active_node_mask.view(1, -1), 0.0)
            pred_logits.append(self.pred_head(hidden))
            mem_logits.append(self.mem_head(hidden))
            ctrl_logits.append(self.ctrl_head(hidden))
            hidden_traj.append(hidden)

        return {
            "pred_logits": torch.stack(pred_logits, dim=1),
            "mem_logits": torch.stack(mem_logits, dim=1),
            "ctrl_logits": torch.stack(ctrl_logits, dim=1),
            "hidden_traj": torch.stack(hidden_traj, dim=1),
        }
