from __future__ import annotations

import torch
import torch.nn as nn


class StandardMoESequenceModel(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, num_experts: int = 4, obs_dim: int = 10, memory_num_classes: int = 8, control_num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh())
        self.router = nn.Linear(64, num_experts)
        self.experts = nn.ModuleList([nn.GRUCell(64, hidden_dim) for _ in range(num_experts)])
        self.pred_head = nn.Linear(hidden_dim, obs_dim)
        self.mem_head = nn.Linear(hidden_dim, memory_num_classes)
        self.ctrl_head = nn.Linear(hidden_dim, control_num_classes)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.active_node_mask: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor, mem_query: torch.Tensor, danger_flag: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = obs.shape
        device = obs.device
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        pred_logits = []
        mem_logits = []
        ctrl_logits = []
        router_logits = []
        hidden_traj = []
        expert_usage = torch.zeros(self.num_experts, device=device)

        for t in range(seq_len):
            input_t = torch.cat([obs[:, t], mem_query[:, t], danger_flag[:, t]], dim=-1)
            encoded = self.encoder(input_t)
            route_logits = self.router(encoded)
            chosen = torch.argmax(route_logits, dim=-1)
            next_hidden = torch.zeros_like(hidden)
            for expert_id, expert in enumerate(self.experts):
                mask = chosen == expert_id
                if mask.sum() == 0:
                    continue
                expert_hidden = expert(encoded[mask], hidden[mask])
                next_hidden[mask] = expert_hidden
                expert_usage[expert_id] += mask.sum()
            hidden = next_hidden
            if self.active_node_mask is not None:
                hidden = hidden.masked_fill(~self.active_node_mask.view(1, -1), 0.0)
            pred_logits.append(self.pred_head(hidden))
            mem_logits.append(self.mem_head(hidden))
            ctrl_logits.append(self.ctrl_head(hidden))
            router_logits.append(route_logits)
            hidden_traj.append(hidden)
        return {
            "pred_logits": torch.stack(pred_logits, dim=1),
            "mem_logits": torch.stack(mem_logits, dim=1),
            "ctrl_logits": torch.stack(ctrl_logits, dim=1),
            "router_logits": torch.stack(router_logits, dim=1),
            "expert_usage": expert_usage,
            "hidden_traj": torch.stack(hidden_traj, dim=1),
        }
