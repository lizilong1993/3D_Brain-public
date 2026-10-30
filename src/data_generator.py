from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .task_space_ecologies import rotation_matrix
from .utils import quantize_to_bins


@dataclass
class WorldBatch:
    obs: torch.Tensor
    mem_query: torch.Tensor
    danger_flag: torch.Tensor
    pred_target: torch.Tensor
    mem_target: torch.Tensor
    ctrl_target: torch.Tensor
    mem_mask: torch.Tensor
    ctrl_mask: torch.Tensor
    state_c: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs,
            "mem_query": self.mem_query,
            "danger_flag": self.danger_flag,
            "pred_target": self.pred_target,
            "mem_target": self.mem_target,
            "ctrl_target": self.ctrl_target,
            "mem_mask": self.mem_mask,
            "ctrl_mask": self.ctrl_mask,
            "state_c": self.state_c,
        }


@dataclass
class TaskSpaceBatch:
    input: torch.Tensor
    task_coords: torch.Tensor
    velocity: torch.Tensor
    heading_angle: torch.Tensor
    goal_coords: torch.Tensor
    pred_target: torch.Tensor
    mem_target: torch.Tensor
    mem_mask: torch.Tensor
    ctrl_target: torch.Tensor
    ctrl_mask: torch.Tensor
    control_demand: torch.Tensor
    metadata: dict

    def as_dict(self) -> dict:
        return {
            "input": self.input,
            "task_coords": self.task_coords,
            "velocity": self.velocity,
            "heading_angle": self.heading_angle,
            "goal_coords": self.goal_coords,
            "pred_target": self.pred_target,
            "mem_target": self.mem_target,
            "mem_mask": self.mem_mask,
            "ctrl_target": self.ctrl_target,
            "ctrl_mask": self.ctrl_mask,
            "control_demand": self.control_demand,
            "metadata": self.metadata,
        }


class SequenceWorldGenerator:
    def __init__(
        self,
        seq_len: int,
        obs_noise_std: float,
        memory_delay: int,
        danger_threshold: float,
        world_type: str,
        control_action_bins: list[float],
        device: torch.device,
        cue_probability: float = 0.08,
        shock_probability: float = 0.08,
        shock_scale_low: float = 0.55,
        shock_scale_high: float = 0.80,
        control_drift: float = 1.08,
        control_gain: float = 0.90,
        control_noise_std: float = 0.03,
    ) -> None:
        self.seq_len = seq_len
        self.obs_noise_std = obs_noise_std
        self.memory_delay = memory_delay
        self.danger_threshold = danger_threshold
        self.world_type = world_type
        self.device = device
        self.cue_probability = cue_probability
        self.shock_probability = shock_probability
        self.shock_scale_low = shock_scale_low
        self.shock_scale_high = shock_scale_high
        self.control_drift = control_drift
        self.control_gain = control_gain
        self.control_noise_std = control_noise_std
        self.action_bins = torch.tensor(control_action_bins, dtype=torch.float32, device=device)

        self.obs_matrix = torch.tensor(
            [
                [0.8, 0.1, 0.0, 0.0],
                [0.0, 0.7, 0.2, 0.0],
                [0.1, 0.0, 0.9, 0.1],
                [0.0, 0.1, 0.1, 0.8],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.A_s = torch.tensor(
            [
                [0.91, 0.03, 0.00, 0.00],
                [0.00, 0.88, 0.04, 0.00],
                [0.02, 0.00, 0.90, 0.02],
                [0.00, 0.02, 0.03, 0.89],
            ],
            dtype=torch.float32,
            device=device,
        )
        token_eye = torch.eye(8, dtype=torch.float32, device=device)
        self.token_embedding = torch.tensor(
            [
                [1.0, 0.0, 0.2, 0.0],
                [0.8, 0.2, 0.0, 0.1],
                [0.1, 0.9, 0.1, 0.0],
                [0.0, 0.8, 0.3, 0.1],
                [0.1, 0.0, 1.0, 0.1],
                [0.0, 0.2, 0.8, 0.3],
                [0.2, 0.1, 0.0, 1.0],
                [0.0, 0.3, 0.2, 0.8],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.token_eye = token_eye

    def _init_latents(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.randn(batch_size, 4, device=self.device) * 0.2
        m = torch.randint(0, 8, (batch_size,), device=self.device)
        c = torch.randn(batch_size, 1, device=self.device) * 0.15
        return s, m, c

    def _render_obs(self, s: torch.Tensor, m: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        obs_s = s @ self.obs_matrix.T
        obs_m = self.token_embedding[m]
        bias = torch.ones(s.shape[0], 1, device=self.device)
        noise = torch.randn(s.shape[0], 8, device=self.device) * self.obs_noise_std
        return torch.cat([obs_s, obs_m], dim=-1) + noise

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        s, m, c = self._init_latents(batch_size)

        obs = torch.zeros(batch_size, self.seq_len, 10, device=self.device)
        mem_query = torch.zeros(batch_size, self.seq_len, 1, device=self.device)
        danger_flag = torch.zeros(batch_size, self.seq_len, 1, device=self.device)
        pred_target = torch.zeros(batch_size, self.seq_len, 10, device=self.device)
        mem_target = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        ctrl_target = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        mem_mask = torch.zeros(batch_size, self.seq_len, device=self.device)
        ctrl_mask = torch.zeros(batch_size, self.seq_len, device=self.device)
        state_c = torch.zeros(batch_size, self.seq_len, 1, device=self.device)

        scheduled_mem_targets = torch.full((batch_size, self.seq_len), -1, dtype=torch.long, device=self.device)

        for t in range(self.seq_len):
            cue_mask = torch.rand(batch_size, device=self.device) < self.cue_probability
            new_tokens = torch.randint(0, 8, (batch_size,), device=self.device)
            m = torch.where(cue_mask, new_tokens, m)

            future_step = t + self.memory_delay
            if future_step < self.seq_len:
                scheduled_mem_targets[cue_mask, future_step] = m[cue_mask]

            query_mask = scheduled_mem_targets[:, t] >= 0
            if query_mask.any():
                mem_query[:, t, 0] = 1.0
                mem_target[query_mask, t] = scheduled_mem_targets[query_mask, t]
                mem_mask[query_mask, t] = 1.0

            eta_t = torch.randn(batch_size, 1, device=self.device) * self.control_noise_std
            shock_trigger = (torch.rand(batch_size, 1, device=self.device) < self.shock_probability).float()
            shock_sign = torch.where(torch.rand(batch_size, 1, device=self.device) < 0.5, -1.0, 1.0)
            shock_amp = self.shock_scale_low + (self.shock_scale_high - self.shock_scale_low) * torch.rand(batch_size, 1, device=self.device)
            shock = shock_trigger * shock_sign * shock_amp

            c_uncontrolled_next = self.control_drift * c + eta_t + shock
            u_star = torch.clamp(self.control_gain * c, min=-1.0, max=1.0)
            ctrl_target[:, t] = quantize_to_bins(u_star.squeeze(-1).unsqueeze(1), self.action_bins).squeeze(1)

            future_proxy = c_uncontrolled_next
            danger = (torch.abs(future_proxy) > self.danger_threshold).float()
            danger_flag[:, t, 0] = danger.squeeze(-1)

            applied_u = self.action_bins[ctrl_target[:, t]].unsqueeze(-1)
            if self.world_type == "B":
                ctrl_mask[:, t] = danger.squeeze(-1)

            obs_t = self._render_obs(s, m, c)
            obs[:, t] = torch.cat([obs_t, c, torch.ones(batch_size, 1, device=self.device)], dim=-1)
            state_c[:, t] = c

            s_next = s @ self.A_s.T + torch.randn(batch_size, 4, device=self.device) * 0.05
            c_next = c_uncontrolled_next - applied_u
            pred_obs = self._render_obs(s_next, m, c_next)
            pred_target[:, t] = torch.cat([pred_obs, c_next, torch.ones(batch_size, 1, device=self.device)], dim=-1)

            s = s_next
            c = c_next

        return WorldBatch(
            obs=obs,
            mem_query=mem_query,
            danger_flag=danger_flag,
            pred_target=pred_target,
            mem_target=mem_target,
            ctrl_target=ctrl_target,
            mem_mask=mem_mask,
            ctrl_mask=ctrl_mask,
            state_c=state_c,
        ).as_dict()

    def sample_dataset(self, num_batches: int, batch_size: int) -> list[dict[str, torch.Tensor]]:
        return [self.sample_batch(batch_size) for _ in range(num_batches)]

    def sanity_check_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        ctrl_targets = batch["ctrl_target"][batch["ctrl_mask"] > 0]
        if ctrl_targets.numel() == 0:
            ctrl_entropy = 0.0
        else:
            counts = torch.bincount(ctrl_targets, minlength=self.action_bins.numel()).float()
            probs = counts / torch.clamp(counts.sum(), min=1.0)
            ctrl_entropy = float((-(probs * torch.log(probs + 1e-12))).sum().item())
        return {
            "mem_positive_rate": float(batch["mem_mask"].float().mean().item()),
            "ctrl_positive_rate": float(batch["ctrl_mask"].float().mean().item()),
            "danger_rate": float(batch["danger_flag"].float().mean().item()),
            "obs_std": float(batch["obs"].std().item()),
            "ctrl_label_entropy": ctrl_entropy,
        }


class TaskSpaceSequenceGenerator:
    def __init__(
        self,
        seq_len: int,
        task_space_dim: int,
        goal_switch_prob: float,
        control_horizon: int,
        control_strength: float,
        noise_std: float,
        periodic_boundary: bool,
        memory_delay: int,
        device: torch.device,
        ecology_name: str,
        heldout_region: str = "upper_right_quadrant",
        coordinate_transform: str = "identity",
        rotation_degrees: float = 0.0,
        scale_factor: float = 1.0,
        nonlinear_warp: float = 0.0,
        train_region_mode: str = "train",
        model_obs_dim: int = 7,
        heading_bins: int = 8,
    ) -> None:
        self.seq_len = seq_len
        self.task_space_dim = task_space_dim
        self.goal_switch_prob = goal_switch_prob
        self.control_horizon = control_horizon
        self.control_strength = control_strength
        self.noise_std = noise_std
        self.periodic_boundary = periodic_boundary
        self.memory_delay = memory_delay
        self.device = device
        self.ecology_name = ecology_name
        self.heldout_region = heldout_region
        self.coordinate_transform = coordinate_transform
        self.rotation_degrees = rotation_degrees
        self.scale_factor = scale_factor
        self.nonlinear_warp = nonlinear_warp
        self.train_region_mode = train_region_mode
        self.model_obs_dim = model_obs_dim
        self.heading_bins = heading_bins

    def _sample_points(self, batch_size: int, region_mode: str) -> torch.Tensor:
        points = torch.empty(batch_size, self.task_space_dim, device=self.device).uniform_(-1.0, 1.0)
        if self.ecology_name == "conceptual_grid_task":
            if region_mode == "train":
                invalid = (points[:, 0] > 0.25) & (points[:, 1] > 0.25)
                while invalid.any():
                    points[invalid] = torch.empty(int(invalid.sum().item()), self.task_space_dim, device=self.device).uniform_(-1.0, 1.0)
                    invalid = (points[:, 0] > 0.25) & (points[:, 1] > 0.25)
            elif region_mode == "heldout_region":
                points[:, 0] = torch.empty(batch_size, device=self.device).uniform_(0.30, 1.0)
                points[:, 1] = torch.empty(batch_size, device=self.device).uniform_(0.30, 1.0)
            elif region_mode == "interpolation":
                points = torch.empty(batch_size, self.task_space_dim, device=self.device).uniform_(-0.2, 0.2)
            elif region_mode == "extrapolation":
                signs = torch.where(torch.rand(batch_size, self.task_space_dim, device=self.device) < 0.5, -1.0, 1.0)
                points = signs * torch.empty(batch_size, self.task_space_dim, device=self.device).uniform_(0.9, 1.3)
                points = torch.clamp(points, -1.5, 1.5)
        return points

    def _wrap_or_clip(self, coords: torch.Tensor) -> torch.Tensor:
        if self.periodic_boundary:
            return ((coords + 1.0) % 2.0) - 1.0
        return torch.clamp(coords, -1.5, 1.5)

    def _transform_coords(self, coords: torch.Tensor, override_transform: str | None = None) -> torch.Tensor:
        transform = override_transform or self.coordinate_transform
        result = coords.clone()
        if transform in {"rotated", "rotated_scaled", "nonlinear_rotated"}:
            rot = rotation_matrix(self.rotation_degrees)
            rot_tensor = torch.tensor(rot, dtype=result.dtype, device=result.device)
            result = result @ rot_tensor.T
        if transform in {"scaled", "rotated_scaled", "nonlinear_rotated"}:
            result = result * self.scale_factor
        if transform in {"nonlinear", "nonlinear_rotated"}:
            result = result + self.nonlinear_warp * torch.stack(
                [torch.sin(math.pi * result[:, 0]), torch.sin(math.pi * result[:, 1])],
                dim=-1,
            )
        return result

    def _goal_bin(self, goal_coords: torch.Tensor) -> torch.Tensor:
        gx = torch.clamp(torch.floor((goal_coords[:, 0] + 1.5) / 0.75), min=0, max=3)
        gy = torch.clamp(torch.floor((goal_coords[:, 1] + 1.5) / 0.75), min=0, max=3)
        return (gx * 4 + gy).long()

    def _heading_bin(self, heading_angle: torch.Tensor) -> torch.Tensor:
        normalized = (heading_angle + math.pi) / (2.0 * math.pi)
        return (torch.floor(normalized * self.heading_bins).long() % self.heading_bins)

    def sample_batch(
        self,
        batch_size: int,
        region_mode: str | None = None,
        coordinate_transform: str | None = None,
        control_strength: float | None = None,
    ) -> dict:
        mode = region_mode or self.train_region_mode
        applied_control_strength = float(self.control_strength if control_strength is None else control_strength)
        state = self._sample_points(batch_size, mode)
        goal = self._sample_points(batch_size, mode)
        velocity = torch.zeros(batch_size, self.task_space_dim, device=self.device)

        inputs = torch.zeros(batch_size, self.seq_len, self.model_obs_dim + 2, device=self.device)
        task_coords = torch.zeros(batch_size, self.seq_len, self.task_space_dim, device=self.device)
        velocities = torch.zeros(batch_size, self.seq_len, self.task_space_dim, device=self.device)
        heading_angles = torch.zeros(batch_size, self.seq_len, device=self.device)
        goal_coords = torch.zeros(batch_size, self.seq_len, self.task_space_dim, device=self.device)
        pred_target = torch.zeros(batch_size, self.seq_len, self.model_obs_dim, device=self.device)
        mem_target = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        mem_mask = torch.zeros(batch_size, self.seq_len, device=self.device)
        ctrl_target = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=self.device)
        ctrl_mask = torch.zeros(batch_size, self.seq_len, device=self.device)
        control_demand = torch.full((batch_size, self.seq_len), applied_control_strength, device=self.device)

        scheduled_mem_targets = torch.full((batch_size, self.seq_len), -1, dtype=torch.long, device=self.device)
        visible_goal = torch.zeros(batch_size, self.task_space_dim, device=self.device)

        for t in range(self.seq_len):
            switch_mask = torch.rand(batch_size, device=self.device) < self.goal_switch_prob
            if switch_mask.any():
                goal[switch_mask] = self._sample_points(int(switch_mask.sum().item()), mode)
                visible_goal[switch_mask] = goal[switch_mask]
                future_step = t + self.memory_delay
                if future_step < self.seq_len:
                    scheduled_mem_targets[switch_mask, future_step] = self._goal_bin(goal[switch_mask])

            query_mask = scheduled_mem_targets[:, t] >= 0
            if query_mask.any():
                mem_target[query_mask, t] = scheduled_mem_targets[query_mask, t]
                mem_mask[query_mask, t] = 1.0

            direction = goal - state
            distance = torch.sqrt(torch.sum(direction * direction, dim=-1, keepdim=True) + 1e-12)
            desired_velocity = direction / distance
            velocity = 0.55 * velocity + 0.45 * applied_control_strength * desired_velocity
            velocity = velocity + torch.randn_like(velocity) * self.noise_std
            next_state = self._wrap_or_clip(state + 0.12 * velocity)
            heading_angle = torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0])
            ctrl_target[:, t] = self._heading_bin(heading_angle)
            ctrl_mask[:, t] = (distance.squeeze(-1) > 0.12).float()

            obs_state = self._transform_coords(state, coordinate_transform)
            next_obs_state = self._transform_coords(next_state, coordinate_transform)
            obs_core = torch.cat(
                [
                    obs_state,
                    velocity,
                    torch.full((batch_size, 1), applied_control_strength, device=self.device),
                    torch.norm(goal - state, dim=-1, keepdim=True),
                    switch_mask.float().unsqueeze(-1),
                ],
                dim=-1,
            )
            next_obs_core = torch.cat(
                [
                    next_obs_state,
                    velocity,
                    torch.full((batch_size, 1), applied_control_strength, device=self.device),
                    torch.norm(goal - next_state, dim=-1, keepdim=True),
                    torch.zeros(batch_size, 1, device=self.device),
                ],
                dim=-1,
            )
            mem_query = query_mask.float().unsqueeze(-1)
            danger_flag = (applied_control_strength > 0.0) * ctrl_mask[:, t].unsqueeze(-1)

            inputs[:, t] = torch.cat([obs_core, mem_query, danger_flag], dim=-1)
            task_coords[:, t] = state
            velocities[:, t] = velocity
            heading_angles[:, t] = heading_angle
            goal_coords[:, t] = goal
            pred_target[:, t] = next_obs_core

            state = next_state
            visible_goal = torch.where(switch_mask.unsqueeze(-1), torch.zeros_like(visible_goal), visible_goal * 0.90)

        metadata = {
            "ecology_name": self.ecology_name,
            "coordinate_transform": coordinate_transform or self.coordinate_transform,
            "heldout_region": self.heldout_region if mode != "train" else "none",
            "region_mode": mode,
        }
        batch = TaskSpaceBatch(
            input=inputs,
            task_coords=task_coords,
            velocity=velocities,
            heading_angle=heading_angles,
            goal_coords=goal_coords,
            pred_target=pred_target,
            mem_target=mem_target,
            mem_mask=mem_mask,
            ctrl_target=ctrl_target,
            ctrl_mask=ctrl_mask,
            control_demand=control_demand,
            metadata=metadata,
        ).as_dict()
        batch["obs"] = inputs[:, :, : self.model_obs_dim]
        batch["mem_query"] = inputs[:, :, self.model_obs_dim : self.model_obs_dim + 1]
        batch["danger_flag"] = inputs[:, :, self.model_obs_dim + 1 : self.model_obs_dim + 2]
        batch["state_c"] = torch.norm(goal_coords - task_coords, dim=-1, keepdim=True)
        return batch

    def sample_dataset(
        self,
        num_batches: int,
        batch_size: int,
        region_mode: str | None = None,
        coordinate_transform: str | None = None,
        control_strength: float | None = None,
    ) -> list[dict]:
        return [
            self.sample_batch(
                batch_size,
                region_mode=region_mode,
                coordinate_transform=coordinate_transform,
                control_strength=control_strength,
            )
            for _ in range(num_batches)
        ]

    def analysis_splits(self, batch_size: int, num_batches: int) -> dict[str, list[dict]]:
        rotated_name = "rotated" if self.coordinate_transform == "identity" else self.coordinate_transform
        return {
            "train_like": self.sample_dataset(num_batches, batch_size, region_mode="train", coordinate_transform=self.coordinate_transform),
            "heldout_region": self.sample_dataset(num_batches, batch_size, region_mode="heldout_region", coordinate_transform=self.coordinate_transform),
            "interpolation": self.sample_dataset(num_batches, batch_size, region_mode="interpolation", coordinate_transform=self.coordinate_transform),
            "extrapolation": self.sample_dataset(num_batches, batch_size, region_mode="extrapolation", coordinate_transform=self.coordinate_transform),
            "rotated": self.sample_dataset(num_batches, batch_size, region_mode="train", coordinate_transform=rotated_name),
        }
