from __future__ import annotations

from dataclasses import dataclass, fields

from .task_space_ecologies import TASK_SPACE_ECOLOGY_REGISTRY


@dataclass(frozen=True)
class EcologySpec:
    name: str
    world_type: str
    seq_len: int = 64
    obs_noise_std: float = 0.05
    memory_delay: int = 12
    danger_threshold: float = 0.8
    cue_probability: float = 0.08
    shock_probability: float = 0.08
    shock_scale_low: float = 0.55
    shock_scale_high: float = 0.80
    control_drift: float = 1.08
    control_gain: float = 0.90
    control_noise_std: float = 0.03


ECOLOGY_REGISTRY: dict[str, EcologySpec] = {
    "world_a": EcologySpec(name="world_a", world_type="A"),
    "world_b": EcologySpec(name="world_b", world_type="B"),
    "low_danger_b": EcologySpec(
        name="low_danger_b",
        world_type="B",
        danger_threshold=0.95,
        shock_probability=0.05,
        shock_scale_low=0.45,
        shock_scale_high=0.70,
    ),
    "latency_stress_b": EcologySpec(
        name="latency_stress_b",
        world_type="B",
        danger_threshold=0.65,
        shock_probability=0.12,
        shock_scale_low=0.65,
        shock_scale_high=0.95,
        control_drift=1.12,
        control_gain=0.95,
    ),
    "memory_stress_b": EcologySpec(
        name="memory_stress_b",
        world_type="B",
        memory_delay=18,
        cue_probability=0.12,
        shock_probability=0.06,
    ),
    "high_noise_b": EcologySpec(
        name="high_noise_b",
        world_type="B",
        obs_noise_std=0.10,
        control_noise_std=0.06,
        shock_probability=0.10,
    ),
    "fast_memory_b": EcologySpec(
        name="fast_memory_b",
        world_type="B",
        memory_delay=6,
        cue_probability=0.10,
        shock_probability=0.08,
    ),
}

TASK_SPACE_ECOLOGY_NAMES = set(TASK_SPACE_ECOLOGY_REGISTRY.keys())


def get_ecology_spec(name: str) -> EcologySpec:
    if name not in ECOLOGY_REGISTRY:
        raise ValueError(f"unknown ecology_name: {name}")
    return ECOLOGY_REGISTRY[name]


def resolve_ecology_config(config: dict) -> dict:
    ecology_name = config.get("ecology_name")
    if ecology_name is None:
        ecology_name = "world_b" if config.get("world_type", "B") == "B" else "world_a"
    spec = get_ecology_spec(ecology_name)
    resolved = dict(config)
    resolved["ecology_name"] = ecology_name
    for field in fields(EcologySpec):
        if field.name == "name":
            continue
        resolved.setdefault(field.name, getattr(spec, field.name))
    return resolved


def is_task_space_ecology_name(name: str) -> bool:
    return name in TASK_SPACE_ECOLOGY_NAMES
