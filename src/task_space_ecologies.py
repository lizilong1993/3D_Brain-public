from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class TaskSpaceEcologySpec:
    name: str
    task_space_dim: int = 2
    sequence_length: int = 64
    goal_switch_prob: float = 0.05
    control_horizon: int = 1
    control_strength: float = 1.0
    noise_std: float = 0.01
    periodic_boundary: bool = False
    memory_delay: int = 8
    heldout_region: str = "upper_right_quadrant"
    coordinate_transform: str = "identity"
    rotation_degrees: float = 0.0
    scale_factor: float = 1.0
    nonlinear_warp: float = 0.0
    train_region_mode: str = "train"


TASK_SPACE_ECOLOGY_REGISTRY: dict[str, TaskSpaceEcologySpec] = {
    "abstract_2d_navigation": TaskSpaceEcologySpec(
        name="abstract_2d_navigation",
        sequence_length=64,
        control_strength=1.0,
        periodic_boundary=True,
        coordinate_transform="identity",
    ),
    "conceptual_grid_task": TaskSpaceEcologySpec(
        name="conceptual_grid_task",
        sequence_length=64,
        control_strength=1.0,
        periodic_boundary=False,
        heldout_region="upper_right_quadrant",
        coordinate_transform="identity",
    ),
    "remapped_task_space": TaskSpaceEcologySpec(
        name="remapped_task_space",
        sequence_length=64,
        control_strength=1.0,
        periodic_boundary=False,
        coordinate_transform="rotated",
        rotation_degrees=35.0,
        scale_factor=1.15,
        nonlinear_warp=0.12,
    ),
}


TASK_SPACE_ECOLOGY_CONFIG_KEYS: tuple[str, ...] = tuple(
    field.name for field in fields(TaskSpaceEcologySpec) if field.name != "name"
)


def get_task_space_ecology_spec(name: str) -> TaskSpaceEcologySpec:
    if name not in TASK_SPACE_ECOLOGY_REGISTRY:
        raise ValueError(f"unknown task-space ecology_name: {name}")
    return TASK_SPACE_ECOLOGY_REGISTRY[name]


def resolve_task_space_config(config: dict) -> dict:
    ecology_name = config.get("ecology_name", "abstract_2d_navigation")
    spec = get_task_space_ecology_spec(ecology_name)
    resolved = {
        field_name: getattr(spec, field_name)
        for field_name in TASK_SPACE_ECOLOGY_CONFIG_KEYS
    }
    resolved.update(config)
    resolved["ecology_name"] = ecology_name
    resolved.setdefault("task_space_mode", True)
    return resolved


def rotation_matrix(degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    radians = math.radians(degrees)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return ((cos_v, -sin_v), (sin_v, cos_v))
