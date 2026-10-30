from __future__ import annotations

import torch

from .baseline_3d_sparse import Sparse3DBaseline
from .baseline_moe import StandardMoESequenceModel
from .dense_rnn import DenseSequenceModel
from .graph_init import build_initial_sparse_graph, build_node_coordinates
from .model_main import BoundaryInteriorSparseRNN, LocalizedSparseAdapter3DSparse


def build_model(config: dict, device: torch.device):
    model_alias = config.get("model_family", config.get("model_type", "main"))
    alias_map = {
        "boundary_interior_3d_sparse": "main",
        "plain_3d_dynamic_sparse": "sparse_baseline",
        "dynamic_sparse_no_geometry": "sparse_baseline",
        "fixed_3d_sparse": "sparse_baseline",
        "moe_matched": "moe",
        "dense_rnn_matched": "dense_rnn",
        "localized_sparse_adapter_3d_sparse": "localized_sparse_adapter",
    }
    model_type = alias_map.get(model_alias, model_alias)

    coords, node_types = build_node_coordinates(config["num_surface"], config["num_interior"], device)
    edge_index, edge_weight = build_initial_sparse_graph(
        coords,
        node_types,
        config["edge_budget"],
        {"surface": config["surface_degree_limit"], "interior": config["interior_degree_limit"]},
        geometry_constrained=bool(config.get("geometry_constrained", True)),
    )
    input_node_ids = torch.arange(
        int(config.get("input_node_start", 0)),
        int(config.get("input_node_start", 0)) + int(config.get("input_node_count", 8)),
        device=device,
    )
    pred_start = int(config.get("pred_readout_start", 8))
    pred_readout_ids = torch.arange(
        pred_start,
        pred_start + int(config.get("pred_readout_count", 8)),
        device=device,
    )
    ctrl_start = int(config.get("ctrl_readout_start", 16))
    ctrl_readout_ids = torch.arange(
        ctrl_start,
        ctrl_start + int(config.get("ctrl_readout_count", 8)),
        device=device,
    )
    if max(input_node_ids.max(), pred_readout_ids.max(), ctrl_readout_ids.max()) >= config["num_nodes"]:
        raise ValueError("readout/input node IDs exceed num_nodes")
    ctrl_readout_strategy = config.get("ctrl_readout_strategy", "legacy_fixed_node_readout")
    if model_type == "main":
        return BoundaryInteriorSparseRNN(
            config["num_nodes"],
            config["num_surface"],
            edge_index,
            edge_weight,
            coords,
            node_types,
            input_node_ids,
            pred_readout_ids,
            ctrl_readout_ids,
            input_dim=config["input_dim"],
            obs_dim=config["obs_dim"],
            memory_num_classes=config["memory_num_classes"],
            control_num_classes=config["control_num_classes"],
            ctrl_readout_strategy=ctrl_readout_strategy,
        ).to(device)
    if model_type == "sparse_baseline":
        return Sparse3DBaseline(
            config["num_nodes"],
            config["num_surface"],
            edge_index,
            edge_weight,
            coords,
            node_types,
            input_node_ids,
            pred_readout_ids,
            ctrl_readout_ids,
            input_dim=config["input_dim"],
            obs_dim=config["obs_dim"],
            memory_num_classes=config["memory_num_classes"],
            control_num_classes=config["control_num_classes"],
        ).to(device)
    if model_type == "moe":
        return StandardMoESequenceModel(
            input_dim=config["input_dim"],
            obs_dim=config["obs_dim"],
            memory_num_classes=config["memory_num_classes"],
            control_num_classes=config["control_num_classes"],
        ).to(device)
    if model_type == "dense_rnn":
        return DenseSequenceModel(
            input_dim=config["input_dim"],
            hidden_dim=int(config.get("dense_hidden_dim", config["num_nodes"])),
            obs_dim=config["obs_dim"],
            memory_num_classes=config["memory_num_classes"],
            control_num_classes=config["control_num_classes"],
        ).to(device)
    if model_type == "localized_sparse_adapter":
        return LocalizedSparseAdapter3DSparse(
            config["num_nodes"],
            config["num_surface"],
            edge_index,
            edge_weight,
            coords,
            node_types,
            input_node_ids,
            pred_readout_ids,
            input_dim=config["input_dim"],
            obs_dim=config["obs_dim"],
            memory_num_classes=config["memory_num_classes"],
            control_num_classes=config["control_num_classes"],
            num_adapter_channels=int(config.get("num_adapter_channels", 4)),
            adapter_topk_nodes=int(config.get("adapter_topk_nodes", 4)),
            ctrl_readout_strategy=ctrl_readout_strategy,
            adapter_sparsity_regularizer=bool(config.get("adapter_sparsity_regularizer", True)),
            adapter_entropy_regularizer=bool(config.get("adapter_entropy_regularizer", True)),
            adapter_l1_weight=float(config.get("adapter_l1_weight", 1e-4)),
            adapter_entropy_weight=float(config.get("adapter_entropy_weight", 1e-4)),
            adapter_temperature=float(config.get("adapter_temperature", 1.0)),
            # Phase-6 convergence regularizer
            adapter_convergence_regularizer=bool(config.get("adapter_convergence_regularizer", False)),
            adapter_channel_diversity_weight=float(config.get("adapter_channel_diversity_weight", 0.01)),
            adapter_entropy_floor_weight=float(config.get("adapter_entropy_floor_weight", 0.005)),
            adapter_min_union_nodes=int(config.get("adapter_min_union_nodes", 4)),
            adapter_regularizer_diagnostic_only=bool(config.get("adapter_regularizer_diagnostic_only", True)),
        ).to(device)
    raise ValueError(f"unknown model_type: {model_type}")
