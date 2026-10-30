"""Interface exclusion registry for deconfounded regionalization.

Defines which nodes are excluded from region candidate pools based on
their role as input/output/readout interfaces. The exclusion list is
determined **before** any training or analysis and is fixed across seeds
for a given architecture configuration.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import ensure_dir, write_json


# Architecture constants (32-node sparse graph)
NUM_NODES = 32
INPUT_NODE_IDS = list(range(0, 8))
PRED_READOUT_IDS = list(range(8, 16))
CTRL_READOUT_IDS_DEFAULT = list(range(16, 24))  # legacy fixed block


def compute_exclusion_registry(
    ctrl_readout_strategy: str = "legacy_fixed_node_readout",
    ctrl_readout_ids: list[int] | None = None,
    exclusion_radius: int = 0,
    num_nodes: int = NUM_NODES,
    adapter_excluded_from_region_candidates: bool = True,
) -> dict:
    """Compute the interface exclusion registry for a given architecture.

    Args:
        ctrl_readout_strategy: One of 'legacy_fixed_node_readout' (fixed
            ctrl_readout_ids block) or 'distributed_projection' (control
            head reads from all nodes, no specific block), or
            'localized_sparse_adapter_readout' (adapter-based readout).
        ctrl_readout_ids: Optional override for ctrl readout node IDs.
        exclusion_radius: Graph-distance radius around readout nodes to
            also exclude (0 = only direct readout nodes).
        num_nodes: Total number of nodes in the architecture.
        adapter_excluded_from_region_candidates: Whether adapter virtual
            units are excluded from region candidates. Always True for
            localized_sparse_adapter_readout.

    Returns:
        A dict containing the exclusion registry with:
        - input_interface_nodes
        - output_interface_nodes (pred_readout)
        - ctrl_readout_interface_nodes
        - pred_readout_interface_nodes
        - mem_readout_interface_nodes (always empty since mem reads all)
        - head_adjacent_exclusion_nodes
        - excluded_from_region_candidates
        - exclusion_policy
        - exclusion_defined_before_region_prediction
        - adapter_virtual_units (for localized_sparse_adapter_readout)
        - adapter_connected_nodes (empty at registry time, populated later)
    """
    input_nodes = list(INPUT_NODE_IDS)
    pred_nodes = list(PRED_READOUT_IDS)

    # Adapter virtual units: these are non-spatial virtual channels,
    # not graph nodes. They are NEVER region candidates.
    adapter_virtual_units: list[int] = []
    if ctrl_readout_strategy == "localized_sparse_adapter_readout":
        # Adapter virtual units are not graph nodes (they have no node_id).
        # We represent them with negative IDs to distinguish from graph nodes.
        # For the 32-node graph, adapter virtual units are virtual channels.
        adapter_virtual_units = []  # virtual, not graph nodes

    if ctrl_readout_strategy == "distributed_projection":
        # Distributed projection: control head reads from ALL hidden nodes.
        # No fixed block exists; ctrl_readout_interface_nodes is empty.
        ctrl_nodes: list[int] = []
        ctrl_readout_has_fixed_node_block = False
    elif ctrl_readout_strategy == "localized_sparse_adapter_readout":
        # Adapter readout: control head reads from adapter virtual channels,
        # NOT from graph nodes. No fixed ctrl_readout_ids block.
        ctrl_nodes = []
        ctrl_readout_has_fixed_node_block = False
    else:
        # Legacy: fixed block of nodes [16-23] (or custom)
        if ctrl_readout_ids is not None:
            ctrl_nodes = list(ctrl_readout_ids)
        else:
            ctrl_nodes = list(CTRL_READOUT_IDS_DEFAULT)
        ctrl_readout_has_fixed_node_block = True

    # Mem head reads from all nodes — no specific interface nodes.
    mem_nodes: list[int] = []

    # Heads with fixed readout nodes
    all_fixed_heads = set(input_nodes + pred_nodes + ctrl_nodes)

    # Head-adjacent exclusion (graph-distance radius)
    # We don't have graph structure in this pure function, so we only
    # document the policy. The actual adjacency extension requires the
    # edge_index tensor and is done in extend_with_adjacent().
    head_adjacent_exclusion_nodes: list[int] = []
    if exclusion_radius > 0:
        head_adjacent_exclusion_nodes = list(all_fixed_heads)

    if ctrl_readout_strategy == "localized_sparse_adapter_readout":
        # For adapter architecture: exclude input/ouput nodes, but NOT
        # ctrl_nodes (there are none). Adapter-connected graph nodes are
        # potential source regions and should NOT be auto-excluded.
        excluded_from_region_candidates = sorted(set(
            list(input_nodes)
            + list(pred_nodes)
            + head_adjacent_exclusion_nodes
        ))
    else:
        excluded_from_region_candidates = sorted(set(
            list(input_nodes)
            + list(pred_nodes)
            + (list(ctrl_nodes) if ctrl_readout_strategy == "legacy_fixed_node_readout" else [])
            + head_adjacent_exclusion_nodes
        ))

    # Candidate pool is all nodes NOT in the exclusion list
    all_nodes_set = set(range(num_nodes))
    candidate_pool = sorted(all_nodes_set - set(excluded_from_region_candidates))

    registry = {
        "input_interface_nodes": input_nodes,
        "output_interface_nodes": pred_nodes,
        "ctrl_readout_interface_nodes": ctrl_nodes,
        "pred_readout_interface_nodes": pred_nodes,
        "mem_readout_interface_nodes": mem_nodes,
        "head_adjacent_exclusion_nodes": head_adjacent_exclusion_nodes,
        "excluded_from_region_candidates": excluded_from_region_candidates,
        "candidate_pool_nodes": candidate_pool,
        "num_nodes_total": num_nodes,
        "num_excluded_nodes": len(excluded_from_region_candidates),
        "num_candidate_nodes": len(candidate_pool),
        "ctrl_readout_strategy": ctrl_readout_strategy,
        "ctrl_readout_has_fixed_node_block": ctrl_readout_has_fixed_node_block,
        "exclusion_radius": exclusion_radius,
        "adapter_virtual_units": adapter_virtual_units,
        "adapter_connected_nodes": [],  # populated post-training from adapter weights
        "adapter_excluded_from_region_candidates": adapter_excluded_from_region_candidates,
        "exclusion_policy": (
            "All nodes that are part of any head's direct readout interface "
            "are excluded from region candidate pools. For the legacy strategy, "
            "ctrl_readout_ids are also excluded. For distributed_projection, "
            "no specific ctrl_readout interface block exists. "
            "For localized_sparse_adapter_readout, adapter virtual units are "
            "excluded (they are not graph nodes), and adapter-connected graph "
            "nodes are potential source regions (not excluded by default)."
        ),
        "exclusion_defined_before_region_prediction": True,
        "input_nodes_excluded": True,
        "output_nodes_excluded": True,
        "ctrl_nodes_excluded": ctrl_readout_strategy == "legacy_fixed_node_readout",
        "mem_nodes_excluded": False,  # mem reads all nodes
    }

    # Audit trail
    audit_why_no_ctrl = (
        "Control readout nodes are the fixed architecture interface; selecting them is the known artifact."
        if ctrl_readout_strategy == "legacy_fixed_node_readout"
        else "N/A (distributed projection: no fixed ctrl block)."
        if ctrl_readout_strategy == "distributed_projection"
        else "N/A (localized_sparse_adapter_readout: control reads from adapter virtual channels, not graph nodes)."
    )
    registry["exclusion_audit"] = {
        "why_no_input_nodes_in_region": "Input nodes receive observation input directly; their activity is trivially task-correlated.",
        "why_no_output_nodes_in_region": "Output nodes are the direct readout interface for prediction; selecting them would conflate readout with region.",
        "why_no_ctrl_nodes_in_region_legacy": audit_why_no_ctrl,
        "why_adapter_virtual_units_excluded": "Adapter virtual units are non-spatial virtual channels, not graph nodes. They cannot be region candidates.",
    }

    return registry


def extend_with_adjacent(
    registry: dict,
    edge_index: torch.Tensor | None = None,
    num_hops: int = 1,
) -> dict:
    """Extend the exclusion registry by adding graph-adjacent nodes.

    For each node in existing exclusion sets, add nodes within `num_hops`
    graph distance along the edge_index.

    Args:
        registry: Output from compute_exclusion_registry().
        edge_index: [2, E] edge index tensor.
        num_hops: Number of hops for adjacency (default 1).

    Returns:
        Updated registry with head_adjacent_exclusion_nodes populated.
    """
    if edge_index is None or num_hops < 1:
        return registry

    excluded = set(registry["excluded_from_region_candidates"])
    frontier = set(excluded)

    for _ in range(num_hops):
        neighbors: set[int] = set()
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            if d in frontier:
                neighbors.add(s)
            if s in frontier:
                neighbors.add(d)
        frontier = neighbors - excluded
        excluded |= frontier

    # Update candidate pool
    all_nodes = set(range(registry["num_nodes_total"]))
    candidate_pool = sorted(all_nodes - excluded)

    registry["excluded_from_region_candidates"] = sorted(excluded)
    registry["candidate_pool_nodes"] = candidate_pool
    registry["head_adjacent_exclusion_nodes"] = sorted(excluded - set(registry["excluded_from_region_candidates"]))
    registry["num_excluded_nodes"] = len(excluded)
    registry["num_candidate_nodes"] = len(candidate_pool)
    registry["exclusion_radius"] = num_hops
    registry["adjacency_extension_applied"] = True

    return registry


def write_registry(
    registry: dict,
    output_dir: str | Path,
    filename: str = "interface_node_registry.json",
) -> Path:
    """Write exclusion registry to JSON file."""
    out_dir = ensure_dir(output_dir)
    path = Path(out_dir) / filename
    write_json(path, registry)
    return path


def load_registry(path: str | Path) -> dict:
    """Load exclusion registry from JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_region_overlap(
    region_nodes: list[int],
    registry: dict,
) -> dict:
    """Check whether a predicted region overlaps with excluded interface nodes.

    Args:
        region_nodes: The predicted region node IDs.
        registry: The exclusion registry dict.

    Returns:
        Contamination check dict with overlap counts and pass/fail.
    """
    region_set = set(region_nodes)
    excluded = set(registry["excluded_from_region_candidates"])
    ctrl_set = set(registry["ctrl_readout_interface_nodes"])
    input_set = set(registry["input_interface_nodes"])
    output_set = set(registry["output_interface_nodes"])
    adapter_units = set(registry.get("adapter_virtual_units", []))
    adapter_connected = set(registry.get("adapter_connected_nodes", []))

    overlap_excluded = region_set & excluded
    overlap_ctrl = region_set & ctrl_set
    overlap_input = region_set & input_set
    overlap_output = region_set & output_set
    overlap_adapter = region_set & adapter_units
    overlap_adapter_connected = region_set & adapter_connected

    return {
        "region_nodes": sorted(region_nodes),
        "region_size": len(region_nodes),
        "overlap_with_excluded_nodes": sorted(overlap_excluded),
        "overlap_count_with_excluded": len(overlap_excluded),
        "overlap_with_ctrl_readout_ids": sorted(overlap_ctrl),
        "overlap_count_with_ctrl": len(overlap_ctrl),
        "overlap_with_input_interface": sorted(overlap_input),
        "overlap_count_with_input": len(overlap_input),
        "overlap_with_output_interface": sorted(overlap_output),
        "overlap_count_with_output": len(overlap_output),
        "overlap_with_adapter_virtual_units": sorted(overlap_adapter),
        "overlap_count_with_adapter": len(overlap_adapter),
        "overlap_with_adapter_connected_nodes": sorted(overlap_adapter_connected),
        "overlap_count_with_adapter_connected": len(overlap_adapter_connected),
        "interface_contamination_pass": len(overlap_excluded) == 0,
        "exclusion_policy": registry.get("exclusion_policy", ""),
        "exclusion_defined_before_region_prediction": registry.get(
            "exclusion_defined_before_region_prediction", True
        ),
    }
