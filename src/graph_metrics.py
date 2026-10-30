from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def _edge_lengths(graph: nx.Graph, coords: torch.Tensor) -> list[float]:
    lengths: list[float] = []
    coords_cpu = coords.detach().cpu()
    for src, dst in graph.edges():
        diff = coords_cpu[int(src)] - coords_cpu[int(dst)]
        lengths.append(float(torch.sqrt(torch.sum(diff * diff)).item()))
    return lengths


def build_directed_analysis_graph(edge_index: torch.Tensor, edge_weight: torch.Tensor) -> nx.DiGraph:
    graph = nx.DiGraph()
    for idx in range(edge_weight.shape[0]):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        weight = float(torch.abs(edge_weight[idx]).item())
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += weight
        else:
            graph.add_edge(src, dst, weight=weight)
    return graph


def _reachable_path_lengths(graph: nx.Graph) -> list[float]:
    lengths: list[float] = []
    all_pairs = nx.all_pairs_dijkstra_path_length(
        graph,
        weight=lambda _u, _v, data: 1.0 / max(data.get("weight", 1e-6), 1e-6),
    )
    for src, payload in all_pairs:
        for dst, distance in payload.items():
            if src == dst:
                continue
            lengths.append(float(distance))
    return lengths


def _harmonic_closeness(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    centrality = nx.harmonic_centrality(
        graph,
        distance=lambda _u, _v, data: 1.0 / max(data.get("weight", 1e-6), 1e-6),
    )
    if not centrality:
        return 0.0
    return float(np.mean(list(centrality.values())))


def _reachability_fraction(graph: nx.DiGraph, source_nodes: list[int], target_nodes: list[int]) -> float:
    if not source_nodes or not target_nodes:
        return 0.0
    reachable = 0
    total = 0
    for src in source_nodes:
        if src not in graph:
            continue
        descendants = nx.descendants(graph, src)
        descendants.add(src)
        for dst in target_nodes:
            total += 1
            if dst in descendants:
                reachable += 1
    return float(reachable / total) if total > 0 else 0.0


def compute_graph_metrics(
    graph: nx.Graph,
    coords: torch.Tensor,
    node_types: torch.Tensor,
    region_map: dict[int, int],
    directed_graph: nx.DiGraph,
    model,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "num_nodes": float(graph.number_of_nodes()),
        "num_edges": float(graph.number_of_edges()),
        "disconnected_handling": "reachable_only_and_harmonic",
    }
    if graph.number_of_nodes() == 0:
        return metrics
    components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    component_sizes = [component.number_of_nodes() for component in components]
    metrics["num_components"] = float(len(components))
    metrics["largest_component_fraction"] = float(max(component_sizes) / max(1, graph.number_of_nodes()))
    weak_components = list(nx.weakly_connected_components(directed_graph)) if directed_graph.number_of_nodes() > 0 else []
    strong_components = list(nx.strongly_connected_components(directed_graph)) if directed_graph.number_of_nodes() > 0 else []
    metrics["largest_weakly_connected_component_fraction"] = float(max((len(nodes) for nodes in weak_components), default=0) / max(1, directed_graph.number_of_nodes()))
    metrics["largest_strongly_connected_component_fraction"] = float(max((len(nodes) for nodes in strong_components), default=0) / max(1, directed_graph.number_of_nodes()))
    lengths = _edge_lengths(graph, coords)
    if lengths:
        metrics["mean_edge_length"] = float(np.mean(lengths))
        metrics["median_edge_length"] = float(np.median(lengths))
        metrics["long_edge_fraction"] = float(np.mean(np.array(lengths) > np.median(lengths)))
    else:
        metrics["mean_edge_length"] = 0.0
        metrics["median_edge_length"] = 0.0
        metrics["long_edge_fraction"] = 0.0
    if region_map:
        communities: dict[int, set[int]] = {}
        for node, region_id in region_map.items():
            communities.setdefault(int(region_id), set()).add(int(node))
        metrics["num_regions"] = float(len(communities))
        metrics["modularity"] = float(nx.algorithms.community.modularity(graph, list(communities.values()), weight="weight"))
    else:
        metrics["num_regions"] = 0.0
        metrics["modularity"] = 0.0
    reachable_lengths = _reachable_path_lengths(graph)
    metrics["reachable_fraction"] = float(len(reachable_lengths) / max(1, graph.number_of_nodes() * max(graph.number_of_nodes() - 1, 1)))
    metrics["mean_shortest_path_reachable_only"] = float(np.mean(reachable_lengths)) if reachable_lengths else 0.0
    metrics["legacy_path_mean"] = metrics["mean_shortest_path_reachable_only"]
    metrics["legacy_path_mean_deprecated"] = True
    metrics["harmonic_closeness"] = _harmonic_closeness(graph)
    node_types_cpu = node_types.detach().cpu()
    surface_surface = 0
    surface_interior = 0
    interior_interior = 0
    for src, dst in graph.edges():
        src_type = int(node_types_cpu[int(src)].item())
        dst_type = int(node_types_cpu[int(dst)].item())
        if src_type == 0 and dst_type == 0:
            surface_surface += 1
        elif src_type == 1 and dst_type == 1:
            interior_interior += 1
        else:
            surface_interior += 1
    total_edges = max(1, graph.number_of_edges())
    metrics["surface_surface_edge_fraction"] = float(surface_surface / total_edges)
    metrics["surface_interior_edge_fraction"] = float(surface_interior / total_edges)
    metrics["interior_interior_edge_fraction"] = float(interior_interior / total_edges)
    input_nodes = model.input_node_ids.detach().cpu().tolist()
    ctrl_nodes = model.ctrl_readout_ids.detach().cpu().tolist()
    pred_nodes = model.pred_readout_ids.detach().cpu().tolist()
    mem_nodes = list(range(model.num_nodes))
    region_nodes = sorted(region_map.keys())
    metrics["input_to_region_reachability"] = _reachability_fraction(directed_graph, input_nodes, region_nodes)
    metrics["region_to_control_head_reachability"] = _reachability_fraction(directed_graph, region_nodes, ctrl_nodes)
    metrics["region_to_pred_head_reachability"] = _reachability_fraction(directed_graph, region_nodes, pred_nodes)
    metrics["region_to_mem_head_reachability"] = _reachability_fraction(directed_graph, region_nodes, mem_nodes)
    return metrics
