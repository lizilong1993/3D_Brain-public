from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from .predict_task_regions import normalize_metric, predict_task_regions_without_lesion as predict_task_regions_nonlesion
from .utils import pairwise_path_length_or_large


def build_analysis_graph(edge_index: torch.Tensor, edge_weight: torch.Tensor) -> nx.Graph:
    graph = nx.Graph()
    for idx in range(edge_weight.shape[0]):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        weight = float(torch.abs(edge_weight[idx]).item())
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += weight
        else:
            graph.add_edge(src, dst, weight=weight)
    return graph


def threshold_graph_by_weight(edge_index: torch.Tensor, edge_weight: torch.Tensor, quantile: float) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = torch.quantile(torch.abs(edge_weight), quantile)
    keep = torch.abs(edge_weight) >= threshold
    return edge_index[:, keep], edge_weight[keep]


def detect_regions_louvain(graph: nx.Graph) -> dict[int, int]:
    communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42)
    region_map: dict[int, int] = {}
    for region_id, community in enumerate(communities):
        for node in community:
            region_map[int(node)] = region_id
    return region_map


def detect_regions_spectral(graph: nx.Graph, num_regions: int) -> dict[int, int]:
    nodes = sorted(graph.nodes())
    if not nodes:
        return {}
    if len(nodes) <= 1:
        return {nodes[0]: 0}
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    num_clusters = max(1, min(num_regions, len(nodes)))
    try:
        clustering = SpectralClustering(n_clusters=num_clusters, affinity="precomputed", random_state=42)
        labels = clustering.fit_predict(adjacency)
        return {node: int(label) for node, label in zip(nodes, labels)}
    except Exception:
        return {node: 0 for node in nodes}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_autocorr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return _safe_corr(x[1:], x[:-1])


def _finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else 0.0


def _density_from_edge_count(edge_count: int, src_count: int, dst_count: int) -> float:
    possible = max(1, src_count * dst_count)
    return float(edge_count / possible)


def _edge_length_lookup(graph: nx.Graph, coords: torch.Tensor) -> dict[tuple[int, int], float]:
    coords_cpu = coords.detach().cpu()
    lookup: dict[tuple[int, int], float] = {}
    for src, dst in graph.edges():
        diff = coords_cpu[int(src)] - coords_cpu[int(dst)]
        length = float(torch.sqrt(torch.sum(diff * diff)).item())
        lookup[(int(src), int(dst))] = length
        lookup[(int(dst), int(src))] = length
    return lookup


def compute_region_statistics(
    graph: nx.Graph,
    region_map: dict[int, int],
    coords: torch.Tensor,
    node_types: torch.Tensor,
    model,
    edge_index: torch.Tensor,
) -> dict:
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=lambda _u, _v, data: 1.0 / max(data.get("weight", 1e-6), 1e-6)))
    region_stats: dict[int, dict] = {}
    ctrl_nodes = model.ctrl_readout_ids.tolist()
    pred_nodes = model.pred_readout_ids.tolist()
    input_nodes = model.input_node_ids.tolist()
    mem_nodes = list(range(model.num_nodes))
    edge_index_cpu = edge_index.detach().cpu()
    node_types_cpu = node_types.detach().cpu()
    edge_lengths = _edge_length_lookup(graph, coords)
    all_lengths = list(edge_lengths.values())
    global_length_median = float(np.median(all_lengths)) if all_lengths else 0.0
    mem_head_weights = torch.norm(model.mem_head.weight.detach().cpu(), dim=0)
    for node, region_id in region_map.items():
        region_stats.setdefault(region_id, {"nodes": []})["nodes"].append(node)
    for region_id, stats in region_stats.items():
        nodes = stats["nodes"]
        node_set = set(int(node) for node in nodes)
        coord_tensor = coords[nodes]
        radial = torch.sqrt(torch.sum(coord_tensor * coord_tensor, dim=1))
        surface_count = int((node_types_cpu[nodes] == 0).sum().item())
        undirected_degree = np.array([graph.degree(node, weight=None) for node in nodes], dtype=np.float64)
        internal_edges = 0
        incident_lengths = []
        src_nodes = edge_index_cpu[0].tolist()
        dst_nodes = edge_index_cpu[1].tolist()
        incoming_from_inputs = 0
        outgoing_to_ctrl = 0
        outgoing_to_pred = 0
        outgoing_within_region = 0
        for src, dst in zip(src_nodes, dst_nodes):
            src = int(src)
            dst = int(dst)
            if src in node_set and dst in node_set:
                outgoing_within_region += 1
            if src in input_nodes and dst in node_set:
                incoming_from_inputs += 1
            if src in node_set and dst in ctrl_nodes:
                outgoing_to_ctrl += 1
            if src in node_set and dst in pred_nodes:
                outgoing_to_pred += 1
            if src in node_set or dst in node_set:
                length = edge_lengths.get((src, dst))
                if length is not None:
                    incident_lengths.append(length)
        internal_edges = graph.subgraph(nodes).number_of_edges()
        stats["size"] = len(nodes)
        stats["mean_radius"] = float(radial.mean().item())
        stats["surface_fraction"] = float(surface_count / max(1, len(nodes)))
        stats["internal_density"] = float(internal_edges / max(1, len(nodes) * max(len(nodes) - 1, 1) / 2))
        stats["incoming_density_from_inputs"] = _density_from_edge_count(incoming_from_inputs, len(input_nodes), len(nodes))
        stats["outgoing_density_to_control_head"] = _density_from_edge_count(outgoing_to_ctrl, len(nodes), len(ctrl_nodes))
        stats["outgoing_density_to_prediction_head"] = _density_from_edge_count(outgoing_to_pred, len(nodes), len(pred_nodes))
        stats["outgoing_density_to_memory_head"] = float(mem_head_weights[nodes].mean().item()) if nodes else 0.0
        stats["mean_edge_length"] = float(np.mean(incident_lengths)) if incident_lengths else 0.0
        stats["long_edge_fraction"] = float(np.mean(np.array(incident_lengths) > global_length_median)) if incident_lengths and global_length_median > 0.0 else 0.0
        stats["degree_mean"] = float(np.mean(undirected_degree)) if undirected_degree.size > 0 else 0.0
        stats["degree_std"] = float(np.std(undirected_degree)) if undirected_degree.size > 0 else 0.0
        ctrl_path = []
        pred_path = []
        mem_path = []
        for src in nodes:
            ctrl_path.extend([pairwise_path_length_or_large(path_lengths, src, dst) for dst in ctrl_nodes])
            pred_path.extend([pairwise_path_length_or_large(path_lengths, src, dst) for dst in pred_nodes])
            mem_path.extend([pairwise_path_length_or_large(path_lengths, src, dst) for dst in mem_nodes])
        stats["ctrl_path_mean"] = _finite_mean([value for value in ctrl_path if value < 1e6])
        stats["pred_path_mean"] = _finite_mean([value for value in pred_path if value < 1e6])
        stats["mem_path_mean"] = _finite_mean([value for value in mem_path if value < 1e6])
        stats["legacy_ctrl_path_mean"] = float(np.mean(ctrl_path)) if ctrl_path else 0.0
        stats["legacy_pred_path_mean"] = float(np.mean(pred_path)) if pred_path else 0.0
        stats["legacy_path_mean_deprecated"] = True
        reachable_ctrl = [value for value in ctrl_path if value < 1e6]
        reachable_pred = [value for value in pred_path if value < 1e6]
        reachable_mem = [value for value in mem_path if value < 1e6]
        stats["region_to_control_head_reachability"] = float(len(reachable_ctrl) / max(1, len(ctrl_path)))
        stats["region_to_prediction_head_reachability"] = float(len(reachable_pred) / max(1, len(pred_path)))
        stats["region_to_memory_head_reachability"] = float(len(reachable_mem) / max(1, len(mem_path)))
        input_paths = []
        for src in input_nodes:
            input_paths.extend([pairwise_path_length_or_large(path_lengths, src, dst) for dst in nodes])
        reachable_input = [value for value in input_paths if value < 1e6]
        stats["input_to_region_reachability"] = float(len(reachable_input) / max(1, len(input_paths)))
        stats["harmonic_closeness"] = float(np.mean([1.0 / (1.0 + value) for value in reachable_mem])) if reachable_mem else 0.0
        outgoing_mask = torch.isin(model.edge_index[0], torch.tensor(nodes, device=model.edge_index.device))
        stats["ctrl_coupling"] = float(torch.abs(model.edge_weight[outgoing_mask]).mean().item()) if outgoing_mask.sum() > 0 else 0.0
    return region_stats


def attach_activity_statistics(
    region_stats: dict[int, dict],
    region_map: dict[int, int],
    hidden_traj: torch.Tensor,
    batch: dict[str, torch.Tensor],
    pred_logits: torch.Tensor | None = None,
) -> tuple[dict[int, dict], bool]:
    if hidden_traj.numel() == 0:
        return region_stats, False
    danger = batch["danger_flag"].detach().cpu().squeeze(-1).numpy()
    ctrl_mask = batch["ctrl_mask"].detach().cpu().numpy()
    mem_query = batch["mem_query"].detach().cpu().squeeze(-1).numpy()
    mem_mask = batch.get("mem_mask", torch.zeros_like(batch["ctrl_mask"]))
    mem_mask_np = mem_mask.detach().cpu().numpy()
    pred_signal = torch.norm(batch.get("pred_target", torch.zeros_like(batch["obs"]))[:, :, :8], dim=-1).detach().cpu().numpy()
    pred_mask = (pred_signal > np.median(pred_signal)).astype(np.float64) if pred_signal.size > 0 else np.zeros_like(pred_signal)
    obs_signal = torch.norm(batch["obs"], dim=-1).detach().cpu().numpy()
    hidden = hidden_traj.detach().cpu().numpy()
    pred_error_signal = None
    if pred_logits is not None and pred_logits.numel() > 0 and pred_logits.shape == batch["pred_target"].shape:
        pred_error_signal = torch.norm(pred_logits.detach().cpu() - batch["pred_target"].detach().cpu(), dim=-1).numpy()
    available = hidden.ndim == 3 and danger.shape == hidden.shape[:2]
    if not available:
        return region_stats, False
    danger_flat = danger.reshape(-1)
    ctrl_flat = ctrl_mask.reshape(-1)
    mem_flat = mem_query.reshape(-1)
    for region_id, stats in region_stats.items():
        nodes = [int(node) for node in stats["nodes"]]
        region_activity = np.mean(np.abs(hidden[:, :, nodes]), axis=2)
        region_flat = region_activity.reshape(-1)
        mean_activity = np.mean(region_activity, axis=0)
        danger_on = region_flat[danger_flat > 0.5]
        danger_off = region_flat[danger_flat <= 0.5]
        ctrl_on = region_flat[ctrl_flat > 0.5]
        ctrl_off = region_flat[ctrl_flat <= 0.5]
        mem_on = region_flat[mem_flat > 0.5]
        mem_off = region_flat[mem_flat <= 0.5]
        stats["activity_features_available"] = True
        stats["mean_activation"] = float(np.mean(region_flat))
        stats["activation_std"] = float(np.std(region_flat))
        stats["danger_response"] = float(np.mean(danger_on) - np.mean(danger_off)) if danger_on.size > 0 and danger_off.size > 0 else 0.0
        stats["danger_flag_response"] = stats["danger_response"]
        stats["control_mask_response"] = float(np.mean(ctrl_on) - np.mean(ctrl_off)) if ctrl_on.size > 0 and ctrl_off.size > 0 else 0.0
        stats["memory_query_response"] = float(np.mean(mem_on) - np.mean(mem_off)) if mem_on.size > 0 and mem_off.size > 0 else 0.0
        stats["danger_flag_correlation"] = _safe_corr(region_flat, danger_flat)
        stats["control_mask_correlation"] = _safe_corr(region_flat, ctrl_flat)
        stats["memory_query_correlation"] = _safe_corr(region_flat, mem_flat)
        stats["mem_mask_correlation"] = _safe_corr(region_flat, mem_mask_np.reshape(-1))
        stats["prediction_target_correlation"] = _safe_corr(region_flat, pred_signal.reshape(-1))
        stats["pred_mask_correlation"] = _safe_corr(region_flat, pred_mask.reshape(-1)) if pred_signal.size > 0 else 0.0
        stats["input_response_correlation"] = _safe_corr(region_flat, obs_signal.reshape(-1))
        if danger_on.size > 0 and danger_off.size > 0:
            from sklearn.metrics import roc_auc_score

            stats["danger_probe_auc"] = float(roc_auc_score(danger_flat, region_flat)) if len(np.unique(danger_flat)) > 1 else 0.0
        else:
            stats["danger_probe_auc"] = 0.0
        onset_values = []
        for batch_index in range(danger.shape[0]):
            for time_index in range(1, danger.shape[1]):
                if danger[batch_index, time_index] > 0.5 and danger[batch_index, time_index - 1] <= 0.5:
                    onset_values.append(region_activity[batch_index, time_index] - region_activity[batch_index, time_index - 1])
        stats["latency_sensitive_activation_timing"] = float(np.mean(onset_values)) if onset_values else 0.0
        stats["temporal_autocorrelation"] = float(np.mean([_safe_autocorr(region_activity[batch_index]) for batch_index in range(region_activity.shape[0])]))
        stats["activation_autocorrelation"] = stats["temporal_autocorrelation"]
        stats["temporal_activation_variance"] = float(np.var(region_activity))
        stats["prediction_activity_response"] = float(np.mean(region_flat[pred_mask.reshape(-1) > 0.5]) - np.mean(region_flat[pred_mask.reshape(-1) <= 0.5])) if pred_signal.size > 0 else 0.0
        stats["prediction_target_response"] = stats["prediction_activity_response"]
        stats["prediction_error_correlation"] = _safe_corr(region_flat, pred_error_signal.reshape(-1)) if pred_error_signal is not None else 0.0
    return region_stats, True


def predict_task_regions_without_lesion(region_stats: dict[int, dict], feature_set: str = "structure_only") -> dict[str, dict]:
    return predict_task_regions_nonlesion(region_stats, feature_set=feature_set)


def identify_task_regions_nonlesion(region_stats: dict[int, dict]) -> dict[str, dict]:
    return predict_task_regions_without_lesion(region_stats, feature_set="structure_only")


def _normalize_metric(values: dict[int, float]) -> dict[int, float]:
    return normalize_metric(values)


def identify_control_region(region_stats: dict[int, dict], lesion_matrix: dict[int | str, dict] | None = None) -> tuple[int, dict]:
    lesion_matrix = lesion_matrix or {}
    lesion_delta_ctrl = {
        int(region_id): float(payload.get("Delta Ctrl", 0.0))
        for region_id, payload in lesion_matrix.items()
    }
    lesion_ls_ctrl = {
        int(region_id): float(payload.get("LS_ctrl", 0.0))
        for region_id, payload in lesion_matrix.items()
    }
    evidence_available = bool(lesion_matrix)
    radius_values = {int(region_id): float(stats.get("mean_radius", 0.0)) for region_id, stats in region_stats.items()}
    coupling_values = {int(region_id): float(stats.get("ctrl_coupling", 0.0)) for region_id, stats in region_stats.items()}
    size_values = {int(region_id): float(stats.get("size", 0.0)) for region_id, stats in region_stats.items()}

    radius_norm = _normalize_metric(radius_values)
    coupling_norm = _normalize_metric(coupling_values)
    size_norm = _normalize_metric(size_values)
    delta_ctrl_norm = _normalize_metric(lesion_delta_ctrl)
    ls_ctrl_norm = _normalize_metric(lesion_ls_ctrl)

    region_evidence: dict[int, dict] = {}
    if evidence_available:
        candidate_ids = []
        for region_id in region_stats:
            rid = int(region_id)
            delta_ctrl = lesion_delta_ctrl.get(rid, 0.0)
            ls_ctrl = lesion_ls_ctrl.get(rid, 0.0)
            if delta_ctrl > 0.02 and ls_ctrl > 0.05:
                candidate_ids.append(rid)
        if not candidate_ids:
            return -1, {"method": "lesion_guided", "reason": "no_region_passed_control_evidence_threshold"}
        best_region = max(
            candidate_ids,
            key=lambda rid: (
                0.55 * ls_ctrl_norm.get(rid, 0.0)
                + 0.3 * delta_ctrl_norm.get(rid, 0.0)
                + 0.1 * radius_norm.get(rid, 0.0)
                + 0.05 * coupling_norm.get(rid, 0.0)
            ),
        )
        for rid in candidate_ids:
            region_evidence[rid] = {
                "delta_ctrl": lesion_delta_ctrl.get(rid, 0.0),
                "ls_ctrl": lesion_ls_ctrl.get(rid, 0.0),
                "mean_radius": radius_values.get(rid, 0.0),
                "ctrl_coupling": coupling_values.get(rid, 0.0),
                "size": size_values.get(rid, 0.0),
                "control_score": (
                    0.55 * ls_ctrl_norm.get(rid, 0.0)
                    + 0.3 * delta_ctrl_norm.get(rid, 0.0)
                    + 0.1 * radius_norm.get(rid, 0.0)
                    + 0.05 * coupling_norm.get(rid, 0.0)
                ),
            }
        return best_region, {
            "method": "lesion_guided",
            "candidate_region_ids": candidate_ids,
            "selected_region_id": best_region,
            "region_evidence": region_evidence,
        }

    best_region = -1
    best_score = None
    radii = [stats["mean_radius"] for stats in region_stats.values()]
    radius_median = float(np.median(radii)) if radii else 0.0
    for region_id, stats in region_stats.items():
        if stats["mean_radius"] < radius_median:
            continue
        score = stats["ctrl_coupling"] + 0.05 * stats["size"]
        if best_score is None or score > best_score:
            best_score = score
            best_region = int(region_id)
    return best_region, {"method": "structural_fallback"}


def compute_res(main_A_stats: dict[int, dict], main_B_stats: dict[int, dict]) -> float:
    size_A = max((stats["size"] for stats in main_A_stats.values()), default=0)
    size_B = max((stats["size"] for stats in main_B_stats.values()), default=0)
    return float(size_B - size_A)


def compute_rprs(before_stats: dict[int, dict], after_stats: dict[int, dict]) -> float:
    size_before = max((stats["size"] for stats in before_stats.values()), default=0)
    size_after = max((stats["size"] for stats in after_stats.values()), default=0)
    return float(size_after - size_before)


def compare_region_maps(region_map_a: dict[int, int], region_map_b: dict[int, int]) -> float:
    shared_nodes = sorted(set(region_map_a.keys()) & set(region_map_b.keys()))
    labels_a = [region_map_a[node] for node in shared_nodes]
    labels_b = [region_map_b[node] for node in shared_nodes]
    return float(adjusted_rand_score(labels_a, labels_b)) if shared_nodes else 0.0
