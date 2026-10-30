from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def plot_graph_3d(coords, edge_index, edge_weight, node_types, region_map, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    region_colors = {rid: idx for idx, rid in enumerate(sorted(set(region_map.values())))} if region_map else {}
    for node_id in range(coords.shape[0]):
        color = plt.cm.tab20(region_colors.get(region_map.get(node_id, -1), 0) % 20) if region_map else ("tab:blue" if node_types[node_id] == 0 else "tab:orange")
        ax.scatter(coords[node_id, 0].item(), coords[node_id, 1].item(), coords[node_id, 2].item(), c=[color], s=12)
    sample_edges = min(edge_index.shape[1], 200)
    for idx in range(sample_edges):
        src = edge_index[0, idx].item()
        dst = edge_index[1, idx].item()
        xs = [coords[src, 0].item(), coords[dst, 0].item()]
        ys = [coords[src, 1].item(), coords[dst, 1].item()]
        zs = [coords[src, 2].item(), coords[dst, 2].item()]
        ax.plot(xs, ys, zs, alpha=0.15, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_region_size_comparison(stats_A, stats_B, save_path):
    sizes_A = sorted([stats["size"] for stats in stats_A.values()], reverse=True)
    sizes_B = sorted([stats["size"] for stats in stats_B.values()], reverse=True)
    max_len = max(len(sizes_A), len(sizes_B), 1)
    sizes_A = sizes_A + [0] * (max_len - len(sizes_A))
    sizes_B = sizes_B + [0] * (max_len - len(sizes_B))
    x = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, sizes_A, width=0.4, label="A")
    ax.bar(x + 0.2, sizes_B, width=0.4, label="B")
    ax.legend()
    ax.set_xlabel("Region rank")
    ax.set_ylabel("Region size")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_lesion_heatmap(lesion_matrix, save_path):
    region_ids = sorted(lesion_matrix.keys())
    data = np.array([[lesion_matrix[r]["Delta Pred"], lesion_matrix[r]["Delta Mem"], lesion_matrix[r]["Delta Ctrl"]] for r in region_ids])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Pred", "Mem", "Ctrl"])
    ax.set_yticks(np.arange(len(region_ids)))
    ax.set_yticklabels([str(r) for r in region_ids])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_retraction_curve(history, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["control_region_size"], marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Control region size")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
