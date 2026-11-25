"""
benchmark.py
Phase 4: Performance Testing & Runtime Visualization.
Benchmarks Merge Sort, 0/1 Knapsack DP, and Dijkstra + NN-TSP across
increasing input sizes and saves 'runtime_chart.png'.
"""

import random
import time
from algorithms import merge_sort_parcels, knapsack_01_pack, nearest_neighbor_tsp


def generate_synthetic_case(num_parcels: int, num_nodes: int, capacity: int):
    rng = random.Random(42 + num_parcels)
    nodes = ["HUB"] + [f"Node_{i}" for i in range(1, num_nodes)]

    # Build a connected random weighted graph
    graph = {node: [] for node in nodes}
    for i in range(1, num_nodes):
        parent = nodes[rng.randint(0, i - 1)]
        w = float(rng.randint(2, 15))
        graph[nodes[i]].append((parent, w))
        graph[parent].append((nodes[i], w))

    # Add extra cross-edges
    for _ in range(num_nodes):
        u, v = rng.sample(nodes, 2)
        w = float(rng.randint(3, 20))
        graph[u].append((v, w))
        graph[v].append((u, w))

    parcels = []
    for idx in range(num_parcels):
        parcels.append({
            "id": f"P{idx + 1:04d}",
            "destination": rng.choice(nodes[1:]),
            "weight": rng.randint(1, max(2, capacity // 6)),
            "value": rng.randint(10, 100),
            "deadline": rng.randint(1, 24),
        })

    return parcels, graph, capacity


def run_benchmarks():
    configs = [
        ("Small (n=10)", 10, 8, 20),
        ("Medium (n=200)", 200, 30, 100),
        ("Large (n=1000)", 1000, 60, 250),
        ("Stress (n=2000)", 2000, 100, 500),
    ]

    results = []
    print(f"{'Dataset':<18} {'Sort (ms)':<12} {'Knapsack (ms)':<15} {'Routing (ms)':<14} {'Total (ms)':<12}")
    print("-" * 72)

    for label, n, v, cap in configs:
        parcels, graph, capacity = generate_synthetic_case(n, v, cap)

        t0 = time.perf_counter()
        sorted_p = merge_sort_parcels(parcels)
        t1 = time.perf_counter()

        _, packed, _ = knapsack_01_pack(sorted_p, capacity)
        t2 = time.perf_counter()

        stops = [p["destination"] for p in packed]
        nearest_neighbor_tsp(graph, "HUB", stops)
        t3 = time.perf_counter()

        sort_ms = (t1 - t0) * 1000
        dp_ms = (t2 - t1) * 1000
        route_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        results.append((label, sort_ms, dp_ms, route_ms, total_ms))
        print(f"{label:<18} {sort_ms:<12.2f} {dp_ms:<15.2f} {route_ms:<14.2f} {total_ms:<12.2f}")

    try:
        import matplotlib.pyplot as plt

        labels = [r[0] for r in results]
        sort_vals = [r[1] for r in results]
        dp_vals = [r[2] for r in results]
        route_vals = [r[3] for r in results]

        plt.figure(figsize=(8, 4.8))
        plt.plot(labels, sort_vals, marker="o", label="Merge Sort O(n log n)")
        plt.plot(labels, dp_vals, marker="s", label="0/1 Knapsack DP O(n*W)")
        plt.plot(labels, route_vals, marker="^", label="Dijkstra + NN-TSP")
        plt.title("Mini-Courier Planner: Algorithmic Runtime Comparison")
        plt.xlabel("Dataset Scale")
        plt.ylabel("Execution Time (ms)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("runtime_chart.png", dpi=150)
        print("\nSaved runtime comparison chart to 'runtime_chart.png'.")
    except ImportError:
        print("\nNote: Install matplotlib ('pip install matplotlib') to save runtime_chart.png.")


if __name__ == "__main__":
    run_benchmarks()