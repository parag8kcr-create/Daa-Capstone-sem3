"""
main.py
Entry point for the Mini-Courier Planner prototype.
Loads sample data, runs Sorting -> 0/1 Knapsack DP -> Dijkstra + NN-TSP,
and prints detailed execution evidence for Phase 2 and Phase 3.
"""

import json
from pathlib import Path
from algorithms import (
    merge_sort_parcels,
    knapsack_01_pack,
    nearest_neighbor_tsp,
)


def load_scenario(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Convert edge list into adjacency list
    graph = {}
    for u, v, w in raw["edges"]:
        graph.setdefault(u, []).append((v, float(w)))
        graph.setdefault(v, []).append((u, float(w)))

    return {
        "depot": raw["depot"],
        "vehicle_capacity": raw["vehicle_capacity"],
        "parcels": raw["parcels"],
        "graph": graph,
    }


def main() -> None:
    data_path = Path(__file__).parent / "data" / "sample_input.json"
    scenario = load_scenario(str(data_path))

    depot = scenario["depot"]
    capacity = scenario["vehicle_capacity"]
    parcels = scenario["parcels"]
    graph = scenario["graph"]

    print("=" * 68)
    print("MINI-COURIER PLANNER : INTEGRATED ALGORITHMIC DISPATCH")
    print("=" * 68)

    # Phase 2A: Parcel Prioritization via Merge Sort
    sorted_parcels = merge_sort_parcels(parcels)
    print("\n[PHASE 2A] Sorted Parcels (by Deadline ASC, Value DESC):")
    print(f"{'ID':<8} {'Destination':<12} {'Deadline(h)':<13} {'Weight(kg)':<12} {'Value':<8}")
    print("-" * 55)
    for p in sorted_parcels:
        print(f"{p['id']:<8} {p['destination']:<12} {p['deadline']:<13} {p['weight']:<12} {p['value']:<8}")

    # Phase 2B: Optimal Trip Packing via 0/1 Knapsack DP
    max_value, packed_parcels, dp_table = knapsack_01_pack(sorted_parcels, capacity)
    total_weight = sum(p["weight"] for p in packed_parcels)

    print(f"\n[PHASE 2B] 0/1 Knapsack Packing Plan (Capacity = {capacity} kg):")
    print(f"DP Table Dimensions : {len(dp_table)} rows x {len(dp_table[0])} cols")
    print(f"Optimal Priority Val: {max_value} points")
    print(f"Total Loaded Weight : {total_weight} / {capacity} kg")
    print("Selected Parcels    :", ", ".join(f"{p['id']} ({p['destination']})" for p in packed_parcels))

    # Phase 3: Route Planning via Dijkstra + Nearest-Neighbor TSP
    delivery_stops = [p["destination"] for p in packed_parcels]
    tour, total_dist, legs = nearest_neighbor_tsp(graph, depot, delivery_stops)

    print("\n[PHASE 3] Shortest-Path Legs & Nearest-Neighbor TSP Route:")
    for src, dst, dist, sub_path in legs:
        path_str = " -> ".join(sub_path)
        print(f"  {src:<10} to {dst:<10} | Dist: {dist:>5.1f} km | Path: {path_str}")

    print("\nFinal Delivery Sequence :", " -> ".join(tour))
    print(f"Total Route Length      : {total_dist:.1f} km")
    print("=" * 68)


if __name__ == "__main__":
    main()