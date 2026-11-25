"""
algorithms.py
Core algorithmic implementations for the Mini-Courier Planner Capstone:
1. Merge Sort (Divide & Conquer) for parcel prioritization
2. 0/1 Knapsack (Dynamic Programming) for optimal trip packing
3. Dijkstra's Algorithm (Shortest Path) for road network distances
4. Nearest-Neighbor TSP Approximation for delivery route sequencing
"""

import heapq
from typing import Any, Dict, List, Optional, Tuple


def merge_sort_parcels(parcels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts parcels by ascending deadline, then descending priority value.
    Time Complexity: O(n log n) | Space Complexity: O(n)
    """
    if len(parcels) <= 1:
        return parcels[:]

    mid = len(parcels) // 2
    left = merge_sort_parcels(parcels[:mid])
    right = merge_sort_parcels(parcels[mid:])
    return _merge(left, right)


def _merge(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    i = j = 0

    while i < len(left) and j < len(right):
        # Primary key: smaller deadline first; Secondary key: higher priority value first
        left_key = (left[i]["deadline"], -left[i]["value"])
        right_key = (right[j]["deadline"], -right[j]["value"])

        if left_key <= right_key:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def knapsack_01_pack(
    parcels: List[Dict[str, Any]], capacity: int
) -> Tuple[int, List[Dict[str, Any]], List[List[int]]]:
    """
    Selects the optimal subset of parcels maximizing total priority value
    without exceeding vehicle weight capacity using 0/1 Knapsack DP.
    Time Complexity: O(n * W) | Space Complexity: O(n * W)
    """
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = parcels[i - 1]["weight"]
        v_i = parcels[i - 1]["value"]
        for w in range(capacity + 1):
            if w_i <= w:
                dp[i][w] = max(dp[i - 1][w], v_i + dp[i - 1][w - w_i])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack through DP table to recover selected parcels
    selected_parcels: List[Dict[str, Any]] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            parcel = parcels[i - 1]
            selected_parcels.append(parcel)
            w -= parcel["weight"]

    selected_parcels.reverse()
    return dp[n][capacity], selected_parcels, dp


def dijkstra_shortest_paths(
    graph: Dict[str, List[Tuple[str, float]]], start: str
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Computes shortest distances and predecessor map from `start` to all reachable nodes.
    Time Complexity: O((V + E) log V) | Space Complexity: O(V)
    """
    distances: Dict[str, float] = {node: float("inf") for node in graph}
    previous: Dict[str, Optional[str]] = {node: None for node in graph}
    distances[start] = 0.0

    min_heap: List[Tuple[float, str]] = [(0.0, start)]

    while min_heap:
        curr_dist, u = heapq.heappop(min_heap)
        if curr_dist > distances[u]:
            continue

        for v, weight in graph.get(u, []):
            new_dist = curr_dist + weight
            if new_dist < distances.get(v, float("inf")):
                distances[v] = new_dist
                previous[v] = u
                heapq.heappush(min_heap, (new_dist, v))

    return distances, previous


def reconstruct_path(
    previous: Dict[str, Optional[str]], start: str, target: str
) -> List[str]:
    """Reconstructs node-by-node path from Dijkstra's predecessor map."""
    path: List[str] = []
    curr: Optional[str] = target
    while curr is not None:
        path.append(curr)
        if curr == start:
            break
        curr = previous.get(curr)
    path.reverse()
    return path if path and path[0] == start else []


def nearest_neighbor_tsp(
    graph: Dict[str, List[Tuple[str, float]]], depot: str, stops: List[str]
) -> Tuple[List[str], float, List[Tuple[str, str, float, List[str]]]]:
    """
    Approximates the shortest delivery tour starting and ending at `depot`
    across all unique `stops` using Nearest-Neighbor heuristic + Dijkstra distances.
    Time Complexity: O(k * (V + E) log V + k^2) where k = len(stops)
    """
    unique_stops = list(dict.fromkeys(s for s in stops if s != depot))
    relevant_nodes = [depot] + unique_stops

    # Precompute all-pairs shortest distances among depot and stops
    dist_matrix: Dict[str, Dict[str, float]] = {}
    prev_matrix: Dict[str, Dict[str, Optional[str]]] = {}
    for node in relevant_nodes:
        dists, prevs = dijkstra_shortest_paths(graph, node)
        dist_matrix[node] = dists
        prev_matrix[node] = prevs

    unvisited = set(unique_stops)
    current = depot
    tour = [depot]
    total_distance = 0.0
    leg_details: List[Tuple[str, str, float, List[str]]] = []

    while unvisited:
        next_stop = min(unvisited, key=lambda node: dist_matrix[current][node])
        leg_dist = dist_matrix[current][next_stop]
        if leg_dist == float("inf"):
            raise ValueError(f"Destination '{next_stop}' is unreachable from '{current}'.")

        sub_path = reconstruct_path(prev_matrix[current], current, next_stop)
        leg_details.append((current, next_stop, leg_dist, sub_path))
        total_distance += leg_dist
        tour.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    # Return to depot
    return_dist = dist_matrix[current][depot]
    if return_dist == float("inf"):
        raise ValueError(f"Depot '{depot}' is unreachable from '{current}'.")

    return_path = reconstruct_path(prev_matrix[current], current, depot)
    leg_details.append((current, depot, return_dist, return_path))
    total_distance += return_dist
    tour.append(depot)

    return tour, total_distance, leg_details