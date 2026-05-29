"""Core implementation of the LG-DWMASAM temporal edge ranking model.

The code follows the formulas extracted from the provided paper:

* directed weighted line graph conversion;
* product, harmonic, and geometric line-edge weight operators;
* TCM, arc clustering, and K-shell/Jaccard neighborhood matrices;
* coefficient-of-variation fusion into TSC;
* temporal inter-layer coupling and the 2|E|T by 2|E|T ASAM;
* leading-eigenvector edge scores plus validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import math
import numpy as np


WeightOperator = Literal["product", "harmonic", "geometric"]


@dataclass(frozen=True, order=True)
class Edge:
    """A directed edge in the original temporal network."""

    source: str
    target: str

    @property
    def label(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class TemporalNetwork:
    """A fixed-universe temporal directed weighted network.

    ``weights[t, i]`` stores the weight of global edge ``i`` in time layer
    ``t``. A zero value means that edge is inactive in that layer.
    """

    times: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.weights.shape != (len(self.times), len(self.edges)):
            raise ValueError(
                "weights must have shape (number_of_times, number_of_edges)"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain non-finite values")
        if np.any(self.weights < 0):
            raise ValueError("LG-DWMASAM expects non-negative edge weights")

    @property
    def time_count(self) -> int:
        return len(self.times)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def active_mask(self) -> np.ndarray:
        return self.weights > 0

    @classmethod
    def from_records(
        cls, records: Iterable[tuple[str, str, str, float]]
    ) -> "TemporalNetwork":
        records = list(records)
        if not records:
            raise ValueError("at least one temporal edge record is required")

        time_order: list[str] = []
        edge_order: list[Edge] = []
        nodes: set[str] = set()
        weights_by_key: dict[tuple[str, Edge], float] = {}

        for time, source, target, weight in records:
            if weight < 0:
                raise ValueError(f"negative weight for {source}->{target} at {time}")
            edge = Edge(str(source), str(target))
            time = str(time)
            nodes.update([edge.source, edge.target])
            if time not in time_order:
                time_order.append(time)
            if edge not in edge_order:
                edge_order.append(edge)
            weights_by_key[(time, edge)] = weights_by_key.get((time, edge), 0.0) + float(
                weight
            )

        times = tuple(sorted(time_order, key=_natural_time_key))
        edges = tuple(edge_order)
        edge_index = {edge: i for i, edge in enumerate(edges)}
        time_index = {time: i for i, time in enumerate(times)}
        weights = np.zeros((len(times), len(edges)), dtype=float)
        for (time, edge), weight in weights_by_key.items():
            weights[time_index[time], edge_index[edge]] = weight

        return cls(
            times=times,
            nodes=tuple(sorted(nodes)),
            edges=edges,
            weights=weights,
        )

    def remove_edges(self, edge_indices: Iterable[int]) -> "TemporalNetwork":
        new_weights = self.weights.copy()
        idx = list(edge_indices)
        if idx:
            new_weights[:, idx] = 0.0
        return TemporalNetwork(
            times=self.times,
            nodes=self.nodes,
            edges=self.edges,
            weights=new_weights,
        )


@dataclass(frozen=True)
class LineGraphLayer:
    time: str
    line_weights: np.ndarray
    directed_adjacency: np.ndarray
    undirected_adjacency: np.ndarray
    p_in: np.ndarray
    transfer_capacity: np.ndarray
    arc_clustering: np.ndarray
    kshell_neighborhood: np.ndarray
    tsc: np.ndarray
    cv_weights: tuple[float, float, float]
    node_capacity: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    network: TemporalNetwork
    layers: tuple[LineGraphLayer, ...]
    asam: np.ndarray
    eigenvalue: float
    eigenvector: np.ndarray
    scores_by_time: np.ndarray
    aggregate_scores: np.ndarray
    ranking: tuple[int, ...]
    monotonicity: float
    lcc_before: float
    lcc_after: float
    deletion_drop_rate: float
    cascade_by_time: tuple[dict[str, float], ...]


def _natural_time_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=np.abs(denominator) > 1e-12)
    return out


def map_line_weight(w_i: float, w_j: float, operator: WeightOperator) -> float:
    if operator == "product":
        return w_i * w_j
    if operator == "harmonic":
        denominator = w_i + w_j
        return 0.0 if denominator <= 0 else (2.0 * w_i * w_j) / denominator
    if operator == "geometric":
        return math.sqrt(w_i * w_j)
    raise ValueError(f"unknown weight operator: {operator}")


def build_line_graph_weights(
    network: TemporalNetwork,
    time_index: int,
    operator: WeightOperator = "product",
    allow_self_transitions: bool = False,
) -> np.ndarray:
    """Build W^L(t) for one temporal layer."""

    edge_count = network.edge_count
    original_weights = network.weights[time_index]
    active = original_weights > 0
    by_source: dict[str, list[int]] = {}
    for j, edge in enumerate(network.edges):
        if active[j]:
            by_source.setdefault(edge.source, []).append(j)

    line_weights = np.zeros((edge_count, edge_count), dtype=float)
    for i, edge_i in enumerate(network.edges):
        if not active[i]:
            continue
        for j in by_source.get(edge_i.target, []):
            if i == j and not allow_self_transitions:
                continue
            line_weights[i, j] = map_line_weight(
                original_weights[i], original_weights[j], operator
            )
    return line_weights


def _directed_adjacency(line_weights: np.ndarray) -> np.ndarray:
    adjacency = (line_weights > 0).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _undirected_adjacency(line_weights: np.ndarray) -> np.ndarray:
    adjacency = ((line_weights > 0) | (line_weights.T > 0)).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def transfer_capacity_matrix(line_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Formula (2-1), (2-2), and (2-3)."""

    col_sums = line_weights.sum(axis=0, keepdims=True)
    row_sums = line_weights.sum(axis=1, keepdims=True)
    p_in = _safe_divide(line_weights, col_sums)
    p_out = _safe_divide(line_weights, row_sums)
    tcm = p_in + p_out.T
    np.fill_diagonal(tcm, 0.0)
    return tcm, p_in


def arc_clustering_matrix(line_weights: np.ndarray) -> np.ndarray:
    """Formula (2-4) for the line-graph arc clustering matrix."""

    n = line_weights.shape[0]
    active = line_weights > 0
    in_degree = active.sum(axis=0).astype(float)
    out_degree = active.sum(axis=1).astype(float)
    cube = np.cbrt(line_weights)
    clustering = np.zeros_like(line_weights, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denominator = in_degree[i] * out_degree[j]
            if denominator <= 0 or line_weights[i, j] <= 0:
                continue
            triangle_flow = np.cbrt(line_weights[j, :] * line_weights[:, i]).sum()
            clustering[i, j] = cube[i, j] * triangle_flow / denominator

    return clustering


def core_numbers_undirected(adjacency: np.ndarray) -> np.ndarray:
    """Compute undirected core numbers without external graph libraries."""

    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    active = adjacency > 0
    remaining = np.ones(n, dtype=bool)
    degree = active.sum(axis=1).astype(int)
    core = np.zeros(n, dtype=float)
    current_core = 0

    for _ in range(n):
        masked_degree = np.where(remaining, degree, np.iinfo(np.int32).max)
        node = int(np.argmin(masked_degree))
        current_core = max(current_core, int(degree[node]))
        core[node] = current_core
        remaining[node] = False
        neighbors = np.flatnonzero(active[node] & remaining)
        degree[neighbors] -= 1

    return core


def jaccard_similarity_matrix(adjacency: np.ndarray) -> np.ndarray:
    active = (adjacency > 0).astype(float)
    common = active @ active
    degree = active.sum(axis=1)
    union = degree[:, None] + degree[None, :] - common
    similarity = _safe_divide(common, union)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def kshell_neighborhood_matrix(
    undirected_adjacency: np.ndarray,
    q_value: float = 1.0,
) -> np.ndarray:
    """Formula (2-5) and (2-6).

    The paper names Q^L(t) but does not define its estimation rule in the
    provided manuscript. This implementation exposes it as ``q_value`` and
    uses 1.0 by default, making the formula reduce to degree plus core level.
    """

    degree = undirected_adjacency.sum(axis=1)
    core = core_numbers_undirected(undirected_adjacency)
    node_strength = degree + core * float(q_value)
    pair_strength = np.outer(node_strength, node_strength)
    similarity = jaccard_similarity_matrix(undirected_adjacency)
    sk = pair_strength * similarity
    np.fill_diagonal(sk, 0.0)
    return sk


def coefficient_of_variation_weights(
    matrices: Sequence[np.ndarray],
    normalize_matrices: bool = True,
) -> tuple[tuple[float, ...], tuple[np.ndarray, ...]]:
    """Compute CV fusion weights for TCM, SK, and C."""

    if not matrices:
        raise ValueError("at least one matrix is required")

    normalized: list[np.ndarray] = []
    cvs: list[float] = []
    for matrix in matrices:
        work = np.array(matrix, dtype=float, copy=True)
        np.fill_diagonal(work, 0.0)
        if normalize_matrices:
            max_value = float(np.max(work)) if work.size else 0.0
            if max_value > 1e-12:
                work = work / max_value
        values = work[~np.eye(work.shape[0], dtype=bool)] if work.size else work
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 0.0
        cv = 0.0 if abs(mean) <= 1e-12 else std / abs(mean)
        normalized.append(work)
        cvs.append(cv)

    total_cv = sum(cvs)
    if total_cv <= 1e-12:
        weights = tuple(1.0 / len(matrices) for _ in matrices)
    else:
        weights = tuple(cv / total_cv for cv in cvs)
    return weights, tuple(normalized)


def node_transmission_capacity(
    line_weights: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Formula (2-8)."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in the open interval (0, 1)")

    max_weight = float(np.max(line_weights)) if line_weights.size else 0.0
    active = line_weights > 0
    degree = active.sum(axis=0) + active.sum(axis=1)
    max_degree = float(np.max(degree)) if degree.size else 0.0
    denominator = max_weight * max_degree
    if denominator <= 1e-12:
        return np.zeros(line_weights.shape[0], dtype=float)

    in_strength = line_weights.sum(axis=0)
    out_strength = line_weights.sum(axis=1)
    return (alpha * in_strength + (1.0 - alpha) * out_strength) / denominator


def inter_layer_similarity(
    capacities: np.ndarray,
    active_mask: np.ndarray,
    start: int,
    end: int,
    require_endpoint_activity: bool = True,
) -> np.ndarray:
    """Formula (2-9) and (2-10)."""

    if end <= start:
        raise ValueError("end must be greater than start")

    span = end - start
    if span == 1:
        similarity = np.sqrt(np.maximum((capacities[end] + capacities[start]) / 2.0, 0.0))
    else:
        terms = capacities[start].copy()
        for idx in range(start + 1, end + 1):
            terms += (capacities[idx] + capacities[idx - 1]) / 2.0
        similarity = np.power(np.maximum(terms / span, 0.0), 1.0 / (span + 1))

    if require_endpoint_activity:
        similarity = similarity * (active_mask[start] & active_mask[end])
    return similarity


def build_asam(
    network: TemporalNetwork,
    layers: Sequence[LineGraphLayer],
    require_endpoint_activity: bool = True,
) -> np.ndarray:
    """Build the 2|E|T by 2|E|T line-graph ASAM block matrix."""

    time_count = network.time_count
    edge_count = network.edge_count
    size = 2 * time_count * edge_count
    asam = np.zeros((size, size), dtype=float)
    capacities = np.vstack([layer.node_capacity for layer in layers])
    active_mask = network.active_mask

    def block_slice(time_index: int, copy_index: int) -> slice:
        block = 2 * time_index + copy_index
        start = block * edge_count
        return slice(start, start + edge_count)

    for t, layer in enumerate(layers):
        first = block_slice(t, 0)
        second = block_slice(t, 1)
        asam[first, first] = layer.undirected_adjacency
        asam[second, second] = layer.undirected_adjacency
        asam[first, second] = layer.tsc
        asam[second, first] = layer.tsc.T

    for start in range(time_count):
        for end in range(start + 1, time_count):
            diag_values = inter_layer_similarity(
                capacities,
                active_mask,
                start,
                end,
                require_endpoint_activity=require_endpoint_activity,
            )
            coupling = np.diag(diag_values)
            for copy_index in (0, 1):
                left = block_slice(start, copy_index)
                right = block_slice(end, copy_index)
                asam[left, right] = coupling
                asam[right, left] = coupling

    return asam


def leading_eigenvector_power(
    matrix: np.ndarray,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Power iteration for the non-negative symmetric ASAM."""

    n = matrix.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=float)

    vector = np.full(n, 1.0 / math.sqrt(n), dtype=float)
    eigenvalue = 0.0
    for _ in range(max_iter):
        next_vector = matrix @ vector
        norm = float(np.linalg.norm(next_vector))
        if norm <= 1e-12:
            return 0.0, np.zeros(n, dtype=float)
        next_vector /= norm
        if np.linalg.norm(next_vector - vector) <= tol:
            vector = next_vector
            break
        vector = next_vector

    eigenvalue = float(vector @ (matrix @ vector))
    return eigenvalue, np.abs(vector)


def run_lg_dwmasam(
    network: TemporalNetwork,
    operator: WeightOperator = "product",
    alpha: float = 0.5,
    kshell_q: float = 1.0,
    normalize_matrices: bool = True,
    top_ratio: float = 0.2,
    cascade_seed_ratio: float = 0.2,
    cascade_threshold: float = 0.35,
    require_endpoint_activity: bool = True,
) -> ExperimentResult:
    """Run the full LG-DWMASAM experiment pipeline."""

    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    if not 0 < cascade_seed_ratio <= 1:
        raise ValueError("cascade_seed_ratio must be in (0, 1]")
    if not 0 <= cascade_threshold <= 1:
        raise ValueError("cascade_threshold must be in [0, 1]")

    layers: list[LineGraphLayer] = []
    for t, time in enumerate(network.times):
        line_weights = build_line_graph_weights(network, t, operator)
        directed = _directed_adjacency(line_weights)
        undirected = _undirected_adjacency(line_weights)
        tcm, p_in = transfer_capacity_matrix(line_weights)
        clustering = arc_clustering_matrix(line_weights)
        sk = kshell_neighborhood_matrix(undirected, q_value=kshell_q)
        cv_weights, normalized = coefficient_of_variation_weights(
            (tcm, sk, clustering),
            normalize_matrices=normalize_matrices,
        )
        tsc = (
            cv_weights[0] * normalized[0]
            + cv_weights[1] * normalized[1]
            + cv_weights[2] * normalized[2]
        )
        np.fill_diagonal(tsc, 0.0)
        capacity = node_transmission_capacity(line_weights, alpha=alpha)
        layers.append(
            LineGraphLayer(
                time=time,
                line_weights=line_weights,
                directed_adjacency=directed,
                undirected_adjacency=undirected,
                p_in=p_in,
                transfer_capacity=tcm,
                arc_clustering=clustering,
                kshell_neighborhood=sk,
                tsc=tsc,
                cv_weights=(
                    float(cv_weights[0]),
                    float(cv_weights[1]),
                    float(cv_weights[2]),
                ),
                node_capacity=capacity,
            )
        )

    asam = build_asam(
        network,
        layers,
        require_endpoint_activity=require_endpoint_activity,
    )
    eigenvalue, eigenvector = leading_eigenvector_power(asam)
    scores_by_time = scores_from_eigenvector(network, eigenvector)
    scores_by_time *= network.active_mask
    aggregate_scores = aggregate_edge_scores(scores_by_time, network.active_mask)
    ranking = tuple(np.argsort(-aggregate_scores, kind="mergesort").tolist())
    monotonicity = ranking_monotonicity(aggregate_scores)

    selected_for_deletion = select_top_edges(aggregate_scores, top_ratio)
    lcc_before, _ = temporal_lcc_utility(network)
    lcc_after, _ = temporal_lcc_utility(network.remove_edges(selected_for_deletion))
    deletion_drop_rate = 0.0
    if lcc_before > 1e-12:
        deletion_drop_rate = (lcc_before - lcc_after) / lcc_before

    seeds = select_top_edges(aggregate_scores, cascade_seed_ratio)
    cascade_results: list[dict[str, float]] = []
    for t, layer in enumerate(layers):
        active_seed_count = sum(1 for idx in seeds if network.weights[t, idx] > 0)
        cascade = cascade_linear_threshold(
            layer.p_in,
            seeds,
            threshold=cascade_threshold,
            active_mask=network.active_mask[t],
        )
        cascade_results.append(
            {
                "time_index": float(t),
                "seed_count": float(active_seed_count),
                "activated_count": float(cascade["activated_count"]),
                "active_fraction": float(cascade["active_fraction"]),
                "depth": float(cascade["depth"]),
            }
        )

    return ExperimentResult(
        network=network,
        layers=tuple(layers),
        asam=asam,
        eigenvalue=eigenvalue,
        eigenvector=eigenvector,
        scores_by_time=scores_by_time,
        aggregate_scores=aggregate_scores,
        ranking=ranking,
        monotonicity=monotonicity,
        lcc_before=lcc_before,
        lcc_after=lcc_after,
        deletion_drop_rate=deletion_drop_rate,
        cascade_by_time=tuple(cascade_results),
    )


def scores_from_eigenvector(network: TemporalNetwork, eigenvector: np.ndarray) -> np.ndarray:
    """Formula (2-11) in the manuscript text."""

    time_count = network.time_count
    edge_count = network.edge_count
    expected = 2 * time_count * edge_count
    if eigenvector.size != expected:
        raise ValueError(f"eigenvector length must be {expected}")

    scores = np.zeros((time_count, edge_count), dtype=float)
    for t in range(time_count):
        base = 2 * edge_count * t
        scores[t] = eigenvector[base : base + edge_count] + eigenvector[
            base + edge_count : base + 2 * edge_count
        ]
    return scores


def aggregate_edge_scores(scores_by_time: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    active_count = active_mask.sum(axis=0)
    totals = scores_by_time.sum(axis=0)
    aggregate = np.zeros(scores_by_time.shape[1], dtype=float)
    np.divide(totals, active_count, out=aggregate, where=active_count > 0)
    return aggregate


def ranking_monotonicity(scores: np.ndarray, decimals: int = 12) -> float:
    """Formula (3-1)."""

    count = scores.size
    if count <= 1:
        return 1.0
    rounded = np.round(scores.astype(float), decimals=decimals)
    _, group_counts = np.unique(rounded, return_counts=True)
    tied_pairs = float(sum(c * (c - 1) for c in group_counts))
    return 1.0 - tied_pairs / float(count * (count - 1))


def select_top_edges(scores: np.ndarray, ratio: float) -> tuple[int, ...]:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    count = max(1, int(math.ceil(scores.size * ratio)))
    order = np.argsort(-scores, kind="mergesort")
    return tuple(int(i) for i in order[:count])


def temporal_lcc_utility(network: TemporalNetwork) -> tuple[float, tuple[float, ...]]:
    """Formula (3-2), averaged across temporal layers."""

    values = tuple(layer_lcc_utility(network, t) for t in range(network.time_count))
    return float(np.mean(values)) if values else 0.0, values


def layer_lcc_utility(network: TemporalNetwork, time_index: int) -> float:
    weights = network.weights[time_index]
    total_weight = float(weights.sum())
    if total_weight <= 1e-12 or network.node_count == 0:
        return 0.0

    adjacency: dict[str, list[str]] = {node: [] for node in network.nodes}
    for edge, weight in zip(network.edges, weights):
        if weight > 0:
            adjacency[edge.source].append(edge.target)

    components = strongly_connected_components(adjacency)
    best = 0.0
    for component in components:
        component_nodes = set(component)
        component_weight = 0.0
        for edge, weight in zip(network.edges, weights):
            if weight > 0 and edge.source in component_nodes and edge.target in component_nodes:
                component_weight += float(weight)
        best = max(best, len(component_nodes) * component_weight)

    return best / (network.node_count * total_weight)


def strongly_connected_components(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan SCC implementation for directed graphs."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            components.append(component)

    for node in adjacency:
        if node not in indices:
            visit(node)

    return components


def cascade_linear_threshold(
    p_in: np.ndarray,
    seeds: Sequence[int],
    threshold: float = 0.35,
    active_mask: np.ndarray | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Formula (3-4) linear-threshold cascade on one line graph layer."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    n = p_in.shape[0]
    if active_mask is None:
        active_mask = np.ones(n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
    active = np.zeros(n, dtype=bool)
    for seed in seeds:
        if 0 <= seed < n and active_mask[seed]:
            active[seed] = True

    if max_steps is None:
        max_steps = max(1, n)

    depth = 0
    for step in range(1, max_steps + 1):
        inactive = (~active) & active_mask
        if not np.any(inactive):
            break
        influence = active.astype(float) @ p_in
        newly_active = inactive & (influence >= threshold)
        if not np.any(newly_active):
            break
        active |= newly_active
        depth = step

    active_total = int(active_mask.sum())
    activated_count = int(active.sum())
    active_fraction = 0.0 if active_total == 0 else activated_count / active_total
    return {
        "activated_count": float(activated_count),
        "active_fraction": float(active_fraction),
        "depth": float(depth),
    }
