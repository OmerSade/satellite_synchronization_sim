"""Graph topology and link-delay model for a LEO-like satellite network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LinkSample:
    """One directed-delay sample for a bidirectional inter-satellite link."""

    delay_ij_ns: float
    delay_ji_ns: float


class SatelliteNetwork:
    """Undirected graph with dynamic link availability and asymmetric delays."""

    def __init__(
        self,
        num_nodes: int,
        adjacency: dict[int, set[int]],
        rng: np.random.Generator,
        base_delay_ns: tuple[float, float] = (150_000.0, 900_000.0),
        asymmetry_std_ns: float = 30.0,
        jitter_std_ns: float = 2.0,
        edge_base_delay: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.adjacency = adjacency
        self.rng = rng
        self.base_delay_ns = base_delay_ns
        self.asymmetry_std_ns = asymmetry_std_ns
        self.jitter_std_ns = jitter_std_ns
        self.disabled_edges: set[tuple[int, int]] = set()
        self._edge_base_delay = (
            {edge: float(delay) for edge, delay in edge_base_delay.items()}
            if edge_base_delay is not None
            else {edge: float(rng.uniform(*base_delay_ns)) for edge in self.edges()}
        )

    @classmethod
    def leo_mesh(
        cls,
        num_nodes: int,
        rng: np.random.Generator,
        ring_degree: int = 4,
        random_links: int | None = None,
        **kwargs: float,
    ) -> "SatelliteNetwork":
        """Create a connected graph that resembles local LEO inter-satellite links.

        Nodes are connected to nearby neighbors on a ring, then sparse cross-links
        are added to mimic inter-plane optical links. This is intentionally not an
        orbital propagator; it is a graph-based testbed for synchronization logic.
        """

        if num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        if ring_degree < 2 or ring_degree % 2 != 0:
            raise ValueError("ring_degree must be an even integer >= 2")

        adjacency = {node: set() for node in range(num_nodes)}
        half_degree = ring_degree // 2
        for node in range(num_nodes):
            for hop in range(1, half_degree + 1):
                cls._add_edge(adjacency, node, (node + hop) % num_nodes)
                cls._add_edge(adjacency, node, (node - hop) % num_nodes)

        if random_links is None:
            random_links = max(num_nodes // 2, 1)
        while random_links > 0:
            i, j = rng.choice(num_nodes, size=2, replace=False)
            if j not in adjacency[int(i)]:
                cls._add_edge(adjacency, int(i), int(j))
                random_links -= 1

        return cls(num_nodes=num_nodes, adjacency=adjacency, rng=rng, **kwargs)

    @staticmethod
    def _add_edge(adjacency: dict[int, set[int]], i: int, j: int) -> None:
        adjacency[i].add(j)
        adjacency[j].add(i)

    @staticmethod
    def _edge_key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def neighbors(self, node: int) -> set[int]:
        """Return currently connected neighbors for ``node``."""

        return {
            neighbor
            for neighbor in self.adjacency[node]
            if self._edge_key(node, neighbor) not in self.disabled_edges
        }

    def edges(self) -> list[tuple[int, int]]:
        """Return sorted undirected edges."""

        return sorted(
            (i, j)
            for i, nbrs in self.adjacency.items()
            for j in nbrs
            if i < j
        )

    def active_edges(self) -> list[tuple[int, int]]:
        """Return sorted edges that are not temporarily disconnected."""

        return [edge for edge in self.edges() if edge not in self.disabled_edges]

    def sample_link(self, i: int, j: int) -> LinkSample:
        """Sample directed propagation delays for an active bidirectional link."""

        edge = self._edge_key(i, j)
        if edge in self.disabled_edges:
            raise ValueError(f"Link {edge} is disabled")
        base = self._edge_base_delay[edge]
        asymmetry = float(self.rng.normal(0.0, self.asymmetry_std_ns))
        jitter_ij = float(self.rng.normal(0.0, self.jitter_std_ns))
        jitter_ji = float(self.rng.normal(0.0, self.jitter_std_ns))
        return LinkSample(
            delay_ij_ns=max(0.0, base + asymmetry / 2.0 + jitter_ij),
            delay_ji_ns=max(0.0, base - asymmetry / 2.0 + jitter_ji),
        )

    def disable_edges(self, edges: Iterable[tuple[int, int]]) -> None:
        """Temporarily disconnect selected links."""

        for i, j in edges:
            self.disabled_edges.add(self._edge_key(i, j))

    def enable_all_edges(self) -> None:
        """Restore all graph links."""

        self.disabled_edges.clear()

    def randomly_toggle_links(self, probability: float) -> None:
        """Randomly disable a fraction of links to model topology changes."""

        self.disabled_edges = {
            edge for edge in self.edges() if self.rng.random() < probability
        }

    def base_delays(self) -> dict[tuple[int, int], float]:
        """Return a copy of deterministic per-edge propagation delays."""

        return dict(self._edge_base_delay)
