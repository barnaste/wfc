"""
Wave Function Collapse (WFC) graph module.

Module Description
==================
Implementation of the WFC bitmap-based procedural image generation algorithm.
This module is concerned with the underlying directed multigraph used in
the storage of tile adjacency data.

Copyright 2024, Stefan Barna, All rights reserved.
"""
from __future__ import annotations
import numpy as np


class _Vertex:
    """A vertex in the WFC graph.
    NOTE that a vertex may be its own neighbour.
    Edge direction must be L/R/U/D (Preconditions).

    Instance Attributes:
        - id: the tile ID representing the tile stored within this vertex
        - neighbours: the tiles adjacent to this tile in the input image,
            stored with directional information
            i.e. (vertex, direction)

    Representation Invariants:
        - all(n[1] in {'L', 'R', 'U', 'D'} for n in self.neighbours)

    REMARK. direction refers to the direction of the target with respect to self.
    """
    id_: np.int64
    neighbours: set[tuple[_Vertex, str]]

    def __init__(self, id_: np.int64) -> None:
        """Initialize a new vertex with the given id."""
        self.id_ = id_
        self.neighbours = set()


class Graph:
    """A directed multigraph, where each edge stores information about direction.
    Used in WFC to store adjacency information among tiles.
    """
    # Private Instance Attributes:
    #   - _vertices: A dictionary mapping tile ids to vertices in this graph.
    _vertices: dict[np.int64, _Vertex]

    def __init__(self, vertices: set = None) -> None:
        """Initialize a graph.
        The graph may be initialized with a set of vertex ids as parameter.
        Otherwise, it will be initialized as an empty graph.
        """
        if vertices is None:
            vertices = set()
        self._vertices = {}

        # iterate over vertices to add them to graph
        for v in vertices:
            self._vertices[v] = _Vertex(v)

    def add_vertex(self, v: np.int64) -> None:
        """Add a vertex to this graph.
        If the vertex is already in the graph, do nothing.
        """
        if v not in self._vertices:
            self._vertices[v] = _Vertex(v)

    def add_edge(self, src: np.int64, targ: np.int64, dir_: str) -> None:
        """Add an edge from vertex src to vertex targ with directional information dir_.
        Raise a ValueError when either of src or targ are not vertices in this graph.

        Preconditions:
            - dir_ in {'L', 'R', 'U', 'D'}
        """
        if src in self._vertices and targ in self._vertices:
            self._vertices[src].neighbours.add((self._vertices[targ], dir_))
        else:
            raise ValueError

    def adjacent(self, src: np.int64, dir_: str) -> set[np.int64]:
        """Return a set consisting of all vertices in this graph that are pointed to
        from the source vertex src, with directional information matching dir_.
        Raise a ValueError when src is not a vertex in this graph.

        Preconditions:
            - dir_ in {'L', 'R', 'U', 'D'}
        """
        if src in self._vertices:
            return {n[0].id_ for n in self._vertices[src].neighbours if n[1] == dir_}
        else:
            raise ValueError


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['__future__', 'numpy'],
        'allowed-io': [],
        'max-line-length': 120
    })
