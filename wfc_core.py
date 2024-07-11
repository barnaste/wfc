"""
Wave Function Collapse (WFC) output image generation module.

Module Description
==================
Implementation of the WFC bitmap-based procedural image generation algorithm.
This module is concerned with generation of the output image based on
adjacency and frequency data.

Copyright 2024, Stefan Barna, All rights reserved.
"""
import heapq
from collections import deque
import numpy as np

from wfc_graph import Graph
from wfc_utilities import get_neighbours
from wfc_visual import Visual


class Core:
    """
    The WFC core. Handles image generation given an adjacency graph, frequnecy rules, and
    desired dimensions of the output file.

    NOTE: As stated in the representation inveriants, all tiles must be registered both in
    the frequency distribution and as a vertex in the adjacency graph.

    Representation Invariants:
        - self._wave[i, j] stores a list of all tileIDs feasible for cell i, j, or is None if
            cell i, j may take on any tileID (i.e. is unrestricted)
        - all(key in self._adj._vertices for key in freq)
        - all(key in freq for key in self._adj._vertices)
    """
    # Instance Attributes:
    #     - _wave: the output "wave" -- a grid storing cell states as tile IDs
    #     - _uncollapsed: the number of uncollapsed tiles in the generated output
    #     - _adj: the adjacency graph storing information regarding which tiles may be adjacent
    #     - _freq: the frequency distribution of tiles within the adjacency graph
    #     - _visual: the Visual used for graphically representing the wave during generation
    #     - _entropyq: a min heap storing wave cells of lowest entropy
    _wave: np.ndarray
    _uncollapsed: int
    _adj: Graph
    _freq: dict
    _visual: Visual
    _entropyq: list

    # NOTE: all functions in this class are PRIVATE with the exception of generate(). They are
    # not meant to be called by any external functions or scripts.
    def __init__(self, adj: Graph, freq: dict, w: int, h: int, vis: Visual) -> None:
        """Initialize a new core to generate an array of size w x h storing the ID of tiles in the output.
            - adj is a directed weighted multigraph that permits loops, storing permitted tile adjacencies.
            - w and h specify the dimensions of the output image.
            - vis is a visualizer for the wave during generation

        Preconditions:
            - w > 0
            - h > 0
        """
        # Initialize the output array with None entries -- entries that are None represent regions
        # of the wave with the default entropy (that is, their entropy has not been reduced by
        # the collapse of nearby cells). This is to reduce memory cost of storing the possible
        # states of each cell.
        self._adj = adj
        self._freq = freq
        self._visual = vis
        self._wave = np.ndarray((h, w), set)
        self._uncollapsed = w * h
        self._entropyq = []

    def __entropy(self, cell: tuple[int, int]) -> float:
        """Compute and return the entropy of the input cell.
        The entropy of a cell is given by the formula
            log(W) - (w1*log(w1) + ... + wn*log(wn)) / W
        where w1, ..., wn are the weights corresponding to each tile the current
        cell may collapse to, and W = w1 + ... + wn.

        Return np.nan if the current cell is unrestricted, and may collapse to any
        tile in self._adj. Note that a collapsed cell will return entropy 0 using
        the formula listed above.

        Preconditions:
            - 0 < cell[0] < self._wave.shape[0]
            - 0 < cell[1] < self._wave.shape[1]
        """
        # If this cell can occupy any state, we return np.nan, representing that
        # the entropy is the largest it can possibly be. This is to prevent the
        # unnecessary computations involved for unrestricted cells.
        if self._wave[cell] is None:
            return np.nan
        # Otherwise, we compute the entropy of this cell based on possible tiles
        # it may collapse to, with weights as specified in the frequency dist.
        else:
            tw = 0  # total weight
            logw = 0  # logarithmic weight
            for state in self._wave[cell]:
                tw += self._freq[state]
                logw += self._freq[state] * np.log2(self._freq[state])
            return np.log2(tw) - (logw / tw)

    def __collapse(self, cell: tuple[int, int]) -> None:
        """Collapses the input cell to a fixed tile on the wave (grid).

        The cell is collapsed to a random tile based on the probability distribution
        derived from neighbouring tiles and the frequency rules.

       Preconditions:
            - 0 < cell[0] < self._wave.shape[0]
            - 0 < cell[1] < self._wave.shape[1]
            - self._wave[cell] is None or len(self._wave[cell]) > 1
        """
        self._uncollapsed -= 1
        # if the cell may collapse to ANY state, use the full frequency distribution
        if self._wave[cell] is None:
            states = list(self._freq.keys())
        # otherwise, limit the distribution to the possible states it may collapse to
        else:
            states = list(self._wave[cell])
        # NOTE: we cast states to a list as we need element order for choice function.
        # Now, regardless of how the states are determined, we collapse the cell.
        weights = [self._freq[s] for s in states]
        tw = sum(weights)
        self._wave[cell] = set(np.random.choice(states, 1, p=[w / tw for w in weights]))

    def __reduce(self, cell: tuple[int, int], fringe: deque) -> int:
        """Reduces cell's states based on its neighbouring cells in the wave. If the cell's states
        are indeed reduced (that is, before and after calling this function the number of tiles
        it may collapse to changes), this cell's neighbours to the fringe stack. Furthermore, the
        new entropy of this cell is pushed to the entropyq heap.

        A cell may collapse during this process, when its number of possible states becomes 1.
        In the case that the cell has 0 possible states after reduction, a contradiction is reached,
        and this function returns 1. Otherwise, it returns 0.

        Preconditions:
            - 0 < cell[0] < self._wave.shape[0]
            - 0 < cell[1] < self._wave.shape[1]
            - cells in fringe are valid positions on the wave
        """
        nb = get_neighbours(cell, self._wave.shape[0], self._wave.shape[1])
        states = set(self._freq.keys())   # fetch all possible tiles as default collection
        # update the possible states of the input cell based on neighbours
        for n in nb:
            # we need not consider the cells that have self._wave[cell] is None, as these will
            # not restrict the intersection at all
            if self._wave[n[0]] is not None:
                states.intersection_update(set.union(*(self._adj.adjacent(s, n[1]) for s in self._wave[n[0]])))

        # compare new possible cell states to the cell states prior to calling this method;
        # if they are different, the cell has changed in entropy -- update everything
        if self._wave[cell] != states:
            # we update everything accordingly
            self._wave[cell] = states
            # Check that a contradiction has not occurred (i.e. the number of states
            # that the cell may collapse to is 0). If it has, return 1.
            if len(states) == 0:
                return 1
            # furthermore, if there is only one remaining state the cell may occupy,
            # we may consider it to have collapsed to that cell
            if len(states) == 1:
                self._uncollapsed -= 1
            # note that we know entropy is not np.nan, as cell has reduced states in this branch,
            # and if len(states) == 1 this cell has collapsed and must not be considered again
            else:
                heapq.heappush(self._entropyq, (self.__entropy(cell), cell))

            for n in nb:
                # NOTE that the same cell may be reduced more than once during a single propagation phase.
                # This is intentional -- the change in a neighbour during propagation may change a cell,
                # which in turn may again affect the neighbour!
                if self._wave[n[0]] is None or len(self._wave[n[0]]) > 1:
                    fringe.append(n[0])

        # If we have gotten to this point, we have not reached a contradiction with this cell,
        # so we may return 0.
        return 0

    def __propagate(self, cell: tuple[int, int]) -> int:
        """Propagate the collapse of a cell throughout the wave, so that the possible
        states each cell may collapse to is updated, and the entropyq is kept up to
        date. Return 1 if there is a contradiction. Return 0 otherwise.

        Preconditions:
            - 0 < cell[0] < self._wave.shape[0]
            - 0 < cell[1] < self._wave.shape[1]
        """
        # add all neighbours to the collapsed cell to the stack of affected cells
        # if they are not already collapsed themselves
        fringe = deque()
        for n in get_neighbours(cell, self._wave.shape[0], self._wave.shape[1]):
            if self._wave[n[0]] is None or len(self._wave[n[0]]) > 1:
                fringe.append(n[0])

        # propagate the collapse to each neighbouring tile while there is anything to propagate
        while len(fringe) > 0:
            next_ = fringe.pop()
            if self.__reduce(next_, fringe) == 1:
                return 1
        return 0

    def generate(self) -> np.ndarray:
        """Generate and return a wave of size w x h storing the ID of tiles in the output image,
        using the adjacency and frequency rules stored as a graph and dictionary in this core.
        """
        # choose the first cell to collapse at random
        cell = (np.random.randint(0, self._wave.shape[0]), np.random.randint(0, self._wave.shape[1]))

        # while there is at least one uncollapsed cell in the output array, continue main loop
        while self._uncollapsed > 0:
            # Keep popping until the cell you're at is not collapsed. It is possible that
            # a collapsed cell exists in the queue, as the entropy it is pushed with may have
            # changed in subsequent cell collapses. We must therefore account for this.
            # NOTE that before first iteration, cell will be collapsed already, and thus
            # this condition is guaranteed to be met
            while self._wave[cell] is not None and len(self._wave[cell]) == 1:
                cell = heapq.heappop(self._entropyq)[1]   # entropyq stores (entropy, cell) tuple
            # collapse the lowest entropy cell, and propagate the effects of this collapse
            self.__collapse(cell)

            # check for a contradiction -- if there is a contradiction, reset the wave
            if self.__propagate(cell) == 1:
                self._entropyq = []
                self._wave = np.ndarray((self._wave.shape[0], self._wave.shape[1]), set)
                self._uncollapsed = self._wave.shape[0] * self._wave.shape[1]

            # at the end of every step, visualize the updated wave
            self._visual.draw(self._wave)

        return self._wave


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'heapq', 'collections', 'wfc_graph', 'wfc_utilities', 'wfc_visual'],
        'allowed-io': [],
        'max-line-length': 120
    })
