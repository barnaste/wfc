"""
Wave Function Collapse (WFC) utilities module.

Module Description
==================
Implementation of the WFC bitmap-based procedural image generation algorithm.
This module is concerned the implementation of various utilities applied
throughout other modules in the WFC algorithm.

Copyright 2024, Stefan Barna, All rights reserved.
"""
import numpy as np


def hash_arr(arr: np.array) -> int:
    """Return a hash value for the input numpy array.
    WARNING. mutating the array after hashing changes the hash value. Once hashed,
    do not mutate an array that is to be accessed by its hash value.
    """
    return hash(np.array_str(arr))


def get_neighbours(cell: tuple[int, int], h: int, w: int) -> set[tuple[tuple[int, int], str]]:
    """Return the coordinates of the neighbours to the input cell on a grid
    of height h and width w. This method accounts for cells on grid border.
    These coordinates are bound to directional data representing the position
    of the input cell with respect to the neighbour (e.g. L for left).

    Preconditions:
        - 0 < cell[0] < h
        - 0 < cell[1] < w
    """
    return {(((cell[0] - 1) % h, cell[1]), 'D'),
            (((cell[0] + 1) % h, cell[1]), 'U'),
            ((cell[0], (cell[1] - 1) % w), 'R'),
            ((cell[0], (cell[1] + 1) % w), 'L')}


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy'],
        'allowed-io': [],
        'max-line-length': 120
    })
