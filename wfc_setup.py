"""
Wave Function Collapse (WFC) data setup module.

Module Description
==================
Implementation of the WFC bitmap-based procedural image generation algorithm.
This module is concerned with input image extraction, mapping tiles to pixel
data and collecting adjacency and frequency rules.

Copyright 2024, Stefan Barna, All rights reserved.
"""
from PIL import Image
import numpy as np

from wfc_graph import Graph
from wfc_utilities import hash_arr, get_neighbours


def extract(path: str, n: int = 2) -> (dict[np.int64, np.array], np.ndarray[np.int64]):
    """Extract image data from the Image found at input path.
    Return a dictionary mapping tile IDs to the n x n tiles extracted from the input image.
    Return also a numpy array of dimensions equal to input image, where cell [i,j] is the ID
    of the n x n tile whose upper left corner is the pixel [i, j] within the input image.

    Preconditions:
        - path is a valid image path
        - if img is the image in path, n <= min(img width, img height)
        - n > 0
    """
    # open image and grab dimensions
    with Image.open(path) as img:
        w, h = img.size
        tileset = np.zeros((h, w), np.int64)
        tileid = {}

        # setup image array with padding to allow reading N x N tiles that
        # exceed image boundaries and wrap to the other side
        aimg = np.asarray(img.convert("RGB"))
        aimg = np.pad(aimg, ((0, n - 1), (0, n - 1), (0, 0)), 'wrap')
        # aimg has three dimensions: height, width, and pixel description
        # we only wish to wrap around height and width

        # iterate over image columns
        for i in range(h):
            # iterate over image row
            for j in range(w):
                # store tile in dictionary -- if already present nothing will happen
                # NOTE: we store only the TOP-LEFT pixel of the tile in the dictionary.
                # This is because, for every tile placed in the output grid, only a single
                # pixel of the tile may be introduced to the image. We choose the top-left
                # for convenience. We do, however, need the entire tile for hashing.
                key = hash_arr(aimg[i:i + n, j:j + n])
                tileid[key] = aimg[i, j]
                # store a reference to the N x N tile in tiles via the hash key
                tileset[i, j] = key

    return tileid, tileset


def gen_rules(tileset: np.ndarray[np.int64]) -> (Graph, dict):
    """Generate and return directed multigraph based on adjacencies in tiles.
    Vertices are all keys within tileid. Two tiles adjacent in the input tiles array
    T1 and T2 are represented by two edge connections: one from T1 to T2, storing
    the direction (L/R/U/D) T2 is with respect to T1, and one from T2 to T1 analogously.
    Return also a dictionary storing the frequency of each tile, mapping tileID -> freq.

    Preconditions:
        - all(len(tiles[i]) == len(tiles[0]) for i in range(len(tiles)))
    """
    # determine unique tiles and frequencies
    tiles, counts = np.unique(tileset, return_counts=True)
    freq = dict(zip(tiles, counts))

    # add all unique tiles as vertices in the graph
    adj = Graph(set(tiles))
    w, h = tileset.shape[1], tileset.shape[0]

    # iterate over every column
    for i in range(h):
        # iterate over every row
        for j in range(w):
            nb = get_neighbours((i, j), h, w)
            # add adjacency data for every tile adjacent to the current one in cardinal directions
            # using direction of the target wrt. the current tile
            for n in nb:
                adj.add_edge(tileset[n[0]], tileset[i, j], n[1])

    return adj, freq


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['PIL', 'numpy', 'wfc_graph', 'wfc_utilities'],
        'allowed-io': [],
        'max-line-length': 120
    })
