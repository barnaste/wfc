"""
Wave Function Collapse (WFC) visualization module.

Module Description
==================
Implementation of the WFC bitmap-based procedural image generation algorithm.
This module is concerned the visualization of the output image, as well as
the graphical representation of the image as it is being generated.

Copyright 2024, Stefan Barna, All rights reserved.
"""
from PIL import Image
import numpy as np
import pygame


class Visual:
    """A wave visualizer, responsible for graphically representing the condition of
    cells in any wave by interpolating over possible states they may collapse to.

    Representation Invariants:
        - all(0 <= p <= 255 for p in self._default)
        - len(self._default) == 3
        - self._flag in {'auto', 'manual'}
        - self._tsize > 0
    """
    # Private Instance Attributes:
    #   - _screen: The pygame surface to which we visualize any wave
    #   - _tileid: A mapping between tileIDs and tile pixel data
    #   - _default: The default pixel data; determined by interpolating between the
    #               pixel data of all possible tiles
    #   - _flag: The set of flags considered when visualizing any wave
    #   - _debug: Whether to feature debug visuals
    _screen: pygame.Surface
    _tileid: dict[np.int64, np.array]
    _default: np.array
    _tsize: int

    # flags and behaviour control
    _flag: str
    _debug: bool

    def __init__(self, w: int, h: int, tileid: dict[np.int64, np.array],
                 tsize: int = 8, flag: str = 'off', debug: bool = False) -> None:
        """Initialize a new visualizer onto a surface compatible with any wave of the given
        width w and height h. Accept an optional tsize tile size parameter indicating the
        number of pixels a cell occupies with any given wave on the screen, such that the
        screen is of dimensions w * tsize x h * tsize.

        Visual accepts an optional flag that control the behaviour of the draw method
            - 'off' (default) disables the visualizer altogether
            - 'manual' pauses after visualization to await keyboard input
            - 'auto' enables a pause for a fixed period after visualization
        It furthermore accepts an optional parameter that enables debug visuals when toggled.
        Debug visuals identify which cells have collapsed through red indicators.

        Preconditions:
            - w > 0
            - h > 0
            - tsize > 0
            - flag in {'off', 'manual', 'auto'}
            - debug or flag != 'off'
        """
        if flag == 'off':
            # if the visualizer is turned off, the draw function is disabled -- we do nothing
            self.draw = lambda x: None
        else:
            # otherwise, we initialize pygame and load relevant information
            pygame.init()
            pygame.display.set_caption("Wave")

            self._screen = pygame.display.set_mode((w * tsize, h * tsize))
            self._tileid = tileid
            self._tsize = tsize
            self._flag = flag
            self._debug = debug

            # determine average pixel data through averaging pixel data of all tiles in tileid
            self._default = np.zeros(3)
            for i in range(3):
                self._default[i] = sum(tileid[key][i] for key in tileid) // len(tileid)

    def draw(self, wave: np.ndarray) -> None:
        """Draw the input wave by translating the tileID data it stores into pixel data.

        Preconditions:
        - wave[i, j] stores a list of all tileIDs feasible for cell [i, j], or is None if
            cell [i, j] may take on any tileID (i.e. is unrestricted)
        - every element of wave[i, j] is a tileID stored in self._tileid
        - all(s in self._tileid for i in range(wave.shape[0]) for j in range(wave.shape[1]) for s in wave[i, j])
        - all(wave.shape[i] == self._screen.get_size()[i] // self._tsize for i in range(2))
        """
        # set background colour to the default tile colour
        self._screen.fill(self._default)
        # iterate over every element of the wave
        for i in range(wave.shape[0]):
            for j in range(wave.shape[1]):
                # if the tile at this index is not default (wave[i, j] is None)
                # colour it in either based on the tile it has collapsed to, or
                # as an average of the tiles it could collapse to
                if wave[i, j] is not None:
                    col = np.zeros(3)
                    for k in range(3):
                        col[k] = sum(self._tileid[key][k] for key in wave[i, j]) // len(wave[i, j])
                    pygame.draw.rect(self._screen, col, (j * self._tsize, i * self._tsize,
                                                         self._tsize, self._tsize))

                    # if the debug visual environment is enabled, indicate whether this cell is collapsed
                    if self._debug and len(wave[i, j]) == 1:
                        pygame.draw.rect(self._screen, (255, 0, 0), (j * self._tsize, i * self._tsize,
                                                                     self._tsize, self._tsize), 1)
        pygame.display.flip()
        # await keyboard input if manual flag is toggled
        if self._flag == 'manual':
            pygame.event.clear()
            while pygame.event.wait().type != pygame.KEYDOWN:
                continue


def render(path: str, tileid: dict[np.int64, np.array], wave: np.ndarray) -> None:
    """Render the wave of tileIDs into an output image at the location given by
    path, where the tileID corresponds to the pixel stored in the tileid mapping.

    As stated in the preconditions, all cells in the wave must be singleton sets,
    where each element is found in the input tileid mapping.

    Preconditions:
        - path is a valid image path
        - len(wave.shape) == 2
        - all(len(wave[i, j]) == 1 for i in range(wave.shape[0]) for j in range(wave.shape[1]))
        - all(s in self._tileid for i in range(wave.shape[0]) for j in range(wave.shape[1]) for s in wave[i, j])
    """
    # we set the output ndarray to contain 4 elements per tile: RGBA
    out = np.ndarray((wave.shape[0], wave.shape[1], 3), np.uint8)
    # iterate over every element in the wave and translate it to a pixel
    for i in range(wave.shape[0]):
        for j in range(wave.shape[1]):
            # casting to set here is, for the most part, an unnecessary step
            # as we know all slements of wave are singleton sets (precondition);
            # unfortunately, it is otherwise picked up as an error in the editor
            out[i, j] = tileid[set(wave[i, j]).pop()]
    # render the output image at the given location
    img = Image.fromarray(out)
    img.save(path)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['PIL', 'numpy', 'pygame'],
        'allowed-io': [],
        'max-line-length': 120
    })
