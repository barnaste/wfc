"""
Wave Function Collapse (WFC) main module.
Implementation of the WFC bitmap-based procedural image generation algorithm;
this module runs the entire algorithm from start to finish, with an optional
path to the input and output image.
"""
import wfc_setup
import wfc_core
import wfc_visual


def main(in_: str = 'images/demo.png', out: str = 'outpot/demo_out.png',
         n: int = 2, w: int = 30, h: int = 30, flag: str = 'auto') -> None:
    """The main function running the WFC algorithm. Accepts optional
    parameters representing input image, output image location, and size
    of tiles to be extracted, as well as height h and width w as
    dimensions for the output image. It additionally accepts a flag
    indicating the form of visaulization desired during generation.

    Preconditions:
        - in_ is a valid image path
        - n > 0
        - w > 0
        - h > 0
        - flag in {'off', 'manual', 'auto'}
        - debug or flag != 'off'
    """
    # load up data and rules
    tileid, tileset = wfc_setup.extract(in_, n)
    adj, freq = wfc_setup.gen_rules(tileset)
    # generate and visualize
    vis = wfc_visual.Visual(w, h, tileid, flag=flag)
    core = wfc_core.Core(adj, freq, w, h, vis)
    wave = core.generate()
    # produce an output image
    wfc_visual.render(out, tileid, wave)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['wfc_setup', 'wfc_core', 'wfc_visual'],
        'allowed-io': [],
        'max-line-length': 120
    })

    # run the project
    main()
