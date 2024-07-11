Please see the `requirements.txt` document that accompanies this submission for a collection of this project’s Python library dependencies. To initiate the algorithm, simply run the `main.py` module. This will run the only method in the module, `main()`. Please ensure that the parameter `in_` is indeed a valid path to the input image. Additionally, the directory for `out` must exist, but the file does not. I encourage experimentation with the remaining parameters to yield different results.

- `n` specifies the dimensions of the N × N tiles to be extracted from the input image
- `w` and `h` specify the dimensions of the output
- `flag` specifies the form of visualisation desired during wave generation, passed to `Visual`; this flag is set to ‘auto’ to automatically refresh the visualisation every time the wave changes, ‘manual’ to await keyboard input after every wave change, and ‘off’ when no in-progress visualisation is desired

For more information, consult the documentation within each module. It is to be noted that the debug toggle is automatically set to `False`. If readers wish to experiment with the code and enable debug mode, they must add the additional argument `debug=True` to the `Visual` initializer.

Once the module is run, a `pygame` window will appear. The behaviour of this window depends on
the flag specified as input to `main()`. This window will not appear at all if the flag is set to `off`. Once wave generation begins, the display will update at the end of every propagation phase, visualising the extent to which the output has been generated. If the screen resets to a monotone colour, the algorithm has reached a contradiction and reset. Once the wave is entirely collapsed, the program terminates and the `pygame` window closes. After this, the ouput image should be found in the location specified as parameter to `main()`.
