# Blockwise Multi-Order Feature Regression for Real-Time Path Tracing Reconstruction

This is the code used in the paper:
"Blockwise Multi-Order Feature Regression for Real-Time Path Tracing Reconstruction".
by Koskela M., Immonen K., Mäkitalo M., Foi A., Viitanen T., Jääskeläinen P., 
Kultala H., and Takala J.

# Datasets

The dataset to run the code is around 19GB because it contain 60 frames
animations of 7 scenes with references and feature buffers all in
single-precision .exr format.

The datasets can be found here: http://www.tuni.fi/vga/bmfr

# Building

Make sure that you have the dataset's "inputs" folder at the location defined
by `INPUT_DATA_PATH`, which can be found in `opencl/bmfr.cpp`

You need to rebuild the project if you change the files in the location without
changing the `INPUT_DATA_PATH`. It changes the `camera_matrices.h` and the
makefile/project does not check its modification date because the path to it is
defined in the `bmfr.cpp` file.

## Linux

Install OpenCL driver and OpenImageIO library.

`make && ./bmfr` in `opencl` folder should build and run the code.

## Windows

Building and running the `bmfr.sln` with Visual Studio 2017 should work out of
the box.

# Notes

Defines in the `bmfr.cpp` file can be used to:
 * Edit some of the BMFR algorithm's parameters
 * Run the code with different inputs
 * Change some of the optimizations for finding the fastest runtime on your
   target hardware