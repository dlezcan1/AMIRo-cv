# AMIRo Computer Vision
## Author: Dimitri Lezcano
This is a repository of computer vision methods (and data used) for Advanced Medical Instrumentations and Robotics 
laboratory. Generally, this contains methods for needle segmentation and shape reconstruction.

### Dependencies
* Python3
* opencv-python
* numpy
* NURBS-python


## Library
* `BSpline1D.py` A 1D continuous BSpline interpolation class
* `Bspline3D.py` A 3D continuous BSpline interpolation class for 3D curves (Not functional)
* `stereo_needle_proc.py` A library of stereo vision image processing functions as well as 3D needle reconstruction algorithms
    * Most functional stereo reconstruction algorithm is from `needle_reconstruction_ref` where it is based on obtaining a reference image
    where the reference image is of the tissue prior to needle insertion subtracted from when the needle is inserted. This is fairly stable for needle segmentation.

### MATLAB
A host of needle shape-sensing functions as well as some geometry functions and testing stereo CV in MATLAB. The version of MATLAB using is R2021a 


## Scripts and Notebooks
There is one computer vision python (Jupyter) notebook,`NeedleCompVision.ipynb`, which is used for testing of image processing algorithms, ideas and stereo reconstruction algorithms.

### `needle_reconstruction.py`
This is a library of stereo imaging classes as well as a script to perform image processing. Requirements of this scipt are that you have a dataset directory in the format of
* `main_directory`
    * `Insertion*/`
        * `0/` the reference insertion images
            * `left.png`
            * `right.png`
        * `XXX/` where XXX is the insertion depth and is a float
            * `left.png`
            * `right.png`
        * ...

The script usage is
```
python \
needle_reconstruction.py [-h] [--insertion-numbers INSERTION_NUMBERS [INSERTION_NUMBERS ...]]
                              [--insertion-depths INSERTION_DEPTHS [INSERTION_DEPTHS ...]]
                              [--show-processed] [--save] [--force-overwrite]
                              [--left-roi TOP_Y TOP_X BOTTOM_Y BOTTOM_X] 
                              [--right-roi TOP_Y TOP_X BOTTOM_Y BOTTOM_X]
                              [--left-blackout LEFT_BLACKOUT [LEFT_BLACKOUT ...]] 
                              [--right-blackout RIGHT_BLACKOUT [RIGHT_BLACKOUT ...]] 
                              [--zoom ZOOM] [--window-size WIDTH HEIGHT] 
                              [--alpha ALPHA] [--subtract-thresh SUBTRACT_THRESH]
                              stereoParamFile dataDirectory

Perform 3D needle reconstruction of the needle insertion experiments.

positional arguments:
  stereoParamFile       Stereo Calibration parameter file
  dataDirectory         Needle Insertion Experiment directory

optional arguments:
  -h, --help            show this help message and exit
  --insertion-numbers INSERTION_NUMBERS [INSERTION_NUMBERS ...]
  --insertion-depths INSERTION_DEPTHS [INSERTION_DEPTHS ...]
                        The insertion depths of the needle to be parsed.
  --show-processed      Show the processed data
  --save                Save the processed data or not
  --force-overwrite     Overwrite previously processed data.
  --left-roi TOP_Y TOP_X BOTTOM_Y BOTTOM_X
                        The left image ROI to use
  --right-roi TOP_Y TOP_X BOTTOM_Y BOTTOM_X
                        The right image ROI to use
  --left-blackout LEFT_BLACKOUT [LEFT_BLACKOUT ...]
                        The blackout regions for the left image
  --right-blackout RIGHT_BLACKOUT [RIGHT_BLACKOUT ...]
                        The blackout regions for the right image
  --zoom ZOOM           The zoom for stereo template matching
  --window-size WIDTH HEIGHT
                        The window size for stereo template matching
  --alpha ALPHA         The alpha parameter for stereo rectification.
  --subtract-thresh SUBTRACT_THRESH
                        The threshold for reference image subtraction.

```

An example of this would be
```
python needle_reconstruction.py --insertion-numbers 1 3 4 --insertion-depths 0 105 110 115 120 --save \
                                --left-roi 50 60 -50 -60 --right-roi 30 40 -30 -40 --zoom 2.5 --window-size 201 51 \
                                --alpha 0.6 --subtract-thresh 60 "path/to/stereo_parameters.mat" "path/to/data/directory"
```
