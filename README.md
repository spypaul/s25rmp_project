# s25rmp
## Prerequisites

The required mesh file should be downloaded from [here](https://sumailsyr-my.sharepoint.com/:u:/g/personal/gkatz01_syr_edu/EUKWyYSia3ZIm-fT52mZBVsBUEYFgXxSADCDQQiRCSegBw?e=KGUUel).  Then, extract the zip archive into a sub-directory of this repository named "meshes".

The [PyBullet](https://pybullet.org/wordpress/) simulator should be installed.

Other required libraries include: Pytorch, numpy, Scipy, and matplotlib. They can be installed through `pip3 install`

## Run

Run this command from within the top-level folder of the repository to start the evaluation:

`python3 ./evaluation.py`

The number of objects can be changed in line 96 of evaluation.py.
