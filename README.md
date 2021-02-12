This repository contains code for reproducing results in the associated manuscript: "**Beyond Euler/Cardan analysis: true glenohumeral axial rotation during arm elevation and rotation**".

#### Installation Instructions

This repository relies on [Anaconda](https://www.anaconda.com/products/individual) for installing dependencies. All commands below should be run from the Anaconda Prompt. After installing Anaconda and cloning the repository, make sure you are in the same directory as `environment.yml` and run:

`conda env create --file environment.yml`

On Windows 10 (and potentially other platforms) Anaconda has issues with `pyzmq`. The easiest way to fix these issues is to start an elevated (Administrator) Anaconda Prompt, then:

```
conda activate true-vs-apparent-axial-rot
pip uninstall pyzmq
pip install pyzmq
```

Since Anaconda has [stopped automatically setting environments up as Jupyter kernels](https://stackoverflow.com/a/44786736/2577053), this step will need to be performed manually:

`python -m ipykernel install --user --name true-vs-apparent-axial-rot --display-name "Python (true-vs-apparent-axial-rot)"`

Whenever running a Jupyter notebook it's important to select `Python (true-vs-apparent-axial-rot)` as the kernel because it will contain the necessary dependencies.

#### Configuration

All code within the repository relies on two configuration files (`logging.ini` and `parameters.json`) to locate the associated data repository, configure analysis parameters, and instantiate logging. The two Jupyter notebooks within the repository assume that these files reside in the `config` directory within the root of the repository. For all other analysis tasks the location of the `config` directory is specified as a command line parameter (so it is feasible for this folder to reside anywhere in the filesystem). Analysis and figure generation tasks must be executed as module scripts:

`python -m true_vs_apparent.figures.plane_difference config`

Template `logging - template.ini` and `parameters - template.json` files are located in the `config` directory and should be copied and renamed to `logging.ini` and `parameters.json`. [Parameters.md](Parameters.md) explains each parameter within `parameters.json`. Each analysis and figure generation task contains Python documentation describing its utility and the parameters that it expects from `parameters.json`.

#### Supporting dataset and data

The associated data repository containing biplane fluoroscopy derived humerus and scapula kinematics can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4536684.svg)](https://doi.org/10.5281/zenodo.4536684). The `biplane_vicon_db_dir` parameter should point to the `database` folder of this repository. This analysis also relies on 8 additional supporting data files which reside in the `data` directory.

#### Organization

The `true_axial_matlab_v3d_python` directory contains code for computing true axial rotation in MATLAB, Visual3D, and Python.

The `config` directory contains configuration files.

The `data` directory contains supporting data files. Two of these indicate the start and end frames that should be utilized when analyzing external rotation at 90&deg; of abduction and external rotation in adduction trials. The remaining 6 files contain glenohumeral and scapulothoracic mean trajectories for coronal plane abduction, scapular plane abduction, and forward elevation from [Ludewig et al.](https://pubmed.ncbi.nlm.nih.gov/19181982/) [1].

The `true_vs_apparent` directory contains code for reproducing the analysis of the associated manuscript. Within `true_vs_apparent` the following packages exist:

* `analysis` - contains code for various analyses that were undertaken to compare true versus apparent axial rotation.
* `figures` - contains code for reproducing the figures in the associated manuscript.
* `common` - contains common code utilized by both `analysis` and `figures`.

#### License

This code is licensed according to the most restrictive license ([GPLv3](https://choosealicense.com/licenses/gpl-3.0/)) of the packages that it utilizes: [spm1d](https://github.com/0todd0000/spm1d).



[1] P.M. Ludewig, V. Phadke, J.P. Braman, D.R. Hassett, C.J. Cieminski, R.F. LaPrade, Motion of the shoulder complex during multiplanar humeral elevation, The Journal of bone and joint surgery. American volume 91(2) (2009) 378-89.