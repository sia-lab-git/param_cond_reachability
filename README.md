# Parameter-Conditioned Reachable Sets for Updating Safety Assurances Online
### [Project Page](https://javierborquez.github.io/ParamCondReachability/) | [Paper](https://arxiv.org/abs/2209.14976)<br>


[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html),
University of Southern California,
and Kensuke Nakamura,
Princeton University

This is the official implementation of the paper "Parameter-Conditioned Reachable Sets for Updating Safety Assurances Online".

## Get started
You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml   (replace torch version with the one compatible with your current CUDA ver.)
conda activate param_cond_brt
```

## High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* training.py contains a generic training routine.
* modules.py contains layers and full neural network modules.
* utils.py contains utility functions.
* diff_operators.py contains implementations of differential operators.
* loss_functions.py contains loss functions for the different experiments.
* ./ExpName_scripts/ contains scripts to reproduce experiment named 'ExpName' in the paper.

## Reproducing experiments

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

To start training for rocketlanding, you can run:
```
python train_hji_parameterconditioned_simplerocketlanding.py --experiment_name rocket1 --pretrain --diffModel --adjust_relative_grads
```
This will regularly save checkpoints in the directory specified by the rootpath in the script. 

Once training is complete to get detailed value functions at different state and time slices run: 
```
python validate_hji_parameterconditioned_simplerocketlanding
```

Or for trajectory simulation run: 
```
python plot_parameterconditioned_simplerocketlanding_trajectory_and_brt
```

## Contact
If you have any questions, please feel free to email the authors.

