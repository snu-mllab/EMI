# EMI: Exploration with Mutual Information
### In ICML 2019

#### Hyoungseok Kim<sup>\* 1 2</sup>, Jaekyeom Kim<sup>\* 1 2</sup>, Yeonwoo Jeong<sup>1 2</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>3</sup>, [Hyun Oh Song](http://mllab.snu.ac.kr/hyunoh/)<sup>1 2</sup>

<sup><a name="equal">*</a>: Equal contribution, <a name="snu">1</a>: Seoul National University, Department of Computer Science and Engineering, <a name="nprc">2</a>: Neural Processing Research Center, <a name="ucb">3</a>: UC Berkeley, Department of Electrical Engineering and Computer Sciences</sup>

<img src="demos/emi_demo.gif" width="500" />

This codebase contains the source code for our paper, [EMI: Exploration with Mutual Information](https://arxiv.org/abs/1810.01176). 

## Citing this work

Please cite if you find our work helpful to your research:

    @inproceedings{kimICML19,
      Author    = {Hyoungseok Kim and Jaekyeom Kim and Yeonwoo Jeong and Sergey Levine and Hyun Oh Song},
      Title     = {EMI: Exploration with Mutual Information},
      Booktitle = {International Conference on Machine Learning (ICML)},
      Year      = {2019}}

## Environment setup

### Prerequisites

A non-virtual machine with the following components:

* Ubuntu 16.04
* CUDA 8.0
* cuDNN 6.0
* Conda

### Setting up Conda environment

* Run ```conda env create -f environment.yml```.
* After activating the created environment by executing ```conda activate rllab3```, run ```pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip```.

### Setting up MuJoCo

* Create a subdirectory, `./vendor/mujoco/`.
* Obtain a MuJoCo license for your machine by following the instructions from [their website](https://www.roboti.us/license.html) if you don't have one. They offer a number of licensing options including 30-day free trials.
* Copy `mjkey.txt`, the license key file, into `./vendor/mujoco/`.
* Get the version 1.31 of the MuJoCo binaries for Linux from [their website](https://www.roboti.us/download/mjpro131_linux.zip). Unzip the file.
* Copy all the files inside the directory `mjpro131/bin/` from the extracted content, into `./vendor/mujoco/`.


## Running experiments

* Before running experiments, activate the conda environment by running ```conda activate rllab3```.
* To train an EMI agent on *SwimmerGather*, run:

      python examples/trpo_emi_mujoco.py
* To train an EMI agent on *SparseHalfCheetah*, run:

      python examples/trpo_emi_mujoco.py --env=SparseHalfCheetah
* To train an EMI agent on *Montezuma's Revenge*, run:

      python examples/trpo_emi_atari.py
* The first run will end with no operations other than creating a config. Run the command again if you see the configuration message.

## Acknowledgements

This work was partially supported by Samsung Advanced Institute of Technology and Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-01367, BabyMind).

## License

MIT License
