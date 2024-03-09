 # Deterministic safety guarantees for learning-based control of monotone nonlinear systems under uncertainty

This repository contains the code for the paper "Deterministic safety guarantees for learning-based control of monotone nonlinear systems under uncertainty" by J.Adamek, M.Heinlein, L.Lüken, S.Lucia. 

**Abstract**:
This paper presents a novel framework to guarantee safety for learning-based control of nonlinear monotone systems under uncertainty. We propose to evaluate online whether a one-step simulation brings a nonlinear system into a robust control invariant (RCI) set. Such evaluation can be very efficiently computed even under the presence of uncertainty for learning-based approximate controllers and monotone systems, which also enable a simple computation of RCI sets. In case the one-step simulation drives the system outside of the RCI set, a fallback strategy is used, which is obtained as a byproduct of the RCI set computation.   
We also develop a method to calculate an $N$-step RCI set to reduce the conservativeness of the proposed strategy and we illustrate the results with a simulation study of a nonlinear monotone system.
-----------------
## Installation
You can install the python environment for this repository by running the following command in the root directory of the repository with the command:
``````
pip install requirements.txt
``````
The code uses Pytorch for the approximate MPC training
and [Do-MPC](https://www.do-mpc.com/en/latest/) as well as [CasADi](https://web.casadi.org/) for the MPC and robust MPC implementation.

## Usage
The main file is found in the subfolder ``n_double_int_mpc``, where you can use the approximate MPC, Robust MPC, MPC as well as the
RCI set as calculated for the paper. If you want to design your own RCI set use  ``RCIS_calc_interactive.py``. The approximate MPC is 
trained using the chain of files starting with ``generate_sampling_plans.py``. The sampling data is generated using the file 
``data_sampling.py`` and ``sampling_to_dataset.py``. The sampling data is then used to train the approximate MPC using the file ``train_approximate_mpc.py``.
Be careful that currently the paths are all set to the default paths, so if you change the settings, you might overwrite the data used within the paper.