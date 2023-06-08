# Code for The Moderating Effect of Instant Runoff Voting

This repository accompanies the paper 

> The Moderating Effect of Instant Runoff Voting <br> Kiran Tomlinson, Johan Ugander, and Jon Kleinberg <br> https://arxiv.org/abs/2303.09734

We have included all code for running simulations, all code for generating plots in the paper, and the results files from our simulations.

## Contents
- `plots`: all simulation data plots in the paper
- `results`: compressed simulation results (526 MB)
- `irv.py`: code for running IRV and plurality simulations
- `plot.py`: code for plotting simulation results
- `volumes.nb`: Mathematica notebook for generating winner region figures

## Compute details
Simulations run in 15 minutes on 100 cores of a server with Intel Xeon Gold 6254 CPUs and 1.5TB RAM (but would run fine on a more modest machine with more time).

Simulations run with:
-  Python 3.8.10
    - numpy 1.22.4
    - scipy 1.8.1
    - tqdm 4.64.0 

Plotting code run with:
- Python 3.10.8
    - numpy 1.23.5
    - scipy 1.9.3
    - matplotlib 3.6.2

## Reproducibility
To uncompress results files and regenerate plots:
```
gunzip results/*.gz
python3 plot.py
```

To rerun simulations:
```
python3 irv --threads [THREADS]
```