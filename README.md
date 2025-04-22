# Covariate-Dependent Graphical Model Estimation via Neural Networks

This is the official repo for the paper titled "Covariate-dependent Graphical Model Estimation via Neural Networks with Statistical Guarantees", by **Jiahe Lin, Yikai Zhang and George Michailidis**, published in _Transactions on Machine Learning Research (TMLR), 2025_. [[Link to paper]](https://openreview.net/pdf?id=beqSqPgE33)

## Setup

The following installs the necessary packages and their dependencies into the venv:
```console
pip install -e .
```

## Repo Outline
* `bin/`: shell scripts for running synthetic data experiments with multiple data replicas; see also **Synthetic Data Experiments** section
* `configs/`:
    - `_synthetic_.yaml`: config file for the parameters used in synthetic data generation;
    - `*.yaml`: config file that stores the hyperparameters used for running the end-to-end training
* `generator/`: scripts used for generating synthetic data
    - `simulator/`:  various simulator objects for synthetic data generation;
    - generated data are automatically saved to `data_sim/${data_str}_seed${data_seed}/`.
* `models/`: main implementation of the models
* `notebook/`: demo for running the model on the S&P 100 stock data (Appendix C of the paper)
* `utils/`: 
    - `datasets/`: `torch.utils.data.dataset.Dataset` objects used for loading the data, depending on their storing format;
    - `utils_train.py`: functions for facilitating each step required for training the neural network;
* `./`:
    - `run_sim.py`: script for running synthetic data experiments with nn-based covariate-dependent graphical model estimation
    - `run_complete.R`: script for running synthetic data experiments using glasso or neighborhood selection (mb). 


## Synthetic Data Experiments

Valid DATA_STR: `GGM1, GGM2, NPN1, NPN2, DAG1, DAG2`. 

* Generate data
    ```console
    ## for Gaussian data
    python -u generator/generate_GGM.py --data_str=$DATA_STR --data_seed=$DATA_SEED
    ## for DAG-based DGP
    python -u generator/generate_DAG.py --data_str=$DATA_STR --data_seed=$DATA_SEED
    ## for non-paranormal based DGP
    python -u generator/generate_NPN.py --data_str=$DATA_STR --data_seed=$DATA_SEED
    ```
    See `configs/_synthetic_.yaml`

    To generate multiple data replicas all at one
    ```console
    cd bin
    bash generate.sh $DATA_STR
    ```

* Run experiments
    
    To run a specific synthetic dataset
    ```console
    ## VERSION is either empty (default, neural network based method) or `reggmm` (linear method)
    python -u run_sim.py --data_str=$DATA_STR --train_size=$TRAIN_SIZE --gpu=$GPU_ID --data_seed=$DATA_SEED --version=$VERSION
    ```
    To run multiple synthetic data replicas in one go:
    ```console
    cd bin
    bash run.sh $DATA_STR $GPU_ID $VERSION
    ```
    To run competitor models that leverage R packages (glasso or neighborhood selection aka mb) can be run through `bin/run-r.sh`.

Finally, to run the entire pipeline for any specific data setting (as specified through $DATA_STR) encompassing generating data, running proposed DNN model and competitors, refer to `bin/run-all.sh`

## Real Data Experiments

See Appendix E of the manuscript for the accessibility of the raw data. 

### The resting-state fMRI Dataset
* Raw data is stored in the format of `.RData`. We first leverage the [reticulate](https://rstudio.github.io/reticulate/) package to convert the data so that it can be read through Python. See `data/fmri-data-convert.R`, which outputs the corresponding `.npz` files.
* See `notebooks/demo-run-fmri-data.ipynb` for the end-to-end model run and results visualization.


### S&P 100 Constituents Dataset
* In this repo, we provide a copy of the processed daily data (see folder `data/stock/*`). Here, $x$ (node values) correspond to the $\beta$-adjusted residual returns (relative to SPX) of the constituent stocks, with the $\beta$ calculated based on a 252-day lookback window; $z$ (covariates) correspond to the returns of SPX, Nasdaq and the level of VIX. 
* See `notebooks/demo-run-stock-data.ipynb` for the end-to-end model run and results visualization.

## Citation and Contact
To cite this work:
```
@article{lin2025covariate,
    title = {Covariate-dependent Graphical Model Estimation via Neural Networks with Statistical Guarantees},
    author = {Lin, Jiahe and Zhang, Yikai and Michailidis, George},
    year = {2025},
    journal = {Transactions on Machine Learning Research},
    issn={2835-8856},
    url = {https://openreview.net/pdf?id=beqSqPgE33}
}
```
For questions on the code implementation, contact [Jiahe Lin](mailto:jiahelin@umich.edu).