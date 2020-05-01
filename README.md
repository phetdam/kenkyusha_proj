# BAC Research Team Spring 2020: Boosting with Noise

![./banner.png](./banner.png)

_last updated: 05-01-2020_  
_file created: 03-23-2020_

Project folder for the BAC Research Team's study of boosting algorithm performance on noisy data sets. This project was motivated by prior experiences with AdaBoost being defeated by label noise, which is a known shortcoming, and theoretical curiosity with regards to gradient boosting, the robustness of boosting to noise, XGBoost, and the XGBoost implementation.

**Remark.** This directory is subject to change at any notice due to its state of active development.

**Remark.** Evaluation times stored in the output pickles should be taken with a grain of salt. The machine running evaluations suffered mechanical damage to its cooling fan and has since exhibited noticeably slower and more erratic performance.

## How to use

**Remark.** The `--warm-start` option has not yet been implemented, so training times may be exceedingly long if you use any of the JSON files from `config`. Please use one of the test configurations in `test/config` instead.

Simply `git clone` this repository onto your local machine and browse around. If you are interested in reproducing our results, please execute `noisyeval.py` with an appropriate JSON configuration file, say `foo.json`, either with `./noisyeval.py foo.json --warm-start` in the `bash` shell or by invoking `python noisyeval.py foo.json --warm-start`. The option `--warm-start` will instruct `noisyeval.py` to only compute results if there is no preexisting output pickle in the directory it writes results to.

Please read `doc/config_format.md` for instructions on how to write your own JSON model configurations.

## Directories

Below are some brief descriptions of the directories in this repository.

### config

The `config` directory contains the JSON configuration files used to produce experimental results.

### data

The `data` directory, as implied, contains all the data used for this project. The raw CSV data files downloaded from OpenML are in `data/csv_raw`, where the [minimally] cleaned CSV data files used for model evaluation are in `data/csv_clean`.

### doc

The `doc` directory contains relevant documentation. File `data_descs.md` contains descriptions for each of the 16 data sets we used in this project, and file `config_format.md` contains instructions on how to interpret and write your own JSON configuration files to feed to `noisyeval.py`, the script we used for evaluating the models.

### kyulib

The `kyulib` directory is a local Python package containing the [little] library code used for this project. Each function and class in the module files, as well as the module files themselves, have docstrings readable using the Python `help` command.

### results

The `results` directory, as implied, contains the main experimental results from running `noisyeval.py` with the main configuration files in `config`. Output files are Python pickles, with PNG files for the average ELA/RLA plots.

**Remark.** For info on ELA and RLA, see [this paper](https://doi.org/10.1016/j.neucom.2014.11.086) or read the docstrings for functions `ela` and `rla` in module `kyulib.metrics`.

### test

The `test` directory contains a copy of a few of the data sets from `data` and some sample configurations, used during project development. `test/config` contains JSON configuration files, `test/data` contains the sample data sets, and `test/results` contains some sample evaluation results. Runtimes for the test configuration are much shorter than those used for to produce the results in the main `results` directory.