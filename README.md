# BAC Research Team Spring 2020: Boosting with Noise

_last updated: 05-22-2020_  
_file created: 03-23-2020_

Project folder for the BAC Research Team's study of boosting algorithm performance on noisy data sets. This project was motivated by prior experiences with AdaBoost being defeated by label noise, which is a known shortcoming, and theoretical curiosity with regards to gradient boosting, the robustness of boosting to noise, XGBoost, and the XGBoost implementation.

**Remark.** Evaluation times stored in the output pickles should be taken with a grain of salt. The machine running evaluations suffered mechanical damage to its cooling fan and has since exhibited noticeably slower and more erratic performance.

Contributors: Derek Huang

## How to use

Simply `git clone` this repository onto your local machine and browse around. If you are interested in reproducing our results, please execute `noisyeval.py` with an appropriate JSON configuration file, say `foo.json`, either with `./noisyeval.py foo.json` in the `bash` shell or by invoking `python noisyeval.py foo.json`. By default, the `warm_start` field in the configuration files is set to 1 (true), so `noisyeval.py` will only compute results if there is no preexisting output pickle in the specified output directory.

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

### slides

The `slides` directory contains a PDF set of slides which provide some background on gradient boosting and XGBoost, explain our experimental setup and results, and state our conclusions and some future considerations. An abridged version, not included here,  was presented to fellow Spring 2020 BAC Advanced Team members.

### test

The `test` directory contains a copy of a few of the data sets from `data` and some sample configurations, used during project development. `test/config` contains JSON configuration files, `test/data` contains the sample data sets, and `test/results` contains some sample evaluation results. Runtimes for the configurations in `test/config` are much shorter than those used to produce the results in the main `results` directory, i.e. in the order of minutes, not days.