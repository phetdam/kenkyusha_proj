# Formatting config files

_last updated: 05-18-2020_  
_file created: 04-28-2020_

A brief guide to interpreting and formatting the JSON evaluation config files for `noisyeval.py`.

## Overview

The configuration files used by `noisy_eval.py` are standard JSON files with specific formatting conventions that are specify different configuration parameters to `noisy_eval.py` , for example where to write results, the global random seed to use, what kinds of noise to add, what quantities to display to `stdout`, etc. Each configuration file consists of a single JSON object containing other JSON objects, arrays, and strings. There are several keys associated with the main JSON object which are required to be present; we detail their required values in the section below.

## Required keys

### data_dir

Indicates to `noisy_eval.py` where the data files for model training are, all of which will be used. Must be assigned a string value interpretable as a valid directory name containing only CSV files that can be read by `pandas.read_csv`. Note that if there are non-CSV files in the directory, `noisy_eval.py` will ignore them and issue a warning.

### results_dir

Indicates to `noisy_eval.py` where to write computation results, including the final model result `pickle` and any associated RLA/ELA plots as appropriately indicated. See also [**ela_fig**](#ela_fig) and [**rla_fig**](#rla_fig). Must be assigned a string value interpretable as a valid directory name.

### test_fraction

Indicates to `noisy_eval.py`, for each data set in the directory specified by **data_dir**, what fraction of the data to use as the validation data that metrics will be computed on. Must be assigned a float in the range of (0, 1). See also [**random_state**](#random_state).

### random_state

Controls the global random seed used by `noisy_eval.py` during the training proces, controlling both the training/validation data split, random seed for any model that has stochastic fitting behavior, and the way that noise applied to a data set is generated. Note that the underlying PRNG is the `numpy` PRNG, so the behavior is not thread-safe. See also [**noise_kinds**](#noise_kinds).

### noise_kinds

**Remark.** Work in progress. Will be updated during the summer.