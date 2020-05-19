# Formatting config files

_last updated: 05-18-2020_  
_file created: 04-28-2020_

A brief guide to interpreting and formatting the JSON evaluation config files for `noisyeval.py`.

**Remark.** Work in progress. Will be updated during the summer.

## Overview

The configuration files used by `noisy_eval.py` are standard JSON files with specific formatting conventions that are specify different configuration parameters to `noisy_eval.py` , for example where to write results, the global random seed to use, what kinds of noise to add, what quantities to display to `stdout`, etc. Each configuration file consists of a single JSON object containing other JSON objects, arrays, and strings. There are several keys associated with the main JSON object which are required to be present; we detail their required values in the section below.

## Required keys

### data_dir

Indicates to `noisy_eval.py` where the data files for model training are, all of which will be used. Must be assigned a string value interpretable as a valid directory name containing only CSV files that can be read by `pandas.read_csv`. Note that if there are non-CSV files in the directory, `noisy_eval.py` will ignore them and issue a warning.  

Example: `"data_dir": "./data/csv_clean"`

### results_dir

Indicates to `noisy_eval.py` where to write computation results, including the final model result `pickle` and any associated RLA/ELA plots. See also [**ela_fig**](#ela_fig) and [**rla_fig**](#rla_fig). Must be assigned a string value interpretable as a valid directory name.

### test_fraction

Indicates to `noisy_eval.py`, for each data set in the directory specified by **data_dir**, what fraction of the data to use as the validation data that metrics will be computed on. Must be assigned a float in the range of (0, 1). See also [**random_state**](#random_state).

### random_state

Controls the global random seed used by `noisy_eval.py` during the training proces, controlling both the training/validation data split, random seed for any model that has stochastic fitting behavior, and the way that noise applied to a data set is generated. Note that the underlying PRNG is the `numpy` PRNG, so the behavior is not thread-safe. Must be assigned a nonnegative integer, which is directly passed to `numpy.random.seed`. See also [**noise_kinds**](#noise_kinds).

### noise_kinds

Indicates to `noisy_eval.py` what kinds of noise to add to each copy of each data set specified by **data_dir**. If $ k $ types of noise are specified, then for each noise level, $ k $ different noisy data sets copies will be made. Must be assigned an array, where each element of the array is a valid string corresponding to a type of noise to introduce. So far, only `"label"` is a supported noise value.  See also [**noise_levels**](#noise_levels).

### noise_levels

Indicates to `noisy_eval.py` the noise level to assign for each of the noisy copies specified by **noise_kinds** made for each data set specified by **data_dir**. Must be assigned an array, where each element of the array is a float in (0, 1). See also [**noise_kinds**](#noise_kinds) above.

## An example

The following JSON object, when placed into a JSON file, is a valid configuration file.

```json
{
    "data_dir": "test/data",
    "results_dir": "test/results",
    "test_fraction": 0.2,
    "random_state": 7,
    "noise_kinds": ["label"],
    "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
    "disp_accs": 0,
    "disp_elas": 0,
    "disp_rlas": 0,
    "disp_avg_elas": 1,
    "disp_avg_rlas": 1,
    "ela_fig": {
	"save_fig": 1,
	"fig_size": [6, 5],
	"fig_dpi": 150,
	"fig_title": "Average ELA with 50 trees, max_depth=6",
	"fig_cmap": "viridis",
	"plot_kwargs": [{}, {}, {"marker": "s", "markersize": 5}]
    },
    "rla_fig": {
	"save_fig": 1,
	"fig_size": [6, 5],
	"fig_dpi": 150,
	"fig_title": "Average RLA with 50 trees, max_depth=6",
	"fig_cmap": "viridis_r",
	"plot_kwargs": [{}, {}, {"marker": "s", "markersize": 5}]
    },
    "warm_start": 1,
    "models": [
	{
	    "name": "adaboost6",
	    "module": "sklearn.ensemble",
	    "model": "AdaBoostClassifier",
	    "params": {
		"base_estimator": {
		    "module": "sklearn.tree",
		    "model": "DecisionTreeClassifier",
		    "params": {
			"criterion": "entropy",
			"max_depth": 6,
			"random_state": 7
		    }
		},
		"n_estimators": 50,
		"random_state": 7
	    }
	},
	{
	    "name": "gboost6",
	    "module": "sklearn.ensemble",
	    "model": "GradientBoostingClassifier",
	    "params": {
		"n_estimators": 50,
		"max_depth": 6,
		"learning_rate": 0.1,
		"random_state": 7
	    }
	},
	{
	    "name": "xgboost6",
	    "module": "xgboost",
	    "model": "XGBClassifier",
	    "params": {
		"n_estimators": 50,
		"max_depth": 6,
		"learning_rate": 0.1,
		"booster": "gbtree",
		"reg_alpha": 0,
		"gamma": 0,
		"reg_lambda": 0.1,
		"n_jobs": 2,
		"random_state": 7
	    }
	}
    ]
}
```