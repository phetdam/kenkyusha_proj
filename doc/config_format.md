# Formatting config files

_last updated: 05-22-2020_  
_file created: 04-28-2020_

A brief guide to interpreting and formatting the JSON evaluation config files for `noisyeval.py`.  

LaTeX rendering achieved using Alexander Rodin's workaround, detailed [here](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b).  

**Remark.** Work in progress. Will be updated during the summer.

## Overview

The configuration files used by `noisy_eval.py` are standard JSON files with specific formatting conventions that are specify different configuration parameters to `noisy_eval.py` , for example where to write results, the global random seed to use, what kinds of noise to add, what quantities to display to `stdout`, etc. Each configuration file consists of a single JSON object containing other JSON objects, arrays, and strings. There are several keys associated with the main JSON object which are required to be present; we detail their required values in the section below.

## Required keys

### data_dir

Indicates to `noisy_eval.py` where the data files for model training are, all of which will be used. Must be assigned a string value interpretable as a valid directory name containing only CSV files that can be read by `pandas.read_csv`. Note that if there are non-CSV files in the directory, `noisy_eval.py` will ignore them and issue a warning.

**Example:**

```json
"data_dir": "./data/csv_clean",
```

### results_dir

Indicates to `noisy_eval.py` where to write computation results, including the final model result `pickle` and any associated RLA/ELA plots. See also [**ela_fig**](#ela_fig) and [**rla_fig**](#rla_fig). Must be assigned a string value interpretable as a valid directory name.

**Example:**

```json
"results_dir": "./results",
```

### test_fraction

Indicates to `noisy_eval.py`, for each data set in the directory specified by **data_dir**, what fraction of the data to use as the validation data that metrics will be computed on. Must be assigned a float in the range of (0, 1). See also [**random_state**](#random_state).

**Example:**

```json
"test_fraction": 0.2,
```

### random_state

Controls the global random seed used by `noisy_eval.py` during the training proces, controlling both the training/validation data split, random seed for any model that has stochastic fitting behavior, and the way that noise applied to a data set is generated. Note that the underlying PRNG is the `numpy` PRNG, so the behavior is not thread-safe. Must be assigned a nonnegative integer, which is directly passed to `numpy.random.seed`. See also [**noise_kinds**](#noise_kinds).

**Example:**

```json
"random_state": 7,
```

### noise_kinds

Indicates to `noisy_eval.py` what kinds of noise to add to each copy of each data set specified by **data_dir**. If <img src="https://render.githubusercontent.com/render/math?math=k"> types of noise are specified, then for each noise level, <img src="https://render.githubusercontent.com/render/math?math=k"> different noisy data sets copies will be made. Must be assigned an array, where each element of the array is a valid string corresponding to a type of noise to introduce.

So far, only `"label"` is a valid, supported noise type.  See also [**noise_levels**](#noise_levels).

**Example:**

```json
"noise_kinds": ["label"],
```

### noise_levels

Indicates to `noisy_eval.py` the noise level to assign for each of the noisy copies specified by **noise_kinds** made for each data set specified by **data_dir**. Must be assigned an array, where each element of the array is a float in (0, 1). See also [**noise_kinds**](#noise_kinds) above.  

**Example:**

```json
"noise_levels": [0.1, 0.2, 0.3, 0.6, 0.9],
```

### disp_accs

Indicates to `noisy_eval.py` that after computation, accuracy matrices for each model should be printed to `stdout`. The matrices are indexed by data set name along the rows and by noise level along the columns, and if there are <img src="https://render.githubusercontent.com/render/math?math=k"> noise kinds specified in **noise_kinds**, then there will be <img src="https://render.githubusercontent.com/render/math?math=k"> accuracy matrices for each model. Must be assigned either 0 for false, 1 for true.

**Example:**

```json
"disp_accs": 0,
```

### disp_elas, disp_rlas

Play similar roles to **disp_accs**, except the matrices are of ELA and RLA, indexed in the same way as described above in [**disp_accs**](#disp_accs). See the paper [here](https://doi.org/10.1016/j.neucom.2014.11.086) for a description of what the two metrics are. Must be assigned either 0 for false, 1 for true.  

**Example:**

```json
"disp_elas": 0,
"disp_rlas": 0,
```

### disp_avg_elas, disp_avg_rlas

Play similar roles to **disp_elas** and **disp_rlas** except with respect to whether or not row vectors of data set macro averages of ELA and RLA will be sent to `stdout`. Must be assigned either 0 for false, 1 for true.

**Example:**

```json
"disp_avg_elas": 1,
"disp_avg_rlas": 1,
```

### ela_fig, rla_fig

Indicates options to be used when painting comparison plots of per-model average ELA/RLA versus noise level across data sets. Must be assigned a JSON object that contains several required keys which are described below.

* **save_fig**

  add filler text here does this change the list level?

### warm_start

### models

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