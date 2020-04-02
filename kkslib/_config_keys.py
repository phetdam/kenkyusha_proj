# contains the special dict key names that are used determine whether the
# given .json object is readable, depending on the context. the function
# .utils.model_from_spec and the main entry point (noisyeval.py) depend on the
# definitions given in this file, so please DO NOT CHANGE THEM!!
#
# Changelog:
#
# 04-02-2020
#
# changed _RESULT_DIR to _RESULTS_DIR. added _TEST_FRACTION and _RANDOM_STATE.
# change the _SAVE_[ACCS | ELAS | RLAS] keys to be prefixed with _DISP instead.
# added key _FIG_SIZE to give size of the average ELA/RLA comparison plots.
#
# 04-01-2020
#
# initial creation. added all important keys, for both the main .json config and
# for the model_from_spec dependent keys, as well as list of all the import
# .json config keys for ease of sanity checking config files.

from sys import stderr

__doc__ = """
internal constants required by .utils.model_from_spec and noisyeval.py.

PLEASE DO NOT CHANGE!!
"""

_MODULE_NAME = "_config_keys"

# for importing all

__all__ = ["_DATA_DIR", "_RESULTS_DIR", "_TEST_FRACTION", "_RANDOM_STATE",
           "_NOISE_KINDS", "_NOISE_LEVELS", "_DISP_ACCS", "_DISP_ELAS",
           "_DISP_RLAS", "_SAVE_ELA_FIG", "_SAVE_RLA_FIG", "_FIG_SIZE",
           "_MODELS", "_MODELS_NAME", "_MODELS_MODULE", "_MODELS_MODEL",
           "_MODELS_PARAMS", "_MAIN_CONFIG_KEYS"]

### special constants ###
# in .json config, specify directory of .csv data files to use
_DATA_DIR = "data_dir"
# in .json config, specify directory to write all results to
_RESULTS_DIR = "results_dir"
# in .json config, specify the fraction of the data to use as test data
_TEST_FRACTION = "test_fraction"
# in .json config, specify the random state to be used when creating noisy
# copies of the data sets or when splitting train/test data
_RANDOM_STATE = "random_state"
# in .json config, specify kinds of noise to add to data
_NOISE_KINDS = "noise_kinds"
# in .json config, specify noise levels in (0, 1) to add to data
_NOISE_LEVELS = "noise_levels"
# in .json config, 1/0 true/false for printing the accuracy matrix for each
# model to stdout during execution of noisyeval.py
_DISP_ACCS = "disp_accs"
# in .json config, 1/0 true/false for printing the ELA matrix for each model to
# stdout during execution of noisyeval.py
_DISP_ELAS = "disp_elas"
# in .json config, 1/0 true/false for printing the RLA matrix for each model to
# stdout during execution of noisyeval.py
_DISP_RLAS = "disp_rlas"
# in .json config, 1/0 true/false for saving average ELA comparison plot
_SAVE_ELA_FIG = "save_ela_fig"
# in .json config, 1/0 true/false for saving average RLA comparison plot
_SAVE_RLA_FIG = "save_rla_fig"
# in .json config, array [width, height] (inches) for ELA/RLA plot dimensions
_FIG_SIZE = "fig_size"
# in .json config, give list of dicts specifying model hyperparameters
_MODELS = "models"
# in _MODELS list, for a dict, gives name of model
_MODELS_NAME = "name"
# in _MODELS list, for a dict, gives module model class is from
_MODELS_MODULE = "module"
# in _MODELS list, for a dict, gives model (class) of model
_MODELS_MODEL = "model"
# in _MODELS list, for a dict, gives model dict of hyperparameters
_MODELS_PARAMS = "params"

# a list of the main constants used .json config to simplify sanity check
_MAIN_CONFIG_KEYS = [_DATA_DIR, _RESULTS_DIR, _TEST_FRACTION, _RANDOM_STATE,
                     _NOISE_KINDS, _NOISE_LEVELS, _DISP_ACCS, _DISP_ELAS,
                     _DISP_RLAS, _SAVE_ELA_FIG, _SAVE_RLA_FIG,
                     _FIG_SIZE, _MODELS]

if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
