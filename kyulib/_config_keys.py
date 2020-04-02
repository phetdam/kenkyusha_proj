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
# added _DISP_AVG_ELAS and _DISP_AVG_RLAS, and removed _SAVE_ELA_FIG and
# _SAVE_RLA_FIG in exchange for more general _ELA_FIG and _RLA_FIG keys which
# hold a dict of parameters for both figures. added several figure-related
# keys to control plot format. moved to kyulib. added _XLA_FIG_KEYS to make it
# easier to check all the params needed to plot an ELA/RLA comparison graph.
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
           "_DISP_RLAS", "_DISP_AVG_ELAS", "_DISP_AVG_RLAS", "_ELA_FIG",
           "_RLA_FIG", "_FIGS_SAVE_FIG", "_FIGS_FIG_SIZE", "_FIGS_FIG_TITLE",
           "_FIGS_FIG_CMAP", "_FIGS_PLOT_KWARGS", "_MODELS", "_MODELS_NAME",
           "_MODELS_MODULE", "_MODELS_MODEL", "_MODELS_PARAMS",
           "_MAIN_CONFIG_KEYS", "_XLA_FIG_KEYS"]

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
# in .json config, 1/0 true/false for printing the vector of average ELAs
_DISP_AVG_ELAS = "disp_avg_elas"
# in .json config, 1/0 true/false for printing the vector of average RLAs
_DISP_AVG_RLAS = "disp_avg_rlas"
# in .json config, give dict specifying params for average ELA comparison plot
_ELA_FIG = "ela_fig"
# in .json config, give dict specifying params for average RLA comparison plot
_RLA_FIG = "rla_fig"
# in .json config, 1/0 true/fast for either figure on whether to save the plot
_FIGS_SAVE_FIG = "save_fig"
# in .json config, array [width, height] (inches) for individual plot dimensions
_FIGS_FIG_SIZE = "fig_size"
# in .json config, gives title of a particular figure (str)
_FIGS_FIG_TITLE = "fig_title"
# in .json config gives color map to use when coloring lines in RLA/ELA plots
_FIGS_FIG_CMAP = "fig_cmap"
# in .json config, kwargs to pass to matplotlib.axes.Axes.plot (dict)
_FIGS_PLOT_KWARGS = "plot_kwargs"
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
                     _DISP_RLAS, _DISP_AVG_ELAS, _DISP_AVG_RLAS, _ELA_FIG,
                     _RLA_FIG, _MODELS]

# a list of the keys used for governing average ELA/RLA comparison plot params
_XLA_FIG_KEYS = [_FIGS_SAVE_FIG, _FIGS_FIG_SIZE, _FIGS_FIG_TITLE,
                 _FIGS_FIG_CMAP, _FIGS_PLOT_KWARGS]

if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
