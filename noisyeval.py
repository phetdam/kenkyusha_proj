#!/bin/python
# main entry point for training and evaluation of models on data sets with
# varying levels of noise. added shebang so that execution via ./noisyeval.py
# can be done as shorthand (if applicable). please run from command line only!
#
# Changelog:
#
# 04-28-2020
#
# updated error messages in _check_json_config to refer to the newly created doc
# file doc/config_format.md, which will be updated later. updated the function
# _paint_xla_plots to send dpi argument to savefig, which is a new option that
# is required in the ela_fig and rla_fig dicts controlling plot parameters.
#
# 04-10-2020
#
# made final check to warm start functionality.
#
# 04-09-2020
#
# reorganized the code in the entire file into separate functions and subs to
# tidy up the flow of the code and make it easier to read. built in the warm
# start functionality but have not yet tested it.
#
# 04-08-2020
#
# modified doc string and started work to accomodate warm starting to reuse past
# computation results if an output file already exists.
#
# 04-05-2020
#
# added reference to kyulib.plotting and inserted plotting function. prints a
# message to stdout for each figure plotted, and prints message to stdout for
# each model evaluation that is completed with model name, class, and time.
# updated to allow individual line kwargs when plotting with plot_avg_xlas.
#
# 04-04-2020
#
# reflected the fact that avg_elas and avg_rlas in accuracies2ela and the
# accuracies2rla functions are now called averages.
#
# 04-02-2020
#
# completed most of the main body. just need to write functions to compute ELA
# and RLA matrices, to compute average ELA and RLA vectors, and to plot the
# average ELA/RLA comparison graphs for the models. corrected some sanity checks
# for configuration file options. added lines to compute ELA/RLA and changed
# output format to individual models to a single large pickle. edited docstring.
# renamed kkslib to kyulib so corrected import statements. modified checking of
# config keys and display option keys and put placeholders for plotting.
#
# 04-01-2020
#
# initial creation. originally called test_file.

from copy import deepcopy
import json
from pandas import read_csv
import pickle
from os import listdir
from os.path import exists, isdir
from sys import argv, exit, stderr
from textwrap import fill
from time import time

# import special constants into namespace
from kyulib._config_keys import *
from kyulib.plotting import plot_avg_xla
from kyulib.utils import DataFrame2Data_Set, model_from_spec, Model_Results
from kyulib.metrics import accuracies2ela, accuracies2rla, compute_accuracies

__doc__ = """
main entry point for evaluation of sklearn-compatible models on data sets with
different levels of noise introduced, with computation of ELA (equalized loss
of accuracy), RLA (relative loss of accuracy) metrics, and optional creation of
comparative line plots of average model ELA/RLA statistics. based on user input
recorded in a properly formatted JSON configuration file, will read data, create
the desired model configurations, evaluate the models, and write the output to a
pickled Python dict and any enabled plot to the specified results directory.

please read doc/config_format.md for instructions on how to interpret and format
the JSON file that governs the behavior of this script. passing the --help flag
will produce some usage instructions and this help blurb."""

_PROGNAME = "noisyeval"
_HELP_FLAG = "--help"
_HELP_STR = """Usage: {0} config_file.json
       {0} [ {1} ]{2}""".format(_PROGNAME, _HELP_FLAG, __doc__)

# constants used as keys for results dict in Model_Results objects
_KEY_ACCS = "accs"
_KEY_ACCS_TIME = "accs_time"
_KEY_ELAS = "elas"
_KEY_RLAS = "rlas"
_KEY_AVG_ELAS = "avg_elas"
_KEY_AVG_RLAS = "avg_rlas"
_KEY_EST = "est"
# put them in a list for convenience (except for _KEY_ACCS_TIME and _KEY_EST)
_MAIN_RESULTS_KEYS = [_KEY_ACCS, _KEY_ELAS, _KEY_RLAS, _KEY_AVG_ELAS,
                      _KEY_AVG_RLAS]

# internal functions. do not call from other methods!
def _load_json_config(cfn):
    """
    given a str file name, which must be a JSON file, i.e. end in .json, try to
    open the file and use json.load to decode the JSON file. will return a tuple
    of a dict or list, depending on what is in the file, and an int error code.
    if there are errors, will print the errors to stderr and then return None
    with the int error code. error codes are in range [0, 2].

    in this context, the returned output should be a dict.

    do NOT call externally!

    parameters:

    cfn      str file name, should end in .json
    """
    # only allow .json file as config
    if cfn.split(".")[-1] != "json":
        print("{0}: configuration file must be a .json file".format(_PROGNAME),
              file = stderr)
        return None, 1
    # try to open and decode the JSON file
    cfg_data = None
    try: cfg_data = json.load(open(cfn, "r"))
    except FileNotFoundError:
        print("{0}: FileNotFoundError: could not locate file {1}"
              .format(_PROGNAME), file = stderr)
        return None, 2
    # this will get raised if u try to open a directory
    except PermissionError:
        print("{0}: PermissionError: please check that {1} is a regular file"
              .format(_PROGNAME), file = stderr)
        return None, 2
    # in the case of JSON decoding error
    except json.JSONDecodeError as je:
        print("{0}: {1}: {2}".format(_PROGNAME, je.__class__.__name__, str(je)),
              file = stderr)
        return None, 1
    # if all checks passed, return cfg_data
    return cfg_data, 0

def _check_json_config(cfg_data):
    """
    given a dict formatted in a JSON-compatible manner, check that the dict
    follows the correct format for the configuration file used by noisyeval. the
    main keys will be checked, as well as keys for plotting parameters and for
    model parameters. returns 0 if no errors, else prints errors to stdout and
    then returns either 1 or 2 as the error code.

    do not call externally.

    parameters:

    cfg_data    dict formatted in a JSON-compatible manner
    """
    # must be a dict
    if not isinstance(cfg_data, dict):
        print("{0}: error: decoded JSON data should be dict, not {1}"
              .format(_PROGNAME, type(cfg_data)), file = stderr)
        return 1
    # check keys in cfg_data
    for _ck in _MAIN_CONFIG_KEYS:
        if _ck not in cfg_data:
            print(fill("{0}: error: missing required key {1}. please read "
                       "README.md for instructions on how to format "
                       "configuration files".format(_PROGNAME, _ck),
                       width = 80), file = stderr)
            return 1
    # check that the data and results directories exist + are directories
    data_dir, results_dir = cfg_data[_DATA_DIR], cfg_data[_RESULTS_DIR]
    if exists(data_dir) == False:
        print("{0}: error: cannot find data directory {1}"
              .format(_PROGNAME, data_dir), file = stderr)
        return 2
    if isdir(data_dir) == False:
        print("{0}: error: {1} is not a directory".format(_PROGNAME, data_dir),
              file = stderr)
        return 2
    if exists(results_dir) == False:
        print("{0}: error: cannot find results directory {1}"
              .format(_PROGNAME, results_dir), file = stderr)
        return 2
    if isdir(results_dir) == False:
        print("{0}: error: {1} is not a directory"
              .format(_PROGNAME, results_dir), file = stderr)
        return 2
    # before we start training the models, check the plotting parameters. no one
    # wants to learn after training for 6 hours that they aren't allowed to make
    # a graph... that would really suck, wouldn't it
    # for each one, make sure that all the required keys are present
    for _ck in _XLA_FIG_KEYS:
        if _ck not in cfg_data[_ELA_FIG]:
            print(fill("{0}: error: {1} missing required key {2}. please read "
                       "doc/config_format.md for instructions on how to format "
                       "average ELA/RLA comparison plot params"
                       .format(_PROGNAME, _ELA_FIG, _ck), width = 80),
                  file = stderr)
            return 1
        if _ck not in cfg_data[_RLA_FIG]:
            print(fill("{0}: error: {1} missing required key {2}. please read "
                       "doc/config_format.md for instructions on how to format "
                       "average ELA/RLA comparison plot params"
                       .format(_PROGNAME, _RLA_FIG, _ck), width = 80),
                  file = stderr)
            return 1
    # check that all required keys are in each dict in the _MODELS entry
    for _mdl in cfg_data[_MODELS]:
        for _mk in _MODELS_KEYS:
            if _mk not in _mdl:
                print(fill("{0}: error: {1} missing required key {2}. please "
                           "read README.md for instructions on how to format "
                           "model params dicts".format(_PROGNAME, _MODELS, _mk),
                           width = 80), file = stderr)
                return 1
    # yay, all checks passed so return 0 for no error
    return 0

def _warm_start_status(cfg_data, cfg_name):
    """
    given a correctly formatted JSON-compatible dict of config data and the name
    (sans extension) of the respective JSON configuration file, check the
    recorded warm starting status. returns (bool, int) tuple where the bool
    indicates whether the start is warm (True) or cold (False), and the int is
    the error code. upon error, bool will be replaced by None.

    do not call externally. cfg_data must already be correctly formatted.

    parameters:

    cfg_data    dict correctly formatted for noisyeval from JSON config file
    cfg_name    str name of the current config file without extension. used by
                the function to search in the results directory specified in
                cfg_data for a pickle file with the matching name that is
                required for the warm start to be possible.
    """
    # skip full sanity checks; should have already been done
    assert isinstance(cfg_data, dict) and isinstance(cfg_name, str)
    # get warm start status
    warm_start = cfg_data[_WARM_START]
    # get results directory
    results_dir = cfg_data[_RESULTS_DIR]
    # if 1, indicates warm start, 0 for no warm start, else error and exit. if
    # output file cannot be found, warm_start set to 0 for cold start.
    if warm_start == 1:
        # look for cfg_name.pickle in results_dir; if not found, no warm start
        # and message will be printed to warn about cold start
        if exists(results_dir + "/" + cfg_name + ".pickle"):
            return True, 0
        else:
            print("WARNING: cold starting. {0}/{1}.pickle does not exist."
                  .format(results_dir, cfg_name), file = stderr)
            return False, 0
    elif warm_start == 0:
        # print cold start warning
        print("WARNING: cold starting. {0}/{1}.pickle will be overwritten"
              .format(results_dir, cfg_name), file = stderr)
        return False, 0
    else:
        print("{0}: error: value for {1} must be 1 or 0, not {2}"
              .format(_PROGNAME, _WARM_START, warm_start), file = stderr)
        return None, 1

def _load_prior_results(cfg_data, cfg_name):
    """
    given a correctly formatted JSON-compatible dict of config data and the name
    (sans extension) of the respective JSON configuration file, which is also
    the name of the pickle file that will be read from, read from the pickle
    results file corresponding to the config file and convert the dict results
    in the unpickled dict into a list of Model_Results.

    do not call externally.

    parameters:

    cfg_data    dict correctly formatted for noisyeval from JSON config file
    cfg_name    str name of current config file, without extension. this is the
                name of the pickle file that will be used.
    """
    # skip full sanity checks; should have been done in previous functions
    assert isinstance(cfg_data, dict) and isinstance(cfg_name, str)
    # get results directory
    results_dir = cfg_data[_RESULTS_DIR]
    # attempt to open the pickle file and load the dict into raw_data
    raw_data = None
    try:
        raw_data = pickle.load(
            open(results_dir + "/" + cfg_name + ".pickle", "rb"))
    except FileNotFoundError:
        print("{0}: FileNotFoundError: could not locate file {1}"
              .format(_PROGNAME), file = stderr)
        return None, 2
    # this will get raised if u try to open a directory
    except PermissionError:
        print("{0}: PermissionError: please check that {1} is a regular file"
              .format(_PROGNAME), file = stderr)
        return None, 2
    # unpickling error
    except pickle.UnpicklingError as ue:
        print("{0}: UnpicklingError: {1}".format(_PROGNAME, str(ue)),
              file = stderr)
        return None, 2
    # if not a dict, print error and exit
    if not isinstance(raw_data, dict):
        print("{0}: error: loaded model results expected to be dict, not {1}"
              .format(_PROGNAME, type(raw_data)), file = stderr)
    # create a list of Model_Results from each key (model) in the dict, where
    # key is the model name and the (unfitted) estimator in the dict. we do a
    # deep copy of the estimator because we will remove its key value mapping
    # from the results dict for each model later on.
    mdl_results = [Model_Results(mn, deepcopy(mr[_KEY_EST])) for mn, mr in
                   raw_data.items()]
    # set results attribute for each Model_Results using dicts from raw_data;
    # the est key will be deleted from each raw_mr, however
    for mr, (_, raw_mr) in zip(mdl_results, raw_data.items()):
        raw_mr.pop(_KEY_EST, None)
        mr.results = raw_mr
    # return list of Model_Results
    return mdl_results

def _evaluate(cfg_data):
    """
    main model evaluation loop. for all the model configurations in cfg_data,
    instantiates a list of Model_Results objects, and given the noise kinds and
    noise levels specified in cfg_data, will compute accuracies, record the
    evaluation time, ELAs, RLAs, and the average ELAs and RLAs. prints a message
    indicating completion to stdout each time a model configuration has been
    fully evaluated and returns the finished list of Model_Results objects.

    do NOT call externally! should only be run within this script as the input
    will have already been checked for sanity in the program main.

    parameters:

    cfg_data     dict correctly formatted for noisyeval from JSON config file
    """
    # for each model param dict in cfg_data, create a Model_Results object; all
    # the Model_Results objects contain the name, unfitted model. all in a list.
    mdl_results = [Model_Results(mpd["name"], model_from_spec(mpd))
                   for mpd in cfg_data[_MODELS]]
    # assign random state to _seed, set to None if < 0
    _seed = cfg_data[_RANDOM_STATE]
    if _seed < 0: _seed = None
    # get data directory
    data_dir = cfg_data[_DATA_DIR]
    # for each data file in the data dir, read it into a DataFrame and then use
    # the file name as the name of the Data_Set created out of the DataFrame.
    # use the random seed and test fraction provided in cfg_data. skip and
    # warn for anything in the directory that is not a .csv file.
    data_sets = []
    for fname in listdir(data_dir):
        if fname.split(".")[-1] == "csv":
            df = read_csv(data_dir + "/" + fname)
            ds = DataFrame2Data_Set(df, fname[:-4],
                                    test_size = cfg_data[_TEST_FRACTION],
                                    random_state = _seed)
            data_sets.append(ds)
        else:
            print("{0}: warning: {1} is not a .csv file. skipping"
                  .format(_PROGNAME, fname), file = stderr)
    # assign _NOISE_KINDS and _NOISE_LEVELS to variables; set to None if empty
    noise_kinds, noise_levels = cfg_data[_NOISE_KINDS], cfg_data[_NOISE_LEVELS]
    if len(noise_kinds) == 0: noise_kinds = None
    if len(noise_levels) == 0: noise_levels = None
    # for each Model_Results object in mdl_results, compute accuracy matrix, ela
    # matrix, rla matrix, average elas, average rlas. save to Model_Results
    for mdl_res in mdl_results:
        # time accuracy computation
        time_a = time()
        # compute accuracies using data_sets, noise kinds, noise levels, and
        # the specified random_state. save to Model_Result.results later
        accs = compute_accuracies(mdl_res.est, data_sets,
                                  noise_kinds = noise_kinds,
                                  noise_levels = noise_levels,
                                  random_state = _seed)
        # set computation time
        accs_time = time() - time_a
        # compute ela and rla matrices
        elas = accuracies2ela(accs)
        rlas = accuracies2rla(accs)
        # compute average elas and average rlas (will be row vector DataFrames)
        avg_elas = accuracies2ela(accs, averages = True)
        avg_rlas = accuracies2rla(accs, averages = True)
        # save accs, accs_time, elas, rlas, average elas and average rlas to
        # Model_Results results attribute using eponymous keys
        mdl_res.results[_KEY_ACCS] = accs
        mdl_res.results[_KEY_ACCS_TIME] = accs_time
        mdl_res.results[_KEY_ELAS] = elas
        mdl_res.results[_KEY_RLAS] = rlas
        mdl_res.results[_KEY_AVG_ELAS] = avg_elas
        mdl_res.results[_KEY_AVG_RLAS] = avg_rlas
        # print message to screen indicating completion
        print("{0}: evaluation completed ({1}, time = {2:.5f} s)"
              .format(mdl_res.name, mdl_res._est.__class__.__name__, accs_time))
    # print extra newline to divide completion messages and model results if any
    print()
    # return model results in mdl_results
    return mdl_results

def _display_results(cfg_data, mdl_results):
    """
    given a correctly formatted JSON-compatible dict of config data and a list
    of Model_Results, based on the values set in the config data dict, for each
    of the Model_Result objects, print a subset of possible results to screen.

    do not call externally. returns None

    parameters:

    cfg_data     dict correctly formatted for noisyeval from JSON config file
    mdl_results  list or iterable of Model_Results to display
    """
    assert isinstance(cfg_data, dict) and hasattr(mdl_results, "__iter__")
    # options for printing acc, ela, rla, avg ela, avg rla to the screen (str)
    disp_opts = [_DISP_ACCS, _DISP_ELAS, _DISP_RLAS, _DISP_AVG_ELAS,
                 _DISP_AVG_RLAS]
    # if options are 1, print to screen. if 0, do nothing. else, warn
    # note: set these options to 1 in config only if matrices are small.
    for opt in disp_opts:
        _flag = cfg_data[opt]
        if (_flag != 0) and (_flag != 1):
            print("{0}: warning: {1} has value {2}; must be 0 or 1"
                  .format(_PROGNAME, opt, _flag), file = stderr)
    # if one of these options is 1, set disp_out to True
    disp_out = False
    for opt in disp_opts:
        if cfg_data[opt] == 1:
            disp_out = True
            break
    if disp_out == True:
        # do for each Model_Result; also print model class name and name
        for mdl_res in mdl_results:
            print("results for {0} {1}:\n"
                  .format(mdl_res._est.__class__.__name__, mdl_res.name))
            # note that the output is in list order, not in config file order.
            # cycle through _MAIN_RESULTS_KEYS to preclude hard-coded refs.
            for _mrk, opt in zip(_MAIN_RESULTS_KEYS, disp_opts):
                # if the keyed value in cfg_data is 1, print to stdout
                if cfg_data[opt] == 1:
                    print("{0}\n".format(mdl_res.results[_mrk]))
    return None

def _paint_xla_plots(cfg_data, mdl_results):
    """
    given a correctly formatted JSON-compatible dict of config data and a list
    of Model_Results, based on the values set in the config data dict, paint
    and save an average ELA and/or average RLA comparison plot, or neither.

    do not call externally. returns None.

    parameters:

    cfg_data     dict correctly formatted for noisyeval from JSON config file
    mdl_results  list or iterable of Model_Results to make xLA plots for
    """
    assert isinstance(cfg_data, dict) and hasattr(mdl_results, "__iter__")
    # using mdl_results, create a list of model names
    mdl_names = [_mr.name for _mr in mdl_results]
    # results directory to write plots to
    results_dir = cfg_data[_RESULTS_DIR]
    # extract dicts for ELA figure options and RLA figure options
    ela_fig_opts = cfg_data[_ELA_FIG]
    rla_fig_opts = cfg_data[_RLA_FIG]
    # check if we want to plot average ELA comparison fig or not. warn if the
    # save value is invalid and don't save.
    save_ela_fig = ela_fig_opts[_FIGS_SAVE_FIG]
    if save_ela_fig == 1:
        # using mdl_results, create a list of the model average ELAs
        mdl_avg_elas = [_mr.results[_KEY_AVG_ELAS] for _mr in mdl_results]
        # send all the other keys in ela_fig_opts to the plotting function
        fig, ax = plot_avg_xla(mdl_names, mdl_avg_elas,
                               title = ela_fig_opts[_FIGS_FIG_TITLE],
                               figsize = ela_fig_opts[_FIGS_FIG_SIZE],
                               cmap = ela_fig_opts[_FIGS_FIG_CMAP],
                               xlabel = "noise level",
                               plot_kwargs = ela_fig_opts[_FIGS_PLOT_KWARGS])
        # save figure using dpi from _FIGS_FIG_DPI in results_dir using out_name
        # and print message upon completion
        ela_out = results_dir + "/" + out_name + "_" + _KEY_AVG_ELAS + ".png"
        fig.savefig(ela_out, dpi = ela_fig_opts[_FIGS_FIG_DPI])
        print("saved avg ELA plot to {0}".format(ela_out))
    elif save_ela_fig == 0: pass
    else:
        print("{0}: warning: {1}[{2}] has value {3}; must be 0 or 1"
              .format(_PROGNAME, _ELA_FIG, _FIGS_SAVE_FIG, save_ela_fig),
              file = stderr)
    # check if we want to plot average RLA comparison fig or not
    save_rla_fig = rla_fig_opts[_FIGS_SAVE_FIG]
    if save_rla_fig == 1:
        # using mdl_results, create a list of the model average RLAs
        mdl_avg_rlas = [_mr.results[_KEY_AVG_RLAS] for _mr in mdl_results]
        # send all the other keys in rla_fig_opts to the plotting function
        fig, ax = plot_avg_xla(mdl_names, mdl_avg_rlas,
                               title = rla_fig_opts[_FIGS_FIG_TITLE],
                               figsize = rla_fig_opts[_FIGS_FIG_SIZE],
                               cmap = rla_fig_opts[_FIGS_FIG_CMAP],
                               xlabel = "noise level",
                               plot_kwargs = rla_fig_opts[_FIGS_PLOT_KWARGS])
        # save figure using dpi from _FIGS_FIG_DPI in results_dir using out_name
        # and print message upon completion
        rla_out = results_dir + "/" + out_name + "_" + _KEY_AVG_RLAS + ".png"
        fig.savefig(rla_out, dpi = rla_fig_opts[_FIGS_FIG_DPI])
        print("saved avg RLA plot to {0}".format(rla_out))
    # do nothing if 0
    elif save_rla_fig == 0: pass
    else:
        print("{0}: warning: {1}[{2}] has value {3}; must be 0 or 1"
              .format(_PROGNAME, _RLA_FIG, _FIGS_SAVE_FIG, save_rla_fig),
              file = stderr)
    return None

# main
if __name__ == "__main__":
    # starting runtime
    rt_start = time()
    # get number of arguments from sys.argv
    argc = len(argv)
    # number of arguments is 2, print to stderr otherwise
    if argc == 1:
        print("{0}: no arguments. type '{0} {1}' for usage"
              .format(_PROGNAME, _HELP_FLAG), file = stderr)
        exit(0)
    elif argc == 2: pass
    else:
        print("{0}: too many arguments. type '{0} {1}' for usage"
              .format(_PROGNAME, _HELP_FLAG), file = stderr)
        exit(1)
    # first check if help; if so, print help and exit
    if argv[1] == _HELP_FLAG:
        print(_HELP_STR)
        exit(0)
    # else, try to open and get config dict. if err_code > 0, then there was an
    # error, either due to incorrect file type (must be JSON file), error in
    # opening file, or a JSON parsing error.
    cfg_data, err_code = _load_json_config(argv[1])
    if err_code > 0: exit(err_code)
    # check format of cfg_data. if err_code > 0, there is a formatting error.
    err_code = _check_json_config(cfg_data)
    if err_code > 0: exit(err_code)
    # get name of output file. argv[1] has full name of config file, so we first
    # need to strip the extension (note: we required .json file) and then strip
    # off either "/" or "\\"; try both since we don't know the system
    out_name = argv[1][:-5].split("/")[-1].split("\\")[-1]
    # get warm start status, i.e. if it is possible to do a warm start. if
    # err_code > 0, there is an error, and we exit.
    warm_start, err_code = _warm_start_status(cfg_data, out_name)
    if err_code > 0: exit(err_code)
    # smart warm starting: first check if out_name is in results_dir. if so,
    # then populate mdl_results (defined below) with Model_Results objects from
    # the dicts stored in out_name.pickle in results_dir. computation steps will
    # be skipped and mdl_results will be used directly for output. else compute.
    # make list for Model_Results but first set to None
    mdl_results = None
    # if warm_start is True, attempt a warm start by loading from the pickle
    if warm_start == True:
        # load the raw pickle from file
        mdl_results = _load_prior_results(cfg_data, out_name)
    # else if out_name.pickle does not exist in results_dir, then we cannot
    # perform a warm start and will have to do all the computation ourselves.
    # Model_Results files will be saved as dicts in one large dict to the
    # results_dir using out_name as the output file name (pickle)
    else:
        # evaluate the models and get results as list of Model_Results. may take
        # a very long time depending on the configuration
        mdl_results = _evaluate(cfg_data)
        # output dict
        models_out = {}
        for mdl_res in mdl_results:
            # convert individual Model_Results to dicts then key by model name.
            # note that estimator is also added to the results; this is useful
            # information and enables warm starting from the pickle
            res = mdl_res.results
            res[_KEY_EST] = mdl_res._est
            models_out[mdl_res.name] = res
        # pickle models_out to out_name.pickle in the results directory
        results_dir = cfg_data[_RESULTS_DIR]
        with open(results_dir + "/" + out_name + ".pickle", "wb") as outf:
            pickle.dump(models_out, outf)
    # based on values set in cfg_data, display model results to screen
    _display_results(cfg_data, mdl_results)
    # based on values set in cfg_data, create + save avg xLA comparison plots
    _paint_xla_plots(cfg_data, mdl_results)
    # record and print total runtime
    rt_total = time() - rt_start
    print("total runtime: {0:.5f} s".format(rt_total))
