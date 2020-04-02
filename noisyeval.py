#!/bin/python
# main entry point for training and evaluation of models on data sets with
# varying levels of noise. added shebang so that execution via ./noisyeval.py
# can be done as shorthand (if applicable). please run from command line only!
#
# Changelog:
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
from kyulib.utils import DataFrame2Data_Set, model_from_spec, Model_Results
from kyulib.metrics import accuracies2ela, accuracies2rla, compute_accuracies

__doc__ = """
main entry point for evaluation of sklearn-compatible models on data sets with
different levels of noise introduced, with computation of ELA (equalized loss
of accuracy), RLA (relative loss of accuracy) metrics, and optional creation of
comparative line plots of average model ELA/RLA statistics.

please read README.md for instructions on how to interpret and format the .json
file that governs the behavior of this script. passing the --help flag will
produce some usage instructions and this help blurb."""

_PROGNAME = "noisyeval"
_HELP_FLAG = "--help"
_HELP_STR = """Usage: {0} config_file.json
       {0} [ {1} ]{2}""".format(_PROGNAME, _HELP_FLAG, __doc__)

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
    # only allow .json file as config
    if argv[1].split(".")[-1] != "json":
        print("{0}: configuration file must be a .json file".format(_PROGNAME),
              file = stderr)
        exit(1)
    # if valid, then try to open
    try: cf = open(argv[1], "r")
    except FileNotFoundError:
        print("{0}: FileNotFoundError: could not locate file {1}"
              .format(_PROGNAME), file = stderr)
        exit(2)
    # this will get raised if u try to open a directory
    except PermissionError:
        print("{0}: PermissionError: please check that {1} is a regular file"
              .format(_PROGNAME), file = stderr)
        exit(2)
    # unpack using json.load then close cf, we don't need it anymore
    cfg_data = json.load(cf)
    cf.close()
    # check keys in cfg_data
    for _ck in _MAIN_CONFIG_KEYS:
        if _ck not in cfg_data:
            print(fill("{0}: error: missing required key {1}. please read README.md "
                       "for instructions on how to format configuration files"
                       .format(_PROGNAME, _ck), width = 80), file = stderr)
            exit(1)
    # check that the data and results directories exist + are directories
    data_dir, results_dir = cfg_data[_DATA_DIR], cfg_data[_RESULTS_DIR]
    if exists(data_dir) == False:
        print("{0}: error: cannot find data directory {1}"
              .format(_PROGNAME, data_dir), file = stderr)
        exit(2)
    if isdir(data_dir) == False:
        print("{0}: error: {1} is not a directory".format(_PROGNAME, data_dir),
              file = stderr)
        exit(2)
    if exists(results_dir) == False:
        print("{0}: error: cannot find results directory {1}"
              .format(_PROGNAME, results_dir), file = stderr)
        exit(2)
    if isdir(results_dir) == False:
        print("{0}: error: {1} is not a directory"
              .format(_PROGNAME, results_dir), file = stderr)
        exit(2)
    # before we start training the models, check the plotting parameters. no one
    # wants to learn after training for 6 hours that they aren't allowed to make
    # a graph... that would really suck, wouldn't it
    # options for making comparison plots of average ELAs
    ela_fig_opts = cfg_data[_ELA_FIG]
    rla_fig_opts = cfg_data[_RLA_FIG]
    # for each one, make sure that all the required keys are present
    for _ck in _XLA_FIG_KEYS:
        if _ck not in ela_fig_opts:
            print(fill("{0}: error: {1} missing required key {2}. please read "
                       "README.md for instructions on how to format average ELA"
                       "/RLA comparison plot params"
                       .format(_PROGNAME, _ELA_FIG, _ck), width = 80),
                  file = stderr)
            exit(1)
    # for each model param dict in cfg_data, create a Model_Results object; all
    # the Model_Results objects contain the name, unfitted model. all in a list.
    mdl_results = [Model_Results(mpd["name"], model_from_spec(mpd))
                   for mpd in cfg_data[_MODELS]]
    # assign random state to _seed, set to None if < 0
    _seed = cfg_data[_RANDOM_STATE]
    if _seed < 0: _seed = None
    # for each data file in the data_dir, read it into a DataFrame and then use
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
        avg_elas = accuracies2ela(accs, avg_ela = True)
        avg_rlas = accuracies2rla(accs, avg_rla = True)
        # save accs, accs_time, elas, rlas, average elas and average rlas to
        # Model_Results results attribute using eponymous keys
        mdl_res.results["accs"] = accs
        mdl_res.results["accs_time"] = accs_time
        mdl_res.results["elas"] = elas
        mdl_res.results["rlas"] = rlas
        mdl_res.results["avg_elas"] = avg_elas
        mdl_res.results["avg_rlas"] = avg_rlas
    # save Model_Results as dicts in one large dict to results_dir using name of
    # the config file as the output file name (pickle)
    models_out = {}
    for mdl_res in mdl_results:
        # convert individual Model_Results to dicts then key by model name
        res = mdl_res.results
        res["est"] = mdl_res._est
        models_out[mdl_res.name] = res
    # pickle models_out. argv[1] has full name of config file, so we first need
    # to strip the extension (note: we required .json file) and then strip off
    # either "/" or "\\"; try both since we don't know the system
    out_file = argv[1][:-5].split("/")[-1].split("\\")[-1]
    with open(results_dir + "/" + out_file + ".pickle", "wb") as outf:
            pickle.dump(models_out, outf)
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
    # if one of these options is 1, then we run the following loop
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
            if cfg_data[_DISP_ACCS] == 1:
                print("{0}\n".format(mdl_res.results["accs"]))
            if cfg_data[_DISP_ELAS] == 1:
                print("{0}\n".format(mdl_res.results["elas"]))
            if cfg_data[_DISP_RLAS] == 1:
                print("{0}\n".format(mdl_res.results["rlas"]))
            if cfg_data[_DISP_AVG_ELAS] == 1:
                print("{0}\n".format(mdl_res.results["avg_elas"]))
            if cfg_data[_DISP_AVG_RLAS] == 1:
                print("{0}\n".format(mdl_res.results["avg_rlas"]))
    # remember ela_fig_opts and rla_fig_opts? we finally come back to them.
    # check if we want to plot average ELA comparison fig or not. warn if the
    # save value is invalid and don't save.
    save_ela_fig = ela_fig_opts[_FIGS_SAVE_FIG]
    if save_ela_fig == 1:
        # send all the other keys in ela_fig_opts to the plotting function
        print("plot and save ela figure")
    elif save_ela_fig == 0: pass
    else:
        print("{0}: warning: {1}[{2}] has value {3}; must be 0 or 1"
              .format(_PROGNAME, _ELA_FIG, _FIGS_SAVE_FIG, save_ela_fig),
              file = stderr)
    # check if we want to plot average RLA comparison fig or not
    save_rla_fig = rla_fig_opts[_FIGS_SAVE_FIG]
    if save_rla_fig == 1:
        # send all the other keys in rla_fig_opts to the plotting function
        print("plot and save rla figure")
    # do nothing if 0
    elif save_rla_fig == 0: pass
    else:
        print("{0}: warning: {1}[{2}] has value {3}; must be 0 or 1"
              .format(_PROGNAME, _RLA_FIG, _FIGS_SAVE_FIG, save_rla_fig),
              file = stderr)
    # record and print total runtime
    rt_total = time() - rt_start
    print("total runtime: {0:.5f} s".format(rt_total))
