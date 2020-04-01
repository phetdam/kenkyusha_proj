# contains metrics for quantifying performance characteristics of models. also
# has the main "training" methods that compute classifier accuracies on data.
#
# Changelog:
#
# 04-01-2020
#
# completed model_from_spec and moved it to utils (wrong module). completed
# compute_accuracies, which computes classifier accuracies on multiple data sets
# with optional noisy copies to be fitted at various levels of noise. made note
# in the docstring for compute_accuracies that accuracies are test accuracies.
#
# 03-31-2020
#
# added message for not running module as script. started work on the function
# compute_accuracies for computing the accuracy matrix of an estimator given
# multiple Data_Set objects, multiple noise kinds and noise levels. started work
# on the model_from_spec which instantiates a model from a JSON-like dict.
#
# 03-30-2020
#
# initial creation. added functions to compute, given two classifier accuracies
# calculated appropriately, relative loss of error and equalized loss of error.

from abc import ABCMeta
from copy import deepcopy
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sys import stderr

from .utils import Data_Set, noisy_copy

__doc__ = """
contains metrics for quantifying model performance characteristics.

dependencies: pandas, sklearn
"""

_MODULE_NAME = "metrics"

def rla(a_zero, a_alpha):
    """
    the relative loss of accuracy metric as referenced in [1]. returns a float
    which will be in the range of [0, 1], describing the percentage loss of
    accuracy in a classifier given a level of data set noise alpha.

    rla is valid only with accuracy measures from a classifier trained on two
    copies of a data set D, one of which has a fraction of noisy examples alpha.

    parameters:

    a_zero     float, baseline classifier accuracy with no added noise
    a_alpha    float, classifier accuracy on data set given noise level alpha,
               where the noise level alpha is in the range [0, 1]

    references:

    [1] Saez, J. A., Luengo, J., & Herrara, F. (2015, May 8). Evaluating the
    classifier behavior with noisy data considering performance and robustness:
    The Equalized Loss of Accuracy measure. Neurocomputing, 176, 26-35. 
    https://doi.org/10.1016/j.neucom.2014.11.086
    """
    _fn = rla.__name__
    # type check
    # note that the baseline accuracy cannot be zero, or metric undefined
    if (not isinstance(a_zero, float)) or ((a_zero <= 0) or (a_zero > 1)):
        raise TypeError("{0}: a_zero: float between (0, 1] expected, {1} "
                        "received".format(_fn, type(a_zero)))
    if (not isinstance(a_alpha, float)) or ((a_alpha < 0) or (a_alpha > 1)):
        raise TypeError("{0}: a_alpha: float between [0, 1] expected, {1} "
                        "received".format(_fn, type(a_alpha)))
    # equivalent to (a_zero - a_alpha) / a_zero, as in original formulation
    return 1 - (a_alpha / a_zero)

def ela(a_zero, a_alpha):
    """
    the equalized loss of accuracy metric as described in [1]. returns a float
    which will be in the range of [0, inf). lower values are better.

    ela is valid only with accuracy measures from a classifier trained on two
    copies of a data set D, one of which has a fraction of noisy examples alpha.

    parameters:

    a_zero     float, baseline classifier accuracy with no added noise
    a_alpha    float, classifier accuracy on data set given noise level alpha,
               where the noise level alpha is in the range [0, 1]

    references:

    [1] Saez, J. A., Luengo, J., & Herrara, F. (2015, May 8). Evaluating the
    classifier behavior with noisy data considering performance and robustness:
    The Equalized Loss of Accuracy measure. Neurocomputing, 176, 26-35. 
    https://doi.org/10.1016/j.neucom.2014.11.086
    """
    _fn = ela.__name__
    # type check
    # note that the baseline accuracy cannot be zero, or metric undefined
    if (not isinstance(a_zero, float)) or ((a_zero <= 0) or (a_zero > 1)):
        raise TypeError("{0}: a_zero: float between (0, 1] expected, {1} "
                        "received".format(_fn, type(a_zero)))
    if (not isinstance(a_alpha, float)) or ((a_alpha < 0) or (a_alpha > 1)):
        raise TypeError("{0}: a_alpha: float between [0, 1] expected, {1} "
                        "received".format(_fn, type(a_alpha)))
    return (1 - a_alpha) / a_zero

def compute_accuracies(est, dsets, noise_kinds = None, noise_levels = None,
                       random_state = None, fit_kwargs = None,
                       score_kwargs = None):
    """
    for an sklearn-style classifier instance with fit, predict, and score
    methods, an iterable of Data_Set objects, and an optional list of noise
    kinds and noise levels, computes test accuracies for each Data_Set as well
    as for noisy copies of each Data_Set, which satisfy each combination of
    noise type and noise level. if there are n_kinds noise types, n_levels noise
    levels, and n_dsets Data_Sets, the method returns a DataFrame row-indexed by
    Data_Set name, column-indexed by noise levels, and will have the final shape
    of (n_dsets, n_kinds * n_levels + 1).

    the column label format is acc_[kind]_[noise fraction]. for reproducibility,
    it is recommended that if applicable, est have a fixed random seed. the
    value for random_state will be used to seed every call to .utils.noisy_copy.

    warning: NOT thread safe! due to the use of numpy.random.seed.
    
    parameters:

    est           sklearn-like classifier instance implementing fit, predict,
                  and score instance methods. call signatures are below.

                  est.fit(X_train, y_train, **kwargs)

                  parameters:

                  X_train    ndarray or DataFrame training feature matrix
                  y_train    ndarray or DataFrame training response vector
                  kwargs    keyword arguments

                  est.predict(X, **kwargs)

                  parameters:

                  X         ndarray or DataFrame validation/test feature matrix
                  kwargs    keyword arguments

                  est.score(X, y, **kwargs)

                  parameters:

                  X         ndarray or DataFrame feature matrix
                  kwargs    keyword arguments

                  only fit() and score() will be called in this method.

    dsets         list, ndarray, Series of .utils.Data_Set objects
    noise_kinds   optional list, ndarray, Series of str noise kinds to indicate
                  what kind of noise kind should be passed to .utils.noisy_copy.
    noise_levels  optional list, ndarray, Series of noise levels in (0, 1) to
                  introduce to each Data_Set in dsets, default None. if
                  noise_levels is not None, then for each Data_Set in dsets,
                  noisy copies of the Data_Set will be made for each noise level
                  and accuracies computed for the noisy copy.
    random_state  optional int seed used to seed the numpy PRNG directly through
                  a call to numpy.random.seed. used only to control the behavior
                  of multiple calls to .utils.noisy_copy if applicable, i.e.
                  noise_kinds and noise_levels are both not None. will call
                  numpy.random.seed only once before the noisy copies are made.
    fit_kwargs    optional kwargs to pass to est.fit
    score_kwargs  optional kwargs to pass to est.score
    """
    _fn = compute_accuracies.__name__
    # type checks
    # automatic ok if est is an instance of ClassifierMixin
    if isinstance(est, ClassifierMixin): pass
    # for more general types, check if the three methods are implemented
    else:
        # check if implements fit, predict, score
        if hasattr(est, "fit") == False:
            raise AttributeError("{0}: est must implement fit method"
                                 .format(_fn))
        if hasattr(est, "predict") == False:
            raise AttributeError("{0}: est must implement predict method"
                                 .format(_fn))
        if hasattr(est, "score") == False:
            raise AttributeError("{0}: est must implement score method"
                                 .format(_fn))
        # leave argspec checking to evaluation time
    # check that dsets is iterable but not a string or dict
    if hasattr(dsets, "__iter__"):
        if isinstance(dsets, str) or isinstance(dsets, dict):
            raise TypeError("{0}: dsets must not be str or dict".format(_fn))
    else: raise TypeError("{0}: dsets must be an iterable".format(_fn))
    # check that dsets is not empty; defer element type checking
    if len(dsets) == 0:
        raise ValueError("{0}: dsets must have length at least 1")
    # check noise_kinds, noise_levels; must be None/not None at the same time
    if (noise_kinds is None) and (noise_levels is None): pass
    else:
        if noise_kinds is None:
            raise ValueError("{0}: noise_kinds is None while noise_levels is "
                             "not None".format(_fn))
        if noise_levels is None:
            raise ValueError("{0}: noise_levels is None while noise_kinds is "
                             "not None".format(_fn))
        # check that both are iterable but not strings or dicts
        if hasattr(noise_kinds, "__iter__"):
            if isinstance(noise_kinds, str) or isinstance(noise_kinds, dict):
                raise TypeError("{0}: noise_kinds must not be str or dict"
                                .format(_fn))
        if hasattr(noise_levels, "__iter__"):
            if isinstance(noise_levels, str) or isinstance(noise_levels, dict):
                raise TypeError("{0}: noise_levels must not be str or dict"
                                .format(_fn))
        # defer type checking of elements to later
    # check random_state
    if (random_state is None) or isinstance(random_state, int): pass
    else: raise TypeError("{0}: random_state must be None or int".format(_fn))
    # check fit and score kwargs; set to empty dict if None
    if fit_kwargs is None: fit_kwargs = {}
    elif isinstance(fit_kwargs, dict): pass
    else: raise TypeError("{0}: fit_kwargs must be None or dict".format(_fn))
    if score_kwargs is None: score_kwargs = {}
    elif isinstance(score_kwargs, dict): pass
    else: raise TypeError("{0}: score_kwargs must be None or dict".format(_fn))
    # get number of data sets, number of noise kinds, number of noise levels
    n_dsets = len(dsets)
    n_kinds = len(noise_kinds) if not noise_kinds is None else 0
    n_levels = len(noise_levels) if not noise_levels is None else 0
    # create  row index of data set names and noise kinds/levels for cols
    row_index = [None for _ in range(n_dsets)]
    for i in range(n_dsets): row_index[i] = dsets[i].name
    col_index = [None for _ in range(n_kinds * n_levels + 1)]
    col_index[0] = "acc_0"
    # noise_kinds and noise_levels are both not be None at the same time
    if not noise_kinds is None:
        for i in range(n_kinds):
            nk = noise_kinds[i]
            # check if str
            if not isinstance(nk, str):
                raise TypeError("{0}: elements of noise_kinds must be str"
                                .format(_fn))
            for j in range(n_levels):
                nl = noise_levels[j]
                # check if float in (0, 1)
                if (not isinstance(nl, float)) or ((nl <= 0) or (nl >= 1)):
                    raise TypeError("{0}: elements of noise_levels must be "
                                    "floats in range (0, 1)".format(_fn))
                col_index[1 + (i * n_levels) + j] = "acc_" + nk + "_" + str(nl)
    # create output DataFrame
    df = DataFrame(data = None, index = row_index, columns = col_index)
    # for each Data_Set in dsets
    for ds in dsets:
        # check if actually a Data_Set
        if not isinstance(ds, Data_Set):
            raise TypeError("{0}: elements of dsets must be Data_Set instances"
                            .format(_fn))
        # get accuracy score for ds, noise level 0. train est on the training
        # data, validate on test data, and write accuracy to column acc_0.
        # note: must create new instance since fitting is inplace
        _est = deepcopy(est)
        _est.fit(ds.X_train, ds.y_train, **fit_kwargs)
        df.loc[ds.name, "acc_0"] = _est.score(ds.X_test, ds.y_test,
                                             **score_kwargs)
        # if noise_kinds is not None, then compute noisy accuracies
        if not noise_kinds is None:
            for nk in noise_kinds:
                for nl in noise_levels:
                    # create noisy copy
                    dsn = noisy_copy(ds, fraction = nl, kind = nk,
                                     random_state = random_state)
                    # get accuracy score for ds with noise type nk, level nl
                    _est = deepcopy(est)
                    _est.fit(dsn.X_train, dsn.y_train, **fit_kwargs)
                    # compute column label for brevity
                    _col = "acc_" + nk + "_" + str(nl)
                    df.loc[ds.name, _col] = _est.score(dsn.X_test, dsn.y_test,
                                                       **score_kwargs)
    # return DataFrame df
    return df
    
if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
