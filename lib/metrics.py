# contains metrics for quantifying performance characteristics of models.
#
# Changelog:
#
# 03-30-2020
#
# initial creation. added functions to compute, given two classifier accuracies
# calculated appropriately, relative loss of error and equalized loss of error.

from numpy import nan

__doc__ = """
contains metrics for quantifying model performance characteristics.
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
    

