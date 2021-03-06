# contains methods for creating plots such as average xLA comparison plots.
#
# Changelog:
#
# 04-09-2020
#
# edited try/except in plot_avg_xla to remove exception chaining.
#
# 04-05-2020
#
# initial creation. added plot_avg_xla, which supports customized figure sizes,
# Axes title, x and y labels, matplotlib.cm colormap, and kwargs for lines.
# added option to pass in kwargs to matplotlib.axes.Axes.plot for each line.
# added forgotten length checking for model names, xLAs, and plot kwargs.

from matplotlib.pyplot import subplots
import matplotlib.cm as cm
from pandas import DataFrame

__doc__ = """
contains methods for making relevant plots such as average xLA comparison plots.

dependencies: matplotlib, pandas
"""

_MODULE_NAME = "plotting"

def plot_avg_xla(mdl_names, mdl_xlas, title = "", figsize = (6, 5),
                 cmap = "viridis", xlabel = "", ylabel = "", plot_kwargs = None):
    """
    gives an iterable of n_models str model names and an iterable of n_models
    DataFrames of average xLAs, with shapes (1, n_levels - 1) where n_levels is
    the number of noise levels and may be different for each model, create a
    line plot of the average xLAs, with each line labeled with a model name.

    the DataFrame columns labels must end in _[level], where level is a float
    noise level in (0, 1), as noise levels are inferred from the column labels.

    returns a fig, ax pair of the Figure and Axes object, respectively.

    parameters:

    mdl_names    iterable of n_models str model names
    mdl_xlas     iterable of n_models DataFrames of shape (1, n_levels - 1),
                 where the DataFrames contains average xLA per noise level.
    title        optional str governing axes title, default ""
    figsize      optional iterable governing figure size in inches, default
                 (6, 5). format is (width, height).
    xlabel       optional str x-axis label, default ""
    ylabel       optional str y-axis label, default ""
    cmap         optional str matplotlib.cm color map name, default "viridis"
    plot_kwargs  optional iterable of n_models dicts containing keyword 
                 arguments to pass to matplotlib.axes.Axes.plot, with one dict
                 corresponding to a name in mdl_names and its xlas DataFrame
                 in mdl_xlas. default None, which passes no keyword arguments.
                 passing an empty list also passes no keyword arguments.
    """
    _fn = plot_avg_xla.__name__
    # sanity checks
    if hasattr(mdl_names, "__iter__"):
        # don't allow strings or dicts
        if isinstance(mdl_names, str) or isinstance(mdl_names, dict):
            raise TypeError("{0}: iterable mdl_names must not be str or dict"
                            .format(_fn))
    else:
        raise TypeError("{0}: mdl_names must be an iterable, not {1}"
                        .format(_fn, type(mdl_names)))
    if hasattr(mdl_xlas, "__iter__"):
        # don't allow strings or dicts
        if isinstance(mdl_xlas, str) or isinstance(mdl_xlas, dict):
            raise TypeError("{0}: iterable mdl_xlas must not be str or dict"
                            .format(_fn))
    else:
        raise TypeError("{0}: mdl_xlas must be an iterable, not {1}"
                        .format(_fn, type(mdl_xlas)))
    if not isinstance(title, str):
        raise TypeError("{0}: title must be str, not {1}"
                        .format(_fn, type(title)))
    # skip figsize; matplotlib will check that for you
    if not isinstance(xlabel, str):
        raise TypeError("{0}: xlabel must be str, not {1}"
                        .format(_fn, type(xlabel)))
    if not isinstance(ylabel, str):
        raise TypeError("{0}: ylabel must be str, not {1}"
                        .format(_fn, type(ylabel)))
    if not isinstance(cmap, str):
        raise TypeError("{0}: cmap must be str, not {1}"
                        .format(_fn, type(cmap)))
    # check if cmap is a valid matplotlib color map
    if not hasattr(cm, cmap):
        raise ValueError("{0}: {1} is not a valid matplotlib.cm colormap"
                         .format(_fn, cmap))
    # check that mdl_names and mdl_xlas have the same length + get length
    if len(mdl_names) != len(mdl_xlas):
        raise ValueError("{0}: mdl_names and mdl_xlas must have the same length"
                         .format(_fn))
    if plot_kwargs is None: pass
    elif hasattr(plot_kwargs, "__iter__"):
        # don't allow strings or dicts
        if isinstance(plot_kwargs, str) or isinstance(plot_kwargs, dict):
            raise TypeError("{0}: iterable plot_kwargs must not be str or dict"
                            .format(_fn))
    else:
        raise TypeError("{0}: plot_kwargs must be an iterable, not {1}"
                        .format(_fn, type(plot_kwargs)))
    # check that length of mdl_names and mdl_xlas are the same length
    if len(mdl_names) != len(mdl_xlas):
        raise ValueError("{0}: mdl_names and mdl_xlas must have the same length"
                         .format(_fn))
    # get number of models n_models
    n_models = len(mdl_names)
    # if plot_kwargs is None or has length zero, make it a list of empty dicts
    # length n_models, else check that its length is equal to n_models
    if (plot_kwargs is None) or (len(plot_kwargs) == 0):
        plot_kwargs = [{} for _ in range(n_models)]
    elif len(plot_kwargs) == n_models: pass
    else:
        raise ValueError("{0}: plot_kwargs must have the same length as "
                         "mdl_names and mdl_xlas".format(_fn))
    # create fig, ax from subplots + set ax title
    fig, ax = subplots(nrows = 1, ncols = 1, figsize = figsize)
    ax.set_title(title)
    # get color map from cm
    cmap_obj = getattr(cm, cmap)
    # get n_models colors from the cmap in cm. to reduce unnecessary contrast
    # between colors, we divide [0, 1] into 2 + n_models divisions and ignore
    # the endpoints. this avoids the extreme standardized colors 0, 1.
    colors = [None for _ in range(n_models)]
    for i in range(n_models):
        colors[i] = cmap_obj((i + 1) / (n_models + 2))
    # for each triple of colors, DataFrames of average xLAs, and plot kwargs
    for lc, xlas, kwargs in zip(colors, mdl_xlas, plot_kwargs):
        # first check if xlas is a proper DataFrame
        if not isinstance(xlas, DataFrame):
            raise TypeError("{0}: elements of mdl_xlas must be DataFrames"
                            .format(_fn))
        # get noise levels from the columns
        levels = [None for _ in range(len(xlas.columns))]
        for i, col in enumerate(xlas.columns):
            # try to split by _ and check if last arg is a float
            nl = col.split("_")[-1]
            try: nl = float(nl)
            # raise from None removes exception chaining
            except ValueError:
                raise ValueError("{0}: columns of xLA DataFrame are incorrectly"
                                 "formatted. please see function docstring"
                                 .format(_fn)) from None
            levels[i] = nl
        # check if kwargs is a dict; if not, raise TypeError
        if not isinstance(kwargs, dict):
            raise TypeError("{0}: elements of plot_kwargs must be dict"
                            .format(_fn))
        # plot the noise levels on x-axis and xlas values on y-axis with kwargs
        ax.plot(levels, xlas.iloc[0, :], color = lc, **kwargs)
    # add legend, xlabel, ylabel to the axes and return fig, ax
    ax.legend(mdl_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
