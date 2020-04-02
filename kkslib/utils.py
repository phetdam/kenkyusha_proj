# contains general utility functions/classes
#
# Changelog:
#
# 04-02-2020
#
# added references to constants in ._config_keys to avoid hardcoding. updated
# docstring for the Data_Set class to provide an easier example using the
# DataFrame2Data_Set function (easier in practice).
#
# 04-01-2020
#
# added function model_from_spec for returning an sklearn-like model instance
# given an appropriately formatted JSON-representable dict. modified noisy_copy
# docstring to note that it is not thread safe due to numpy.random.seed call.
# added missing importlib import for module_from_spec and modified the
# module_from_spec to call itself recursively to handle nested estimators.
#
# 03-31-2020
#
# updated docstring for Data_Set class and checked over all the functions. added
# explicit check for random_state being None or not in noisy_copy.
#
# 03-30-2020
#
# moved to lib directory so that we can refer to module as part of a package.
# added the convenience function DataFrame2Data_Set to make it easier to convert
# from a DataFrame to a Data_Set, as DataFrames are common data structures.
# added function to create copies of Data_Sets with a specified fraction of
# label (only implemented type) noise to the training/test sets. also modified
# Data_Set to have attributes indicating amount artificially introduced noise.
#
# 03-27-2020
#
# modified Data_Set class to add splitting of the input feature matrix and
# response into training and test data portions using sklearn's train_test_split
# method. also takes a random seed to determine the exact split.
#
# 03-26-2020
#
# added implementation for the Data_Set wrapper class. 
#
# 03-25-2020
#
# initial creation. added __doc__ and _MODULE_NAME. no substantial code.

from copy import deepcopy
from importlib import import_module
from numpy import arange, array, array_equal, hstack, ndarray, setdiff1d, \
    unique, union1d
from numpy.random import choice, seed
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sys import stderr
from textwrap import fill

# import keyword constants for model_from_spec
from ._config_keys import _MODELS_MODULE, _MODELS_MODEL, _MODELS_PARAMS

__doc__ = """
contains general utility functions/classes for loading/converting data, etc.

dependencies: numpy, pandas, sklearn
"""

_MODULE_NAME = "utils"

class Data_Set:
    """
    data set wrapper class. recommended to construct from pandas.DataFrame
    using DataFrame2Data_Set, which is a friendlier interface. allows user to
    specify what fraction of the data should be the holdout data.

    built on top of sklearn.model_selection.train_test_split.

    example:

    from numpy import array
    from utils import Data_Set, DataFrame2Data_Set
    # from pandas DataFrame df
    data1 = Data_Set(df.iloc[:, :-1].values, df.iloc[:, -1].values, 
                     array(df.columns[:-1]), "response", "data1", 
                     test_size = 0.3, random_state = 7)
    # using DataFrame2Data_Set
    data2 = DataFram2Data_Set(df, "data2", test_size = 0.4, random_state = 1)

    attributes:

    X_train               ndarray feature matrix for training set
    y_train               ndarray response vector for training set
    X_train               ndarray feature matrix for test set
    y_train               ndarray response vector for test set
    features              ndarray of str feature labels
    response              str response label
    name                  str name identifying the data set
    n_samples             total number of data samples in the Data_Set
    n_features            number of feature columns
    test_size             fraction of n_samples that are test samples
    random_state          int seed used to seed the PRNG when making train/test
                          splits using sklearn's train_test_split; can be None
    added_noise_fraction  fraction of data points in both training and test sets
                          that have noise applied to them
    added_noise_type      kind of noise applied; ex. label noise
    """
    def __init__(self, X_in, y_in, X_labs, y_lab, dname, test_size = 0.2,
                 random_state = None, _no_check = False):
        """
        constructor for Data_Set object.

        parameters:

        X_in            ndarray feature matrix, shape (n_samples, n_features)
        y_in            ndarray response vector, shape (n_samples,)
        X_labs          ndarray of feature labels, shape (n_features,)
        y_lab           str response label
        dname           str data set name
        test_size       optional float in (0, 1) for test fraction, default 0.2
        random_state    optional int to seed PRNG seed, default None

        do not use the following parameters unless you are a developer.

        _no_check       boolean, default False. if True skips type checking.
        """
        # type check, skip if _no_check is True
        if _no_check == False:
            if not isinstance(X_in, ndarray):
                raise TypeError("{0}: X_in: ndarray expected, {1} received"
                                .format(Data_Set.__name__, type(X_in)))
            if not isinstance(y_in, ndarray):
                raise TypeError("{0}: y_in: ndarray expected, {1} received"
                                .format(Data_Set.__name__, type(y_in)))
            if not isinstance(X_labs, ndarray):
                raise TypeError("{0}: X_labs: ndarray expected, {1} received"
                                .format(Data_Set.__name__, type(X_labs)))
            if not isinstance(y_lab, str):
                raise TypeError("{0}: y_lab: str expected, {1} received"
                                .format(Data_Set.__name__, type(y_lab)))
            if not isinstance(dname, str):
                raise TypeError("{0}: dname: str expected, {1} received"
                                .format(Data_Set.__name__, type(dname)))
            if (not isinstance(test_size, float)) or \
               ((test_size <= 0) or (test_size >= 1)):
                raise TypeError("{0}: test_size: float between (0, 1) expected,"
                                " {1} received".format(Data_Set.__name__,
                                                       type(test_size)))
            if (not isinstance(random_state, int)) and \
               (not random_state is None):
                raise TypeError("{0}: random_state: int or None expected, {1} "
                                "received".format(Data_Set.__name__,
                                                  type(random_state)))
        # check that shapes match up
        if X_in.shape[0] != y_in.shape[0]:
            raise ValueError("{0}: X_in and y_in row mismatch"
                             .format(Data_Set.__name__))
        if X_in.shape[1] != X_labs.shape[0]:
            raise ValueError("{0}: X_in columns and X_labs count mismatch"
                             .format(Data_Set.__name__))
        # split X_in and y_in into X_train, X_test, y_train, y_test
        Xtn, Xtt, ytn, ytt = train_test_split(X_in, y_in, test_size = test_size,
                                              random_state = random_state)
        # set instance attributes
        self.X_train, self.X_test = Xtn, Xtt
        self.y_train, self.y_test = ytn, ytt
        self.features = X_labs
        self.response = y_lab
        self.n_samples = X_in.shape[0]
        self.n_features = X_in.shape[1]
        self.name = dname
        self.test_size = test_size
        self.random_state = random_state
        self.added_noise_fraction = 0.0
        self.added_noise_type = None

    # instance methods
    def to_DataFrames(self):
        """
        returns the Data_Set data as a tuple of two DataFrames, one with the
        training data and one with the test data. For both DataFrames, y 
        columns are appended as the last column of X matrices and the response
        labels appended as the last element of the feature labels.
        """
        # list of all features + appended response label
        labels = list(self.features) + [self.response]
        # need to reshape y to get correct shape to broadcast when stacking
        # training data
        df_tn = DataFrame(data = hstack(
            (self.X_train, self.y_train.reshape([self.y_train.shape[0], 1]))),
                          columns = labels)
        # test data
        df_tt = DataFrame(data = hstack(
            (self.X_test, self.y_test.reshape([self.y_test.shape[0], 1]))),
                          columns = labels)
        return df_tn, df_tt

    # define repr and str format
    def __repr__(self):
        return fill("{0}(name = {1}, n_samples = {2}, n_features = {3}, "
                    "test_size = {4}, random_state = {5}, added_noise_fraction"
                    " = {6}, added_noise_type = {7}, response = {8}, "
                    "features = {9})"
                    .format(Data_Set.__name__, self.name, self.n_samples,
                            self.n_features, self.test_size,
                            self.random_state, self.added_noise_fraction,
                            self.added_noise_type, self.response,
                            list(self.features)), width = 80,
                    subsequent_indent = (len(Data_Set.__name__) + 1) * " ")

    def __str__(self): return self.__repr__()

class Model_Results:
    """
    flexible wrapper to hold various model validation results.

    attributes:

    name       model name, read-only str
    est        unfitted sklearn-like estimator instance implementing fit,
               predict, and score functions. read-only; accessing the property
               will return a deep copy of the estimator. call signatures:

               est.fit(X_train, y_train, **kwargs)

               parameters:

               X_train    ndarray or DataFrame training feature matrix
               y_train    ndarray or iterable training response vector
               kwargs     fit() keyword arguments

               est.predict(X, **kwargs)

               parameters:

               X         ndarray or DataFrame validation/test feature matrix
               kwargs    predict() keyword arguments

               est.score(X, y, **kwargs)

               parameters:

               X         ndarray or DataFrame feature matrix
               kwargs    score() keyword arguments

    results    dict holding estimator results; access individual results by key.
    """
    def __init__(self, name, est):
        """
        constructor for Model_Results object.

        parameters:

        name      str model name
        est       sklearn-like estimator instance implementing fit, predict, and
                  score functions. see class docstring for details.
        """
        # type checks
        if not isinstance(name, str):
            raise TypeError("{0}: name must be str"
                            .format(Model_Results.__name__))
        if not hasattr(est, "fit"):
            raise AttributeError("{0}: est must implement fit method"
                                 .format(Model_Results.__name__))
        if not hasattr(est, "predict"):
            raise AttributeError("{0}: est must implement predict method"
                                 .format(Model_Results.__name__))
        if not hasattr(est, "score"):
            raise AttributeError("{0}: est must implement score method"
                                 .format(Model_Results.__name__))
        # set attributes
        self._name = name
        self._est = est
        self.results = {}

    # read-only access to name and est
    @property
    def name(self): return self._name
    
    @property
    def est(self): return deepcopy(self._est)
    
    # define repr and str format
    def __repr__(self):
        return fill("{0}(name = {1}, est = {2}, results_keys = {3})"
                    .format(Model_Results.__name__, self._name, self._est,
                            list(self.results.keys())), width = 80,
                    subsequent_indent = (len(Model_Results.__name__) + 1) * " ")

    def __str__(self): return self.__repr__()

def DataFrame2Data_Set(df, dname, response = None, test_size = 0.2,
                       random_state = None):
    """
    given a pandas DataFrame, use the data in the DataFrame to create a Data_Set
    instance with separate training and test data sets. returns Data_Set.

    parameters:

    df            DataFrame, shape (n_samples, n_features + 1)
    dname         str name for the Data_Set
    response      optional str to indicate which column is the response column,
                  default None to treat the last column as the response column
    test_size     optional float in (0, 1) to indicate fraction to set aside as
                  test data; default value is 0.2
    random_state  optional int to seed the PRNG, default None
    """
    _fn = DataFrame2Data_Set.__name__
    # type checks
    if not isinstance(df, DataFrame):
        raise TypeError("{0}: df: DataFrame expected, {1} received"
                        .format(_fn, type(df)))
    if not isinstance(dname, str):
        raise TypeError("{0}: dname: str expected, {1} received"
                        .format(_fn, type(dname)))        
    if (not response is None) and (not isinstance(response, str)):
        raise TypeError("{0}: response: str expected, {1} received"
                        .format(_fn, type(str)))
    if (not isinstance(test_size, float)) or \
       ((test_size <= 0) or (test_size >= 1)):
        raise TypeError("{0}: test_size: float between (0, 1) expected, {1} "
                        "received".format(_fn, type(test_size)))
    if (not isinstance(random_state, int)) and (not random_state is None):
        raise TypeError("{0}: random_state: int or None expected, {1} received"
                        .format(_fn, type(test_size)))
    # if response is None or last column, easy to make X and y
    X, y = None, None
    if (response is None) or (response == df.columns[-1]):
        X, y = df.iloc[:, :-1].copy(), df.iloc[:, -1].copy()
    # else
    else:
        # index of response columns
        ri = None
        # get index of response in df.columns
        for i, col in enumerate(df.columns):
            if col == response:
                ri = i
                break
        # if ri still None, raise KeyError
        if ri is None:
            raise KeyError("{0}: column {1} does not exist".format(_fn, response))
        # get separate halves and then join
        X_1, X_2 = df.iloc[:, :ri], df.iloc[:, (ri + 1):]
        X = X_1.join(X_2)
        y = df.loc[:, response].copy()
    # return new Data_Set; disable input checking for speed. note that X.columns
    # is actually a pandas Index, not an ndarray, hence the conversion
    return Data_Set(X.values, y.values, array(X.columns), response, dname,
                    test_size = test_size, random_state = random_state,
                    _no_check = True)

def noisy_copy(ds, fraction = 0.2, kind = "label", random_state = None):
    """
    given a Data_Set, create a copy and add noise to a specified fraction of the
    samples in the copy. returns a Data_Set that is a copy of the original, with
    the specified noise added to a specified fraction of the train/test samples.

    currently, only "label" noise is supported.

    not thread safe due to call to numpy.random.seed!

    parameters:

    ds              Data_Set
    fraction        optional float in (0, 1) indicating what fraction of the 
                    training and test sets should be affected, default 0.2
    kind            optional str, the kind of noise added to ds, default "label"
    random_state    optional int to seed the numpy PRNG, default None

    below is a description of the noise types supported by the "kind" kwarg.

    label    given an affected sample with label k out of m many labels,
             randomly select one of the m - 1 labels to assign to the sample.
    """
    _fn = noisy_copy.__name__
    # type checks
    if not isinstance(ds, Data_Set):
        raise TypeError("{0}: ds: Data_Set expected, {1} received"
                        .format(_fn, type(ds)))
    if (not isinstance(fraction, float)) or \
       ((fraction <= 0) or (fraction >= 1)):
        raise TypeError("{0}: fraction: float between (0, 1) expected, {1} "
                        "received".format(_fn, type(fraction)))
    if not isinstance(kind, str):
        raise TypeError("{0}: kind: str expected, {1} received"
                        .format(_fn, type(str)))
    if (not isinstance(random_state, int)) and (not random_state is None):
        raise TypeError("{0}: random_state: int or None expected, {1} received"
                        .format(_fn, type(random_state)))
    # create deep copy the original data set ds
    ds_copy = deepcopy(ds)
    # get number of training and test samples
    n_tn, n_tt = ds.X_train.shape[0], ds.X_test.shape[0]
    # create index arrays for training and test data sets
    idx_tn, idx_tt = arange(0, n_tn), arange(0, n_tt)
    # use random_state to seed numpy's PRNG for reproducibility if not None
    if not random_state is None: seed(seed = random_state)
    # get training and test set indices to add noise to; if fractional, truncate
    nidx_tn = choice(idx_tn, size = int(fraction * n_tn))
    nidx_tt = choice(idx_tt, size = int(fraction * n_tt))
    # do different things based on the noise type. only "label" supported.
    if kind == "label":
        # get unique response values in the training and test sets; union the
        # two arrays if they are not equal
        uniq_tn, uniq_tt, uniqs = unique(ds.y_train), unique(ds.y_test), None
        if array_equal(uniq_tn, uniq_tt) == True: uniqs = uniq_tn
        else: uniqs = union1d(uniq_tn, uniq_tt)
        # if lengths of uniq_tn and uniq_tt equal the number of training and
        # test samples respectively, warn that ds may be a regression data set
        if (uniq_tn.shape[0] == ds.y_train.shape[0]) and \
           (uniq_tt.shape[0] == ds.y_test.shape[0]):
            print("{0}: warning: Data_Set {1} may be a regression data set"
                  .format(_fn, ds.name), file = stderr)
        # for each index i in nidx_tn, nidx_tt, uniformly sample from uniqs
        # except for the value taken by y_train[i] or y_test[i], and then modify
        # the label of y_train[i] or y_test[i] with the sampled label. use
        # setdiff1d to get set difference of uniqs and label; we know values are
        # unique so set assume_unique = True to speed up calculation
        for i in nidx_tn:
            ds_copy.y_train[i] = choice(setdiff1d(uniqs, [ds_copy.y_train[i]],
                                                  assume_unique = True))
        for i in nidx_tt:
            ds_copy.y_test[i] = choice(setdiff1d(uniqs, [ds_copy.y_test[i]],
                                                 assume_unique = True))
        # done
    # else raise ValueError
    else:
        raise ValueError("{0}: unknown kind {1}. see docstring for supported "
                         "noise types".format(_fn, kind))
    # change attributes to reflect fraction of affected samples + noise type
    ds_copy.added_noise_fraction = fraction
    ds_copy.added_noise_type = kind
    # return
    return ds_copy

def model_from_spec(spec, _depth = 1):
    """
    given an appropriate JSON-formatted dict specifying an sklearn-like model,
    return a model instance with the specified hyperparameters.

    only works if the module the class is defined in is on the search path.

    parameters:

    spec    dict. format:

            {"module": str, "model": str, "params": {"param1": param1, ...}}

            params may also be an empty dict. note that if any of the values in
            the params dict are also dicts, then they will be treated as params
            dicts for a nested model (like for meta-estimators) and the function
            will call itself recursively to recover that inner model.
    _depth  do not use: internal parameter indicating recursion depth.
    """
    _fn = model_from_spec.__name__
    # if _depth exceeds 2, break infinite loop
    if _depth > 2:
        raise RuntimeError("{0}: exceeded maximum recursion depth (2)"
                           .format(_fn))
    if not isinstance(spec, dict):
        raise TypeError("{0}: spec must be properly formatted JSON-style dict"
                        .format(_fn))
    # check if all required keys are present
    if _MODELS_MODULE not in spec:
        raise KeyError("{0}: spec missing required key \"{1}\""
                       .format(_fn, _MODELS_MODULE))
    if _MODELS_MODEL not in spec:
        raise KeyError("{0}: spec missing required key \"{1}\""
                       .format(_fn, _MODELS_MODEL))
    if _MODELS_PARAMS not in spec:
        raise KeyError("{0}: spec missing required key \"{1}\""
                       .format(_fn, _MODELS_PARAMS))
    # get module name, model name, and params dict
    n_module = spec[_MODELS_MODULE]
    n_model, params = spec[_MODELS_MODEL], spec[_MODELS_PARAMS]
    # try to import the relevant module first
    try: _mdl = import_module(n_module)
    except ImportError as ie:
        raise ImportError("{0}: could not import module {1}"
                          .format(_fn, n_module)) from ie
    # check if module has the desired attribute; if not raise AttributeError
    if not hasattr(_mdl, n_model):
        raise AttributeError("{0}: module {1} does not have attribute {2}"
                             .format(_fn, n_module, n_model))
    # for each key, value pair in params, check if value is dict. if so, call
    # model_from_spec recursively (assume dict is valid format)
    for k, v in params.items():
        if isinstance(v, dict):
            params[k] = model_from_spec(v, _depth = _depth + 1)
    # instantiate model with hyperparamaters; lazy error catching
    est = getattr(_mdl, n_model)(**params)
    # delete the module reference after and return est
    del _mdl
    return est

if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
