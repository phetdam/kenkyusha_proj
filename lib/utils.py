# contains general utility functions/classes
#
# Changelog:
#
# 03-30-2020
#
# moved to lib directory so that we can refer to module as part of a package.
# added the convenience function DataFrame2Data_Set to make it easier to convert
# from a DataFrame to a Data_Set, as DataFrames are common data structures.
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

from numpy import array, hstack, ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from textwrap import fill

__doc__ = """
contains general utility functions/classes for loading/converting data, etc.
"""

_MODULE_NAME = "utils"

class Data_Set:
    """
    data set wrapper class. recommended to construct from pandas.DataFrame or
    at the bare minimum numpy.array as .shape is accessed. allows user to
    specify what fraction of the data should be the holdout data.

    built on top of sklearn.model_selection.train_test_split.

    example:

    from numpy import array
    from utils import Data_Set
    # from pandas DataFrame df
    data1 = Data_Set(df.iloc[:, :-1].values, df.iloc[:, -1].values, 
                     array(df.columns[:-1]), "response", "df_data", 
                     test_size = 0.3, random_state = 7)
    # from numpy arrays X, y, and a list of labels
    data2 = Data_Set(X, y, array(["lab1", "lab2", "lab3"]), random_state = 5)
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
                    "response = {4}, test_size = {5}, random_state = {6}, "
                    "features = {7})"
                    .format(Data_Set.__name__, self.name, self.n_samples,
                            self.n_features, self.response, self.test_size,
                            self.random_state, list(self.features)), width = 80,
                    subsequent_indent = (len(Data_Set.__name__) + 1) * " ")

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
    if not isinstance(random_state, int):
        raise TypeError("{0}: random_state: int expected, {1} received"
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

    parameters:

    ds              Data_Set
    fraction        optional float in (0, 1) indicating what fraction of the 
                    training and test sets should be affected, default 0.2
    kind            optional str, the kind of noise added to ds, default "label"
    random_state    optional int to seed the PRNG, default None

    below is a description of the noise types supported by the "kind" kwarg.

    label    given an affected sample with label k out of m many labels,
             randomly select one of the m - 1 labels to assign to the sample.
    """
    pass
        

if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
