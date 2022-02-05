.. config_format.rst

   last updated: 2022-2-04
   file created: 2020-04-28

Formatting config files
=======================

A brief guide to interpreting and formatting the JSON evaluation config files
for ``noisyeval.py``.  

LaTeX rendered [#]_ using Alexander Rodin's hacky workaround, detailed here__.

.. __: https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b

.. [#] Actually, LaTeX rendering was removed from this document, as the hack
   used does not allow the LaTeX to render properly when using GitHub in dark
   mode. LaTeX rendering in reStructuredText files on GitHub is the topic of a
   few popular issues for the ``github/markup`` repo, i.e. 83__, 274__, 897__.
   Whether or not LaTeX rendering will be supported is still up in the air.

.. __: https://github.com/github/markup/issues/83

.. __: https://github.com/github/markup/issues/274

.. __: https://github.com/github/markup/issues/897

Overview
--------

The config files used by ``noisy_eval.py`` are standard JSON files with
specific formatting conventions that are specify different config parameters
to ``noisy_eval.py``, for example where to write results, the global random
seed to use, what kinds of noise to add, what quantities to display to
``stdout``, etc. Each config file consists of a single JSON object containing
other JSON objects, arrays, and strings. There are several keys associated with
the main JSON object which are required to be present; we detail their required
values in the section below.

Required keys
-------------

``data_dir``
~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` where the data files for model training are,
all of which will be used. Must be assigned a string value interpretable as a
valid directory name containing only CSV files that can be read by
``pandas.read_csv``. Note that if there are non-CSV files in the directory,
``noisy_eval.py`` will ignore these files and print a warning to ``stderr``.

**Example:**

.. code:: json

   "data_dir": "./data/csv_clean",

``results_dir``
~~~~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` where to write computation results, including
the final model result ``pickle`` and any associated RLA/ELA plots. See also
`ela_fig, rla_fig`_. Must be assigned a string value interpretable as a valid
directory name.

**Example:**

.. code:: json

   "results_dir": "./results",

``test_fraction``
~~~~~~~~~~~~~~~~~

Indicates to ``noisy_eval.py``, for each data set in the directory specified
by `data_dir`_, what fraction of the data to use as the validation data that
metrics will be computed on. Must be assigned a float in the range of (0, 1).
See also `random_state`_.

**Example:**

.. code:: json

   "test_fraction": 0.2,

``random_state``
~~~~~~~~~~~~~~~~

Controls the global random seed used by ``noisy_eval.py`` during the training
process, controlling both the training/validation data split, random seed for
any model that has stochastic fitting behavior, and the way that noise applied
to a data set is generated. Note that the underlying PRNG is the NumPy PRNG, so
the behavior is not thread-safe.

Must be assigned a nonnegative integer, which is directly passed to
``numpy.random.seed``. See also `noise_kinds`_, the following section.

**Example:**

.. code:: json

   "random_state": 7,

``noise_kinds``
~~~~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` what kinds of noise to add to each copy of each
data set specified by `data_dir`_. If ``k`` types of noise are specified, then
for each noise level, ``k`` different noisy data sets copies will be made. Must
be assigned an array, where each element of the array is a valid string
corresponding to a type of noise to introduce to the data. So far, only
``"label"`` is supported. See also `noise_levels`_.

**Example:**

.. code:: json

   "noise_kinds": ["label"],

``noise_levels``
~~~~~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` the noise level to assign for each of the noisy
copies specified by `noise_kinds`_ made for each data set specified by
``data_dir``. Must be assigned an array, where each element of the array is a
float in (0, 1). See also `noise_kinds`_ above.  

**Example:**

.. code:: json

   "noise_levels": [0.1, 0.2, 0.3, 0.6, 0.9],

``disp_accs``
~~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` that after computation, accuracy matrices for
each model should be printed to ``stdout``. The matrices are indexed by data
set name along the rows and by noise level along the columns, and if there are
``k`` noise kinds specified in ``noise_kinds``, then there will be ``k``
accuracy matrices for each model. Must be assigned either 0 for false, 1 for
true.

**Example:**

.. code:: json

   "disp_accs": 0,

``disp_elas``, ``disp_rlas``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Play similar roles to `disp_accs`_, except the matrices are of ELA and RLA,
indexed in the same way as described above in `disp_accs`_. See the paper
here__ for a description of what the two metrics are. Both keys must be
assigned either 0 for false, 1 for true.  

.. __: https://doi.org/10.1016/j.neucom.2014.11.086

**Example:**

.. code:: json

   "disp_elas": 0,
   "disp_rlas": 0,

``disp_avg_elas``, ``disp_avg_rlas``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Play similar roles to `disp_elas, disp_rlas`_ except with respect to whether or
not row vectors of data set macro averages of ELA and RLA will be sent to
``stdout``. Must be assigned either 0 for false, 1 for true.

**Example:**

.. code:: json

   "disp_avg_elas": 1,
   "disp_avg_rlas": 1,

``ela_fig``, ``rla_fig``
~~~~~~~~~~~~~~~~~~~~~~~~

Indicates options to be used when painting comparison plots of per-model
average ELA/RLA versus noise level across data sets. Must be assigned a JSON
object that contains several required keys which are described below.

``save_fig``
   Indicates to ``noisy_eval.py`` whether or not the figure should be painted
   and saved. Must be assigned either 1 for true to paint and save the image as
   a PNG file, or 0 to not produce the image upon completion.

``fig_size``
   Specifies the size in inches of the figure if ``save_fig`` has value 1. Must
   be an array of two positive integers.

``fig_dpi``
   Specifies the DPI of the figure if ``save_fig`` has value 1. Must be a
   positive integer. The typical value is 100.

``fig_title``
   Specifies the figure's title if ``save_fig`` has value 1. Must be a string.
   LaTeX can be included in the string between dollar signs, i.e. ``$``, as
   long as ``\`` is properly escaped, i.e. with ``\\``. Otherwise, your LaTeX
   won't render properly.

``fig_cmap``
   Specifies the color map used to paint the lines in the figure if ``save_fig``
   has value 1. Must be a string that is a valid color map from
   ``matplotlib.cm``. A good standard color map choice is ``"viridis"``.

``plot_kwargs``
   Specifies per-line keyword args to pass to ``matplotlib.axes.Axes.plot`` if
   ``save_fig`` has value 1. Must be an array, either empty if no keyword args
   are needed, or containing the same number of JSON objects as the number of
   models present in the config file. See `models`_ for details on specifying
   models in a config file. If no keyword arguments are to be specified for a
   line/model, an empty JSON object can be used, else write valid key/value
   pairs in the JSON object that are interpretable by
   ``matplotlib.axes.Axes.plot``.

**Example:**

.. code:: json

   "ela_fig": {
       "save_fig": 1,
       "fig_size": [6, 5],
       "fig_dpi": 100,
       "fig_title": "Average ELA with 50 trees, max_depth=6",
       "fig_cmap": "viridis",
       "plot_kwargs": [{}, {}, {"marker": "s", "markersize": 5}]
   },

``warm_start``
~~~~~~~~~~~~~~

Indicates to ``noisy_eval.py`` whether to perform a warm start or not. Given a
configuration file ``foo.json``, if the ``pickle`` file ``foo.pickle`` exists
in the results directory specified by ``results_dir``, warm starting is defined
as reusing the results of ``foo.pickle`` when painting the plots with the
options specified in `ela_fig, rla_fig`_. The benefit of warm starting is that
after computing results, one can modify plotting options in `ela_fig, rla_fig`_
to change plot aesthetics without having to recompute all the results again.
Must be assigned either 1 to warm start, 0 to always cold start. It is
recommended to set `warm_start`_ to 1 and simply delete the old ``pickle`` file
if new results need to be computed.

**Example:**

.. code:: json

   "warm_start": 1,

``models``
~~~~~~~~~~

Specifies the models to evaluate on the data sets specified by `data_dir`_
over the noise kinds and levels specified by `noise_kinds`_ and
`noise_levels`_. Must be an array of objects, where each object contains
several required keys as specified below.

``name``
   A name to uniquely identify the model, which will also be the legend label
   assigned to the line plotted in the average ELA/RLA comparison figures, if
   they are to be saved. Must be assigned a string. Like with ``fig_title`` in
   `ela_fig, rla_fig`_, LaTeX can be included in the string between ``$`` so
   long as ``\`` is properly escaped as ``\\`` to preserve LaTeX commands.

``module``
   Specifies the Python module the model class belongs to. Must be assigned a
   string.

``model``
   Specifies the class name of the desired model, excluding the module name.
   Note that only classes that implement an ``sklearn``-like interface can be
   used with ``noisy_eval.py``, as the computation methods assume that every
   model implements the instance methoeds ``fit``, ``score``, and ``predict``.
   Must be assigned a string as a value.

``params``
  Specifies any hyperparameters used for creating an ``sklearn``-like model
  class instance. Note that in the case that a hyperparameter is another
  ``sklearn``-like model instance, one can specify this case by assigning to a
  string key an object with keys ``module``, ``model``, and ``params``.
  ``noisy_eval.py`` will be able to interpret this JSON object as a request for
  a model instance as a hyperparameter. Must be assigned a JSON object, where
  each key/value pair corresponds to a named hyperparameter and its associated
  value. Please see the example below.

See also `data_dir`_, `noise_kinds`_, `noise_levels`_ for configuration of data
files, noise types, and noise levels.

**Example:**

Note the syntax used for the ``AdaBoostClassifier``. The ``base_estimator``
hyperparameter requires a ``sklearn`` model instance, which cannot be stored
in a JSON file. However, since ``base_estimator`` has been assigned a JSON
object, ``noisy_eval.py`` knows to interprets the JSON object as specification
for an ``sklearn``-like class instance. In this case, an instance of the
``DecisionTreeClassifier`` from ``sklearn.tree`` with the specified values for
``criterion``, ``max_depth``, and ``random_state`` will be created and passed
to ``base_estimator`` upon creation of an ``AdaBoostClassifier``.

.. code:: json

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
       }
   ]

A full example
--------------

The following JSON object, when placed into a JSON file, is a valid
configuration file. You may wish to copy and paste the example below and edit
the fields as necessary to facilitate the writing of your own config files.

.. code:: json

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
           "plot_kwargs": [
               {},
               {},
               {
                   "marker": "s",
                   "markersize": 5
               }
           ]
       },
       "rla_fig": {
           "save_fig": 1,
           "fig_size": [6, 5],
           "fig_dpi": 150,
           "fig_title": "Average RLA with 50 trees, max_depth=6",
           "fig_cmap": "viridis_r",
           "plot_kwargs": [
               {},
               {},
               {
                   "marker": "s",
                   "markersize": 5
               }
           ]
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