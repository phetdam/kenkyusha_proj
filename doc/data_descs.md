# Data set descriptions

_last updated: 05-22-2020_  
_file created: 04-07-2020_

Brief descriptions and OpenML links to the 16 data sets we used to evaluate average model performance. Minimal cleaning was performed only when necessary to binarize categorical features and remove any missing values.

## breast-w

The Breast Cancer Wisconsin (Original) data set. 699 data points, 9 features, and 2 target classes. However, after cleaning out input rows with missing values encoded as '?', number of remaining data points dropped to 683. OpenML link [here](https://www.openml.org/d/15).

## diabetes

The Pima Indians diabetes data set. 768 data points, 8 features, and 2 target classes. OpenML link [here](https://www.openml.org/d/37).

## iris

The classic iris data set. 150 data points, 4 features, and 3 target classes. OpenML link [here](https://www.openml.org/d/61).

## isolet

The Isolated Letter Speech Recognition data set. 7797 data points, 617 features, and 26 target classes. OpenML link [here](https://www.openml.org/d/300).

## MiceProtein

Expression levels for 77 types of mice proteins measured in the cerebral cortex of 8 classes of mice. 1080 data points, 81 features, and 8 target classes. However, after removing rows with '?' entries and dropping ignored features, number of data points dropped to 552, and the number of features dropped to 77. OpenML link [here](https://www.openml.org/d/40966). 

## mushroom

Descriptions of hypothetical samples of 23 different species of Agaricus and Lepiota mushrooms. 8124 data points, 22 features, and 2 target classes. However, after binarizing the features, which were all categorical, the number of features increased to a total of 118 one-hot features. Note that the `stalk-root` feature, which was the only feature that contained missing values, had the presence of missing values coded as another one-hot feature. OpenML link [here](https://www.openml.org/d/24).

## nursery

Data set derived from a hierarchical decision model used for ranking nursery school applications in the 1980s when there was excessive nursery school enrollment in Ljubljana, Slovenia. 12960 data points, 8 features, and 5 target classes. However, after binarizing the features, which were all categorical, the number of features increased to 27 one-hot features. OpenML link [here](https://www.openml.org/d/26).

## optdigits

Handwritten digits for optical recognition, from UCI. 5620 data points, 64 features, 10 target classes. OpenML link [here](https://www.openml.org/d/28).

## PhishingWebsites

Data set of features used for predicting whether a website is a phishing website or not, from UCI. 11055 data points, 30 features, and 2 target classes. However, after binarizing the features, which were all categorical, the number of features increased to 68 one-hot features. Information about the original features can be found in the doc [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Phishing%20Websites%20Features.docx). OpenML link [here](https://www.openml.org/d/4534).

## semeion

The Semeion Handwritten Digit Data Set. 1593 data points, 256 features, 10 target classes. OpenML link [here](https://www.openml.org/d/1501).

## spambase

Spam email data set, from UCI. 4601 data points, 57 features, 2 target classes. OpenML link [here](https://www.openml.org/d/44).

## splice

Primate splice-junction gene sequences. 3190 data points, 60 features, 3 target classes. However, after binarizing the features, which were all categorical, the number of features increased to 287 one-hot features. OpenML link [here](https://www.openml.org/d/46).

## steel-plates-fault

Data set of steel plate faults. 1941 data points, 33 features, 2 target classes. OpenML link [here](https://www.openml.org/d/1504).

## vertebra-column

Two-class verterbra column classification. 310 data points, 6 features, 2 target classes. OpenML link [here](https://www.openml.org/d/1524).

## wdbc

The Breast Cancer Wisconsin (Diagnostic) data set. 569 data points, 30 features, 2 target classes. OpenML link [here](https://www.openml.org/d/1510).

## yeast

Data set for predicting cellular localization sites, presumably of yeast-like bacteria as implied by the name, from UCI. Further information about the features and links to relvevant papers can be found at its UCI Machine Learning Repository page [here](http://archive.ics.uci.edu/ml/datasets/Yeast). 1484 data points, 8 features, 2 target classes. OpenML link [here](https://www.openml.org/d/181).

