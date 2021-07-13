############ Modeling Functions for In Vino Veritas ##################
# This module has Functions pertaining to feature selection, creating models,
# and evaluating models

import pandas as pd 
import numpy as np 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt 
import seaborn as sns


###############
def select_kbest(X, y, k, score_func=f_classif):
    '''
    takes in the predictors (X), the target (y), and the number of features to select (k) 
    and returns the names (in a list) of the top k selected features based on the SelectKBest class
    Optional arg: score_func. Default is f_regression. other options ex: f_classif 
    '''
    # create selector
    f_selector = SelectKBest(score_func=score_func, k=k)
    
    #fit to X and y
    f_selector.fit(X, y)
    
    # return the list of the column names that are the top k selected features
    return list(X.columns[f_selector.get_support()])

###############
def all_aboard_the_X_train(X_cols, y_col, train, validate, test):
    '''
    X_cols = list of column names you want as your features
    y_col = string that is the name of your target column
    train = the name of your train dataframe
    validate = the name of your validate dataframe
    test = the name of your test dataframe
    outputs X_train and y_train, X_validate and y_validate, and X_test and y_test
    6 variables come out! So have that ready
    '''
    
    # do the capital X lowercase y thing for train test and split
    # X is the data frame of the features, y is a series of the target
    X_train, y_train = train[X_cols], train[y_col]
    X_validate, y_validate = validate[X_cols], validate[y_col]
    X_test, y_test = test[X_cols], test[y_col]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

##############

def print_metrics(model, y, pred, set_name = 'This Set'):
    
    print(model)
    print(f"~~~~~{set_name} Scores~~~~~~")
    print(classification_report(y, pred))
    print(confusion_matrix(y, pred))
    print()

########## Credit https://github.com/DTrimarchi10/confusion_matrix
########## A confusion matrix function from Dennis Trimarch
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
