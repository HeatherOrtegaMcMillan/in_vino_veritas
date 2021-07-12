#~~~~~~~~~~ Explore module for In Vino Veritas Project ~~~~~~~~~~~~~~~~

# import modules
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

################ Plot Distributions and Box Plots
def explore_univariate(df, figsize = (18,3)):
    '''
    This function is for exploring. Takes in a dataframe with variables 
    you would like to see the box plot of.
    Input the dataframe (either fully, or using .drop) 
    with ONLY the columns you want to see plotted.
    Optional argument figsize. Default it's small.    
    '''

    for col in list(df):
        plt.figure(figsize=figsize)
        plt.subplot(121)
        sns.boxplot(x = col, data = df)
        plt.title(f'Box Plot of {col}')

        plt.subplot(122)
        sns.histplot(data = df, x = col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

############## 


def plot_against_target(df, target, var_list, figsize = (10,5), hue = None):
    '''
    Takes in dataframe, target and varialbe list, and plots against target. 
    '''
    for var in var_list:
        plt.figure(figsize = (figsize))
        sns.regplot(data = df, x = var, y = target, 
                    line_kws={'color': 'orange'})
        plt.show()

############
def eval_ttest_1samp(t, p, alpha = 0.05):
    '''
    This function evaluates ttests, Takes a t a p and the alpha is 
    automatically set to 0.05
    '''
    if p < alpha:
        print("We can reject our null hypothesis")
    if p > alpha:
        print("We cannot reject our null hypothesis")
    if t > 0:
        print('The T is positive so the sample is greater than the population mean')
    if t < 0:
        print('The T is negative so the sample is less than the population mean')

#############

def eval_stats_test(p, alpha = 0.05):
    '''
    This function is a generic stats evaluation function
    Prints out if we can reject the null hypothesis or not
    '''
    if p < alpha:
        print("We can reject our null hypothesis")
    if p > alpha:
        print("We cannot reject our null hypothesis")