# ~~~~~~~~~~~~ Prepare File for In Vino Veritas Project ~~~~~~~~~~~~~~~

# imports 
import pandas as pd
from sklearn.model_selection import train_test_split


##################

def white_or_red(white_df, red_df):
    '''
    This function takes in the red wine dataframe and the white wine dataframe
    Adds a column to each called 'is_white'
    Assigns 1 (true) to all rows in the white_df
    and 0 to all rows in the red_df
    returns both dataframes
    '''
    white_df['is_white'] = 1
    red_df['is_white'] = 0
    
    return white_df, red_df


################## 

def join_red_and_white(white_df, red_df):
    '''
    This function takes in the white df and the red df
    Concats them together
    Returns the full df
    '''
    
    full_df = pd.concat([red_df, white_df])

    return full_df

###############

def rename_columns(df):
    '''
    This function takes in a dataframe
    Replaces all ' ' (spaces) with '_'
    Returns the dataframe with the new names as col names
    '''
    # use list comp to replace with _
    # assign to columns of df
    df.columns = [col.replace(' ', '_') for col in list(df)]
    
    return df

#############

def the_good_and_the_bad(df):
    '''
    This function takes in the wine dataframe and adds 2 columns
    is_good and is_bad. Returns the df
    '''
    df['is_bad'] = (df.quality <=4).astype(int) # all wines that are 4 or lower are 'bad'
    df['is_good'] = (df.quality  >= 7).astype(int) # all wines that are 7 or higher are 'good'
    
    return df

############# Splitting Function

def banana_split(df):
    '''
    args: df
    This function take in the telco_churn data data acquired by aquire.py, get_telco_data(),
    performs a split.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=713)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=713)
    print(f'train --> {train.shape}')
    print(f'validate --> {validate.shape}')
    print(f'test --> {test.shape}')
    return train, validate, test