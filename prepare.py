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

#############

def quality_bin_maker(df):
    '''
    This function creates one column with the quality split into bad, average and good
    Quality binned
    0 == bad
    1 == average
    2 == good
    Returns dataframe with quality_bins returned
    '''
    df['quality_bins'] = pd.cut(df.quality,bins = (0,4,6,10), labels=[0,1,2]).astype(int)
    
    wine_categories = {0: 'bad_wine', 1: 'avg_wine', 2 :'good_wine'}

    df['quality_bins_str'] = df.quality_bins.replace(wine_categories)

    return df

############# PREPARE FUNCTION

def prepare_wine_df(red_df, white_df):

    white_df, red_df = white_or_red(white_df, red_df)

    df = join_red_and_white(white_df, red_df)

    df = rename_columns(df)

    df = the_good_and_the_bad(df)

    df = quality_bin_maker(df)

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

#############


def my_scaler(train, validate, test, col_names, scaler, scaler_name):
    
    '''
    This function takes in the train validate and test dataframes, columns you want to scale (as a list), a scaler (i.e. MinMaxScaler(), with whatever paramaters you need),
    scaler_name as a string.
    col_names: list of columns to scale
    Scaler_name, should be what you want in the name of your new dataframe columns.
    Adds columns to the train validate and test dataframes. 
    Outputs scaler for doing inverse transforms.
    Ouputs a list of the new column names (what you can use to create the X_train).
    
    example: min_max_scaler, scaled_cols_list = my_scaler(train, validate, test, MinMaxScaler(), 'scaled_min_max')
    
    '''
    
    #create the scaler (input here should be minmax scaler)
    mm_scaler = scaler
    
    # make empty list for return
    scaled_cols_list = []
    
    # loop through columns in col names
    for col in col_names:
        
        #fit and transform to train, add to new column on train df
        train[f'{col}_{scaler_name}'] = mm_scaler.fit_transform(train[[col]]) 
        
        #df['col'].values.reshape(-1, 1)
        
        #transform cols from validate and test (only fit on train)
        validate[f'{col}_{scaler_name}']= mm_scaler.transform(validate[[col]])
        test[f'{col}_{scaler_name}']= mm_scaler.transform(test[[col]])
        
        #add new column name to the list that will get returned
        scaled_cols_list.append(f'{col}_{scaler_name}')
    
    #confirmation print
    print('Your scaled columns have been added to your train validate and test dataframes.')
    
    #returns scaler, and a list of column names that can be used in X_train, X_validate and X_test.
    return scaler, scaled_cols_list 
