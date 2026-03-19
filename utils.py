import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def preprocessing(red_file, white_file, output_file=None, test_size = 0.2, rd_state=42):
    """
    Args: 
    red_file: the name of data containing info of red wine
    white_file: the name of data containing info of white wine
    output_file: the name of file that you want to save the final concatenated data
    test_size: the ratio of test data
    rd_state: random seed for reproductibility
    output:
    X_train: -
    y_train: -
    X_test: -
    y_test -
    """
    final_data = []
    # read the data file
    for file in (red_file, white_file):
        with open(file, mode='r', newline='') as infile:
            reader = csv.reader(infile, delimiter=';')
            rows = list(reader)

        header = rows[0]
        header.insert(-1, 'red')

        is_red = 1 if file==red_file else 0

        for row in rows[1:]:
            row.insert(-1, is_red)

        # only include one header row
        if is_red == 1:
            final_data.extend(rows)
        else:
            final_data.extend(rows[1:])

    if output_file != None:
        with open(output_file, mode='', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=';')
            writer.writerows(final_data)

    # turn the list into a datafram
    headers = final_data[0]
    values = final_data[1:]
    data = pd.DataFrame(values, columns=headers)
    data = data.apply(pd.to_numeric)

    # Basic data preprocessing and summary
    summary = data.describe()
    print(summary)

    # Check for missing data
    print(data.isnull().sum())

    # Check for the share of red and white wine
    print(data['red'].value_counts())

    # Shuffule the data frac=1 means all shuffled, rd_state for reproductibility, reset index true to reassign the index
    data_shuffled = data.sample(frac=1, random_state=rd_state).reset_index(drop=True)

    # Train test split
    train_data, test_data = train_test_split(data_shuffled, test_size=test_size, random_state=rd_state)
    X_train = train_data.iloc[:,0:12]
    y_train = train_data.iloc[:,12]
    X_test = test_data.iloc[:,0:12]
    y_test = test_data.iloc[:,12]

    return X_train, y_train, X_test, y_test


def normalization(X_train, X_test, is_minmax = False):
    """
    Args: 
    is_minmax: True means using minmaxscaling, else use standardsclaer
    X_train: -
    X_test: -
    Output:
    X_train_normalized: normalized training set
    X_test_normalized: normalized test set
    scaler: for later use
    """
    if is_minmax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized, scaler

def hp_search_grid(alg_type, y_train):
    """
    Args:
    alg_type: "knn", "lb"-local bayesian, "ann", "bfr"-basis function regression
    y_train: -
    """
    n = len(y_train)
    if alg_type == 'knn':
        sqrt_n = int(np.sqrt(len(y_train)))
        k_values = np.concatenate([
        np.arange(1, sqrt_n, 2),
        np.logspace(np.log10(sqrt_n), np.log10(n//2), 20)
        ]).astype(int)
        k_values = np.unique(k_values)
        return {'k': k_values}
    
    elif alg_type == 'ann':
        return {'num_layers':[1,2,3], 
                'num_units':[32,64,128]}
    
    elif alg_type == 'bfr':
        return {'width':[], 
                'center':[], 
                'regularization':[]}
    
    elif alg_type == 'lb':
        sqrt_n = int(np.sqrt(len(y_train)))
        k_values = np.concatenate([
        np.arange(1, sqrt_n, 2),
        np.logspace(np.log10(sqrt_n), np.log10(n//2), 20)
        ]).astype(int)
        k_values = np.unique(k_values)
        return {'k': k_values, 
                'weights': ['uniform', 'distance'],
                'alpha_1': np.logspace(-6, 0, 4), # regularization for bias
                'lambda_1': np.logspace(-6, 0, 4)} # regularization for weights, assuming Gaussian prior
    else:
        print("you might key in the wrong alg name")
        raise KeyError









