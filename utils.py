import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    X = data_shuffled.drop('quality', axis=1)
    y = data_shuffled['quality']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=rd_state,
        stratify=X['red'],       # 0=white, 1=red
    )
    return X_train, y_train, X_test, y_test

def normalization(X_train, X_test, is_minmax=False):
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
    X_train_cont = X_train.iloc[:, :-1]
    X_train_cat = X_train.iloc[:, -1:].values
    
    X_test_cont = X_test.iloc[:, :-1]
    X_test_cat = X_test.iloc[:, -1:].values
    
    if is_minmax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_train_cont_norm = scaler.fit_transform(X_train_cont)
    X_test_cont_norm = scaler.transform(X_test_cont)
    
    X_train_normalized = np.hstack([X_train_cont_norm, X_train_cat])
    X_test_normalized = np.hstack([X_test_cont_norm, X_test_cat])

    return X_train_normalized, X_test_normalized, scaler

def soft_label(y_true, sigma=0.5, min_score=3, max_score=9):
    y_true = float(y_true)

    if sigma <= 0:
        return y_true

    scores = np.arange(min_score, max_score + 1, dtype=np.float64)
    weights = np.exp(-0.5 * ((scores - y_true) / sigma) ** 2)
    weights /= weights.sum()
    return float(np.dot(weights, scores))

def smooth_labels(y, sigma=0.5, min_score=3, max_score=9):
    if hasattr(y, "to_numpy"):
        y = y.to_numpy(dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)

    y = y.reshape(-1)

    if sigma <= 0:
        return y.reshape(-1, 1)

    y_smooth = np.array(
        [soft_label(label, sigma=sigma, min_score=min_score, max_score=max_score) for label in y],
        dtype=np.float64,
    )
    return y_smooth.reshape(-1, 1)

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
        """
            'hidden_widths': [8, 16, 32, 64],
            'architectures': [
                                [8],
                                [16],
                                [32],
                                [64],
                                [16, 8],
                                [32, 16],
                                [64, 16],
                                [64, 32],
                            ],
        """
        return {
            'hidden_widths': [16, 32, 64, 128],
            'architectures': [
                                [32],
                                [64],
                                [128],

                                [32, 16],
                                [64, 32],
                                [128, 64],
                                [128, 32], 
                                [64, 16],
                                [128, 16],

                                [128, 64, 32],
                                [64, 32, 16],  
                            ],
        }
    
    elif alg_type == 'bfr':
        return {
        'width':[0.001, 0.01, 0.1, 1, 10],
        'center':[5, 10, 25, 50, 100],
        'degree': [1,2,3,4],
        'regularization':[0.001, 0.01, 0.1, 1, 10]
        }
    
    elif alg_type == 'lb':
        min_k = 25 
        sqrt_n = int(np.sqrt(len(y_train)))
        
        k_values = np.concatenate([
            np.linspace(min_k, sqrt_n, 10),
            np.logspace(np.log10(sqrt_n), np.log10(len(y_train)//5), 10)
        ]).astype(int)
        k_values = np.unique(k_values)
        return {'k': k_values, 
                'weights': ['uniform', 'distance'],
                'alpha_1': np.logspace(-6, 0, 4), # regularization for bias
                'lambda_1': np.logspace(-6, 0, 4)} # regularization for weights, assuming Gaussian prior
    else:
        print("you might key in the wrong alg name")
        raise KeyError

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    hits = np.sum(np.abs(np.round(y_pred) - y_true) <= 1)
    acc_plus_minus_1 = hits / len(y_true)

    return mse, rmse, mae, r2, acc_plus_minus_1


def plot_k_metrics(test_predictions, metrics_type = 'mse'):
    """
    Plot metrics vs. k
    """
    ks = list(test_predictions.keys())
    total_metrics = [test_predictions[k]['metrics'][metrics_type] for k in ks]
    red_metrics = [test_predictions[k]['red_metrics'][metrics_type] for k in ks]
    white_metrics = [test_predictions[k]['white_metrics'][metrics_type] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, total_metrics, 'k-o', label=f'Total {metrics_type.upper()}', linewidth=2)
    plt.plot(ks, red_metrics, 'r--', label=f'Red Wine {metrics_type.upper()}')
    plt.plot(ks, white_metrics, 'b--', label=f'White Wine {metrics_type.upper()}')
    
    # notate the smallest/largest
    if metrics_type in ('mse','rmse','mae'):
        plt.axvline(x=ks[np.argmin(total_metrics)], color='black', alpha=0.8, linestyle=':')
        plt.axvline(x=ks[np.argmin(red_metrics)], color='r', alpha=0.8, linestyle=':')
        plt.axvline(x=ks[np.argmin(white_metrics)], color='b', alpha=0.8, linestyle=':')
    else:
        plt.axvline(x=ks[np.argmax(total_metrics)], color='black', alpha=0.8, linestyle=':')
        plt.axvline(x=ks[np.argmax(red_metrics)], color='r', alpha=0.8, linestyle=':')
        plt.axvline(x=ks[np.argmax(white_metrics)], color='b', alpha=0.8, linestyle=':')
    
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel(f'{metrics_type.upper()}')
    plt.title('Heterogeneity between Red and White Wine')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuals(y_true, y_pred, is_red):
    """
    Residual plot
    """
    residuals = y_true - y_pred
    df = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals,
        'Type': ['Red' if r else 'White' for r in is_red]
    })
    
    g = sns.JointGrid(data=df, x="Predicted", y="Residuals", hue="Type")
    g.plot_joint(sns.scatterplot, alpha=0.5)
    g.plot_marginals(sns.kdeplot, fill=True)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.show()
