from utils import preprocessing as pp
from utils import normalization as nz
from utils import hp_search_grid as hp
from utils import calculate_metrics as cm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import BayesianRidge
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def local_prediction(i, inx_i, dist_i, normed_X_train, y_train, normed_X_test_i, weight, reg):
    """
    
    """
    # extract local data point
    X_local = normed_X_train[inx_i]
    y_local = y_train[inx_i]
    
    # calculate distance based average
    sw = 1 / (dist_i + 1e-6) if weight == 'distance' else None
    
    # fit the model and predict
    lbr = BayesianRidge(alpha_1=reg[0], lambda_1=reg[1])
    lbr.fit(X_local, y_local, sample_weight=sw)
    pred = lbr.predict(normed_X_test_i.reshape(1, -1))[0]
    final_lambda = lbr.lambda_
    final_alpha = lbr.alpha_
    
    # return
    return {
            "pred": pred,
            "lambda": final_lambda,
            "alpha": final_alpha
        }

# file name
red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file = 'winequality-modified.csv'

# preprocessing and train test split
X_train, y_train, X_test, y_test = pp(red_file, white_file)
normed_X_train, normed_X_test, _ = nz(X_train, X_test)
y_train_np = y_train.values
#X_train_np = normed_X_train if isinstance(normed_X_train, np.ndarray) else normed_X_train.values

# initialize hyperparam search grid
hps = hp('lb', y_train)
near_neigh = hps['k']
weight_type = hps['weights']
alpha = hps['alpha_1']
lbda = hps['lambda_1']
reg_set = []
for a in alpha:
    for l in lbda:
        reg_set.append([a,l])

# start training
# loop over k
test_predictions = {}
for k in near_neigh:
    best_mse_for_k = float('inf')
    best_record_for_k = None

    nn_engine = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean')
    nn_engine.fit(normed_X_train)
    dist, inx = nn_engine.kneighbors(normed_X_test, return_distance=True)
    for weight in weight_type:
        for reg in reg_set:
            print(f"Running: k={k}, weight={weight}, reg={reg}")
            results = Parallel(n_jobs=-1)(
                delayed(local_prediction)(
                    i, inx[i], dist[i], normed_X_train, y_train_np, normed_X_test[i], weight, reg
                ) 
                for i in range(len(y_test))
            )
            test_prediction = [r['pred'] for r in results]
            lambdas = [r['lambda'] for r in results]
            alphas = [r['alpha'] for r in results]

            mse, rmse, mae, r2, acc1 = cm(y_test, test_prediction)

            if mse < best_mse_for_k:
                best_mse_for_k = mse
                best_record_for_k = {
                    "best_weight": weight,
                    "best_reg": reg,
                    "metrics": {
                        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "acc1": acc1
                    },
                    "avg_posterior": {
                        "lambda": np.mean(lambdas),
                        "alpha": np.mean(alphas)
                    },
                    "y_pred": test_prediction
                }

            test_predictions[k] = best_record_for_k


# Find the best K, and other necessary params
result_df = pd.Dataframe(test_predictions)



