from utils import preprocessing as pp
from utils import normalization as nz
from utils import hp_search_grid as hp
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import Parallel, delayed
import numpy as np

def local_prediction(i, inx_i, dist_i, normed_X_train, y_train, normed_X_test_i, weight, reg):
    """
    针对第 i 个测试点进行局部回归预测
    """
    # 提取局部数据 (确保从训练集取！)
    X_local = normed_X_train[inx_i]
    y_local = y_train[inx_i]
    
    # 计算权重
    sw = 1 / (dist_i + 1e-6) if weight == 'distance' else None
    
    # 拟合局部模型
    lbr = BayesianRidge(alpha_1=reg[0], lambda_1=reg[1])
    lbr.fit(X_local, y_local, sample_weight=sw)
    
    # 预测并返回标量
    return lbr.predict(normed_X_test_i.reshape(1, -1))[0]

# file name
red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file = 'winequality-modified.csv'

# preprocessing and train test split
X_train, y_train, X_test, y_test = pp(red_file, white_file)
normed_X_train, normed_X_test, _ = nz(X_train, X_test)
y_train_np = y_train.values
X_train_np = normed_X_train if isinstance(normed_X_train, np.ndarray) else normed_X_train.values

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
    nn_engine = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean')
    nn_engine.fit(normed_X_train)
    dist, inx = nn_engine.kneighbors(normed_X_test, return_distance=True)
    for weight in weight_type:
        for reg in reg_set:
            print(f"Running: k={k}, weight={weight}, reg={reg}")
            test_prediction = Parallel(n_jobs=-1)(
                delayed(local_prediction)(
                    i, inx[i], dist[i], X_train_np, y_train_np, normed_X_test[i], weight, reg
                ) 
                for i in range(len(y_test))
            )

            mse = mean_squared_error(y_test, test_prediction)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, test_prediction)
            r2 = r2_score(y_test, test_prediction)
            hits = np.sum(np.abs(np.round(test_prediction) - y_test) <= 1)
            acc_plus_minus_1 = hits / len(y_test)
            test_predictions[(k,weight,tuple(reg))] = {"test prediciotn": test_prediction, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "acc1": acc_plus_minus_1}

            


