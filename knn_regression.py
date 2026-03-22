from utils import preprocessing as pp
from utils import normalization as nz
from utils import hp_search_grid as hp
from utils import calculate_metrics as cm
from utils import plot_k_metrics as pk
from utils import plot_residuals as pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def knn_predict(normed_X_train, y_train_np, normed_X_test, k, weight):
    """
    """
    n_test = normed_X_test.shape[0]
    y_pred = np.empty(n_test)
    is_red = normed_X_test[:, -1].copy()
    for i in range(n_test):
        diff = normed_X_train - normed_X_test[i] 
        dists = np.sqrt((diff ** 2).sum(axis=1))
        nn_idx = np.argpartition(dists, k)[:k] 
        nn_dists = dists[nn_idx]
        nn_labels = y_train_np[nn_idx]

        if weight == 'distance':
            w = 1.0 / (nn_dists + 1e-8)
            y_pred[i] = np.dot(w, nn_labels) / w.sum()
        else:
            y_pred[i] = nn_labels.mean()

    return y_pred, is_red

red_file   = 'winequality-red.csv'
white_file = 'winequality-white.csv'

X_train, y_train, X_test, y_test = pp(red_file, white_file)
normed_X_train, normed_X_test, _ = nz(X_train, X_test)
 
y_train_np = y_train.values
y_test_np  = y_test.values

hps         = hp('knn', y_train)
near_neigh  = hps['k']
weight_type = ['uniform', 'distance'] 

test_predictions = {}

for k in near_neigh:
    best_mse_for_k    = float('inf')
    best_record_for_k = None

    for weight in weight_type:
        print(f"Running: k={k}, weight={weight}")

        y_pred, is_red = knn_predict(normed_X_train, y_train_np, normed_X_test, k=k, weight=weight)
        mse, rmse, mae, r2, acc1 = cm(y_test_np, y_pred)
        red_mask   = is_red.astype(bool)
        white_mask = ~red_mask

        y_pred_red   = y_pred[red_mask];   y_test_red   = y_test_np[red_mask]
        y_pred_white = y_pred[white_mask]; y_test_white = y_test_np[white_mask]

        mse_red,   rmse_red,   mae_red,   r2_red,   acc1_red   = cm(y_test_red,   y_pred_red)
        mse_white, rmse_white, mae_white, r2_white, acc1_white = cm(y_test_white, y_pred_white)

        if mse < best_mse_for_k:
            best_mse_for_k = mse
            best_record_for_k = {
                "best_weight": weight,
                "metrics": {
                    "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "acc1": acc1
                },
                "red_metrics": {
                    "mse": mse_red, "rmse": rmse_red, "mae": mae_red,
                    "r2": r2_red,   "acc1": acc1_red
                },
                "white_metrics": {
                    "mse": mse_white, "rmse": rmse_white, "mae": mae_white,
                    "r2": r2_white,   "acc1": acc1_white
                },
                "y_pred":  y_pred.tolist(),
                "is_red":  is_red.tolist(),
            }
    test_predictions[k] = best_record_for_k
    rec = best_record_for_k
    print(
        f"  → best weight={rec['best_weight']}  "
        f"MSE={rec['metrics']['mse']:.4f}  "
        f"R²={rec['metrics']['r2']:.4f}  "
        f"Acc±1={rec['metrics']['acc1']:.4f}"
    )

rows = []
for k, rec in test_predictions.items():
    row = {"k": k, "best_weight": rec["best_weight"]}
    row.update({f"overall_{m}": v for m, v in rec["metrics"].items()})
    row.update({f"red_{m}":     v for m, v in rec["red_metrics"].items()})
    row.update({f"white_{m}":   v for m, v in rec["white_metrics"].items()})
    rows.append(row)
 
result_df = pd.DataFrame(rows)
result_df.to_csv("knn_regression.csv", index=False, encoding="utf-8-sig")
print("\nResults saved to knn_regression.csv")

for metric in ("mse", "rmse", "mae", "r2", "acc1"):
    pk(test_predictions, metrics_type=metric)

best_k = min(test_predictions, key=lambda k: test_predictions[k]["metrics"]["mse"])
best_rec = test_predictions[best_k]
print(f"\nBest k={best_k}  weight={best_rec['best_weight']}")
print(f"  MSE={best_rec['metrics']['mse']:.4f}  "
      f"RMSE={best_rec['metrics']['rmse']:.4f}  "
      f"MAE={best_rec['metrics']['mae']:.4f}  "
      f"R²={best_rec['metrics']['r2']:.4f}  "
      f"Acc±1={best_rec['metrics']['acc1']:.4f}")
 
pr(
    y_true=y_test_np,
    y_pred=np.array(best_rec["y_pred"]),
    is_red=np.array(best_rec["is_red"])
)

y_pred_best = np.array(best_rec["y_pred"])
is_red_best = np.array(best_rec["is_red"]).astype(bool)
 
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test_np[~is_red_best], y_pred_best[~is_red_best],
           alpha=0.35, color="steelblue", label="White", s=12)
ax.scatter(y_test_np[is_red_best],  y_pred_best[is_red_best],
           alpha=0.35, color="firebrick", label="Red", s=12)
lims = [y_test_np.min() - 0.2, y_test_np.max() + 0.2]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("True Quality")
ax.set_ylabel("Predicted Quality")
ax.set_title(f"KNN Regression — Predicted vs Actual\n(k={best_k}, weight={best_rec['best_weight']})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
