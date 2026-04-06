from utils import preprocessing as pp
from utils import normalization as nz
from utils import hp_search_grid as hp
from utils import calculate_metrics as cm
from utils import plot_k_metrics as pk
from utils import plot_residuals as pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def knn_predict(normed_X_train, y_train_np, normed_X_test, k, weight):
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

def run_cv_k_weight(normed_X_all, y_all, k, weight, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    fold_red_results = []
    fold_white_results = []

    for train_idx, val_idx in kf.split(normed_X_all):
        X_tr, X_val = normed_X_all[train_idx], normed_X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        y_pred, is_red = knn_predict(X_tr, y_tr, X_val, k=k, weight=weight)

        fold_results.append(cm(y_val, y_pred))

        red_mask   = is_red.astype(bool)
        white_mask = ~red_mask

        if red_mask.sum() > 0:
            fold_red_results.append(cm(y_val[red_mask],   y_pred[red_mask]))
        if white_mask.sum() > 0:
            fold_white_results.append(cm(y_val[white_mask], y_pred[white_mask]))

    def avg(rows):
        arr = np.array(rows)
        return dict(zip(("mse", "rmse", "mae", "r2", "acc1"), arr.mean(axis=0)))
    return avg(fold_results), avg(fold_red_results), avg(fold_white_results)

def cross_validate(normed_X_all, y_all, near_neigh, weight_type, n_splits=5, random_state=42):
    cv_preds = {}
    for k in near_neigh:
        best_mse_for_k    = float('inf')
        best_record_for_k = None
 
        for weight in weight_type:
            print(f"  [CV] k={k}, weight={weight}")
 
            avg_m, avg_red, avg_white = run_cv_k_weight( normed_X_all, y_all, k=k, weight=weight, n_splits=n_splits, random_state=random_state)
 
            if avg_m["mse"] < best_mse_for_k:
                best_mse_for_k    = avg_m["mse"]
                best_record_for_k = {
                    "best_weight":   weight,
                    "metrics":       avg_m,
                    "red_metrics":   avg_red,
                    "white_metrics": avg_white,
                }
 
        cv_preds[k] = best_record_for_k
        rec = best_record_for_k
        print(
            f"  → CV best weight={rec['best_weight']}  "
            f"MSE={rec['metrics']['mse']:.4f}  "
            f"R²={rec['metrics']['r2']:.4f}  "
            f"Acc±1={rec['metrics']['acc1']:.4f}"
        )
 
    return cv_preds


METRICS = ("mse", "rmse", "mae", "r2", "acc1")
METRIC_FMT = {
    "mse":  ".4f",
    "rmse": ".4f",
    "mae":  ".4f",
    "r2":   ".4f",
    "acc1": ".4f",
}

def _fmt(val, metric):
    return format(val, METRIC_FMT[metric])

def print_best_record(label, k, rec, stratify_label):
    print(f"\n{'─'*60}")
    print(f"  {label} [{stratify_label}]   k={k}   weight={rec['best_weight']}")
    print(f"{'─'*60}")
    header = f"  {'Metric':<8}" + f"{'Overall':>10}" + f"{'Red':>10}" + f"{'White':>10}"
    print(header)
    print(f"  {'':-<8}{'':->10}{'':->10}{'':->10}")
    for m in METRICS:
        ov  = _fmt(rec['metrics'][m], m)
        red = _fmt(rec['red_metrics'][m], m)
        wht = _fmt(rec['white_metrics'][m], m)
        print(f"  {m.upper():<8}{ov:>10}{red:>10}{wht:>10}")


def print_comparison_table(best_k_cv, cv_rec, best_k_test, test_rec, stratify_label):
    print(f"\n{'═'*70}")
    print(f"  KNN REGRESSION [{stratify_label}] — CV vs Standard Test Comparison")
    print(f"{'═'*70}")
    print(
        f"  {'Metric':<8}"
        f"{'CV Overall':>14}{'CV Red':>10}{'CV White':>10}"
        f"  |"
        f"{'Test Overall':>14}{'Test Red':>10}{'Test White':>10}"
    )
    print(
        f"  {'(k=' + str(best_k_cv) + ', ' + cv_rec['best_weight'] + ')':<8}"
        f"{'':>14}{'':>10}{'':>10}"
        f"  |"
        f"{'(k=' + str(best_k_test) + ', ' + test_rec['best_weight'] + ')':<14}"
        f"{'':>10}{'':>10}{'':>10}"
    )
    print(f"  {'':-<8}{'':->14}{'':->10}{'':->10}  |{'':->14}{'':->10}{'':->10}")
    for m in METRICS:
        cv_ov  = _fmt(cv_rec['metrics'][m], m)
        cv_red = _fmt(cv_rec['red_metrics'][m], m)
        cv_wht = _fmt(cv_rec['white_metrics'][m], m)
        ts_ov  = _fmt(test_rec['metrics'][m], m)
        ts_red = _fmt(test_rec['red_metrics'][m], m)
        ts_wht = _fmt(test_rec['white_metrics'][m],m)
        print(
            f"  {m.upper():<8}"
            f"{cv_ov:>14}{cv_red:>10}{cv_wht:>10}"
            f"  |"
            f"{ts_ov:>14}{ts_red:>10}{ts_wht:>10}"
        )
    print(f"{'═'*70}\n")

red_file   = 'winequality-red.csv'
white_file = 'winequality-white.csv'
for stratify in [True, False]:
    stratify_label = "stratified" if stratify else "non_stratified"
    print("\n" + "-" * 60)
    print(f"  RUN: {stratify_label.upper()}")
    print("-" * 60)

    X_train, y_train, X_test, y_test = pp(red_file, white_file, stratify=stratify)
    normed_X_train, normed_X_test, _ = nz(X_train, X_test)
 
    y_train_np = y_train.values
    y_test_np  = y_test.values

    hps         = hp('knn', y_train)
    near_neigh  = hps['k']
    weight_type = ['uniform', 'distance']

    print("\n" + "=" * 60)
    print(f"  STANDARD TEST EVALUATION [{stratify_label}]")
    print("=" * 60)

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
    result_df.to_csv(f"knn_regression_{stratify_label}.csv", index=False, encoding="utf-8-sig")
    print(f"\nResults saved to knn_regression_{stratify_label}.csv")


    best_k_test = min(test_predictions, key=lambda k: test_predictions[k]["metrics"]["mse"])
    best_rec_test = test_predictions[best_k_test]
    print_best_record("STANDARD TEST", best_k_test, best_rec_test, stratify_label)


    print("\n" + "=" * 60)
    print(f"  5-FOLD CROSS-VALIDATION [{stratify_label}]")
    print("=" * 60)
 
    cv_predictions = cross_validate(
        normed_X_train, y_train_np,
        near_neigh=near_neigh,
        weight_type=weight_type,
        n_splits=5,
        random_state=42,
    )

    cv_rows = []
    for k, rec in cv_predictions.items():
        row = {"k": k, "best_weight": rec["best_weight"]}
        row.update({f"cv_overall_{m}": v for m, v in rec["metrics"].items()})
        row.update({f"cv_red_{m}":     v for m, v in rec["red_metrics"].items()})
        row.update({f"cv_white_{m}":   v for m, v in rec["white_metrics"].items()})
        cv_rows.append(row)
 
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(f"knn_regression_cv_{stratify_label}.csv", index=False, encoding="utf-8-sig")
    print(f"\nCross-validation results saved to knn_regression_cv_{stratify_label}.csv")
 
    best_k_cv = min(cv_predictions, key=lambda k: cv_predictions[k]["metrics"]["mse"])
    best_rec_cv = cv_predictions[best_k_cv]
    print_best_record("5-FOLD CV", best_k_cv, best_rec_cv, stratify_label)
 
    print_comparison_table(best_k_cv, best_rec_cv, best_k_test, best_rec_test, stratify_label)

    for metric in ("mse", "rmse", "mae", "r2", "acc1"):
        pk(test_predictions, metrics_type=metric)
    
    pr(
        y_true=y_test_np,
        y_pred=np.array(best_rec_test["y_pred"]),
        is_red=np.array(best_rec_test["is_red"])
    )

    y_pred_best = np.array(best_rec_test["y_pred"])
    is_red_best = np.array(best_rec_test["is_red"]).astype(bool)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test_np[~is_red_best], y_pred_best[~is_red_best],
               alpha=0.35, color="steelblue", label="White", s=12)
    ax.scatter(y_test_np[is_red_best],  y_pred_best[is_red_best],
               alpha=0.35, color="firebrick", label="Red", s=12)
    lims = [y_test_np.min() - 0.2, y_test_np.max() + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True Quality")
    ax.set_ylabel("Predicted Quality")
    ax.set_title(
        f"KNN Regression — Predicted vs Actual [{stratify_label}]\n"
        f"(k={best_k_test}, weight={best_rec_test['best_weight']})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
