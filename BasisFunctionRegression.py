import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import utils as u
import matplotlib.pyplot as plt
import os
import math
from typing import Any

from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.metrics import pairwise_distances

import pprint
import csv
import pandas as pd
import seaborn as sns
import time

from collections import Counter

# configs

red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file1 = 'bfr_results/winequality-modified-poly.csv'
output_file2 = 'bfr_results/winequality-modified-rbf.csv'

status_file = 'bfr_results/bfr_status.txt'

def write_status(msg):
    with open(status_file, "a") as f:
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}", file=f)

def getdata(stratify=True):
    # raw data
    X_train, y_train, X_test, y_test = u.preprocessing(red_file, white_file, stratify=stratify)

    redwines = X_test['red'].values == 1

    # general data
    X_train_norm, X_test_norm, scaler = u.normalization(X_train, X_test)
    general_data = {
        "X_train": X_train_norm, 
        "y_train": y_train, 
        "X_test": X_test_norm, 
        "y_test": y_test,
        "test_is_red": redwines
    }


    # red wine
    redwines = X_train['red'].values == 1
    X_train_redwine = X_train[redwines]
    y_train_redwine = y_train[redwines]

    redwines = X_test['red'].values == 1
    X_test_redwine = X_test[redwines]
    y_test_redwine = y_test[redwines]

    X_train_redwine_norm, X_test_redwine_norm, scaler = u.normalization(X_train_redwine, X_test_redwine)

    redwine_data = {
        "X_train": X_train_redwine_norm, 
        "y_train": y_train_redwine, 
        "X_test": X_test_redwine_norm, 
        "y_test": y_test_redwine
    }

    # white wine

    whitewines = X_train['red'].values == 0
    X_train_whitewine = X_train[whitewines]
    y_train_whitewine = y_train[whitewines]

    whitewines = X_test['red'].values == 0
    X_test_whitewine = X_test[whitewines]
    y_test_whitewine = y_test[whitewines]

    X_train_whitewine_norm, X_test_whitewine_norm, scaler = u.normalization(X_train_whitewine, X_test_whitewine)

    whitewine_data = {
        "X_train": X_train_whitewine_norm, 
        "y_train": y_train_whitewine, 
        "X_test": X_test_whitewine_norm, 
        "y_test": y_test_whitewine
    }

    return general_data, redwine_data, whitewine_data


def trainpoly(data, hp):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    polyresults = {
        "hp": [],
        "mse": [],
    }

    # calculate total combinations
    total = len(hp["degree"]) * len(hp["regularization"]) 
    hpcount = 0
    count = 0
    threshold = math.ceil(float(total) / 20)


    for d, r in product(hp["degree"], hp["regularization"]):

        polymodel = Pipeline([
            ('poly', PolynomialFeatures(degree=d)),
            ('ridge', Ridge(alpha=r))
        ])

        mse = u.kfoldtrain(X_train, y_train, polymodel)

        polyhp = {
            "degree": d,
            "regularization": r
        }

        polyresults["hp"].append(polyhp)
        polyresults["mse"].append( mse)

        hpcount += 1
        count += 1
        if count >= threshold:
            count = 0
            pct = (hpcount / total) * 100
            write_status(f"{hpcount}/{total} ({pct:.1f}%)")

    return polyresults

def getoptimalpolyhp(polyresults, data):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    idx = np.argmin(polyresults["mse"])

    d = polyresults["hp"][idx]["degree"]
    r = polyresults["hp"][idx]["regularization"]
    polymodel = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('ridge', Ridge(alpha=r))
    ])

    start = time.time()
    polymodel.fit(X_train, y_train)
    end = time.time()
    traintime = float(end - start) / 60

    y_pred = polymodel.predict(X_test)
    mse, rmse, mae, r2, acc_plus_minus_1 = u.calculate_metrics(y_test, y_pred)

    results = {
        "degree": d,
        "regularization": r,
        "valid_mse": polyresults["mse"][idx],
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acc_plus_minus_1": acc_plus_minus_1,
        "y_pred": y_pred,
        "time": traintime
    }

    return results


def trainrbf(data, hp):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    rbfresults = {
        "hp": [],
        "mse": [],
    }

    # calculate total combinations
    total = len(hp["width"]) * len(hp["center"]) * len(hp["regularization"]) 
    hpcount = 0
    count = 0
    threshold = math.ceil(float(total) / 20)

    for w, c, r in product(hp["width"], hp["center"], hp["regularization"]):

        rbfmodel = Pipeline([
            ('nystrom', Nystroem(kernel='rbf', gamma=w, n_components=c, random_state=42)),  
            ('ridge', Ridge(alpha=r))
        ])

        mse = u.kfoldtrain(X_train, y_train, rbfmodel)

        rbfhp = {
            "width": w,
            "center": c,
            "regularization": r
        }

        rbfresults["hp"].append(rbfhp)
        rbfresults["mse"].append( mse )

        hpcount += 1
        count += 1
        if count >= threshold:
            count = 0
            pct = (hpcount / total) * 100
            write_status(f"{hpcount}/{total} ({pct:.1f}%)")

    return rbfresults




def getoptimalrbfhp(rbfresults, data):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    idx = np.argmin(rbfresults["mse"])

    w = rbfresults["hp"][idx]["width"]
    c = rbfresults["hp"][idx]["center"]
    r = rbfresults["hp"][idx]["regularization"]
    rbfmodel = Pipeline([
        ('nystrom', Nystroem(kernel='rbf', gamma=w, n_components=c, random_state=42)),  
        ('ridge', Ridge(alpha=r))
    ])

    start = time.time()
    rbfmodel.fit(X_train, y_train)
    end = time.time()
    traintime = float(end - start) / 60

    y_pred = rbfmodel.predict(X_test)
    mse, rmse, mae, r2, acc_plus_minus_1 = u.calculate_metrics(y_test, y_pred)

    results = {
        "width": w,
        "center": c,
        "regularization": r,
        "valid_mse": rbfresults["mse"][idx],
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acc_plus_minus_1": acc_plus_minus_1,
        "y_pred": y_pred,
        "time": traintime
    }

    return results



def plot_comparison(results, results2, outputfilename):
    metrics = ["mse", "rmse", "mae", "r2"]
    x = np.arange(len(metrics))
    width = 0.35

    vals = []
    vals2 = []

    for m in metrics:
        vals.append(results[m])
        vals2.append(results2[m])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, vals, width, label=results["label"])
    ax.bar(x + width/2, vals2, width, label=results2["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title(results["label"] + " vs " + results2["label"])

    if os.path.exists('/Users/minkijiang/Documents/UTSC/year 3/semester 2/CSCC11/CSCC11-Wine-Quality-Regression/bfr_results'):
        fig.savefig("/Users/minkijiang/Documents/UTSC/year 3/semester 2/CSCC11/CSCC11-Wine-Quality-Regression/bfr_results/" + outputfilename, dpi=300)

"""
if not stratify:
        u.plot_residuals(general_data["y_test"], results["PolynomialRegression"]["y_pred"], general_data["test_is_red"],\
            "UnStratified Residuals (Polynomial)", "bfr_results/unstratified_poly_residuals.png")
        u.plot_pred_quality(results["PolynomialRegression"]["y_pred"], general_data["test_is_red"],\
            "UnStratified Predicted Quality Distribution: Red vs White (Polynomial)", "bfr_results/unstratified_poly_predplot.png")

        u.plot_residuals(general_data["y_test"], results["RBFRegression"]["y_pred"], general_data["test_is_red"],\
            "UnStratified Residuals (RBF)", "bfr_results/unstratified_rbf_residuals.png")
        u.plot_pred_quality(results["RBFRegression"]["y_pred"], general_data["test_is_red"],\
            "UnStratified Predicted Quality Distribution: Red vs White (RBF)", "bfr_results/unstratified_rbf_predplot.png")

        results["PolynomialRegression"].pop("y_pred")
        results["RBFRegression"].pop("y_pred")

        with open("bfr_results/bfr_unstratify_general_poly_results.txt", "w") as f:
            pprint.pprint(results["PolynomialRegression"], stream=f)
        with open("bfr_results/bfr_unstratify_general_rbf_results.txt", "w") as f:
            pprint.pprint(results["RBFRegression"], stream=f)
    else:
        u.plot_residuals(general_data["y_test"], results["PolynomialRegression"]["y_pred"], general_data["test_is_red"],\
            "Stratified Residuals", "bfr_results/stratified_poly_residuals.png")
        u.plot_pred_quality(results["PolynomialRegression"]["y_pred"], general_data["test_is_red"],\
            "Stratified Predicted Quality Distribution: Red vs White (Polynomial)", "bfr_results/stratified_poly_predplot.png")

        u.plot_residuals(general_data["y_test"], results["RBFRegression"]["y_pred"], general_data["test_is_red"],\
            "Stratified Residuals (RBF)", "bfr_results/stratified_rbf_residuals.png")
        u.plot_pred_quality(results["RBFRegression"]["y_pred"], general_data["test_is_red"],\
            "Stratified Predicted Quality Distribution: Red vs White (RBF)", "bfr_results/stratified_rbf_predplot.png")

        results["PolynomialRegression"].pop("y_pred")
        results["RBFRegression"].pop("y_pred")

        with open("bfr_results/bfr_stratify_general_poly_results.txt", "w") as f:
            pprint.pprint(results["PolynomialRegression"], stream=f)
        with open("bfr_results/bfr_stratify_general_rbf_results.txt", "w") as f:
            pprint.pprint(results["RBFRegression"], stream=f)
"""


def getgeneralresult(general_data, stratify=True):
    hp = u.hp_search_grid("bfr", general_data["y_train"], general_data["X_train"])

    write_status("Started Poly")
    polyresults = trainpoly(general_data, hp)
    write_status("Done Poly")

    write_status("Started RBF")
    rbfresults = trainrbf(general_data, hp)
    write_status("Done RBF")

    bestpolyhp = getoptimalpolyhp(polyresults, general_data)

    bestrbfhp = getoptimalrbfhp(rbfresults, general_data)

    results = {
        "PolynomialRegression": bestpolyhp,
        "RBFRegression": bestrbfhp
    }

    if not stratify:
        results["PolynomialRegression"].pop("y_pred")
        results["RBFRegression"].pop("y_pred")

        with open("bfr_results/bfr_unstratify_general_poly_results.txt", "w") as f:
            pprint.pprint(results["PolynomialRegression"], stream=f)
        with open("bfr_results/bfr_unstratify_general_rbf_results.txt", "w") as f:
            pprint.pprint(results["RBFRegression"], stream=f)
    else:

        u.save_predictions(general_data["y_test"], results["PolynomialRegression"]["y_pred"], general_data["test_is_red"], output_file1)
        u.save_predictions(general_data["y_test"], results["RBFRegression"]["y_pred"], general_data["test_is_red"], output_file2)

        results["PolynomialRegression"].pop("y_pred")
        results["RBFRegression"].pop("y_pred")

        with open("bfr_results/bfr_stratify_general_poly_results.txt", "w") as f:
            pprint.pprint(results["PolynomialRegression"], stream=f)
        with open("bfr_results/bfr_stratify_general_rbf_results.txt", "w") as f:
            pprint.pprint(results["RBFRegression"], stream=f)

    return results

    


def getredandwhitewineresult(redwine_data, whitewine_data):
    #hp = u.hp_search_grid("bfr", general_data["y_train"])

    redwine_hp = u.hp_search_grid("bfr", redwine_data["y_train"], redwine_data["X_train"])
    write_status("Started Poly")
    redwine_polyresults = trainpoly(redwine_data, redwine_hp)
    write_status("Done Poly")

    write_status("Started RBF")
    redwine_rbfresults = trainrbf(redwine_data, redwine_hp)
    write_status("Done RBF")

    whitewine_hp = u.hp_search_grid("bfr", whitewine_data["y_train"], whitewine_data["X_train"])
    write_status("Started Poly")
    whitewine_polyresults = trainpoly(whitewine_data, whitewine_hp)
    write_status("Done Poly")

    write_status("Started RBF")
    whitewine_rbfresults = trainrbf(whitewine_data, whitewine_hp)
    write_status("Done RBF")

    redwine_bestpolyhp = getoptimalpolyhp(redwine_polyresults, redwine_data)
    redwine_bestrbfhp = getoptimalrbfhp(redwine_rbfresults, redwine_data)

    whitewine_bestpolyhp = getoptimalpolyhp(whitewine_polyresults, whitewine_data)
    whitewine_bestrbfhp = getoptimalrbfhp(whitewine_rbfresults, whitewine_data)

    results = {
        "Red Wine": {
            "PolynomialRegression": redwine_bestpolyhp,
            "RBFRegression": redwine_bestrbfhp
        },
        "White Wine": {
            "PolynomialRegression": whitewine_bestpolyhp,
            "RBFRegression": whitewine_bestrbfhp
        }
        
    }

    results["Red Wine"]["PolynomialRegression"].pop("y_pred")
    results["Red Wine"]["RBFRegression"].pop("y_pred")
    results["White Wine"]["PolynomialRegression"].pop("y_pred")
    results["White Wine"]["RBFRegression"].pop("y_pred")

    with open("bfr_results/bfr_redwhite_results.txt", "w") as f:
        pprint.pprint(results, stream=f)

    return results

def accuracy_plot(general_data, redwine_data, whitewine_data, filename):
    y_true, y_true_red, y_true_white = general_data["y_true"], redwine_data["y_true"], whitewine_data["y_true"]
    y_pred, y_pred_red, y_pred_white = general_data["y_pred"], redwine_data["y_pred"], whitewine_data["y_pred"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, yt, yp, title in zip(axes,
        [y_true, y_true_red, y_true_white],
        [y_pred, y_pred_red, y_pred_white],
        ["Overall", "Red Wine", "White Wine"]):

        ax.scatter(yt, yp, alpha=0.3)
        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--')
        ax.set_xlabel("Actual Quality")
        ax.set_ylabel("Predicted Quality")
        ax.set_title(f"Predicted vs Actual ({title})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_pred_quality(y_pred, is_red, title, outputfilename=None):
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred[is_red == 0], bins=15, alpha=0.6, label="White", color="blue")
    plt.hist(y_pred[is_red == 1], bins=15, alpha=0.6, label="Red", color="red")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    if title != None:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if outputfilename == None:
        plt.show()
    else: 
        plt.savefig(outputfilename, dpi=150, bbox_inches='tight')

    plt.close()

def accuracy_per_score(redwine_data, whitewine_data, filename):
    y_true_red, y_true_white = redwine_data["y_true"], whitewine_data["y_true"]
    y_pred_red, y_pred_white = redwine_data["y_pred"], whitewine_data["y_pred"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for yt, yp, label, color in zip(
        [y_true_red, y_true_white],
        [y_pred_red, y_pred_white],
        ["Red", "White"],
        ["red", "blue"]):

        scores = sorted(set(yt.astype(int)))
        accs = []
        for score in scores:
            mask = yt.astype(int) == score
            if mask.sum() == 0:
                accs.append(0)
                continue
            hits = np.abs(np.round(yp[mask]) - yt[mask]) <= 1
            accs.append(hits.mean())

        ax.plot(scores, accs, marker='o', label=label, color=color)

    ax.set_xlabel("Actual Quality Score")
    ax.set_ylabel("Accuracy ")
    ax.set_title("Accuracy per Quality Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def heatmap(redwine_data, whitewine_data, filename):
    y_true_red, y_true_white = redwine_data["y_true"], whitewine_data["y_true"]
    y_pred_red, y_pred_white = redwine_data["y_pred"], whitewine_data["y_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, yt, yp, title in zip(axes,
        [y_true_red, y_true_white],
        [y_pred_red, y_pred_white],
        ["Red Wine", "White Wine"]):

        scores = sorted(set(yt.astype(int)))
        matrix = np.zeros((len(scores), len(scores)))

        for true, pred in zip(yt.astype(int), np.round(yp).astype(int)):
            if pred in scores:
                i = scores.index(true)
                j = scores.index(pred)
                matrix[i, j] += 1

        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(len(scores)))
        ax.set_yticks(range(len(scores)))
        ax.set_xticklabels(scores)
        ax.set_yticklabels(scores)
        ax.set_xlabel("Predicted Quality")
        ax.set_ylabel("Actual Quality")
        ax.set_title(f"Prediction Heatmap ({title})")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def trainall():
    open("bfr_results/bfr_status.txt", "w").close() # clear file
    write_status("Started")

    write_status("Started Extracting Training Data")
    stratify_general_data, redwine_data, whitewine_data = getdata(stratify=True)
    unstratify_general_data, _ , _ = getdata(stratify=False)
    write_status("Done Extracting Training Data")

    write_status("Started UnStratified General")
    start = time.time()
    unstrat_results = getgeneralresult(unstratify_general_data, stratify=False)
    end = time.time()
    with open("bfr_results/bfr_unstratify_general_poly_results.txt", "a") as f:
        print(f"\n\nPoly Training Time: {unstrat_results["PolynomialRegression"]["time"]: .4f} minutes\n", file=f)
        print(f"\n\nTotal Training Time (of all hyperparameters of both poly/rbf): {float(end - start) / 60:.4f} minutes\n", file=f)
    with open("bfr_results/bfr_unstratify_general_rbf_results.txt", "a") as f:
        print(f"\n\nRBF Training Time: {unstrat_results["RBFRegression"]["time"]: .4f} minutes\n", file=f)
        print(f"\n\nTotal Training Time (of all hyperparameters of both poly/rbf): {float(end - start) / 60:.4f} minutes\n", file=f)
    write_status(f"Finished UnStratified General - {float(end - start) / 60:.4f} minutes")

    write_status("Started Stratified General")
    start = time.time()
    strat_results = getgeneralresult(stratify_general_data, stratify=True)
    end = time.time()
    with open("bfr_results/bfr_stratify_general_poly_results.txt", "a") as f:
        print(f"\n\nPoly Training Time: {strat_results["PolynomialRegression"]["time"]: .4f} minutes\n", file=f)
        print(f"\n\nTotal Training Time (of all hyperparameters of both poly/rbf): {float(end - start) / 60:.4f} minutes\n", file=f)
    with open("bfr_results/bfr_stratify_general_rbf_results.txt", "a") as f:
        print(f"\n\nRBF Training Time: {strat_results["RBFRegression"]["time"]: .4f} minutes\n", file=f)
        print(f"\n\nTotal Training Time (of all hyperparameters of both poly/rbf): {float(end - start) / 60:.4f} minutes\n", file=f)
    write_status(f"Finished Stratified General - {float(end - start) / 60:.4f} minutes")

    unstrat_results["PolynomialRegression"]["label"] = "Polynomial"
    unstrat_results["RBFRegression"]["label"] = "RBF"
    plot_comparison(unstrat_results["PolynomialRegression"], unstrat_results["RBFRegression"], "unstratpolyrbfcomparison.png")

    strat_results["PolynomialRegression"]["label"] = "Polynomial"
    strat_results["RBFRegression"]["label"] = "RBF"
    plot_comparison(strat_results["PolynomialRegression"], strat_results["RBFRegression"], "stratpolyrbfcomparison.png")

    write_status("Started Red/White")
    results = getredandwhitewineresult(redwine_data, whitewine_data)
    write_status("Finished Red/White")

    results["Red Wine"]["PolynomialRegression"]["label"] = "Red Wine"
    results["Red Wine"]["RBFRegression"]["label"] = "White Wine"
    plot_comparison(results["Red Wine"]["PolynomialRegression"], results["Red Wine"]["RBFRegression"], "polyredwhitecomparison.png")

    results["White Wine"]["PolynomialRegression"]["label"] = "Red Wine"
    results["White Wine"]["RBFRegression"]["label"] = "White Wine"
    plot_comparison(results["White Wine"]["PolynomialRegression"], results["White Wine"]["RBFRegression"], "rbfredwhitecomparison.png")


    write_status("All Done")


def graphs():
    poly_general_data, poly_redwine_data, poly_whitewine_data = u.extract_csv(output_file1)
    rbf_general_data, rbf_redwine_data, rbf_whitewine_data = u.extract_csv(output_file2)

    accuracy_plot(poly_general_data, poly_redwine_data, poly_whitewine_data, "bfr_results/poly_pred_vs_actual.png")
    accuracy_plot(rbf_general_data, rbf_redwine_data, rbf_whitewine_data, "bfr_results/rbf_pred_vs_actual.png")

    accuracy_per_score(poly_redwine_data, poly_whitewine_data, "bfr_results/poly_accuracy_per_score.png")
    accuracy_per_score(rbf_redwine_data, rbf_whitewine_data, "bfr_results/rbf_accuracy_per_score.png")

    heatmap(poly_redwine_data, poly_whitewine_data, "bfr_results/poly_prediction_heatmap.png")
    heatmap(rbf_redwine_data, rbf_whitewine_data, "bfr_results/rbf_prediction_heatmap.png")

    u.plot_residuals(poly_general_data["y_true"], poly_general_data["y_pred"], poly_general_data["is_red"],\
            None, "bfr_results/stratified_poly_residuals.png")
    plot_pred_quality(poly_general_data["y_pred"], poly_general_data["is_red"],\
        "Stratified Predicted Quality Distribution: Red vs White (Polynomial)", "bfr_results/stratified_poly_predplot.png")

    u.plot_residuals(rbf_general_data["y_true"], rbf_general_data["y_pred"], rbf_general_data["is_red"],\
            None, "bfr_results/stratified_rbf_residuals.png")
    plot_pred_quality(rbf_general_data["y_pred"], rbf_general_data["is_red"],\
        "Stratified Predicted Quality Distribution: Red vs White (Polynomial)", "bfr_results/stratified_poly_predplot.png")



if __name__ == "__main__":
    graphs()
    


# nohup caffeinate -i python BasisFunctionRegression.py > bfr_output.log 2>&1 &
# ps aux | grep BasisFunctionRegression.py
# tail -n 50 bfr_output.log  
# kill <PID>  

# cat bfr_results/bfr_status.txt


    





    


    

