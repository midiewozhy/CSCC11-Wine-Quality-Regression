from pysr import PySRRegressor
import utils as u
import numpy as np
import pprint

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
from itertools import product
from sklearn.metrics import pairwise_distances
import time

# configs

red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file = 'sr_results/winequality-modified.csv'

status_file = 'sr_results/sr_status.txt'

def getdata(stratify=True):
    # raw data
    X_train, y_train, X_test, y_test = u.preprocessing(red_file, white_file, stratify=stratify)

    redwines = X_test['red'].values == 1
    feature_names = list(X_train.columns)

    # general data
    X_train_norm, X_test_norm, scaler = u.normalization(X_train, X_test)
    general_data = {
        "X_train": X_train_norm, 
        "y_train": y_train, 
        "X_test": X_test_norm, 
        "y_test": y_test,
        "test_is_red": redwines,
        "feature_names": feature_names
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
        "y_test": y_test_redwine,
        "feature_names": feature_names
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
        "y_test": y_test_whitewine,
        "feature_names": feature_names
    }

    return general_data, redwine_data, whitewine_data

def write_status(msg):
    with open(status_file, "a") as f:
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}", file=f)

def trainsr(data, hp):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    srresults = {
        "hp": [],
        "mse": [],
        "equation": []
    }

    # calculate total combinations
    total = len(hp["niterations"]) * len(hp["maxsize"]) * len(hp["population_size"]) * len(hp["binary_operators"]) * len(hp["unary_operators"])
    hpcount = 0
    count = 0
    threshold = math.ceil(float(total) / 20)

    for niterations, maxsize, population_size, binary_operators, unary_operators in \
    product(hp["niterations"], hp["maxsize"], hp["population_size"], hp["binary_operators"], hp["unary_operators"]):

        # symbolic regression model
        srmodel = PySRRegressor(
            model_selection="best",
            niterations=niterations,                        
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            maxsize=maxsize,                            
            population_size=population_size,
            progress=False
        )

        mse = u.kfoldtrain(X_train, y_train, srmodel, data["feature_names"])

        srhp = {
            "niterations": niterations,
            "maxsize": maxsize,
            "population_size": population_size,
            "binary_operators": binary_operators,
            "unary_operators": unary_operators
        }

        srresults["hp"].append(srhp)
        srresults["mse"].append(mse)
        srresults["equation"].append(str(srmodel.sympy()))

        hpcount += 1
        count += 1
        if count >= threshold:
            count = 0
            pct = (hpcount / total) * 100
            write_status(f"{hpcount}/{total} ({pct:.1f}%)")

    return srresults

def getoptimalsrhp(srresults, data):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    idx = np.argmin(srresults["mse"])

    niterations = srresults["hp"][idx]["niterations"]
    binary_operators = srresults["hp"][idx]["binary_operators"]
    unary_operators = srresults["hp"][idx]["unary_operators"]
    maxsize = srresults["hp"][idx]["maxsize"]
    population_size = srresults["hp"][idx]["population_size"]
    srmodel = PySRRegressor(
        model_selection="best",
        niterations=niterations,                        
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        maxsize=maxsize,                            
        population_size=population_size,
        progress=False
    )

    start = time.time()
    srmodel.fit(X_train, y_train)
    end = time.time()
    traintime = float(end - start) / 60

    y_pred = srmodel.predict(X_test)
    mse, rmse, mae, r2, acc_plus_minus_1 = u.calculate_metrics(y_test, y_pred)

    results = {
        "hp": srresults["hp"][idx],
        "valid_mse": srresults["mse"][idx],
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acc_plus_minus_1": acc_plus_minus_1,
        "y_pred": y_pred,
        "equation": srresults["equation"][idx],
        "time": traintime
    }

    return results


"""
    if not stratify:
        u.plot_residuals(general_data["y_test"], bestsrhp["y_pred"], general_data["test_is_red"],\
            "UnStratified Residuals", "sr_results/unstratified_residuals.png")
        u.plot_pred_quality(bestsrhp["y_pred"], general_data["test_is_red"],\
            "UnStratified Predicted Quality Distribution: Red vs White", "sr_results/unstratified_predplot.png")

        bestsrhp.pop("y_pred")
        with open("sr_results/sr_unstratify_general_results.txt", "w") as f:
            pprint.pprint(bestsrhp, stream=f)
    else:
        u.plot_residuals(general_data["y_test"], bestsrhp["y_pred"], general_data["test_is_red"],\
            "Stratified Residuals", "sr_results/stratified_residuals.png")
        u.plot_pred_quality(bestsrhp["y_pred"], general_data["test_is_red"],\
            "Stratified Predicted Quality Distribution: Red vs White", "sr_results/stratified_predplot.png")

        save_predictions(general_data["y_test"], bestsrhp["y_pred"], general_data["test_is_red"], output_file)

        bestsrhp.pop("y_pred")
        with open("sr_results/sr_stratify_general_results.txt", "w") as f:
            pprint.pprint(bestsrhp, stream=f)

"""

def getgeneralresult(general_data, stratify=True):
    hp = u.hp_search_grid("sr", y_train = general_data["y_train"], X_train = general_data["X_train"])

    srresults = trainsr(general_data, hp)
    bestsrhp = getoptimalsrhp(srresults, general_data)

    if not stratify:
        bestsrhp.pop("y_pred")
        with open("sr_results/sr_unstratify_general_results.txt", "w") as f:
            pprint.pprint(bestsrhp, stream=f)
    else:
        u.save_predictions(general_data["y_test"], bestsrhp["y_pred"], general_data["test_is_red"], output_file)
        bestsrhp.pop("y_pred")
        with open("sr_results/sr_stratify_general_results.txt", "w") as f:
            pprint.pprint(bestsrhp, stream=f)


    return bestsrhp




def getredandwhitewineresult(redwine_data, whitewine_data):

    redwine_hp = u.hp_search_grid("sr", y_train = redwine_data["y_train"], X_train = redwine_data["X_train"])
    redwine_srresults = trainsr(redwine_data, redwine_hp)

    whitewine_hp = u.hp_search_grid("sr", y_train = whitewine_data["y_train"], X_train = whitewine_data["X_train"])
    whitewine_srresults = trainsr(whitewine_data, whitewine_hp)

    redwine_besthp = getoptimalsrhp(redwine_srresults, redwine_data)
    whitewine_besthp = getoptimalsrhp(whitewine_srresults, whitewine_data)

    results = {
        "Red Wine": redwine_besthp,
        "White Wine": whitewine_besthp
    }

    results["Red Wine"].pop("y_pred")
    results["White Wine"].pop("y_pred")
    with open("sr_results/sr_redwhite_results.txt", "w") as f:
        pprint.pprint(results, stream=f)

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

    if os.path.exists('/Users/minkijiang/Documents/UTSC/year 3/semester 2/CSCC11/CSCC11-Wine-Quality-Regression/sr_results'):
        fig.savefig("/Users/minkijiang/Documents/UTSC/year 3/semester 2/CSCC11/CSCC11-Wine-Quality-Regression/sr_results/" + outputfilename, dpi=300)



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
    open("sr_results/sr_status.txt", "w").close() # clear file
    write_status("Started")

    write_status("Started Extracting Training Data")
    stratify_general_data, redwine_data, whitewine_data = getdata(stratify=True)
    unstratify_general_data, _ , _ = getdata(stratify=False)
    write_status("Done Extracted Training Data")


    write_status("Started UnStratified General")
    start = time.time()
    unstratify_bestsrhp = getgeneralresult(unstratify_general_data, stratify=False)
    end = time.time()
    with open("sr_results/sr_unstratify_general_results.txt", "a") as f:
        print(f"\n\nTraining Time: {unstratify_bestsrhp['time']: .4f} minutes", file=f)
        print(f"Total Training Time (for all hyperparameters): {float(end - start) / 60:.4f} minutes\n", file=f)
    write_status(f"Finished UnStratified General - {float(end - start) / 60:.4f} minutes")

    
    write_status("Started Stratified General")
    start = time.time()
    stratify_bestsrhp = getgeneralresult(stratify_general_data, stratify=True)
    end = time.time()
    with open("sr_results/sr_stratify_general_results.txt", "a") as f:
        print(f"\n\nTraining Time: {stratify_bestsrhp['time']: .4f} minutes", file=f)
        print(f"Total Training Time (for all hyperparameters): {float(end - start) / 60:.4f} minutes\n", file=f)
    write_status(f"Finished Stratified General - {float(end - start) / 60:.4f} minutes")

    unstratify_bestsrhp["label"] = "unStratified"
    stratify_bestsrhp["label"] = "Stratified"
    plot_comparison(unstratify_bestsrhp, stratify_bestsrhp, "stratcomparison.png")


    write_status("Started Red/White")
    results = getredandwhitewineresult(redwine_data, whitewine_data)
    write_status("Finished Red/White")

    results["Red Wine"]["label"] = "Red Wine"
    results["White Wine"]["label"] = "White Wine"
    plot_comparison(results["Red Wine"], results["White Wine"], "redwhitecomparison.png")

    write_status("All Done")

def graphs():
    general_data, redwine_data, whitewine_data = u.extract_csv(output_file)

    accuracy_plot(general_data, redwine_data, whitewine_data, "sr_results/pred_vs_actual.png")
    accuracy_per_score(redwine_data, whitewine_data, "sr_results/accuracy_per_score.png")
    heatmap(redwine_data, whitewine_data, "sr_results/prediction_heatmap.png")

    u.plot_residuals(general_data["y_true"], general_data["y_pred"], general_data["is_red"],\
            None, "sr_results/stratified_residuals.png")
    plot_pred_quality(general_data["y_pred"], general_data["is_red"],\
        "Stratified Predicted Quality Distribution: Red vs White ", "sr_results/stratified_predplot.png")



if __name__ == "__main__":
    graphs()
    


# nohup caffeinate -i python SymbolicRegression.py > sr_output.log 2>&1 &
# ps aux | grep SymbolicRegression.py
# tail -n 50 sr_output.log  
# kill <PID>  
# pkill -9 -f SymbolicRegression.py
# pkill -9 -f caffeinate

# cat sr_results/sr_status.txt

# nohup python SymbolicRegression.py > sr_output.log 2>&1 & disown
# caffeinate -dims &

# python3 -m venv env
# source env/bin/activate

# pip3 install -r requirements.txt










    