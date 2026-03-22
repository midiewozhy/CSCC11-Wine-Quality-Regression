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

import pprint

# configs

red_file = 'winequality-red.csv'
white_file = 'winequality-white.csv'
output_file = 'winequality-modified.csv'

def getdata():
    # raw data
    X_train, y_train, X_test, y_test = u.preprocessing(red_file, white_file)

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
        "rmse": [],
        "mae": [],
        "r2": [],
        "acc_plus_minus_1": [],
        "y_pred": []
    }

    for d, r in product(hp["degree"], hp["regularization"]):
        polymodel = Pipeline([
            ('poly', PolynomialFeatures(degree=d)),
            ('ridge', Ridge(alpha=r))
        ])
        polymodel.fit(X_train, y_train)
        y_pred = polymodel.predict(X_test)
        mse, rmse, mae, r2, acc_plus_minus_1 = u.calculate_metrics(y_test, y_pred)

        polyhp = {
            "degree": d,
            "regularization": r
        }

        polyresults["hp"].append(polyhp)
        polyresults["mse"].append(mse)
        polyresults["rmse"].append(rmse)
        polyresults["mae"].append(mae)
        polyresults["r2"].append(r2)
        polyresults["acc_plus_minus_1"].append(acc_plus_minus_1)
        polyresults["y_pred"].append(y_pred)

    return polyresults


def trainrbf(data, hp):
    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    rbfresults = {
        "hp": [],
        "mse": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "acc_plus_minus_1": [],
        "y_pred": []
    }

    for w, c, r in product(hp["width"], hp["center"], hp["regularization"]):
        rbfmodel = Pipeline([
            ('nystrom', Nystroem(kernel='rbf', gamma=w, n_components=c, random_state=42)),  
            ('ridge', Ridge(alpha=r))
        ])
        rbfmodel.fit(X_train, y_train)
        y_pred = rbfmodel.predict(X_test)

        mse, rmse, mae, r2, acc_plus_minus_1 = u.calculate_metrics(y_test, y_pred)

        rbfhp = {
            "width": w,
            "center": c,
            "regularization": r
        }

        rbfresults["hp"].append(rbfhp)
        rbfresults["mse"].append(mse)
        rbfresults["rmse"].append(rmse)
        rbfresults["mae"].append(mae)
        rbfresults["r2"].append(r2)
        rbfresults["acc_plus_minus_1"].append(acc_plus_minus_1)
        rbfresults["y_pred"].append(y_pred)

    return rbfresults


def getoptimalpolyhp(polyresults):
    idx = np.argmin(polyresults["rmse"])

    results = {
        "degree": polyresults["hp"][idx]["degree"],
        "regularization": polyresults["hp"][idx]["regularization"],
        "mse": polyresults["mse"][idx],
        "rmse": polyresults["rmse"][idx],
        "mae": polyresults["mae"][idx],
        "r2": polyresults["r2"][idx],
        "acc_plus_minus_1": polyresults["acc_plus_minus_1"][idx],
        "y_pred": polyresults["y_pred"][idx]
    }

    return results


def getoptimalrbfhp(rbfresults):
    idx = np.argmin(rbfresults["rmse"])

    results = {
        "width": rbfresults["hp"][idx]["width"],
        "center": rbfresults["hp"][idx]["center"],
        "regularization": rbfresults["hp"][idx]["regularization"],
        "mse": rbfresults["mse"][idx],
        "rmse": rbfresults["rmse"][idx],
        "mae": rbfresults["mae"][idx],
        "r2": rbfresults["r2"][idx],
        "acc_plus_minus_1": rbfresults["acc_plus_minus_1"][idx],
        "y_pred": rbfresults["y_pred"][idx]
    }

    return results


def getgeneralresult():
    general_data, redwine_data, whitewine_data = getdata()
    hp = u.hp_search_grid("bfr", general_data["y_train"])

    polyresults = trainpoly(general_data, hp)
    rbfresults = trainrbf(general_data, hp)

    bestpolyhp = getoptimalpolyhp(polyresults)

    bestrbfhp = getoptimalrbfhp(rbfresults)

    results = {
        "PolynomialRegression": bestpolyhp,
        "RBFRegression": bestrbfhp
    }

    u.plot_residuals(general_data["y_test"], results["PolynomialRegression"]["y_pred"], general_data["test_is_red"])
    u.plot_residuals(general_data["y_test"], results["RBFRegression"]["y_pred"], general_data["test_is_red"])

    return results

    


def getredandwhitewineresult():
    general_data, redwine_data, whitewine_data = getdata()
    hp = u.hp_search_grid("bfr", general_data["y_train"])

    redwine_polyresults = trainpoly(redwine_data, hp)
    redwine_rbfresults = trainrbf(redwine_data, hp)

    whitewine_polyresults = trainpoly(whitewine_data, hp)
    whitewine_rbfresults = trainrbf(whitewine_data, hp)

    redwine_bestpolyhp = getoptimalpolyhp(redwine_polyresults)
    redwine_bestrbfhp = getoptimalrbfhp(redwine_rbfresults)

    whitewine_bestpolyhp = getoptimalpolyhp(whitewine_polyresults)
    whitewine_bestrbfhp = getoptimalrbfhp(whitewine_rbfresults)

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

    

    return results




if __name__ == "__main__":
    results = getredandwhitewineresult()
    pprint.pprint(results)

    print("\n\n\n")

    results = getgeneralresult()
    pprint.pprint(results)


    





    


    

