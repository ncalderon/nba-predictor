from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils as utils

exp_results = []


def plot_experiment_results(results):
    results_df = pd.DataFrame(results[0])
    for idx, result in enumerate(results[1:]):
        result_df = pd.DataFrame(result)
        results_df = pd.concat([results_df, result_df], ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(25, 10))

    ax = sns.boxplot(x="model", y="test_balanced_accuracy", data=results_df, ax=ax1[0]).set_title("balanced_accuracy")
    ax = sns.boxplot(x="model", y="test_precision", data=results_df, ax=ax1[1]).set_title("precision")
    ax = sns.boxplot(x="model", y="test_recall", data=results_df, ax=ax1[2]).set_title("recall")

    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_train", y="test_balanced_accuracy", hue="model", ax=ax2[0]).set_title(
        "balanced_accuracy")
    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_train", y="test_precision", hue="model", ax=ax2[1]).set_title(
        "precision")
    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_train", y="test_recall", hue="model", ax=ax2[2]).set_title("recall")
    fig.savefig("./plots/tscv_exp.png")
    return results_df


def run_experiment_using_cross_validate(exp_name, df, models, tscv, train_splits, X, y):
    # Evaluate each model in turn
    results = []
    names = []
    print("Start experiment using TimeSeriesSplit")
    for name, model in models:
        cv_results = cross_validate(model,
                                    X,
                                    y.ravel(),
                                    cv=tscv,
                                    scoring=['balanced_accuracy', 'precision', "recall"]
                                    , return_train_score=True
                                    )

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            "balanced_accuracy_mean": cv_results["test_balanced_accuracy"].mean(),
            "balanced_accuracy_std": cv_results["test_balanced_accuracy"].std(),
            "precision_mean": cv_results["test_precision"].mean(),
            "precision_std": cv_results["test_precision"].std(),
            "recall_mean": cv_results["test_recall"].mean(),
            "recall_std": cv_results["test_recall"].std(),
        }
        exp_results.append(exp_result)
        print(f'{name}')
        print(
            f'test_balanced_accuracy: {cv_results["test_balanced_accuracy"].mean()} - {cv_results["test_balanced_accuracy"].std()}')
        print(f'test_precision: {cv_results["test_precision"].mean()} - {cv_results["test_precision"].std()}')
        print(f'test_recall: {cv_results["test_recall"].mean()} - {cv_results["test_recall"].std()}')
        cv_results["model"] = [name] * train_splits
        cv_results["season_train"] = list(df.SEASON.unique()[-train_splits:])
        results.append(cv_results)

        names.append(name)
    print("Done")
    return names, results


def run_experiment(exp_name, df, models, tscv, train_splits, X, y, scale=False):
    # Evaluate each model in turn
    results = []
    names = []
    print("Running experiment", exp_name)
    for name, model in models:
        cv_results = {
            "test_balanced_accuracy": [],
            "test_precision": [],
            "test_recall": []
        }
        for train_index, test_index in tscv.split(X=X):
            if scale:
                X[train_index], X[test_index] = utils.feature_scaling(X[train_index], X[test_index], 5)
            fit_info = model.fit(X[train_index], y[train_index].ravel())
            predictions = model.predict(X=X[test_index])
            balanced_accuracy = balanced_accuracy_score(y[test_index], predictions)
            precision = model.score(X[test_index], y[test_index].ravel())
            recall = recall_score(y[test_index], predictions)
            cv_results["test_balanced_accuracy"].append(balanced_accuracy)
            cv_results["test_precision"].append(precision)
            cv_results["test_recall"].append(recall)

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            "balanced_accuracy_mean": np.mean(cv_results["test_balanced_accuracy"]),
            "balanced_accuracy_std": np.std(cv_results["test_balanced_accuracy"]),
            "precision_mean": np.mean(cv_results["test_precision"]),
            "precision_std": np.std(cv_results["test_precision"]),
            "recall_mean": np.mean(cv_results["test_recall"]),
            "recall_std": np.std(cv_results["test_recall"]),
        }
        exp_results.append(exp_result)
        print(f'{name}')
        print(
            f'balanced_accuracy: {np.mean(cv_results["test_balanced_accuracy"])} - {np.std(cv_results["test_balanced_accuracy"])}')
        print(f'precision: {np.mean(cv_results["test_precision"])} - {np.std(cv_results["test_precision"])}')
        print(f'recall: {np.mean(cv_results["test_recall"])} - {np.std(cv_results["test_recall"])}')
        cv_results["model"] = [name] * train_splits
        cv_results["season_train"] = list(df.SEASON.unique()[-train_splits:])
        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results
