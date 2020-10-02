import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, \
    recall_score, \
    balanced_accuracy_score, \
    f1_score, \
    roc_auc_score
from sklearn.model_selection import cross_validate
import utils as utils

exp_results = []
visualizers = []


def get_models():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)))
    models.append(('SVM', SVC(kernel='linear', random_state=0)))
    #models.append(('KSVM', SVC(kernel='rbf', random_state=0)))
    #models.append(('NB', GaussianNB()))
    #models.append(('DT', DecisionTreeClassifier(criterion='entropy', random_state=0)))
    #models.append(('SGD', SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)))
    models.append(("RF", RandomForestClassifier(n_estimators=200,
                                                max_features="sqrt",
                                                max_depth=5,
                                                n_jobs=-1,
                                                random_state=0)))
    models.append(("GB", GradientBoostingClassifier(n_estimators=200,
                                                    max_depth=5,
                                                    max_features="sqrt",
                                                    random_state=0)))
    models.append(("XGB", xgb.XGBClassifier(
        random_state=0,
        max_depth=20,
        n_estimators=200
    )))

    models.append(("LGB", lgb.LGBMClassifier(
        random_state=0,
        max_depth=20,
        objective='binary',
        metric='binary_logloss',
        n_estimators=200,
        num_leaves=300
    )))
    # models.append(("CB", CatBoostClassifier(
    #     depth=10,
    #     n_estimators=200,
    #     random_state=0,
    #     min_data_in_leaf=80,
    #     learning_rate=0.1,
    #     l2_leaf_reg=9
    # )))
    return models


def map_results_to_df(results):
    results_df = pd.DataFrame(results[0])
    for idx, result in enumerate(results[1:]):
        result_df = pd.DataFrame(result)
        results_df = pd.concat([results_df, result_df], ignore_index=True)
    return results_df


def plot_to_compare_experiments(results_total, metric="test_balanced_accuracy", figsize=(25, 10), use_pointplot=False):
    row_size, column_size = len(results_total) // 4 + 1, 3
    idx = 0
    fig, ax_rows = plt.subplots(len(results_total), figsize=figsize)
    fig.suptitle(metric)
    while idx < len(results_total):
        for ax_row in ax_rows:
            if idx >= len(results_total):
                break
            result = results_total[idx]
            results_df = map_results_to_df(result[1])

            if use_pointplot:
                a = sns.pointplot(data=results_df,
                                  kind="point", x="season_test", y=metric, hue="model",
                                  ax=ax_row)
            else:
                a = sns.boxplot(x="model", y=metric, data=results_df, ax=ax_row)
            a.set_xlabel(None)
            a.set_title(result[0])
            a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            idx += 1

            # for ax_column in ax_row:
    # Put the legend out of the figure

    # fig.savefig(f"./plots/{metric}.png")


def plot_experiment_results(exp_name, results, figsize=(25, 10)):
    results_df = map_results_to_df(results)
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=figsize)

    ax = sns.boxplot(x="model", y="test_balanced_accuracy", data=results_df, ax=ax1[0]).set_title("balanced_accuracy")
    ax = sns.boxplot(x="model", y="test_precision", data=results_df, ax=ax1[1]).set_title("precision")
    ax = sns.boxplot(x="model", y="test_recall", data=results_df, ax=ax1[2]).set_title("recall")

    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_test", y="test_balanced_accuracy", hue="model", ax=ax2[0]).set_title(
        "balanced_accuracy")
    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_test", y="test_precision", hue="model", ax=ax2[1]).set_title(
        "precision")
    ax = sns.pointplot(data=results_df,
                       kind="point", x="season_test", y="test_recall", hue="model", ax=ax2[2]).set_title("recall")
    # fig.savefig(f"./plots/{exp_name}_exp.png")
    return results_df


def run_experiment(exp_name, models, folds, train_seasons, test_seasons, X, y, scale=False
                   , verbose=False):
    # Evaluate each model in turn
    results = []
    names = []
    print("Running experiment", exp_name)
    for name, model in models:
        cv_results = {
            "test_balanced_accuracy": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_roc_auc": [],
        }

        cv_results["season_train"] = []
        for train_idx, test_idx in folds:
            if scale:
                X[train_idx], X[test_idx] = utils.feature_scaling(X[train_idx], X[test_idx], 5)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()
            y_true = y_test
            fit_info = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            precision = precision_score(y_true, y_pred)
            cv_results["test_precision"].append(precision)

            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            cv_results["test_balanced_accuracy"].append(balanced_accuracy)

            recall = recall_score(y_true, y_pred, average='weighted')
            cv_results["test_recall"].append(recall)

            f1 = f1_score(y_true, y_pred, average='weighted')
            cv_results["test_f1"].append(f1)

            roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
            cv_results["test_roc_auc"].append(roc_auc)

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            "precision_mean": np.mean(cv_results["test_precision"]),
            "precision_std": np.std(cv_results["test_precision"]),
            "balanced_accuracy_mean": np.mean(cv_results["test_balanced_accuracy"]),
            "balanced_accuracy_std": np.std(cv_results["test_balanced_accuracy"]),
            "recall_mean": np.mean(cv_results["test_recall"]),
            "recall_std": np.std(cv_results["test_recall"]),
            "f1_mean": np.mean(cv_results["test_f1"]),
            "f1_std": np.std(cv_results["test_f1"]),
            "roc_auc_mean": np.mean(cv_results["test_roc_auc"]),
            "roc_auc_std": np.std(cv_results["test_roc_auc"]),
        }
        exp_results.append(exp_result)
        if verbose:
            print(f'{name}')
            print(
                f'balanced_accuracy: {np.mean(cv_results["test_balanced_accuracy"])}'
                f' - {np.std(cv_results["test_balanced_accuracy"])}')
            print(f'precision: {np.mean(cv_results["test_precision"])}'
                  f' - {np.std(cv_results["test_precision"])}')
            print(f'recall: {np.mean(cv_results["test_recall"])}'
                  f' - {np.std(cv_results["test_recall"])}')
            print(f'f1: {np.mean(cv_results["test_f1"])} - {np.std(cv_results["test_f1"])}')
            print(f'roc_auc: {np.mean(cv_results["test_roc_auc"])}'
                  f' - {np.std(cv_results["test_roc_auc"])}')
        else:
            print(
                f'{name}: balanced_accuracy: {np.mean(cv_results["test_balanced_accuracy"])}'
                f' - {np.std(cv_results["test_balanced_accuracy"])}')

        cv_results["model"] = [name] * len(folds)
        cv_results["season_train"] = train_seasons
        cv_results["season_test"] = test_seasons

        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results


if __name__ == '__main__':
    results_total = utils.deserialize_object("results_total")
    plot_to_compare_experiments(results_total)
