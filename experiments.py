from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils as utils
from yellowbrick.classifier import classification_report

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
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)))
    models.append(('SVM', SVC(kernel='linear', random_state=0)))
    models.append(('KSVM', SVC(kernel='rbf', random_state=0)))
    models.append(('NB', GaussianNB()))
    models.append(('DT', DecisionTreeClassifier(criterion='entropy', random_state=0)))
    models.append(('SGD', SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)))
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
        max_depth=5, n_estimators=200, random_state=0
    )))

    models.append(("LGB", lgb.LGBMClassifier(
        max_depth=5, n_estimators=200, random_state=0, min_data_in_leaf=80, num_leaves=32
    )))
    return models


def plot_experiment_results(exp_name, results, figsize=(25, 10)):
    results_df = pd.DataFrame(results[0])
    for idx, result in enumerate(results[1:]):
        result_df = pd.DataFrame(result)
        results_df = pd.concat([results_df, result_df], ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=figsize)

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
    fig.savefig(f"./plots/{exp_name}_exp.png")
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


def run_experiment(exp_name, df, models, tscv, train_splits, X, y, scale=False, custom_tscv=(False, [0])):
    if type(train_splits) is tuple:
        result_size = train_splits[0]
        seasons = train_splits[1]
    else:
        result_size = train_splits
        seasons = list(df.SEASON.unique()[-train_splits:])

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
        if custom_tscv[0]:
            tscv_list = tscv
        else:
            tscv_list = list(tscv.split(X=X))

        for train_index, test_index in tscv_list:
            if scale:
                X[train_index], X[test_index] = utils.feature_scaling(X[train_index], X[test_index], 5)

            visualizer = classification_report(
                model, X[train_index], y[train_index].ravel(), X[test_index], y[test_index].ravel(), classes=[0, 1],
                support=True
            )
            fit_info = model.fit(X[train_index], y[train_index].ravel())
            predictions = model.predict(X[test_index])
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
        cv_results["model"] = [name] * result_size
        cv_results["season_train"] = [season + q for season in seasons for q in custom_tscv[1]]

        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results
