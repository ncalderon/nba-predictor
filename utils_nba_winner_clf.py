from collections import defaultdict

from sklearn.pipeline import Pipeline
import numpy as np
import utils
from boruta import BorutaPy
import model.config as conf

exp_results = []
visualizers = []


def get_clf_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    import lightgbm as lgb
    import xgboost as xgb

    models = [
        ('LR', LogisticRegression(random_state=0, max_iter=10000))
        , ('KNN', KNeighborsClassifier(n_neighbors=20)),
        ('DT', DecisionTreeClassifier(criterion='entropy', random_state=0)),
        ('SVM', SVC(kernel='linear', random_state=0)),
        ("RF", RandomForestClassifier(n_estimators=200,
                                      max_depth=20,
                                      n_jobs=-1,
                                      random_state=0)),
        ("XGB", xgb.XGBClassifier(
            random_state=0,
            max_depth=20,
            n_estimators=200
        )), ("LGB", lgb.LGBMClassifier(
            random_state=0,
            max_depth=20,
            n_estimators=200
        ))]

    return models


def calculate_clf_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, \
        balanced_accuracy_score, \
        roc_auc_score
    cv_results = {}
    precision = precision_score(y_true, y_pred)
    cv_results["precision"] = precision

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cv_results["balanced_accuracy"] = balanced_accuracy

    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    cv_results["roc_auc"] = roc_auc

    return cv_results


def feature_selection(model, X_train, y_train):
    boruta_selector = BorutaPy(model, n_estimators=200, random_state=0)
    boruta_selector.fit(np.array(X_train), np.array(y_train))
    boruta_ranking = boruta_selector.ranking_
    selected_features = np.array(conf.X_NUM_COLS)[boruta_ranking <= 2]
    return selected_features


def run_experiment(exp_name, models, folds, train_seasons, test_seasons, X, y,
                   preprocessor=None):
    results = []
    names = []
    print("Running experiment", exp_name)
    for name, current_model in models:
        cv_results = defaultdict(list)

        for train_idx, test_idx in folds:
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            y_true = y_test

            # feature_selection(model, X_train, y_train)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', current_model)])
            model = pipeline
            fit_info = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_metric_results = calculate_clf_metrics(y_true, y_pred)
            for key, value in fold_metric_results.items():
                cv_results[key].append(value)

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            **utils.agg_metrics(cv_results.keys(), cv_results)
        }

        # print(exp_result)

        exp_results.append(exp_result)

        cv_results["model"] = [name] * len(folds)
        cv_results["season_train"] = train_seasons
        cv_results["season_test"] = test_seasons

        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results
