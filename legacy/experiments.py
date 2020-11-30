import pprint
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils as utils

pp = pprint.PrettyPrinter(width=41, compact=True)
exp_results = []
reg_exp_results = []
visualizers = []


def get_reg_models():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression, \
        SGDRegressor, \
        Lasso, Ridge, ElasticNet, LassoLars
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor


    models = []
    models.append(('KN', MultiOutputRegressor(KNeighborsRegressor(n_neighbors=20))))
    models.append(('SVM-rbf', MultiOutputRegressor(SVR(kernel='rbf'))))
    models.append(('SVM-linear', MultiOutputRegressor(SVR(kernel='linear'))))
    models.append(('DT', MultiOutputRegressor(DecisionTreeRegressor())))
    models.append(('LinearRegression', MultiOutputRegressor(LinearRegression())))
    models.append(('SGD', MultiOutputRegressor(SGDRegressor(random_state=0))))
    models.append(('Lasso', MultiOutputRegressor(Lasso(random_state=0))))
    models.append(('Ridge', MultiOutputRegressor(Ridge(random_state=0))))
    models.append(('ElasticNet', MultiOutputRegressor(ElasticNet(random_state=0))))
    models.append(('LassoLars', MultiOutputRegressor(LassoLars(random_state=0))))
    models.append(('RF', MultiOutputRegressor(RandomForestRegressor(random_state=0, n_estimators=200,
                                               max_depth=20,
                                               n_jobs=-1))))
    models.append(('GBR', MultiOutputRegressor(GradientBoostingRegressor(random_state=0))))
    models.append(('LGB', MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=200,
                                        max_depth=20))))
    models.append(('XGB', MultiOutputRegressor(XGBRegressor(random_state=0, n_estimators=200,
                                       max_depth=20))))
    return models


def get_clf_models():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import lightgbm as lgb
    import xgboost as xgb

    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))
    models.append(('SVM', SVC(kernel='linear', random_state=0,
                              C=63.513891775842986,
                              gamma=76.1465194934807,
                              degree=0.4300244876201068)))
    # models.append(('KSVM', SVC(kernel='rbf', random_state=0)))
    # models.append(('NB', GaussianNB()))
    # models.append(('DT', DecisionTreeClassifier(criterion='entropy', random_state=0)))
    # models.append(('SGD', SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)))
    models.append(("RF", RandomForestClassifier(n_estimators=200,
                                                max_depth=20,
                                                n_jobs=-1,
                                                random_state=0)))
    models.append(("GB", GradientBoostingClassifier(n_estimators=200,
                                                    max_depth=20,
                                                    random_state=0)))
    models.append(("XGB", xgb.XGBClassifier(
        random_state=0,
        max_depth=20,
        n_estimators=200
    )))

    models.append(("LGB", lgb.LGBMClassifier(
        random_state=0,
        max_depth=20,
        # objective='binary',
        # metric='binary_logloss',
        n_estimators=200,
        # num_leaves=300
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


def plot_to_compare_experiments(results_total, metric="balanced_accuracy", figsize=(25, 10), use_pointplot=False):
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

    ax = sns.boxplot(x="model",
                     y="balanced_accuracy",
                     data=results_df,
                     ax=ax1[0]).set_title("balanced_accuracy")
    ax = sns.boxplot(x="model",
                     y="precision",
                     data=results_df,
                     ax=ax1[1]).set_title("precision")
    ax = sns.boxplot(x="model",
                     y="recall",
                     data=results_df,
                     ax=ax1[2]).set_title("recall")

    ax = sns.pointplot(data=results_df,
                       kind="point",
                       x="season_test",
                       y="balanced_accuracy",
                       hue="model",
                       ax=ax2[0]).set_title(
        "balanced_accuracy")

    ax = sns.pointplot(data=results_df,
                       kind="point",
                       x="season_test",
                       y="precision",
                       hue="model", ax=ax2[1]).set_title(
        "precision")

    ax = sns.pointplot(data=results_df,
                       kind="point",
                       x="season_test",
                       y="recall",
                       hue="model",
                       ax=ax2[2]).set_title("recall")
    # fig.savefig(f"./plots/{exp_name}_exp.png")
    return results_df


def calculate_clf_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, \
        recall_score, \
        balanced_accuracy_score, \
        f1_score, \
        roc_auc_score
    cv_results = {}
    precision = precision_score(y_true, y_pred)
    cv_results["precision"] = precision

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cv_results["balanced_accuracy"] = balanced_accuracy

    recall = recall_score(y_true, y_pred, average='weighted')
    cv_results["recall"] = recall

    f1 = f1_score(y_true, y_pred, average='weighted')
    cv_results["f1"] = f1

    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    cv_results["roc_auc"] = roc_auc

    return cv_results


def calculate_reg_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, \
        mean_squared_error, mean_squared_log_error

    cv_results = {}
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    cv_results["mae"] = mae

    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    cv_results["mse"] = mse

    return cv_results


def print_exp_progress(result):
    pp.pprint(result)


def run_experiment(exp_name, models, folds, train_seasons, test_seasons, X, y,
                   preprocessor=None,
                   #print_exp_progress=None,
                   calculate_metrics_func=calculate_clf_metrics,
                   algorithm_type='clf'
                   ):
    results = []
    names = []
    print("Running experiment", exp_name)
    for name, current_model in models:
        cv_results = defaultdict(list)

        for train_idx, test_idx in folds:
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            #X_train, X_test = utils.scale_X(X, y, train_idx, test_idx)
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            y_true = y_test

            pipeline = Pipeline(steps=[
                                    ('preprocessor', preprocessor),
                                    ('model', current_model)])
            if algorithm_type == 'reg':
                model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
            else:
                model = pipeline
            fit_info = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_metric_results = calculate_metrics_func(y_true, y_pred)
            for key, value in fold_metric_results.items():
                cv_results[key].append(value)

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            **agg_metrics(cv_results.keys(), cv_results)
        }

        if algorithm_type == 'reg':
            reg_exp_results.append(exp_result)
        else:
            exp_results.append(exp_result)

        cv_results["model"] = [name] * len(folds)
        cv_results["season_train"] = train_seasons
        cv_results["season_test"] = test_seasons

        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results


def agg_metrics(metrics, results):
    metric_agg = {}
    for metric in metrics:
        metric_agg[metric + "_mean"] = np.mean(results[metric])
        metric_agg[metric + "_std"] = np.std(results[metric])
    return metric_agg


if __name__ == '__main__':
    import model.config as model_config
    import model.train as train

    df = utils.deserialize_object("df")

    exp_prefix = "reg"
    exp_group_name = "reg_exp"
    reg_results_total = []
    exp_results = []
    exp_X_columns = model_config.X_COLUMNS
    exp_y_columns = model_config.Y_COLUMNS[:-1]

    reg_models = get_reg_models()

    sscv = utils.SeasonSeriesSplit(df)
    df_sscv = sscv.get_df()
    X = df_sscv[exp_X_columns]
    y = df_sscv[exp_y_columns]

    experiment_name = f"{exp_prefix}scaled_data"

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('numerical', num_pipeline, model_config.X_NUM_COLS)
    ], remainder='passthrough')
    # transformed_data = preprocessor.fit_transform(df)

    folds, train_seasons, test_seasons = sscv.split(train_size=1, test_size=1)
    params = (experiment_name, reg_models, folds, train_seasons, test_seasons, X, y, preprocessor
              , calculate_reg_metrics, 'reg'
              )
    names, results = run_experiment(*params)
    #results_total.append((experiment_name, results))

    # exp_prefix = "reg"
    # exp_group_name = "reg_exp"
    # reg_results_total = []
    # exp_results = []
    # exp_X_columns = model_config.X_COLUMNS
    # exp_y_columns = model_config.Y_COLUMNS[:-1]
    # sscv = utils.SeasonSeriesSplit(data_processed)
    # reg_models = get_reg_models()
    # experiment_name = f"{exp_prefix}1_season"
    # folds, train_seasons, test_seasons = sscv.split(train_size=1, test_size=1)
    # X, y = train.X_y_values(df, exp_X_columns, exp_y_columns)
    #
    # params = (
    # experiment_name, reg_models, folds, train_seasons, test_seasons, X, y, StandardScaler(), None
    # ,calculate_reg_metrics, False)
    # names, results = run_experiment(*params)

