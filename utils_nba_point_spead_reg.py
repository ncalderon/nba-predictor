from collections import defaultdict

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils as utils

exp_results = []
visualizers = []


def get_reg_models():
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    models = [
        ('LinearRegression', LinearRegression()),
        ('KN', KNeighborsRegressor(n_neighbors=20)),
        ('SVM-rbf', SVR(kernel='rbf')),
        ('SVM-linear', SVR(kernel='linear')),
        ('RF', RandomForestRegressor(random_state=0, n_estimators=200,
                                     max_depth=20,
                                     n_jobs=-1)),
        ('LGB', LGBMRegressor(random_state=0, n_estimators=200,
                              max_depth=20)),
        ('XGB', XGBRegressor(random_state=0, n_estimators=200,
                             max_depth=20))]
    return models


def calculate_reg_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, \
        mean_squared_error
    from math import sqrt

    cv_results = {}
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    cv_results["mae"] = mae

    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    cv_results["mse"] = mse

    rmse = sqrt(mse)
    cv_results["rmse"] = rmse

    return cv_results


def run_experiment(exp_name, models, folds, train_seasons, test_seasons, X, y,
                   preprocessor=None, targetTransformer=None
                   ):
    results = []
    names = []
    print("Running experiment", exp_name)
    for name, current_model in models:
        cv_results = defaultdict(list)

        for train_idx, test_idx in folds:
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            # X_train, X_test = utils.scale_X(X, y, train_idx, test_idx)
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            y_true = y_test

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', current_model)])

            if preprocessor is None:
                model = pipeline
            else:
                model = TransformedTargetRegressor(regressor=pipeline, transformer=targetTransformer)

            fit_info = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_metric_results = calculate_reg_metrics(y_true, y_pred)
            for key, value in fold_metric_results.items():
                cv_results[key].append(value)

        exp_result = {
            "exp_name": exp_name,
            "model": name,
            **utils.agg_metrics(cv_results.keys(), cv_results)
        }

        exp_results.append(exp_result)

        cv_results["model"] = [name] * len(folds)
        cv_results["season_train"] = train_seasons
        cv_results["season_test"] = test_seasons

        results.append(cv_results)
        names.append(name)
    print("Done")
    return names, results


if __name__ == '__main__':
    import model.config as model_config

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
    # results_total.append((experiment_name, results))

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
