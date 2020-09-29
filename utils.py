#!/usr/bin/env python
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import model.train as train
import model.dataset.config as config
import model.config as model_config
import model.dataset.game_matchup as gm


def do_experiment(experiment):
    train_test_split()
    feature_scaling()
    experiment()


def do_logistic_regression():
    model_logistic_regression()
    print_precission_logistic_regression()


def train_test_split(train_size=0.75):
    global X_train, X_test, y_train, y_test
    train_size_qty = int(len(X) * train_size)
    X_train, X_test, y_train, y_test = X[0:train_size_qty], \
                                       X[train_size_qty:len(X)], \
                                       y[0:train_size_qty], \
                                       y[train_size_qty:len(X)]
    return X_train, X_test, y_train, y_test


def train_test_series_split(df, test_length, split_by_quarter=False):
    train_start_idx = df.index[0]
    train_end_idx = df.index[-test_length] + 1
    test_idx_from = train_end_idx
    test_start_idx = train_end_idx
    if (split_by_quarter):
        test_end_idx = test_idx_from + test_length
        return X[train_start_idx:train_end_idx], \
               y[train_start_idx:train_end_idx], \
               X[test_start_idx:test_end_idx], \
               y[test_start_idx:test_end_idx]

    test_end_idx = test_idx_from + int(test_length * 0.25)

    yield X[train_start_idx:train_end_idx], \
          y[train_start_idx:train_end_idx], \
          X[test_start_idx:test_end_idx], \
          y[test_start_idx:test_end_idx]

    for test_size in [0.50, 0.75, 1]:
        train_end_idx = test_end_idx
        test_start_idx = train_end_idx
        test_end_idx = test_idx_from + int(test_length * test_size)
        yield X[train_start_idx:train_end_idx], \
              y[train_start_idx:train_end_idx], \
              X[test_start_idx:test_end_idx], \
              y[test_start_idx:test_end_idx]


def ds_split(df, train_season_size=2):
    seasons = df.SEASON.unique()[:-train_season_size]
    for i in range(0, len(seasons)):
        season = seasons[i]
        current_df = df[df.SEASON.isin(seasons[i:train_season_size + 1 + i])]
        test_length = len(df[df.SEASON == (season + train_season_size)])
        yield f"{season}-{season + 1}", current_df, test_length


def save_results(name, results):
    experiments_file = open(f"{config.DATA_PATH}{name}_experiment_results.pkl", "wb")
    pickle.dump(results, experiments_file)
    experiments_file.close()


def do_experiments(exp_name, models, df, train_season_size=2, split_by_quarter=False):
    print(f"Start processing experiment: {exp_name}...")
    from sklearn.metrics import precision_score, balanced_accuracy_score
    global X, y
    X, y = train.X_y_values(df=df, y_columns=model_config.y_columns[-1])
    df.reset_index(inplace=True)
    # results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
    results = {}
    for name, model in models:
        results[name] = {}
        for name_ds, ds, test_length in ds_split(df, train_season_size=train_season_size):
            for X_train, y_train, X_test, y_test in train_test_series_split(ds, test_length, split_by_quarter):
                model.fit(X=X_train, y=y_train.ravel())
                y_pred = model.predict(X_test)
                result = {
                    "balanced_accuracy_score": 100 * balanced_accuracy_score(y_test, y_pred),
                    "precision": 100 * precision_score(y_test, y_pred),
                    "recall": 100 * precision_score(y_test, y_pred)
                }
                if name_ds not in results[name]:
                    results[name][name_ds] = []
                results[name][name_ds].append(result)
                results[name][name_ds].append(result)
                results[name][name_ds].append(result)
        print(f"model: {name}. Results: {results[name]}")
        save_results(name, results)
    print(results)
    print(f"Saving results...")
    save_results(exp_name, results)
    print(f"Done.")
    return results


# class CustomTimeSeriesSplit:
#     df: DataFrame
# 
#     def __init__(self, df_input):
#         global df
#         df = df_input.reset_index()
# 
#     def split(self, train_size=1, quarters=[0.25, 0.50, 0.75, 1]):
#         global df
#         test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
#         seasons = df.SEASON.unique()[:-1]
#         for i in range(0, len(seasons)):
#             season = seasons[i]
#             current_df = df[df.SEASON.isin(seasons[i:train_size + i])]
#             n_games_next_season = len(df[df.SEASON == (season + train_size)])
# 
#             train_start_idx = current_df.index[0]
#             train_end_idx = current_df.index[-1] + 1
#             test_idx_from = train_end_idx
#             test_start_idx = train_end_idx
#             test_end_idx = test_idx_from + int(n_games_next_season * quarters[0])
#             yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx,
#                                                                                   dtype=int)
#             for test_size in quarters[1:]:
#                 train_end_idx = test_end_idx
#                 test_start_idx = train_end_idx
#                 test_end_idx = test_idx_from + int(n_games_next_season * test_size)
#                 yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx,
#                                                                                       dtype=int)

class SeasonTimeSeriesSplit:
    df: DataFrame

    def __init__(self, df_input):
        global df
        df = df_input.reset_index()

    def split(self, train_size=1, test_size=1):
        global df
        test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
        seasons = df.SEASON.unique()[:-1]
        for season, idx in seasons:
            season_df = df[df.SEASON.isin(seasons[i:train_size + i])]
        for i in range(0, len(seasons)):
            season = seasons[i]
            current_df = df[df.SEASON.isin(seasons[i:train_size + i])]
            n_games_next_season = len(df[df.SEASON == (season + train_size)])

            train_start_idx = current_df.index[0]
            train_end_idx = current_df.index[-1] + 1
            test_idx_from = train_end_idx
            test_start_idx = train_end_idx
            test_end_idx = test_idx_from + int(n_games_next_season * quarters[0])
            yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx,
                                                                                  dtype=int)
            for test_size in quarters[1:]:
                train_end_idx = test_end_idx
                test_start_idx = train_end_idx
                test_end_idx = test_idx_from + int(n_games_next_season * test_size)
                yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx,
                                                                                      dtype=int)




def feature_scaling(X_train, X_test, start):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, start:] = sc.fit_transform(X_train[:, start:])
    X_test[:, start:] = sc.transform(X_test[:, start:])
    return X_train, X_test


def model_logistic_regression():
    global log_clf
    from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import cross_val_score
    log_clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    log_clf.fit(X_train, y_train.ravel())
    # print("test")
    # score = cross_val_score(log_clf, X=X_train, y=y_train, cv=3)
    # print(score.mean())


def print_precission_logistic_regression():
    from sklearn.metrics import precision_score, recall_score
    y_pred = log_clf.predict(X_test)

    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
    # print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


def execute_experiments(df):
    models = [
        # ('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
        # ('SVM', SVC(kernel='linear', random_state=0)),
        # ('KSVM', SVC(kernel='rbf', random_state=0)),
        ('NB', GaussianNB()),
        ('DT', DecisionTreeClassifier(criterion='entropy', random_state=0)),
        ("RF", RandomForestClassifier(n_estimators=500,
                                      max_features="sqrt",
                                      max_depth=15,
                                      n_jobs=-1,
                                      random_state=0)),
        # ("GB", GradientBoostingClassifier(n_estimators=500,
        #                                 max_depth=15,
        #                                 max_features="sqrt",
        #                                 random_state=0))
    ]
    do_experiments("2season_predict1", models, df=df)

    do_experiments("1season_and_q_predict_q", models, df=df, train_season_size=1, split_by_quarter=True)


def load_experiment_results():
    season2_predict1 = open("data/2season_predict1_experiment_results.pkl", 'rb')
    season2_predict1_results = pickle.load(season2_predict1)
    season1_and_q_predict_q = open("data/1season_and_q_predict_q_experiment_results.pkl", 'rb')
    season1_and_q_predict_q_results = pickle.load(season1_and_q_predict_q)
    return season2_predict1_results, season1_and_q_predict_q_results


def serialize_object(filename, obj):
    pickle.dump(obj, open(f"data/{filename}.p", "wb"))


def deserialize_object(filename):
    return pickle.load(open(f"data/{filename}.p", "rb"))


if __name__ == '__main__':
    pass
