#!/usr/bin/env python
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import model.dataset.game_matchup as games_matchup_d

DATA_PATH = 'data'

SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

TEAMS_DS = f"{DATA_PATH}/teams.csv"
TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

RANKING_DS = f"{DATA_PATH}/ranking.csv"
RANKING_PROCESSED_DS = f"{DATA_PATH}/ranking.processed.feather"

GAMES_DS = f"{DATA_PATH}/games.csv"
GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"

columns = [
    "GAME_DATE_EST",
    "HOME_TEAM_NAME",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_NAME",
    "VISITOR_TEAM_ID",
    "GAME_STATUS_TEXT",
    "SEASON",
    "HT_RANK",
    "HT_CLASS",
    "HT_HW",
    "HT_HL",
    "HT_VW",
    "HT_VL",
    "HT_LAST10_W",
    "HT_LAST10_L",
    "HT_LAST10_MATCHUP_W",
    "HT_LAST10_MATCHUP_L",
    "HT_OVERALL_OFF_POINTS",
    "HT_OVERALL_DEF_POINTS",
    "HT_OVERALL_OFF_FG",
    "HT_OVERALL_DEF_FG",
    "HT_OVERALL_OFF_3P",
    "HT_OVERALL_DEF_3P",
    "HT_OVERALL_OFF_FT",
    "HT_OVERALL_DEF_FT",
    "HT_OVERALL_OFF_REB",
    "HT_OVERALL_DEF_REB",
    "HT_AWAY_POINTS",
    "HT_AWAY_FG",
    "HT_AWAY_3P",
    "HT_AWAY_FT",
    "HT_AWAY_REB",
    "VT_RANK",
    "VT_CLASS",
    "VT_HW",
    "VT_HL",
    "VT_VW",
    "VT_VL",
    "VT_LAST10_W",
    "VT_LAST10_L",
    "VT_LAST10_MATCHUP_W",
    "VT_LAST10_MATCHUP_L",
    "VT_OVERALL_OFF_POINTS",
    "VT_OVERALL_DEF_POINTS",
    "VT_OVERALL_OFF_FG",
    "VT_OVERALL_DEF_FG",
    "VT_OVERALL_OFF_3P",
    "VT_OVERALL_DEF_3P",
    "VT_OVERALL_OFF_FT",
    "VT_OVERALL_DEF_FT",
    "VT_OVERALL_OFF_REB",
    "VT_OVERALL_DEF_REB",
    "VT_AWAY_POINTS",
    "VT_AWAY_FG",
    "VT_AWAY_3P",
    "VT_AWAY_FT",
    "VT_AWAY_REB",
    "PTS_home",
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home",
    "AST_home",
    "REB_home",
    "PTS_away",
    "FG_PCT_away",
    "FT_PCT_away",
    "FG3_PCT_away",
    "AST_away",
    "REB_away",
    "HOME_TEAM_WINS"
]

X_columns = [
    "SEASON",
    "HT_RANK",
    "HT_CLASS",
    "HT_HW",
    "HT_HL",
    "HT_VW",
    "HT_VL",
    "HT_LAST10_W",
    "HT_LAST10_L",
    "HT_LAST10_MATCHUP_W",
    "HT_LAST10_MATCHUP_L",
    "HT_OVERALL_OFF_POINTS",
    "HT_OVERALL_DEF_POINTS",
    "HT_OVERALL_OFF_FG",
    "HT_OVERALL_DEF_FG",
    "HT_OVERALL_OFF_3P",
    "HT_OVERALL_DEF_3P",
    "HT_OVERALL_OFF_FT",
    "HT_OVERALL_DEF_FT",
    "HT_OVERALL_OFF_REB",
    "HT_OVERALL_DEF_REB",
    "HT_AWAY_POINTS",
    "HT_AWAY_FG",
    "HT_AWAY_3P",
    "HT_AWAY_FT",
    "HT_AWAY_REB",
    "VT_RANK",
    "VT_CLASS",
    "VT_HW",
    "VT_HL",
    "VT_VW",
    "VT_VL",
    "VT_LAST10_W",
    "VT_LAST10_L",
    "VT_LAST10_MATCHUP_W",
    "VT_LAST10_MATCHUP_L",
    "VT_OVERALL_OFF_POINTS",
    "VT_OVERALL_DEF_POINTS",
    "VT_OVERALL_OFF_FG",
    "VT_OVERALL_DEF_FG",
    "VT_OVERALL_OFF_3P",
    "VT_OVERALL_DEF_3P",
    "VT_OVERALL_OFF_FT",
    "VT_OVERALL_DEF_FT",
    "VT_OVERALL_OFF_REB",
    "VT_OVERALL_DEF_REB",
    "VT_AWAY_POINTS",
    "VT_AWAY_FG",
    "VT_AWAY_3P",
    "VT_AWAY_FT",
    "VT_AWAY_REB",
    "PTS_home",
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home"
]

y_columns = [
    "PTS_home",
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home",
    "AST_home",
    "REB_home",
    "PTS_away",
    "FG_PCT_away",
    "FT_PCT_away",
    "FG3_PCT_away",
    "AST_away",
    "REB_away",
    "HOME_TEAM_WINS"
]


def load_df():
    games, season_games, teams, seasons, rankings = games_matchup_d.load_datasets()
    games_matchup = pd.read_feather(GAMES_PROCESSED_DS)
    games_matchup = games_matchup.set_index(["GAME_ID"])
    games_matchup = games_matchup.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])
    return games, season_games, teams, seasons, rankings, games_matchup


def load_experiment_dataset(ds_path):
    df = pd.read_feather(ds_path)
    df.set_index(["GAME_DATE_EST"], inplace=True)
    df.sort_index(inplace=True)
    return df


def do_experiment(df, experiment):
    X_y_values(df)
    train_test_split()
    feature_scaling()
    experiment()


def do_logistic_regression():
    model_logistic_regression()
    print_precission_logistic_regression()


def X_y_values(df):
    global X, y
    X = df.loc[:, X_columns].values
    y = df.loc[:, ["HOME_TEAM_WINS"]].values


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
    experiments_file = open(f"data/{name}_experiment_results.pkl", "wb")
    pickle.dump(results, experiments_file)
    experiments_file.close()


def do_experiments(name, models, df, train_season_size=2, split_by_quarter=False):
    import collections
    from sklearn.metrics import precision_score, balanced_accuracy_score
    X_y_values(df)
    df.reset_index(inplace=True)
    #results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
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
    save_results(name, results)
    return results


def time_series_cv_split(df: DataFrame, train_size=1):
    df = df.reset_index()
    test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
    seasons = df.SEASON.unique()[:-1]
    for i in range(0, len(seasons)):
        season = seasons[i]
        current_df = df[df.SEASON.isin(seasons[i:train_size + i])]
        n_games_next_season = len(df[df.SEASON == (season + train_size)])

        train_start_idx = current_df.index[0]
        train_end_idx = current_df.index[-1] + 1
        test_idx_from = train_end_idx
        test_start_idx = train_end_idx
        test_end_idx = test_idx_from + int(n_games_next_season * 0.25)
        yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx, dtype=int)

        for test_size in [0.50, 0.75, 1]:
            train_end_idx = test_end_idx
            test_start_idx = train_end_idx
            test_end_idx = test_idx_from + int(n_games_next_season * test_size)
            yield np.arange(train_start_idx, train_end_idx, dtype=int), np.arange(test_start_idx, test_end_idx,
                                                                                  dtype=int)


def feature_scaling():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, :] = sc.fit_transform(X_train[:, :])
    X_test[:, :] = sc.transform(X_test[:, :])
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


if __name__ == '__main__':
    games, season_games, teams, seasons, rankings, games_matchup = load_df()
    df = games_matchup
    df = df[df.SEASON.isin(df.SEASON.unique()[-10:])]
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
         #("GB", GradientBoostingClassifier(n_estimators=500,
         #                                 max_depth=15,
         #                                 max_features="sqrt",
         #                                 random_state=0))
    ]
    do_experiments("2season_predict1", models, df=df)

    do_experiments("2season_predict1", models, df=df)
