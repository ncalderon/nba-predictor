#!/usr/bin/env python
from datetime import datetime

import numpy as np
import pandas as pd

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
    global df
    games, season_games, teams, seasons, rankings = games_matchup_d.load_datasets()
    df = season_games
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


def do_experiments(experiments=[do_logistic_regression]):
    for i, experiment in enumerate(experiments):
        print(f"Experiment: {experiment.__name__}")
        do_experiment(df, f"{idx + 1}", experiment)


if __name__ == '__main__':
    pass


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


def time_series_train_test_split():
    fromIndex, toIndex, fromIndex, testToIndex = 0, 0, 0, 0

    for season in df.SEASON.unique()[:-1]:
        n_games_season = len(df[df.SEASON == season])
        n_games_next_season = len(df[df.SEASON == season + 1])

        toIndex = fromIndex + n_games_season
        testFromIndex = toIndex
        testToIndex = testFromIndex + int(n_games_next_season * 0.25)

        yield np.arange(fromIndex, toIndex, dtype=int), np.arange(testFromIndex, testToIndex, dtype=int)

        for train_size in [0.50, 0.75]:
            toIndex = testToIndex
            testFromIndex = toIndex
            testToIndex = testFromIndex + (n_games_next_season - int(n_games_next_season * train_size))
            yield np.arange(fromIndex, toIndex, dtype=int), np.arange(testFromIndex, testToIndex, dtype=int)

        toIndex = testToIndex
        testFromIndex = toIndex
        testToIndex = testFromIndex + n_games_next_season
        yield np.arange(fromIndex, toIndex, dtype=int), np.arange(testFromIndex, testToIndex, dtype=int)

        fromIndex += n_games_season



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
