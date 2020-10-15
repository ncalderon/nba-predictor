#!/usr/bin/env python
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pandas import DataFrame

import model.dataset.config as config
import model.config as model_config

def save_results(name, results):
    experiments_file = open(f"{config.DATA_PATH}{name}_experiment_results.pkl", "wb")
    pickle.dump(results, experiments_file)
    experiments_file.close()


class SeasonSeriesSplit:
    df: DataFrame
    season_quarters = []

    def __init__(self, df_input):
        self.df = df_input.reset_index()

    def __quarter_split(self, skip=[]):
        self.season_quarters = []
        df_gr = self.df.groupby("SEASON")
        quarters = [0.25, 0.50, 0.75, 1]
        for season, group in df_gr:
            season_size = len(group)
            start = 0
            end = 0
            for q in quarters:
                if q in skip:
                    continue
                start = end
                end = int(season_size * q)
                self.season_quarters.append(
                    (group[start:end].index.values, season + q - 0.25)
                )

    def get_df(self):
        return self.df

    def split(self, train_size=1, test_size=1):
        test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
        seasons = self.df.SEASON.unique()
        folds = []
        train_seasons = []
        test_seasons = []
        for i in range(0, len(seasons), test_size):
            train_to = i + train_size
            test_to = train_to + test_size
            train_df = self.df[self.df.SEASON.isin(seasons[i:train_to])]
            test_df = self.df[self.df.SEASON.isin(seasons[train_to:test_to])]
            if len(test_df) <= 0:
                continue

            train_seasons.append("-".join(map(lambda x: str(x)[2:], seasons[i:train_to])))
            test_seasons.append("-".join(map(lambda x: str(x)[2:], seasons[train_to:test_to])))

            folds.append(
                (train_df.index.values,
                 test_df.index.values
                 ))
        return folds, train_seasons, test_seasons

    def quarter_split(self, train_size=3, test_size=1, skip=[]):
        test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
        if skip:
            self.__quarter_split(skip)
        else:
            self.__quarter_split()
        folds = []
        train_seasons = []
        test_seasons = []
        q_split = len(self.season_quarters)
        start = -1
        end = 0
        while q_split >= train_size + test_size:
            start += 1
            end = start + train_size
            start_test = end
            end_test = start_test + test_size
            q_split -= 1
            train_seasons.append("-".join(map(lambda x: str(x)[2:],
                                              [x1 for x, x1 in self.season_quarters[start:end]]
                                              )))
            test_seasons.append("-".join(map(lambda x: str(x)[2:],
                                             [x1 for x, x1 in self.season_quarters[start_test:end_test]]
                                             )))
            folds.append(
                (
                    [y for x, x1 in self.season_quarters[start:end] for y in x]
                    , [y for x, x1 in self.season_quarters[start_test:end_test] for y in x]
                )
            )

        return folds, train_seasons, test_seasons


def feature_scaling(X_train, X_test, start):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, start:] = sc.fit_transform(X_train[:, start:])
    X_test[:, start:] = sc.transform(X_test[:, start:])
    return X_train, X_test


def scale_X(X, y, train_idx, test_idx):
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X[model_config.X_NUM_COLS].loc[train_idx], y.loc[train_idx])
    X_ordinal_vals = X[model_config.X_ORDINAL_COLS].loc[train_idx].values
    X_train_transformed = np.concatenate((X_transformed, X_ordinal_vals), axis=1)

    X_test_transformed = sc.transform(X[model_config.X_NUM_COLS].loc[test_idx])
    X_test_ordinal_vals = X[model_config.X_ORDINAL_COLS].loc[test_idx].values
    X_test_transformed = np.concatenate((X_test_transformed, X_test_ordinal_vals), axis=1)
    return X_train_transformed, X_test_transformed


def scale_Y(X, train_idx, test_idx):
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X[model_config.X_NUM_COLS].loc[train_idx])
    X_ordinal_vals = X[model_config.X_ORDINAL_COLS].loc[train_idx].values
    X_train_transformed = np.concatenate((X_transformed, X_ordinal_vals), axis=1)

    X_test_transformed = sc.transform(X[model_config.X_NUM_COLS].loc[test_idx])
    X_test_ordinal_vals = X[model_config.X_ORDINAL_COLS].loc[test_idx].values
    X_test_transformed = np.concatenate((X_test_transformed, X_test_ordinal_vals), axis=1)
    return X_train_transformed, X_test_transformed

def serialize_object(filename, obj):
    pickle.dump(obj, open(f"data/{filename}.p", "wb"))


def deserialize_object(filename):
    return pickle.load(open(f"data/{filename}.p", "rb"))


if __name__ == '__main__':
    df = deserialize_object("df")
    sscv = SeasonSeriesSplit(df)
    folds, train_seasons, test_seasons = sscv.split(train_size=1, test_size=1)
    pass
