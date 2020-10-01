#!/usr/bin/env python
import pickle

from pandas import DataFrame

import model.dataset.config as config


def save_results(name, results):
    experiments_file = open(f"{config.DATA_PATH}{name}_experiment_results.pkl", "wb")
    pickle.dump(results, experiments_file)
    experiments_file.close()

class SeasonSeriesSplit:
    df: DataFrame
    season_quarters = []

    def __init__(self, df_input):
        self.df = df_input.reset_index()
        self.__quarter_split()

    def __quarter_split(self):
        df_gr = self.df.groupby("SEASON")
        quarters = [0.25, 0.50, 0.75, 1]
        for season, group in df_gr:
            season_size = len(group)
            start = 0
            end = 0
            for q in quarters:
                start = end
                end = int(season_size * q)
                self.season_quarters.append(
                    (group[start:end].index.values, season + q - 0.25)
                )

    def split(self, train_size=1, test_size=1):
        test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0
        seasons = self.df.SEASON.unique()[:-test_size]
        folds = []
        train_seasons = []
        test_seasons = []
        for i in range(0, len(seasons) - test_size, test_size):
            train_to = i + train_size
            test_to = train_to + test_size
            train_df = self.df[self.df.SEASON.isin(seasons[i:train_to])]
            test_df = self.df[self.df.SEASON.isin(seasons[train_to:test_to])]
            if len(test_df) <= 0:
                continue

            train_seasons.append("-".join(map(str, seasons[i:train_to])))
            test_seasons.append("-".join(map(str, seasons[train_to:test_to])))

            folds.append(
                (train_df.index.values,
                 test_df.index.values
                 ))
        return folds, train_seasons, test_seasons

    def quarter_split(self, train_size=3, test_size=1):
        test_idx_from, train_start_idx, train_end_idx, test_start_idx, test_end_idx = 0, 0, 0, 0, 0

        folds = []
        train_seasons = []
        test_seasons = []
        q_split = len(self.season_quarters)
        start = 0
        end = 0
        while q_split > 0:
            start = end
            end = start + train_size
            start_test = end
            end_test = start_test + test_size
            q_split -= train_size + test_size
            train_seasons.append("-".join(map(str,
                                              [x1 for x, x1 in self.season_quarters[start:end]]
                                              )))
            test_seasons.append("-".join(map(str,
                                             [x1 for x, x1 in self.season_quarters[start_test:end_test]]
                                             )))
            folds.append(
                (
                  [y for x, x1 in self.season_quarters[start:end] for y in x]
                  ,[y for x, x1 in self.season_quarters[start_test:end_test] for y in x]
                 )
            )

        return folds, train_seasons, test_seasons


def feature_scaling(X_train, X_test, start):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[:, start:] = sc.fit_transform(X_train[:, start:])
    X_test[:, start:] = sc.transform(X_test[:, start:])
    return X_train, X_test


def serialize_object(filename, obj):
    pickle.dump(obj, open(f"data/{filename}.p", "wb"))


def deserialize_object(filename):
    return pickle.load(open(f"data/{filename}.p", "rb"))


if __name__ == '__main__':
    df = deserialize_object("df")
    sscv = SeasonSeriesSplit(df)
    sscv.quarter_split(3, 1)
    sscv.quarter_split(3, 1)
    pass
