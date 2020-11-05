#!/usr/bin/env python
import pickle
import pprint

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

import model.dataset.config as config

pp = pprint.PrettyPrinter(width=41, compact=True)


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


def serialize_object(filename, obj):
    pickle.dump(obj, open(f"data/{filename}.p", "wb"))


def deserialize_object(filename, default=None):
    try:
        return pickle.load(open(f"data/{filename}.p", "rb"))
    except FileNotFoundError:
        return default


def save_results(name, results):
    experiments_file = open(f"{config.DATA_PATH}{name}_experiment_results.pkl", "wb")
    pickle.dump(results, experiments_file)
    experiments_file.close()


def map_results_to_df(results):
    results_df = pd.DataFrame(results[0])
    for idx, result in enumerate(results[1:]):
        result_df = pd.DataFrame(result)
        results_df = pd.concat([results_df, result_df], ignore_index=True)
    return results_df


def agg_metrics(metrics, results):
    metric_agg = {}
    for metric in metrics:
        metric_agg[metric + "_mean"] = np.mean(results[metric])
        metric_agg[metric + "_std"] = np.std(results[metric])
    return metric_agg


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
                a = sns.lineplot(data=results_df,
                                 x="season_test", y=metric, hue="model", style="model",
                                 markers=True, dashes=False,
                                  ax=ax_row)
            else:
                a = sns.boxplot(x="model", y=metric, data=results_df, ax=ax_row)
            a.set_xlabel(None)
            a.set_title(result[0])
            a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            idx += 1


def print_exp_progress(result):
    pp.pprint(result)


if __name__ == '__main__':
    df = deserialize_object("df")
    sscv = SeasonSeriesSplit(df)
    folds, train_seasons, test_seasons = sscv.split(train_size=1, test_size=1)
    pass
