from nba_api.stats.endpoints import leaguegamelog
import model.dataset.data as data
import model.dataset.config as config

import pandas as pd
import numpy as np


def create_calculate_fields(df):
    fields = ['W_L_CUM_HOME',
              'W_L_L10_CUM_HOME',
              'FGM_ML5_HOME',
              'FGA_ML5_HOME',
              'FG_PCT_ML5_HOME',
              'FG3M_ML5_HOME',
              'FG3A_ML5_HOME',
              'FG3_PCT_ML5_HOME',
              'FTM_ML5_HOME',
              'FTA_ML5_HOME',
              'FT_PCT_ML5_HOME',
              'OREB_ML5_HOME',
              'DREB_ML5_HOME',
              'REB_ML5_HOME',
              'AST_ML5_HOME',
              'STL_ML5_HOME',
              'BLK_ML5_HOME',
              'TOV_ML5_HOME',
              'PF_ML5_HOME',
              'PTS_ML5_HOME',
              'PLUS_MINUS_ML5_HOME',
              'W_L_ML5_HOME',
              'FGM_MEAN_HOME',
              'FGA_MEAN_HOME',
              'FG_PCT_MEAN_HOME',
              'FG3M_MEAN_HOME',
              'FG3A_MEAN_HOME',
              'FG3_PCT_MEAN_HOME',
              'FTM_MEAN_HOME',
              'FTA_MEAN_HOME',
              'FT_PCT_MEAN_HOME',
              'OREB_MEAN_HOME',
              'DREB_MEAN_HOME',
              'REB_MEAN_HOME',
              'AST_MEAN_HOME',
              'STL_MEAN_HOME',
              'BLK_MEAN_HOME',
              'TOV_MEAN_HOME',
              'PF_MEAN_HOME',
              'PTS_MEAN_HOME',
              'PLUS_MINUS_MEAN_HOME',
              'FGM_L10_HOME',
              'FGA_L10_HOME',
              'FG_PCT_L10_HOME',
              'FG3M_L10_HOME',
              'FG3A_L10_HOME',
              'FG3_PCT_L10_HOME',
              'FTM_L10_HOME',
              'FTA_L10_HOME',
              'FT_PCT_L10_HOME',
              'OREB_L10_HOME',
              'DREB_L10_HOME',
              'REB_L10_HOME',
              'AST_L10_HOME',
              'STL_L10_HOME',
              'BLK_L10_HOME',
              'TOV_L10_HOME',
              'PF_L10_HOME',
              'PTS_L10_HOME',
              'PLUS_MINUS_L10_HOME']

    for field in fields:
        df[f'HOME_{field[:-5]}'] = df[field] - df[f'{field[:-5]}_AWAY']


def matchup_field_by_id(row, df):
    game_df = df.loc[row.name]
    return '-'.join(map(str, sorted(game_df.TEAM_ID.values.tolist())))


def create_raw_season_games_df():
    print("Create raw season games dataset. ")
    seasons = data.load_seasons().SEASON.unique()
    print("seasons: ", len(seasons))
    print(seasons)
    raw_season_games = pd.DataFrame()
    for season in seasons:
        season_games = leaguegamelog.LeagueGameLog(season_type_all_star="Regular Season"
                                                   , season=season).get_data_frames()[0] \
            .set_index('GAME_ID').sort_values(by=['GAME_DATE'])

        season_games.dropna(inplace=True)
        season_games.drop(columns=['VIDEO_AVAILABLE'], axis=1, inplace=True)

        season_games["W_L"] = np.where(season_games['WL'] == 'W', 1, -1)

        season_games_sum = season_games.groupby(by=["TEAM_ID"])[['W_L']] \
            .expanding().sum().reset_index(level=0)
        season_games = pd.merge(season_games, season_games_sum, suffixes=['', '_CUM'],
                                on=['GAME_ID', 'TEAM_ID'])

        season_games_l10_sum = season_games.groupby(by=["TEAM_ID"])[['W_L']] \
            .rolling(window=10, min_periods=1).sum().reset_index(level=0)

        season_games = pd.merge(season_games, season_games_l10_sum, suffixes=['', '_L10_CUM'],
                                on=['GAME_ID', 'TEAM_ID'])

        raw_season_games = pd.concat([raw_season_games, season_games])

    raw_season_games["UNIQUE_MATCHUP"] = raw_season_games.apply(lambda row: matchup_field_by_id(row, raw_season_games),
                                                                axis=1)

    matchup_season_games_mean = raw_season_games.groupby(by=["TEAM_ID", "UNIQUE_MATCHUP"])[
        ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
         'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
        .rolling(window=5, min_periods=1).mean().reset_index(level=0).reset_index(level=0)
    raw_season_games = pd.merge(raw_season_games, matchup_season_games_mean, suffixes=['', '_ML5'],
                                on=['GAME_ID', 'TEAM_ID', 'UNIQUE_MATCHUP'])

    matchup_season_games_w_l_cum = raw_season_games.groupby(by=["TEAM_ID", "UNIQUE_MATCHUP"])[['W_L']] \
        .rolling(window=5, min_periods=1).sum().reset_index(level=0).reset_index(level=0)
    raw_season_games = pd.merge(raw_season_games, matchup_season_games_w_l_cum, suffixes=['', '_ML5'],
                                on=['GAME_ID', 'TEAM_ID', 'UNIQUE_MATCHUP'])

    raw_season_games = raw_season_games.T.drop_duplicates().T
    raw_season_games.reset_index(inplace=True)
    raw_season_games.to_feather(config.RAW_SEASON_GAMES_DS)
    raw_season_games.to_csv(config.RAW_SEASON_GAMES_DS_CSV)
    print("Process done")


def create_season_game_df(raw_season_games):
    print("Create season games dataset. ")
    rankings = data.load_rankings()
    seasons = data.load_seasons().SEASON_YEAR.unique()
    season_games = pd.DataFrame()

    for season in seasons:
        next_season = raw_season_games[raw_season_games.SEASON_ID.str[-4:] == str(season)]

        season_games_mean = next_season.groupby(by=["TEAM_ID"])[
            ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
             'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
            .expanding().mean().reset_index(level=0)
        next_season = pd.merge(next_season, season_games_mean, suffixes=['', '_MEAN'], on=['GAME_ID', 'TEAM_ID'])

        season_l10_games_mean = next_season.groupby(by=["TEAM_ID"])[
            ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
             'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
            .rolling(window=10, min_periods=1).mean().reset_index(level=0)
        next_season = pd.merge(next_season, season_l10_games_mean, suffixes=['', '_L10'], on=['GAME_ID', 'TEAM_ID'])

        season_home_rows = next_season[next_season.MATCHUP.str.contains('vs.')]
        season_away_rows = next_season[next_season.MATCHUP.str.contains('@')]

        # Join every row to all others with the same game ID.
        joined = pd.merge(season_home_rows, season_away_rows, suffixes=['_HOME', '_AWAY'],
                          on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
        # Filter out any row that is joined to itself.
        result = joined[joined.TEAM_ID_HOME != joined.TEAM_ID_AWAY]
        season_games = pd.concat([season_games, result])

    season_games = season_games.T.drop_duplicates().T
    season_games["HOME_WINS"] = np.where(season_games['WL_HOME'] == 'W', 1, 0)
    season_games["HOME_POINT_SPREAD"] = season_games['PTS_HOME'] - season_games['PTS_AWAY']
    season_games["SEASON"] = season_games.SEASON_ID.str[-4:].astype(int)
    season_games["GAME_DATE_EST"] = season_games.GAME_DATE
    # season_games['RANKING_HOME'] = season_games.apply(lambda row: calculate_ranking(row, 'HOME'),
    #                                                   axis=1)
    # season_games['RANKING_AWAY'] = season_games.apply(lambda row: calculate_ranking(row, 'AWAY'),
    #                                                   axis=1)
    #
    # season_games['HOME_RANKING'] = season_games.apply(lambda row: calculate_ranking(row, 'BOTH'),
    #                                                   axis=1)
    season_games.reset_index(inplace=True)
    season_games.to_feather(config.SEASON_GAMES_DS)
    season_games.to_csv(config.SEASON_GAMES_DS_CSV)
    print("Process done")


def load_season_games_dataset():
    season_games = pd.read_feather(config.SEASON_GAMES_DS)
    season_games['GAME_DATE'] = pd.to_datetime(season_games['GAME_DATE'])
    season_games['GAME_DATE_EST'] = pd.to_datetime(season_games['GAME_DATE_EST'])
    season_games.set_index(["GAME_ID"], inplace=True)
    season_games.sort_values(by=['GAME_DATE', 'GAME_ID'], inplace=True)
    return season_games


def load_raw_season_games_dataset():
    raw_season_games = pd.read_feather(config.RAW_SEASON_GAMES_DS)
    raw_season_games.set_index(["GAME_ID"], inplace=True)
    raw_season_games.sort_values(by=['GAME_DATE', 'GAME_ID'], inplace=True)
    return raw_season_games


if __name__ == '__main__':
    # create_raw_season_games_df()
    create_season_game_df(load_raw_season_games_dataset())
