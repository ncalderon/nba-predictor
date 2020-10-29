from nba_api.stats.endpoints import leaguegamelog
import model.dataset.data as data
import model.dataset.config as config

import pandas as pd
import numpy as np


def matchup_field_by_id(row, df):
    game_df = df.loc[row.name]
    return '-'.join(map(str, sorted(game_df.TEAM_ID.values.tolist())))


def create_raw_season_games_df():
    print("Create raw season games dataset. ")
    seasons = data.load_seasons().SEASON.unique()
    raw_season_games = pd.DataFrame()
    for season in seasons:
        next_season = leaguegamelog.LeagueGameLog(season_type_all_star="Regular Season"
                                                  , season=season).get_data_frames()[0] \
            .set_index('GAME_ID').sort_values(by=['GAME_DATE'])
        next_season.dropna(inplace=True)
        next_season.drop(columns=['VIDEO_AVAILABLE'], axis=1, inplace=True)
        next_season["W_L"] = np.where(next_season['WL'] == 'W', 1, -1)
        season_games_sum = next_season.groupby(by=["TEAM_ID"])[['W_L']] \
            .expanding().sum().reset_index(level=0)
        next_season = pd.merge(next_season, season_games_sum, suffixes=['', '_CUM'],
                               on=['GAME_ID', 'TEAM_ID'])

        raw_season_games = pd.concat([raw_season_games, next_season])

    raw_season_games.reset_index(inplace=True)
    raw_season_games.to_feather(config.RAW_SEASON_GAMES_DS)
    raw_season_games.reset_index(inplace=True)
    raw_season_games.to_csv(config.RAW_SEASON_GAMES_DS_CSV)

def create_season_game_df():
    print("Create season games dataset. ")
    seasons = data.load_seasons().SEASON.unique()
    season_games = pd.DataFrame()
    raw_season_games = pd.DataFrame()
    for season in seasons:
        next_season = leaguegamelog.LeagueGameLog(season_type_all_star="Regular Season"
                                                  , season=season).get_data_frames()[0] \
            .set_index('GAME_ID').sort_values(by=['GAME_DATE'])

        # .sort_values(by=['SEASON_ID', 'GAME_DATE', 'GAME_ID'])
        next_season.dropna(inplace=True)
        next_season.drop(columns=['VIDEO_AVAILABLE'], axis=1, inplace=True)

        next_season["W_L"] = np.where(next_season['WL'] == 'W', 1, -1)
        season_games_sum = next_season.groupby(by=["TEAM_ID"])[['W_L']] \
            .expanding().sum().reset_index(level=0)
        next_season = pd.merge(next_season, season_games_sum, suffixes=['', '_CUM'],
                               on=['GAME_ID', 'TEAM_ID'])

        next_season["UNIQUE_MATCHUP"] = next_season.apply(lambda row: matchup_field_by_id(row, next_season),
                                                                axis=1)
        raw_season_games = pd.concat([raw_season_games, next_season])
        matchup_season_games_mean = raw_season_games.groupby(by=["TEAM_ID", "UNIQUE_MATCHUP"])[
             ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
              'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
             .rolling(window=5, min_periods=1).mean().reset_index(level=0).reset_index(level=0)
        #matchup_season_games_mean = matchup_season_games_mean[matchup_season_games_mean.index in next_season.index]
        next_season = pd.merge(next_season, matchup_season_games_mean, suffixes=['', '_MML5'],
                                on=['GAME_ID', 'TEAM_ID'])

        matchup_season_games_w_l_cum = raw_season_games.groupby(by=["TEAM_ID", "UNIQUE_MATCHUP"])[['W_L']]\
            .reset_index(level=0).reset_index(level=0)
        next_season = pd.merge(next_season, matchup_season_games_w_l_cum, suffixes=['', '_MML5'],
                               on=['GAME_ID', 'TEAM_ID'])

        season_games_mean = next_season.groupby(by=["TEAM_ID"])[
            ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
             'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
            .expanding().mean().reset_index(level=0)
        next_season = pd.merge(next_season, season_games_mean, suffixes=['', '_MEAN'], on=['GAME_ID', 'TEAM_ID'])

        next_season.groupby(by=["TEAM_ID"])[
            ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
             'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']] \
            .rolling(window=10, min_periods=1).median().reset_index(level=0)
        next_season = pd.merge(next_season, season_games_mean, suffixes=['', '_L10'], on=['GAME_ID', 'TEAM_ID'])

        matchup_season_games_w_l_cum = next_season.groupby(by=["TEAM_ID", "UNIQUE_MATCHUP"])[['W_L']] \
            .reset_index(level=0).reset_index(level=0)
        next_season = pd.merge(next_season, matchup_season_games_w_l_cum, suffixes=['', '_MML5'],
                               on=['GAME_ID', 'TEAM_ID'])


        season_home_rows = next_season[next_season.MATCHUP.str.contains('vs.')]
        season_away_rows = next_season[next_season.MATCHUP.str.contains('@')]
        # Join every row to all others with the same game ID.
        joined = pd.merge(season_home_rows, season_away_rows, suffixes=['_HOME', '_AWAY'],
                          on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
        # Filter out any row that is joined to itself.
        result = joined[joined.TEAM_ID_HOME != joined.TEAM_ID_AWAY]
        season_games = pd.concat([season_games, result])

    # season_games["MATCHUP"] = season_games.apply(lambda row:
    #                                              '-'.join(
    #                                                  sorted(
    #                                                      [row.TEAM_ABBREVIATION_HOME, row.TEAM_ABBREVIATION_AWAY])),
    #                                              axis=1)

    # TODO: PENDIENTE
    # l10m = season_games.groupby(by=["MATCHUP"])[
    #     ['MIN_HOME', 'FGM_HOME', 'FGA_HOME', 'FG_PCT_HOME', 'FG3M_HOME', 'FG3A_HOME', 'FG3_PCT_HOME', 'FTM_HOME',
    #      'FTA_HOME', 'FT_PCT_HOME', 'OREB_HOME', 'DREB_HOME', 'REB_HOME', 'AST_HOME', 'STL_HOME', 'BLK_HOME',
    #      'TOV_HOME', 'PF_HOME', 'PTS_HOME', 'PLUS_MINUS_HOME', 'MIN_AWAY', 'FGM_AWAY', 'FGA_AWAY', 'FG_PCT_AWAY',
    #      'FG3M_AWAY', 'FG3A_AWAY', 'FG3_PCT_AWAY', 'FTM_AWAY', 'FTA_AWAY', 'FT_PCT_AWAY', 'OREB_AWAY', 'DREB_AWAY',
    #      'REB_AWAY', 'AST_AWAY', 'STL_AWAY', 'BLK_AWAY', 'TOV_AWAY', 'PF_AWAY', 'PTS_AWAY', 'PLUS_MINUS_AWAY']] \
    #     .rolling(window=10, min_periods=1).median().reset_index(level=0)
    #
    # season_games = pd.merge(season_games, l10m, suffixes=['', '_L10M'], on=['GAME_ID'])

    season_games.reset_index(inplace=True)
    season_games.to_feather(config.SEASON_GAMES_DS)
    season_games.reset_index(inplace=True)
    season_games.to_csv(config.SEASON_GAMES_DS_CSV)
    print("Process done")


def load_season_games_dataset():
    season_games = pd.read_feather(config.SEASON_GAMES_DS)
    season_games.set_index(["GAME_ID"], inplace=True)
    season_games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'], inplace=True)
    return season_games


if __name__ == '__main__':
    create_season_game_df()
