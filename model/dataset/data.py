#!/usr/bin/env python
import pandas as pd
import model.dataset.config as config


def __add_team_name_column(games, teams):
    """
    To games dataframe:

    - Add team name columns
    - Drop team ID columns
    - Set GAME_ID as INDEX
    - Sort by GAME_DATE and GAME_ID

    :param games:
    :param teams:

    :return:

    A new dataframe with those changes

    """
    games_df = games.reset_index()
    teams_df = teams.drop(columns=["NICKNAME", "CITY"])
    result_df = games_df.merge(teams_df, left_on='HOME_TEAM_ID', right_on='TEAM_ID', suffixes=['_games', '_teams'])
    result_df = result_df.drop(columns=["TEAM_ID"])
    result_df = result_df.rename(columns={"NAME": "HOME_TEAM_NAME"})
    result_df = result_df.merge(teams_df, left_on='VISITOR_TEAM_ID', right_on='TEAM_ID', suffixes=['_games', '_teams'])
    result_df = result_df.drop(columns=["TEAM_ID"])
    result_df = result_df.rename(columns={"NAME": "VISITOR_TEAM_NAME"})
    result_df = result_df.set_index("GAME_ID")
    result_df = result_df.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])
    return result_df


def load_teams():
    """
    Load teams processed dataset as a dataframe
    :return:

    teams DataFrame
    """
    teams = pd.read_feather(config.TEAMS_PROCESSED_DS)
    return teams


def load_games():
    """
    1. Load raw games dataset as it was downloaded from https://www.kaggle.com/nathanlauga/nba-games
    2. Sort rows by GAME_DATE and GAME_ID
    3. Load teams
    4. Add team name column to games DataFrame

    :return:

    Games DataFrame
    """
    games = pd.read_csv(config.GAMES_DS,
                        usecols=["GAME_ID", 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                                 'SEASON', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home',
                                 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away',
                                 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
                                 'HOME_TEAM_WINS'], parse_dates=["GAME_DATE_EST"]
                        , infer_datetime_format=True, index_col="GAME_ID")
    games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'], inplace=True)
    teams = load_teams()
    games = __add_team_name_column(games, teams)
    return games


def load_seasons():
    """
    Load season processed dataset as a dataframe

    :return:

    seasons dataframe
    """
    seasons = pd.read_feather(config.SEASONS_PROCESSED_DS)
    return seasons


def load_rankings():
    """
    Load rankings dataset. This dataset has the standing position of each team
    by each date of each season

    :return:

    ranking DataFrame
    """
    rankings = pd.read_csv(config.RANKING_DS, parse_dates=["STANDINGSDATE"],
                           usecols=['TEAM_ID', 'LEAGUE_ID', 'SEASON_ID', 'STANDINGSDATE', 'CONFERENCE',
                                    'TEAM', 'G', 'W', 'L', 'W_PCT', 'HOME_RECORD', 'ROAD_RECORD'],
                           infer_datetime_format=True,
                           index_col=["STANDINGSDATE"])
    rankings.sort_index(inplace=True)
    return rankings


def __create_season_games_df(games, seasons):
    """
    Filter out of the games dataframe these games:

    - Playoff games
    - Preseason games

    So, create a DataFrame with only season games

    :param games:
    :param seasons:
    :return:

    season_gaems DataFrame

    """
    row = seasons.iloc[0]
    season_games = games[(games.SEASON == row.SEASON) & \
                         (games.GAME_DATE_EST >= row.SEASON_START) & \
                         (games.GAME_DATE_EST <= row.SEASON_END)
                         ]
    for i in range(1, len(seasons)):
        row = seasons.iloc[i]
        temp = games[(games.SEASON == row.SEASON) & \
                     (games.GAME_DATE_EST >= row.SEASON_START) & \
                     (games.GAME_DATE_EST <= row.SEASON_END)
                     ]
        season_games = pd.concat([season_games, temp])
    return season_games


def create_season_games_df():
    """
    Load games and teams dataframe and then call create_seasongames

    :return:
    season_games DataFrame
    """
    return __create_season_games_df(load_games(), load_seasons())
