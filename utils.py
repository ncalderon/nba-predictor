#!/usr/bin/env python
import pandas as pd
import model.dataset.game_matchup as games_matchup_d
import model.dataset.seasons as seasons_d
import model.dataset.teams as teams_d
from datetime import datetime

DATA_PATH = 'data'

SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

TEAMS_DS = f"{DATA_PATH}/teams.csv"
TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

RANKING_DS = f"{DATA_PATH}/ranking.csv"
RANKING_PROCESSED_DS = f"{DATA_PATH}/ranking.processed.feather"

GAMES_DS = f"{DATA_PATH}/games.csv"
GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"


def load_df():
    games = pd.read_csv(GAMES_DS, parse_dates=["GAME_DATE_EST"],
                        usecols=["GAME_ID", 'GAME_DATE_EST', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                                 'SEASON', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home',
                                 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away',
                                 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
                                 'HOME_TEAM_WINS']
                        , infer_datetime_format=True, index_col="GAME_ID")
    games = games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])

    teams = pd.read_feather(TEAMS_PROCESSED_DS)

    seasons = pd.read_feather(SEASONS_PROCESSED_DS)

    rankings = pd.read_csv(RANKING_DS, parse_dates=["STANDINGSDATE"],
                           usecols=['TEAM_ID', 'LEAGUE_ID', 'SEASON_ID', 'STANDINGSDATE', 'CONFERENCE',
                                    'TEAM', 'G', 'W', 'L', 'W_PCT', 'HOME_RECORD', 'ROAD_RECORD'],
                           infer_datetime_format=True,
                           index_col=["STANDINGSDATE"])
    rankings.sort_index(inplace=True)

    games_matchup = pd.read_feather(GAMES_PROCESSED_DS)
    games_matchup = games_matchup.set_index(["GAME_ID"])
    games_matchup = games_matchup.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])
    return games, teams, seasons, rankings, games_matchup


def get_n_season_games_matchup_df(season_qty):
    year = datetime.now().year - season_qty
    games_matchup_d.create_dataframe(start=year, end=year)
