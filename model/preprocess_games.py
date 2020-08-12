import numpy as np
import pandas as pd

DATA_PATH = 'data'

SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

TEAMS_DS = f"{DATA_PATH}/teams.processed.feather"
TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

GAMES_DS = f"{DATA_PATH}/games.csv"
GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"

def change_path():
    global SEASONS_PROCESSED_DS, TEAMS_DS, TEAMS_PROCESSED_DS, GAMES_DS, GAMES_PROCESSED_DS
    SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

    TEAMS_DS = f"{DATA_PATH}/teams.processed.feather"
    TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

    GAMES_DS = f"{DATA_PATH}/games.csv"
    GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"

GAME_COLUMNS = [
    'DATE',
    'ID',
    # HOME TEAM
    'HT',
    'HT_RANK',
    'HT_CLASS',
    'HT_HW',
    'HT_HL',
    'HT_VW',
    'HT_VL',
    'HT_LAST10_W',
    'HT_LAST10_L',
    'HT_LAST10_MATCHUP_W',
    'HT_LAST10_MATCHUP_L',
    'HT_OVERALL_OFF_POINTS',
    'HT_OVERALL_DEF_POINTS',
    'HT_OVERALL_OFF_FG',
    'HT_OVERALL_DEF_FG',
    'HT_OVERALL_OFF_3P',
    'HT_OVERALL_DEF_3P',
    'HT_OVERALL_OFF_FT',
    'HT_OVERALL_DEF_FT',
    'HT_OVERALL_OFF_REB',
    'HT_OVERALL_DEF_REB',
    'HT_OVERALL_OFF_POINTS',
    'HT_AWAY_DEF_POINTS',
    'HT_AWAY_OFF_FG',
    'HT_AWAY_DEF_FG',
    'HT_AWAY_OFF_3P',
    'HT_AWAY_DEF_3P',
    'HT_AWAY_OFF_FT',
    'HT_AWAY_DEF_FT',
    'HT_AWAY_OFF_REB',
    'HT_AWAY_DEF_REB',
    # AWAY TEAM
    'AT',
    'AT_RANK',
    'AT_CLASS',
    'AT_HW',
    'AT_HL',
    'AT_VW',
    'AT_VL',
    'AT_LAST10_W',
    'AT_LAST10_L',
    'AT_LAST10_MATCHUP_W',
    'AT_LAST10_MATCHUP_L',
    'AT_OVERALL_OFF_POINTS',
    'AT_OVERALL_DEF_POINTS',
    'AT_OVERALL_OFF_FG',
    'AT_OVERALL_DEF_FG',
    'AT_OVERALL_OFF_3P',
    'AT_OVERALL_DEF_3P',
    'AT_OVERALL_OFF_FT',
    'AT_OVERALL_DEF_FT',
    'AT_OVERALL_OFF_REB',
    'AT_OVERALL_DEF_REB',
    'AT_OVERALL_OFF_POINTS',
    'AT_AWAY_DEF_POINTS',
    'AT_AWAY_OFF_FG',
    'AT_AWAY_DEF_FG',
    'AT_AWAY_OFF_3P',
    'AT_AWAY_DEF_3P',
    'AT_AWAY_OFF_FT',
    'AT_AWAY_DEF_FT',
    'AT_AWAY_OFF_REB',
    'AT_AWAY_DEF_REB',
]

def load_datasets():
    global games, teams, seasons
    games = pd.read_csv("../nba-games/games.csv",parse_dates=["GAME_DATE_EST"]
                    ,infer_datetime_format=True)
    teams = pd.read_feather(TEAMS_PROCESSED_DS)
    seasons = pd.read_feather(SEASONS_PROCESSED_DS)


if __name__ == "__main__":
    DATA_PATH = "../data"
    change_path()
    load_datasets()
    print(games.head())
