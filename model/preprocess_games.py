import numpy as np
import pandas as pd
from pandas import DataFrame

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
    games = pd.read_csv(GAMES_DS, parse_dates=["GAME_DATE_EST"]
                        , infer_datetime_format=True, index_col="GAME_ID")
    games = games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])
    teams = pd.read_feather(TEAMS_PROCESSED_DS)
    seasons = pd.read_feather(SEASONS_PROCESSED_DS)


def get_team(row, games):
    pass

def get_rank(row, games):
    return None


def get_class(row, games):
    return None


def get_hw(row, games):
    query =


def get_hl(row, games):
    pass


def get_vw(row, games):
    pass


def get_vl(row, games):
    pass


def get_last10_w(row, games):
    pass


def get_last10_l(row, games):
    pass


def get_last10_matchup_w(row, games):
    pass


def get_last10_matchup_l(row, games):
    pass


def get_overall_off_points(row, games):
    pass


def get_overall_def_points(row, games):
    pass


def get_overall_off_fg(row, games):
    pass


def get_overall_def_fg(row, games):
    pass


def get_overall_off_3p(row, games):
    pass


def get_overall_def_3p(row, games):
    pass


def get_overall_off_ft(row, games):
    pass


def get_overall_def_ft(row, games):
    pass


def get_overall_off_reb(row, games):
    pass


def get_overall_def_reb(row, games):
    pass


def get_overall_off_points(row, games):
    pass


def get_away_def_points(row, games):
    pass


def get_away_off_fg(row, games):
    pass


def get_away_def_fg(row, games):
    pass


def get_away_off_3p(row, games):
    pass


def get_away_def_3p(row, games):
    pass


def get_away_off_ft(row, games):
    pass


def get_away_def_ft(row, games):
    pass


def get_away_off_reb(row, games):
    pass


def get_away_def_reb(row, games):
    pass


def process(games: DataFrame):
    games_processed: DataFrame = pd.DataFrame(columns=GAME_COLUMNS)
    game_processed = {}
    for i in range(len(games)):
        row = games.iloc[i, :]
        print(games.iloc[i, :])

        game_processed["ID"] = row.name
        game_processed['DATE'] = row["GAME_DATE_EST"]
        # # HOME TEAM
        game_processed['HT'] = row["HOME_TEAM_ID"]
        game_processed['HT_RANK'] = get_rank(row, games)
        game_processed['HT_CLASS'] = get_class(row, games)
        game_processed['HT_HW'] = get_hw(row, games)
        # games_processed['HT_HL'] = get_ht_hl(row, games)
        # games_processed['HT_VW'] = get_ht_vw(row, games)
        # games_processed['HT_VL'] = get_ht_vl(row, games)
        # games_processed['HT_LAST10_W'] = get_ht_last10_w(row, games)
        # games_processed['HT_LAST10_L'] = get_ht_last10_l(row, games)
        # games_processed['HT_LAST10_MATCHUP_W'] = get_ht_last10_matchup_w(row, games)
        # games_processed['HT_LAST10_MATCHUP_L'] = get_ht_last10_matchup_l(row, games)
        # games_processed['HT_OVERALL_OFF_POINTS'] = get_ht_overall_off_points(row, games)
        # games_processed['HT_OVERALL_DEF_POINTS'] = get_ht_overall_def_points(row, games)
        # games_processed['HT_OVERALL_OFF_FG'] = get_ht_overall_off_fg(row, games)
        # games_processed['HT_OVERALL_DEF_FG'] = get_ht_overall_def_fg(row, games)
        # games_processed['HT_OVERALL_OFF_3P'] = get_ht_overall_off_3p(row, games)
        # games_processed['HT_OVERALL_DEF_3P'] = get_ht_overall_def_3p(row, games)
        # games_processed['HT_OVERALL_OFF_FT'] = get_ht_overall_off_ft(row, games)
        # games_processed['HT_OVERALL_DEF_FT'] = get_ht_overall_def_ft(row, games)
        # games_processed['HT_OVERALL_OFF_REB'] = get_ht_overall_off_reb(row, games)
        # games_processed['HT_OVERALL_DEF_REB'] = get_ht_overall_def_reb(row, games)
        # games_processed['HT_OVERALL_OFF_POINTS'] = get_ht_overall_off_points(row, games)
        # games_processed['HT_AWAY_DEF_POINTS'] = get_ht_away_def_points(row, games)
        # games_processed['HT_AWAY_OFF_FG'] = get_ht_away_off_fg(row, games)
        # games_processed['HT_AWAY_DEF_FG'] = get_ht_away_def_fg(row, games)
        # games_processed['HT_AWAY_OFF_3P'] = get_ht_away_off_3p(row, games)
        # games_processed['HT_AWAY_DEF_3P'] = get_ht_away_def_3p(row, games)
        # games_processed['HT_AWAY_OFF_FT'] = get_ht_away_off_ft(row, games)
        # games_processed['HT_AWAY_DEF_FT'] = get_ht_away_def_ft(row, games)
        # games_processed['HT_AWAY_OFF_REB'] = get_ht_away_off_reb(row, games)
        # games_processed['HT_AWAY_DEF_REB'] = get_ht_away_def_reb(row, games)
        # games_processed[]# AWAY TEA] = get_ AWAY(row, games
        # games_processed['AT'] = get_at(row, games)
        # games_processed['AT_RANK'] = get_at_rank(row, games)
        # games_processed['AT_CLASS'] = get_at_class(row, games)
        # games_processed['AT_HW'] = get_at_hw(row, games)
        # games_processed['AT_HL'] = get_at_hl(row, games)
        # games_processed['AT_VW'] = get_at_vw(row, games)
        # games_processed['AT_VL'] = get_at_vl(row, games)
        # games_processed['AT_LAST10_W'] = get_at_last10_w(row, games)
        # games_processed['AT_LAST10_L'] = get_at_last10_l(row, games)
        # games_processed['AT_LAST10_MATCHUP_W'] = get_at_last10_matchup_w(row, games)
        # games_processed['AT_LAST10_MATCHUP_L'] = get_at_last10_matchup_l(row, games)
        # games_processed['AT_OVERALL_OFF_POINTS'] = get_at_overall_off_points(row, games)
        # games_processed['AT_OVERALL_DEF_POINTS'] = get_at_overall_def_points(row, games)
        # games_processed['AT_OVERALL_OFF_FG'] = get_at_overall_off_fg(row, games)
        # games_processed['AT_OVERALL_DEF_FG'] = get_at_overall_def_fg(row, games)
        # games_processed['AT_OVERALL_OFF_3P'] = get_at_overall_off_3p(row, games)
        # games_processed['AT_OVERALL_DEF_3P'] = get_at_overall_def_3p(row, games)
        # games_processed['AT_OVERALL_OFF_FT'] = get_at_overall_off_ft(row, games)
        # games_processed['AT_OVERALL_DEF_FT'] = get_at_overall_def_ft(row, games)
        # games_processed['AT_OVERALL_OFF_REB'] = get_at_overall_off_reb(row, games)
        # games_processed['AT_OVERALL_DEF_REB'] = get_at_overall_def_reb(row, games)
        # games_processed['AT_OVERALL_OFF_POINTS'] = get_at_overall_off_points(row, games)
        # games_processed['AT_AWAY_DEF_POINTS'] = get_at_away_def_points(row, games)
        # games_processed['AT_AWAY_OFF_FG'] = get_at_away_off_fg(row, games)
        # games_processed['AT_AWAY_DEF_FG'] = get_at_away_def_fg(row, games)
        # games_processed['AT_AWAY_OFF_3P'] = get_at_away_off_3p(row, games)
        # games_processed['AT_AWAY_DEF_3P'] = get_at_away_def_3p(row, games)
        # games_processed['AT_AWAY_OFF_FT'] = get_at_away_off_ft(row, games)
        # games_processed['AT_AWAY_DEF_FT'] = get_at_away_def_ft(row, games)
        # games_processed['AT_AWAY_OFF_REB'] = get_at_away_off_reb(row, games)
        # games_processed['AT_AWAY_DEF_REB'] = get_at_away_def_reb(row, games)

    return games_processed


def process_games_by_seasons(games: DataFrame):
    pass


if __name__ == "__main__":
    DATA_PATH = "../data"
    change_path()
    load_datasets()
    # process(games=games)
