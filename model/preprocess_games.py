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
    global games, season_games, teams, seasons
    games = pd.read_csv(GAMES_DS, parse_dates=["GAME_DATE_EST"]
                        , infer_datetime_format=True, index_col="GAME_ID")
    games = games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'])
    teams = pd.read_feather(TEAMS_PROCESSED_DS)
    seasons = pd.read_feather(SEASONS_PROCESSED_DS)
    season_games = get_season_games(games, seasons)

def get_season_games(games, seasons):
    row = seasons.iloc[0]
    season_games = games[(games.SEASON == row.SEASON) & \
                        (games.GAME_DATE_EST >= row.SEASON_START) & \
                         (games.GAME_DATE_EST <= row.SEASON_END)
                         ]
    for i in range(1, len(seasons)):
        nonlocal season_games
        row = seasons.iloc[i]
        temp = games[(games.SEASON == row.SEASON) & \
                             (games.GAME_DATE_EST >= row.SEASON_START) & \
                             (games.GAME_DATE_EST <= row.SEASON_END)
                             ]
        season_games = pd.concat([season_games, temp])

    return season_games


def get_acc_data(home_team_id: int, visitor_team_id: int, season_team_games: DataFrame, last10_matchup: DataFrame):
    acc_data = {}
    previous_ht_games = season_team_games[(season_team_games.HOME_TEAM_ID == home_team_id)]
    hw = previous_ht_games.HOME_TEAM_WINS.sum()
    hl = previous_ht_games.HOME_TEAM_WINS.count() - hw
    acc_data["HT_HW"] = hw
    acc_data["HT_HL"] = hl

    previous_vt_games = season_team_games[(season_team_games.VISITOR_TEAM_ID == home_team_id)]

    vl = previous_vt_games.HOME_TEAM_WINS.sum()
    vw = previous_vt_games.HOME_TEAM_WINS.count() - vl
    acc_data["HT_VW"] = vw
    acc_data["HT_VL"] = vl

    last10_games = season_team_games.tail(10)
    last10_hw = last10_games.HOME_TEAM_WINS.sum()
    last10_hl = last10_games.HOME_TEAM_WINS.count() - last10_hw
    last10_vl = last10_games.HOME_TEAM_WINS.sum()
    last10_vw = last10_games.HOME_TEAM_WINS.count() - vl
    last10_w = last10_hw + last10_vw
    last10_l = last10_hl + last10_vl

    acc_data["HT_LAST10_W"] = last10_w
    acc_data["HT_LAST10_L"] = last10_l

    last10_matchup_hw = last10_matchup.HOME_TEAM_WINS.sum()
    last10_matchup_hl = last10_matchup.HOME_TEAM_WINS.count() - last10_hw
    last10_matchup_vl = last10_matchup.HOME_TEAM_WINS.sum()
    last10_matchup_vw = last10_matchup.HOME_TEAM_WINS.count() - vl
    last10_matchup_w = last10_matchup_hw + last10_matchup_vw
    last10_matchup_l = last10_matchup_hl + last10_matchup_vl
    acc_data["HT_LAST10_MATCHUP_W"] = last10_matchup_w
    acc_data["HT_LAST10_MATCHUP_L"] = last10_matchup_l

    acc_data["HT_OVERALL_OFF_POINTS"] = pd.concat([previous_ht_games.PTS_home, previous_vt_games.PTS_away], axis=0).mean()
    acc_data["HT_OVERALL_DEF_POINTS"] = pd.concat([previous_ht_games.PTS_away, previous_vt_games.PTS_home], axis=0).mean()

    acc_data["HT_OVERALL_OFF_FG"] = pd.concat([previous_ht_games.FG_PCT_home, previous_vt_games.FG_PCT_away],
                                                  axis=0).mean()
    acc_data["HT_OVERALL_DEF_FG"] = pd.concat([previous_ht_games.FG_PCT_away, previous_vt_games.FG_PCT_home],
                                                  axis=0).mean()

    acc_data["HT_OVERALL_OFF_3P"] = pd.concat([previous_ht_games.FG3_PCT_home, previous_vt_games.FG3_PCT_away],
                                              axis=0).mean()
    acc_data["HT_OVERALL_DEF_3P"] = pd.concat([previous_ht_games.FG3_PCT_away, previous_vt_games.FG3_PCT_home],
                                              axis=0).mean()

    acc_data["HT_OVERALL_OFF_FT"] = pd.concat([previous_ht_games.FT_PCT_home, previous_vt_games.FT_PCT_away],
                                              axis=0).mean()
    acc_data["HT_OVERALL_DEF_FT"] = pd.concat([previous_ht_games.FT_PCT_away, previous_vt_games.FT_PCT_home],
                                              axis=0).mean()

    acc_data["HT_OVERALL_OFF_REB"] = pd.concat([previous_ht_games.REB_home, previous_vt_games.REB_away],
                                              axis=0).mean()
    acc_data["HT_OVERALL_DEF_REB"] = pd.concat([previous_ht_games.REB_away, previous_vt_games.REB_home],
                                              axis=0).mean()

    acc_data["HT_AWAY_POINTS"] = previous_vt_games.PTS_away.mean()
    acc_data["HT_AWAY_FG"] = previous_vt_games.FG_PCT_away.mean()
    acc_data["HT_AWAY_3P"] = previous_vt_games.FG3_PCT_away.mean()
    acc_data["HT_AWAY_FT"] = previous_vt_games.FT_PCT_away.mean()
    acc_data["HT_AWAY_REB"] = previous_vt_games.REB_away.mean()


def get_team(row, season_games):
    pass


def get_rank(team_id: int, season_games: DataFrame, until_game=-1):
    return None


def get_class(team_id: int, season_games: DataFrame, until_game=-1):
    return None


def get_last10_w(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_last10_l(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_last10_matchup_w(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_last10_matchup_l(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_def_points(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_fg(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_def_fg(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_3p(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_def_3p(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_ft(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_def_ft(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_reb(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_def_reb(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_def_points(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_off_fg(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_def_fg(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_off_3p(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_def_3p(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_off_ft(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_def_ft(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_off_reb(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def get_away_def_reb(team_id: int, season_games: DataFrame, until_game=-1):
    pass


def process(season_games: DataFrame):
    games_processed: DataFrame = pd.DataFrame(columns=GAME_COLUMNS)

    for i in range(len(season_games)):
        game_processed = {}
        row = games.iloc[i, :]
        print(games.iloc[i, :])

        game_processed["ID"] = row.name
        game_processed['DATE'] = row["GAME_DATE_EST"]
        # # HOME TEAM
        game_processed['HT'] = row["HOME_TEAM_ID"]
        game_processed['HT_RANK'] = None
        game_processed['HT_CLASS'] = None
        game_processed['HT_HW'] = None

        # games_processed['HT_HL'] = get_ht_hl(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_VW'] = get_ht_vw(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_VL'] = get_ht_vl(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_LAST10_W'] = get_ht_last10_w(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_LAST10_L'] = get_ht_last10_l(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_LAST10_MATCHUP_W'] = get_ht_last10_matchup_w(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_LAST10_MATCHUP_L'] = get_ht_last10_matchup_l(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_POINTS'] = get_ht_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_DEF_POINTS'] = get_ht_overall_def_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_FG'] = get_ht_overall_off_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_DEF_FG'] = get_ht_overall_def_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_3P'] = get_ht_overall_off_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_DEF_3P'] = get_ht_overall_def_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_FT'] = get_ht_overall_off_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_DEF_FT'] = get_ht_overall_def_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_REB'] = get_ht_overall_off_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_DEF_REB'] = get_ht_overall_def_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_OVERALL_OFF_POINTS'] = get_ht_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_DEF_POINTS'] = get_ht_away_def_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_OFF_FG'] = get_ht_away_off_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_DEF_FG'] = get_ht_away_def_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_OFF_3P'] = get_ht_away_off_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_DEF_3P'] = get_ht_away_def_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_OFF_FT'] = get_ht_away_off_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_DEF_FT'] = get_ht_away_def_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_OFF_REB'] = get_ht_away_off_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['HT_AWAY_DEF_REB'] = get_ht_away_def_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed[]# AWAY TEA] = get_ AWAY(row, games
        # games_processed['AT'] = get_at(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_RANK'] = get_at_rank(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_CLASS'] = get_at_class(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_HW'] = get_at_hw(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_HL'] = get_at_hl(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_VW'] = get_at_vw(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_VL'] = get_at_vl(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_LAST10_W'] = get_at_last10_w(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_LAST10_L'] = get_at_last10_l(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_LAST10_MATCHUP_W'] = get_at_last10_matchup_w(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_LAST10_MATCHUP_L'] = get_at_last10_matchup_l(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_POINTS'] = get_at_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_DEF_POINTS'] = get_at_overall_def_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_FG'] = get_at_overall_off_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_DEF_FG'] = get_at_overall_def_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_3P'] = get_at_overall_off_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_DEF_3P'] = get_at_overall_def_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_FT'] = get_at_overall_off_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_DEF_FT'] = get_at_overall_def_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_REB'] = get_at_overall_off_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_DEF_REB'] = get_at_overall_def_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_OVERALL_OFF_POINTS'] = get_at_overall_off_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_DEF_POINTS'] = get_at_away_def_points(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_OFF_FG'] = get_at_away_off_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_DEF_FG'] = get_at_away_def_fg(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_OFF_3P'] = get_at_away_off_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_DEF_3P'] = get_at_away_def_3p(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_OFF_FT'] = get_at_away_off_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_DEF_FT'] = get_at_away_def_ft(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_OFF_REB'] = get_at_away_off_reb(team_id: int, season_games: DataFrame, until_game=-1)
        # games_processed['AT_AWAY_DEF_REB'] = get_at_away_def_reb(team_id: int, season_games: DataFrame, until_game=-1)

    return games_processed


def process_games_by_seasons(games: DataFrame):
    pass


if __name__ == "__main__":
    DATA_PATH = "../data"
    change_path()
    load_datasets()
    # process(games=games)
