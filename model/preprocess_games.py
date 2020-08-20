import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

DATA_PATH = 'data'

SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

TEAMS_DS = f"{DATA_PATH}/teams.processed.feather"
TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

GAMES_DS = f"{DATA_PATH}/games.csv"
GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"
GAMES_PROCESSED_DS_CSV = f"{DATA_PATH}/games.processed.csv"


def change_path():
    global SEASONS_PROCESSED_DS, TEAMS_DS, TEAMS_PROCESSED_DS, GAMES_DS, GAMES_PROCESSED_DS
    SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

    TEAMS_DS = f"{DATA_PATH}/teams.processed.feather"
    TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

    GAMES_DS = f"{DATA_PATH}/games.csv"
    GAMES_PROCESSED_DS = f"{DATA_PATH}/games.processed.feather"


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
        row = seasons.iloc[i]
        temp = games[(games.SEASON == row.SEASON) & \
                     (games.GAME_DATE_EST >= row.SEASON_START) & \
                     (games.GAME_DATE_EST <= row.SEASON_END)
                     ]
        season_games = pd.concat([season_games, temp])
    return season_games


def get_balance_last_games(team_id: int, last_games: DataFrame):
    if last_games.empty:
        return 0, 0
    h_games = last_games[last_games.HOME_TEAM_ID == team_id]
    hw = h_games.HOME_TEAM_WINS.sum()
    hl = len(h_games) - hw

    v_games = last_games[last_games.VISITOR_TEAM_ID == team_id]
    vl = v_games.HOME_TEAM_WINS.sum()
    vw = len(v_games) - vl

    w = hw + vw
    l = hl + vl
    return w, l

def get_balance_previous_season_games(team_id: int, previous_season_games: DataFrame, is_visitor = False):
    if previous_season_games.empty:
        return 0, 0
    else:
        if not is_visitor:
            w = previous_season_games.HOME_TEAM_WINS.sum()
            l = previous_season_games.HOME_TEAM_WINS.count() - w
        else:
            l = previous_season_games.HOME_TEAM_WINS.sum()
            w = previous_season_games.HOME_TEAM_WINS.count() - l
        return w, l

def get_acc_data(team_id: int, season_team_games: DataFrame, last10_matchup: DataFrame, is_visit=False):
    prefix_key = "HT" if not is_visit else "VT"
    acc_data = {f"{prefix_key}": team_id, f"{prefix_key}_RANK": None, f"{prefix_key}_CLASS": None}

    previous_ht_games = season_team_games[(season_team_games.HOME_TEAM_ID == team_id)]
    acc_data[f"{prefix_key}_HW"], acc_data[f"{prefix_key}_HL"] = get_balance_previous_season_games(team_id, previous_ht_games)

    previous_vt_games = season_team_games[(season_team_games.VISITOR_TEAM_ID == team_id)]
    acc_data[f"{prefix_key}_VW"], acc_data[f"{prefix_key}_VL"] =  get_balance_previous_season_games(team_id, previous_vt_games, True)

    last10_games = season_team_games.tail(10)

    acc_data[f"{prefix_key}_LAST10_W"], acc_data[f"{prefix_key}_LAST10_L"] = get_balance_last_games(team_id,
                                                                                                    last10_games)

    acc_data[f"{prefix_key}_LAST10_MATCHUP_W"], acc_data[f"{prefix_key}_LAST10_MATCHUP_L"] = get_balance_last_games(
        team_id,
        last10_matchup)

    if previous_ht_games.empty:
        previous_ht_games = pd.DataFrame({
            "PTS_home": 0, "PTS_away": 0,
            "FG_PCT_home": 0, "FG_PCT_away": 0,
            "FG3_PCT_home": 0, "FG3_PCT_away": 0,
            "FT_PCT_home": 0, "FT_PCT_away": 0,
            "REB_home": 0, "REB_away": 0
        }, index=[0])

    if previous_vt_games.empty:
        previous_vt_games = pd.DataFrame({
            "PTS_home": 0, "PTS_away": 0,
            "FG_PCT_home": 0, "FG_PCT_away": 0,
            "FG3_PCT_home": 0, "FG3_PCT_away": 0,
            "FT_PCT_home": 0, "FT_PCT_away": 0,
            "REB_home": 0, "REB_away": 0
        }, index=[0])

    acc_data[f"{prefix_key}_OVERALL_OFF_POINTS"] = pd.concat([previous_ht_games.PTS_home, previous_vt_games.PTS_away],
                                                             axis=0).mean().round(decimals=3)
    acc_data[f"{prefix_key}_OVERALL_DEF_POINTS"] = pd.concat([previous_ht_games.PTS_away, previous_vt_games.PTS_home],
                                                             axis=0).mean().round(decimals=3)

    acc_data[f"{prefix_key}_OVERALL_OFF_FG"] = pd.concat([previous_ht_games.FG_PCT_home, previous_vt_games.FG_PCT_away],
                                                         axis=0).mean().round(decimals=3)
    acc_data[f"{prefix_key}_OVERALL_DEF_FG"] = pd.concat([previous_ht_games.FG_PCT_away, previous_vt_games.FG_PCT_home],
                                                         axis=0).mean().round(decimals=3)

    acc_data[f"{prefix_key}_OVERALL_OFF_3P"] = pd.concat(
        [previous_ht_games.FG3_PCT_home, previous_vt_games.FG3_PCT_away],
        axis=0).mean().round(decimals=3)
    acc_data[f"{prefix_key}_OVERALL_DEF_3P"] = pd.concat(
        [previous_ht_games.FG3_PCT_away, previous_vt_games.FG3_PCT_home],
        axis=0).mean().round(decimals=3)

    acc_data[f"{prefix_key}_OVERALL_OFF_FT"] = pd.concat([previous_ht_games.FT_PCT_home, previous_vt_games.FT_PCT_away],
                                                         axis=0).mean().round(decimals=3)
    acc_data[f"{prefix_key}_OVERALL_DEF_FT"] = pd.concat([previous_ht_games.FT_PCT_away, previous_vt_games.FT_PCT_home],
                                                         axis=0).mean().round(decimals=3)

    acc_data[f"{prefix_key}_OVERALL_OFF_REB"] = pd.concat([previous_ht_games.REB_home, previous_vt_games.REB_away],
                                                          axis=0).mean().round(decimals=3)
    acc_data[f"{prefix_key}_OVERALL_DEF_REB"] = pd.concat([previous_ht_games.REB_away, previous_vt_games.REB_home],
                                                          axis=0).mean().round(decimals=3)

    acc_data[f"{prefix_key}_AWAY_POINTS"] = previous_vt_games.PTS_away.mean().round(decimals=3)
    acc_data[f"{prefix_key}_AWAY_FG"] = previous_vt_games.FG_PCT_away.mean().round(decimals=3)
    acc_data[f"{prefix_key}_AWAY_3P"] = previous_vt_games.FG3_PCT_away.mean().round(decimals=3)
    acc_data[f"{prefix_key}_AWAY_FT"] = previous_vt_games.FT_PCT_away.mean().round(decimals=3)
    acc_data[f"{prefix_key}_AWAY_REB"] = previous_vt_games.REB_away.mean().round(decimals=3)
    return acc_data


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


def get_matchup_report(game, previous_games):
    game_processed = {}
    game_id = game.name
    game_date = game.GAME_DATE_EST
    home_team_id = game.HOME_TEAM_ID
    visitor_team_id = game.VISITOR_TEAM_ID
    season_year = game.SEASON

    game_processed["ID"] = game_id
    game_processed['DATE'] = game_date

    query = ((season_games.HOME_TEAM_ID == home_team_id) | (season_games.VISITOR_TEAM_ID == home_team_id)) & \
            ((season_games.HOME_TEAM_ID == visitor_team_id) | (season_games.VISITOR_TEAM_ID == visitor_team_id))
    last10_matchup = previous_games[query].tail(10)

    query = (season_games.SEASON == season_year) & ((season_games.HOME_TEAM_ID == home_team_id) |
             (season_games.VISITOR_TEAM_ID == home_team_id
              ))
    home_team_season_games = previous_games[query]

    home_team_data = get_acc_data(team_id=home_team_id, season_team_games=home_team_season_games,
                                  last10_matchup=last10_matchup)

    query = (season_games.SEASON == season_year) & \
            ((season_games.HOME_TEAM_ID == visitor_team_id) |
             (season_games.VISITOR_TEAM_ID == visitor_team_id)
             )
    visitor_team_season_games = previous_games[query]

    visitor_team_data = get_acc_data(team_id=visitor_team_id, season_team_games=visitor_team_season_games,
                                     last10_matchup=last10_matchup, is_visit=True)

    game_processed = {**game_processed, **home_team_data, **visitor_team_data}
    return game_processed


def get_games_matchup_report(season_games: DataFrame, until_season: int = 2015):
    print(f"Season games: {len(season_games)}")
    games_matchup = []
    for i in tqdm(reversed(range(len(season_games) - 1))):
        row = season_games.iloc[i, :]
        if row.SEASON == until_season:
            break
        print(f"Game: {i}, {row.name}")
        previous_games = season_games[:i]
        games_matchup.append(get_matchup_report(row, previous_games))
    return games_matchup


def process_games_by_seasons(games: DataFrame):
    pass


if __name__ == "__main__":
    DATA_PATH = "../data"
    change_path()
    load_datasets()
    games_matchup_report_df: DataFrame = pd.DataFrame(get_games_matchup_report(season_games=season_games))
    games_matchup_report_df.to_feather(GAMES_PROCESSED_DS)
    games_matchup_report_df.to_csv(GAMES_PROCESSED_DS_CSV)
