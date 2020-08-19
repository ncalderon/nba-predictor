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


def get_acc_data(team_id: int, season_team_games: DataFrame, last10_matchup: DataFrame):
    acc_data = {}
    previous_ht_games = season_team_games[(season_team_games.HOME_TEAM_ID == team_id)]
    hw = previous_ht_games.HOME_TEAM_WINS.sum()
    hl = previous_ht_games.HOME_TEAM_WINS.count() - hw
    acc_data["HT_HW"] = hw
    acc_data["HT_HL"] = hl

    previous_vt_games = season_team_games[(season_team_games.VISITOR_TEAM_ID == team_id)]

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

    acc_data["HT_OVERALL_OFF_POINTS"] = pd.concat([previous_ht_games.PTS_home, previous_vt_games.PTS_away],
                                                  axis=0).mean()
    acc_data["HT_OVERALL_DEF_POINTS"] = pd.concat([previous_ht_games.PTS_away, previous_vt_games.PTS_home],
                                                  axis=0).mean()

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


def process(season_games: DataFrame):
    games_processed: DataFrame = pd.DataFrame(columns=GAME_COLUMNS)

    for i in reversed(range(len(season_games) - 1)):
        game_processed = {}
        row = season_games.iloc[i, :]
        previous_games = season_games[:i]

        home_team_id = row.HOME_TEAM_ID
        visitor_team_id = row.VISITOR_TEAM_ID
        season_year = row.SEASON

        game_processed["ID"] = row.name
        game_processed['DATE'] = row["GAME_DATE_EST"]
        # # HOME TEAM
        game_processed['HT'] = row["HOME_TEAM_ID"]
        game_processed['HT_RANK'] = None
        game_processed['HT_CLASS'] = None
        game_processed['HT_HW'] = None

        query = ((season_games.HOME_TEAM_ID == home_team_id) | (season_games.VISITOR_TEAM_ID == home_team_id)) & \
                ((season_games.HOME_TEAM_ID == visitor_team_id) | (season_games.VISITOR_TEAM_ID == visitor_team_id))
        last10_matchup = previous_games[query].tail(10)

        query = (season_games.SEASON == season_year) & \
                ((season_games.HOME_TEAM_ID == home_team_id) | \
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
                                         last10_matchup=last10_matchup)

    return games_processed


def process_games_by_seasons(games: DataFrame):
    pass


if __name__ == "__main__":
    DATA_PATH = "../data"
    change_path()
    load_datasets()
    process(season_games=season_games)
