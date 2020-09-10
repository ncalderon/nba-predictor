#!/usr/bin/env python
import sys
import model.dataset.config as config
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import data as data


def __get_balance_last_games(team_id: int, last_games: DataFrame):
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


def __get_balance_previous_season_games(team_id: int, previous_season_games: DataFrame, is_visitor=False):
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


def __get_acc_data(team_id: int, season_team_games: DataFrame, last10_matchup: DataFrame, today_rankings: DataFrame,
                   is_visit=False):
    prefix_key = "HT" if not is_visit else "VT"
    team_rank = today_rankings[today_rankings.TEAM_ID == team_id].index[0]
    acc_data = {f"{prefix_key}": team_id,
                f"{prefix_key}_RANK": team_rank
        , f"{prefix_key}_CLASS": 2 if team_rank >= 20 else 1 if (team_rank >= 10) else 0
                }

    previous_ht_games = season_team_games[(season_team_games.HOME_TEAM_ID == team_id)]
    acc_data[f"{prefix_key}_HW"], acc_data[f"{prefix_key}_HL"] = __get_balance_previous_season_games(team_id,
                                                                                                     previous_ht_games)

    previous_vt_games = season_team_games[(season_team_games.VISITOR_TEAM_ID == team_id)]
    acc_data[f"{prefix_key}_VW"], acc_data[f"{prefix_key}_VL"] = __get_balance_previous_season_games(team_id,
                                                                                                     previous_vt_games,
                                                                                                     True)

    last10_games = season_team_games.tail(10)

    acc_data[f"{prefix_key}_LAST10_W"], acc_data[f"{prefix_key}_LAST10_L"] = __get_balance_last_games(team_id,
                                                                                                      last10_games)

    acc_data[f"{prefix_key}_LAST10_MATCHUP_W"], acc_data[f"{prefix_key}_LAST10_MATCHUP_L"] = __get_balance_last_games(
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


def get_last_season_games(previous_games, season_year, home_team_id, visitor_team_id):
    home_team_season_games = previous_games[previous_games.SEASON.isin([season_year - 1, season_year]) &
                                            ((previous_games.HOME_TEAM_ID == home_team_id) |
                                             (previous_games.VISITOR_TEAM_ID == home_team_id)
                                             )]
    visitor_team_season_games = previous_games[previous_games.SEASON.isin([season_year - 1, season_year]) &
                                               ((previous_games.HOME_TEAM_ID == visitor_team_id) |
                                                (previous_games.VISITOR_TEAM_ID == visitor_team_id)
                                                )]
    return home_team_season_games, visitor_team_season_games


def __get_game_matchup(game, previous_games):
    game_processed = {}
    game_id = game.name
    game_date = game.GAME_DATE_EST
    home_team_id = game.HOME_TEAM_ID
    visitor_team_id = game.VISITOR_TEAM_ID
    season_year = game.SEASON

    today_ranking_df = rankings.loc[rankings.index == game_date].sort_values("W_PCT")
    today_ranking_df.reset_index(inplace=True)
    game_processed["GAME_ID"] = game_id
    # game_processed['SEASON'] = season_year

    for key in game.keys():
        game_processed[key] = game[key]

    query = ((previous_games.HOME_TEAM_ID == home_team_id) | (previous_games.VISITOR_TEAM_ID == home_team_id)) & \
            ((previous_games.HOME_TEAM_ID == visitor_team_id) | (previous_games.VISITOR_TEAM_ID == visitor_team_id))
    last10_matchup = previous_games[query].tail(10)

    query = previous_games.SEASON.eq(season_year) & ((previous_games.HOME_TEAM_ID == home_team_id) |
                                                     (previous_games.VISITOR_TEAM_ID == home_team_id
                                                      ))
    home_team_season_games = previous_games[query]
    if not len(home_team_season_games) > 10:
        home_team_season_games, visitor_team_season_games = get_last_season_games(previous_games, season_year,
                                                                                  home_team_id, visitor_team_id)
    else:
        query = previous_games.SEASON.eq(season_year) & ((previous_games.HOME_TEAM_ID == visitor_team_id) |
                                                         (previous_games.VISITOR_TEAM_ID == visitor_team_id)
                                                         )
        visitor_team_season_games = previous_games[query]
        if not len(visitor_team_season_games) > 10:
            home_team_season_games, visitor_team_season_games = get_last_season_games(previous_games, season_year,
                                                                                      home_team_id, visitor_team_id)

    home_team_data = __get_acc_data(team_id=home_team_id, season_team_games=home_team_season_games,
                                    last10_matchup=last10_matchup, today_rankings=today_ranking_df)

    visitor_team_data = __get_acc_data(team_id=visitor_team_id, season_team_games=visitor_team_season_games,
                                       last10_matchup=last10_matchup, today_rankings=today_ranking_df, is_visit=True)

    game_processed = {**home_team_data, **visitor_team_data, **game_processed}
    return game_processed


def __change_column_order(games_matchup_report_df):
    return games_matchup_report_df[config.columns]


def __get_games_matchup(season_games: DataFrame):
    print(f"Season games: {len(season_games)}")
    games_matchup = []
    seasons = season_games.SEASON.unique()
    idx_begin = len(season_games[season_games.SEASON == seasons[0]])
    with tqdm(total=len(season_games) - idx_begin) as pbar:
        for i in reversed(range(idx_begin, len(season_games))):
            row = season_games.iloc[i, :]
            previous_games = season_games[:i]
            try:
                games_matchup.append(__get_game_matchup(row, previous_games))
            except:
                print(f"Game: {i}, {row.name}")
                print("Unexpected error:", sys.exc_info()[0])
                raise
            pbar.update(1)
    return games_matchup


def __create_dataframe(start: int = 2016, end: int = 2018):
    print("Load datasets: teams, seasons, ranking")
    global season_games, rankings
    rankings = data.load_rankings()
    season_games = data.get_season_games()
    print("Processing...")
    query = ((season_games.SEASON >= start) & (season_games.SEASON <= end))
    df: DataFrame = pd.DataFrame(
        __get_games_matchup(
            season_games=season_games[query]
        )
    )
    df = __change_column_order(df)
    return df


def create_matchup_games_dataset(start: int = 2016, end: int = 2018):
    print("Create matchup games dataset. ")
    df = __create_dataframe(start, end)
    df.to_feather(config.GAMES_MATCHUP_DS)
    df.to_csv(config.GAMES_MATCHUP_DS_CSV)
    print("Process done")


def load_game_matchup_dataset():
    games_matchup = pd.read_feather(config.GAMES_MATCHUP_DS)
    games_matchup.set_index(["GAME_ID"], inplace=True)
    games_matchup.sort_values(by=['GAME_DATE_EST', 'GAME_ID'], inplace=True)
    return games_matchup


if __name__ == '__main__':
    create_matchup_games_dataset(start=2000, end=2018)
