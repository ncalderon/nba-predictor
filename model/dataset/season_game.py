import data

def create_season_game_df():
    import pandas as pd
    from nba_api.stats.endpoints import leaguegamelog
    import model.dataset.data as data

    seasons = data.load_seasons().SEASON.unique()
    season_games = pd.DataFrame()
    for season in seasons[-11:]:
        next_season = leaguegamelog.LeagueGameLog(season_type_all_star="Regular Season"
                                                  , season=season).get_data_frames()[0].sort_values(
            by=['SEASON_ID', 'GAME_DATE', 'GAME_ID'])
        season_home_rows = next_season[next_season.MATCHUP.str.contains('vs.')]
        season_away_rows = next_season[next_season.MATCHUP.str.contains('@')]
        # Join every row to all others with the same game ID.
        joined = pd.merge(season_home_rows, season_away_rows, suffixes=['_HOME', '_AWAY'],
                          on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
        # Filter out any row that is joined to itself.
        result = joined[joined.TEAM_ID_HOME != joined.TEAM_ID_AWAY]
        season_games = pd.concat([season_games, result])
    season_games

if __name__ == '__main__':
    seasons = data.load_seasons()
    print(seasons["SEASON"].unique())