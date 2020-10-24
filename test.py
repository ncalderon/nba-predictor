from nba_api.stats.endpoints import teamgamelogs

if __name__ == '__main__':
    # Query for games where the Celtics were playing
    12 / 25 / 2018
    all_games = teamgamelogs.TeamGameLogs(season_type_nullable="Regular Season"
                                          , date_from_nullable='01/01/2010',
                                          date_to_nullable='01/01/2022').get_data_frames()[0]