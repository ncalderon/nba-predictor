import pandas as pd

DATA_PATH = '../data'
TEAMS_FILE = f"{DATA_PATH}/teams.csv"
TEAMS_PROCESSED_FILE = f"{DATA_PATH}/teams.processed.feather"
TEAMS_COLUMNS = ['TEAM_ID', 'NAME', 'NICKNAME', 'CITY']

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


def preprocess_teams():
    columns = ['TEAM_ID', 'ABBREVIATION', 'NICKNAME', 'CITY']
    teams = pd.read_csv(TEAMS_FILE, usecols=columns)
    teams.rename(columns={"ABBREVIATION": "NAME"}, inplace=True)
    teams.to_feather(TEAMS_PROCESSED_FILE)


def preprocess_games():
    columns = ['GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID',
               'VISITOR_TEAM_ID', 'SEASON', 'TEAM_ID_home', 'PTS_home', 'FG_PCT_home',
               'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'TEAM_ID_away',
               'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away',
               'REB_away', 'HOME_TEAM_WINS']

def season_meta():
    




if __name__ == "__main__":
    preprocess_teams()
    print(pd.read_feather(TEAMS_PROCESSED_FILE))



