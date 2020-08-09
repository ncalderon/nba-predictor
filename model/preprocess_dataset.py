import pandas as pd

DATA_PATH = '../data'
TEAMS_FILE = f"{DATA_PATH}/teams.csv"
TEAMS_PROCESSED_FILE = f"{DATA_PATH}/teams.processed.feather"


def preprocess_teams():
    columns = ['TEAM_ID', 'ABBREVIATION', 'NICKNAME', 'CITY']
    teams = pd.read_csv(TEAMS_FILE, usecols=columns)
    teams.rename(columns={"ABBREVIATION": "NAME"}, inplace=True)
    teams.to_feather(TEAMS_PROCESSED_FILE)

def preprocess_games():
    columns = ['GAME_DATE_EST',
               'GAME_ID',
               'HOME_TEAM_WINS',
               'HOME_TEAM',
               'HOME_TEAM_WIN_LAST_10_GAMES',
               'VISITOR_TEAM_WINS',
               'VISITOR_TEAM',
               'VISITOR_TEAM_WIN_LAST_10_GAMES']

if __name__ == "__main__":
    preprocess_teams()
    print(pd.read_feather(TEAMS_PROCESSED_FILE))
