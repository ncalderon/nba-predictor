#!/usr/bin/env python
import pandas as pd

DATA_PATH = 'data'
TEAMS_FILE = f"{DATA_PATH}/teams.csv"
TEAMS_PROCESSED_FILE = f"{DATA_PATH}/teams.processed.feather"
TEAMS_COLUMNS = ['TEAM_ID', 'NAME', 'NICKNAME', 'CITY']


def create_teams_dataset():
    print("Create new teams dataset from current one.")
    columns = ['TEAM_ID', 'ABBREVIATION', 'NICKNAME', 'CITY']
    teams = pd.read_csv(TEAMS_FILE, usecols=columns)
    teams.rename(columns={"ABBREVIATION": "NAME"}, inplace=True)
    print(f"Dataset size: {len(teams)}")
    teams.to_feather(TEAMS_PROCESSED_FILE)
    print(f"Teams dateset created: {TEAMS_PROCESSED_FILE},{TEAMS_FILE}")


if __name__ == '__main__':
    create_teams_dataset()
