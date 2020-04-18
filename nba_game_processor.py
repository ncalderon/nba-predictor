import pandas as pd


def transform():
    data = pd.read_csv("../nba-games/games.csv")
    print(data.head())


if __name__ == "__main__":
    transform()
