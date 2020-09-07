#!/usr/bin/env python
import requests
from datetime import datetime
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://en.wikipedia.org/wiki/"
DATA_PATH = 'data'
SEASONS_FILE = f"{DATA_PATH}/seasons.csv"
SEASONS_PROCESSED_FILE = f"{DATA_PATH}/seasons.processed.feather"
START_SEASON = 2002
END_SEASON = datetime.now().year - 1


def __fetch_season_info(year):
    url = fr"{URL}{year}-{str(year + 1)[-2:]}_NBA_season"
    r = requests.get(url)
    status_200_ok = r.status_code == 200
    n_error = 0

    while not status_200_ok and n_error < 5:
        print(r.status_code, url)
        sleep(1)
        response = requests.get(url)
        status_200_ok = response.status_code == 200
        n_error += 1
    print(r.status_code, url)

    if n_error > 5:
        raise Exception(f"Cannot fetch data for season: {year}")

    return r.text


def __create_row_from_content(content, year):
    row = {}
    soup = BeautifulSoup(content, "html.parser")
    for column in soup.find("table", class_="infobox").find_all("th", {"scope": "row"}):
        field = column \
            .get_text() \
            .strip() \
            .upper() \
            .replace("-", "") \
            .replace("  ", " ") \
            .replace(" ", "_")

        column_sibling = column.next_sibling
        if len(column) is None:
            row[field] = ""
        else:
            text = column_sibling.get_text()
            if field == 'DURATION':
                val = text
                idx = val.find(fr"{year + 1}") + 4
                val = val[:idx].replace(" ", "")
                dates = val.split("–")
                row["SEASON_START"] = datetime.strptime(dates[0], "%B%d,%Y")
                row["SEASON_END"] = datetime.strptime(dates[1], "%B%d,%Y")
            row[field] = text
    return row


def create_seasons_dataset():
    print(fr"Create season dataset by fetching info from: {URL}")
    seasons = []
    for year in range(START_SEASON, END_SEASON):
        row = __create_row_from_content(__fetch_season_info(year), year)
        row["SEASON"] = year
        seasons.append(row)
        print(f"Season year: {year} processed. Values: {row}")

    seasons_df = pd.DataFrame(seasons)
    print(f"Dataset size: {len(seasons_df)}")
    seasons_df.to_feather(SEASONS_PROCESSED_FILE)
    seasons_df.to_csv(SEASONS_FILE)
    print(f"Seasons dateset created: {SEASONS_PROCESSED_FILE},{SEASONS_FILE}")


if __name__ == '__main__':
    create_seasons_dataset()
