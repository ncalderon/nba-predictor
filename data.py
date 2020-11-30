#!/usr/bin/env python
import model.dataset.season_game as sg

if __name__ == '__main__':
    sg.create_raw_season_games_df()
    sg.create_season_game_df(sg.load_raw_season_games_dataset())
