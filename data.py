#!/usr/bin/env python
import model.dataset.seasons as season
import model.dataset.teams as teams
import model.dataset.game_matchup as matchup_games


if __name__ == '__main__':
    season.create_seasons_dataset()
    teams.create_teams_dataset()
