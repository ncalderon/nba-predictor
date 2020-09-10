DATA_PATH = 'data'

SEASONS_PROCESSED_DS = f"{DATA_PATH}/seasons.processed.feather"

TEAMS_DS = f"{DATA_PATH}/teams.processed.feather"
TEAMS_PROCESSED_DS = f"{DATA_PATH}/teams.processed.feather"

RANKING_DS = f"{DATA_PATH}/ranking.csv"
RANKING_PROCESSED_DS = f"{DATA_PATH}/ranking.processed.feather"

GAMES_DS = f"{DATA_PATH}/games.csv"

GAMES_MATCHUP_DS = f"{DATA_PATH}/games_machup.feather"
GAMES_MATCHUP_DS_CSV = f"{DATA_PATH}/games_machup.csv"

columns = [
    "GAME_DATE_EST",
    "HOME_TEAM_NAME",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_NAME",
    "VISITOR_TEAM_ID",
    "SEASON",
    "HT_RANK",
    "HT_CLASS",
    "HT_HW",
    "HT_HL",
    "HT_VW",
    "HT_VL",
    "HT_LAST10_W",
    "HT_LAST10_L",
    "HT_LAST10_MATCHUP_W",
    "HT_LAST10_MATCHUP_L",
    "HT_OVERALL_OFF_POINTS",
    "HT_OVERALL_DEF_POINTS",
    "HT_OVERALL_OFF_FG",
    "HT_OVERALL_DEF_FG",
    "HT_OVERALL_OFF_3P",
    "HT_OVERALL_DEF_3P",
    "HT_OVERALL_OFF_FT",
    "HT_OVERALL_DEF_FT",
    "HT_OVERALL_OFF_REB",
    "HT_OVERALL_DEF_REB",
    "HT_AWAY_POINTS",
    "HT_AWAY_FG",
    "HT_AWAY_3P",
    "HT_AWAY_FT",
    "HT_AWAY_REB",
    "VT_RANK",
    "VT_CLASS",
    "VT_HW",
    "VT_HL",
    "VT_VW",
    "VT_VL",
    "VT_LAST10_W",
    "VT_LAST10_L",
    "VT_LAST10_MATCHUP_W",
    "VT_LAST10_MATCHUP_L",
    "VT_OVERALL_OFF_POINTS",
    "VT_OVERALL_DEF_POINTS",
    "VT_OVERALL_OFF_FG",
    "VT_OVERALL_DEF_FG",
    "VT_OVERALL_OFF_3P",
    "VT_OVERALL_DEF_3P",
    "VT_OVERALL_OFF_FT",
    "VT_OVERALL_DEF_FT",
    "VT_OVERALL_OFF_REB",
    "VT_OVERALL_DEF_REB",
    "VT_AWAY_POINTS",
    "VT_AWAY_FG",
    "VT_AWAY_3P",
    "VT_AWAY_FT",
    "VT_AWAY_REB",
    "PTS_home",
    "FG_PCT_home",
    "FT_PCT_home",
    "FG3_PCT_home",
    "AST_home",
    "REB_home",
    "PTS_away",
    "FG_PCT_away",
    "FT_PCT_away",
    "FG3_PCT_away",
    "AST_away",
    "REB_away",
    "HOME_TEAM_WINS"
]