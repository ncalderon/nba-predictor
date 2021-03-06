X_COLS_BEST_BORUTA = [
    'AST_AGAINST_MEAN_HOME',
    'FG_AGAINST_MEAN_HOME',
    'AST_AGAINST_MEAN_AWAY',
    'FG_AGAINST_MEAN_AWAY',
    'FG_MEAN_AWAY',
    'FG_MEAN_HOME',
    'HOME_AST_AGAINST_MEAN',
    'HOME_AST_MEAN',
    'HOME_BLK_AGAINST_MEAN',
    'HOME_FG3_MEAN',
    'HOME_FG_AGAINST_MEAN',
    'HOME_FG_MEAN',
    'HOME_FG_MEAN_L10',
    'HOME_FG_MEAN_ML10',
    'HOME_PLUS_MINUS_MEAN',
    'HOME_PLUS_MINUS_MEAN_L10',
    'HOME_PLUS_MINUS_MEAN_ML10',
    'HOME_PTS_AGAINST_MEAN',
    'HOME_PTS_MEAN',
    'HOME_PTS_MEAN_L10',
    'HOME_PYTHAGOREAN_EXPECTATION',
    'HOME_PYTHAGOREAN_EXPECTATION_L10',
    'HOME_PYTHAGOREAN_EXPECTATION_ML10',
    'HOME_REB_AGAINST_MEAN',
    'HOME_W_L_CUM',
    'PLUS_MINUS_MEAN_AWAY',
    'PLUS_MINUS_MEAN_HOME',
    'PLUS_MINUS_MEAN_L10_AWAY',
    'PLUS_MINUS_MEAN_L10_HOME',
    'PLUS_MINUS_MEAN_ML10_AWAY',
    'PLUS_MINUS_MEAN_ML10_HOME',
    'PTS_AGAINST_MEAN_HOME',
    'PTS_MEAN_HOME',
    'PTS_MEAN_L10_HOME',
    'PTS_AGAINST_MEAN_AWAY',
    'PTS_MEAN_AWAY',
    'PTS_MEAN_L10_AWAY',
    'PYTHAGOREAN_EXPECTATION_AWAY',
    'PYTHAGOREAN_EXPECTATION_HOME',
    'PYTHAGOREAN_EXPECTATION_L10_AWAY',
    'PYTHAGOREAN_EXPECTATION_L10_HOME'
]

X_REG_COLS_BEST_BORUTA = [
    'HOME_PLUS_MINUS_MEAN',
    'HOME_PLUS_MINUS_MEAN_L10',
    'HOME_PYTHAGOREAN_EXPECTATION',
    'HOME_W_L_CUM',
    'PLUS_MINUS_MEAN_HOME',
    'PLUS_MINUS_MEAN_AWAY']

X_COLS_NO_HIGH_CORR = [
    'AST_AGAINST_MEAN_AWAY',
    'AST_AGAINST_MEAN_HOME',
    'AST_AGAINST_MEAN_L10_AWAY',
    'AST_AGAINST_MEAN_L10_HOME',
    'AST_MEAN_AWAY',
    'AST_MEAN_HOME',
    'AST_MEAN_L10_AWAY',
    'AST_MEAN_L10_HOME',
    'BLK_AGAINST_MEAN_AWAY',
    'BLK_AGAINST_MEAN_HOME',
    'BLK_AGAINST_MEAN_L10_AWAY',
    'BLK_AGAINST_MEAN_L10_HOME',
    'BLK_MEAN_AWAY',
    'BLK_MEAN_HOME',
    'BLK_MEAN_L10_AWAY',
    'BLK_MEAN_L10_HOME',
    'FG3_AGAINST_MEAN_AWAY',
    'FG3_AGAINST_MEAN_HOME',
    'FG3_AGAINST_MEAN_L10_AWAY',
    'FG3_AGAINST_MEAN_L10_HOME',
    'FG3_MEAN_AWAY',
    'FG3_MEAN_HOME',
    'FG3_MEAN_L10_AWAY',
    'FG3_MEAN_L10_HOME',
    'FG_AGAINST_MEAN_AWAY',
    'FG_AGAINST_MEAN_HOME',
    'FG_AGAINST_MEAN_L10_AWAY',
    'FG_AGAINST_MEAN_L10_HOME',
    'FG_MEAN_AWAY',
    'FG_MEAN_HOME',
    'FG_MEAN_L10_AWAY',
    'FG_MEAN_L10_HOME',
    'FT_AGAINST_MEAN_AWAY',
    'FT_AGAINST_MEAN_HOME',
    'FT_AGAINST_MEAN_L10_AWAY',
    'FT_AGAINST_MEAN_L10_HOME',
    'FT_MEAN_AWAY',
    'FT_MEAN_HOME',
    'FT_MEAN_L10_AWAY',
    'FT_MEAN_L10_HOME',
    'HOME_AST_AGAINST_MEAN',
    'HOME_AST_MEAN',
    'HOME_BLK_AGAINST_MEAN',
    'HOME_BLK_MEAN',
    'HOME_FG3_AGAINST_MEAN',
    'HOME_FG3_AGAINST_MEAN_L10',
    'HOME_FG3_MEAN',
    'HOME_FG3_MEAN_L10',
    'HOME_FG3_MEAN_ML10',
    'HOME_FG_AGAINST_MEAN',
    'HOME_FG_AGAINST_MEAN_L10',
    'HOME_FG_MEAN',
    'HOME_FG_MEAN_L10',
    'HOME_FG_MEAN_ML10',
    'HOME_FT_AGAINST_MEAN',
    'HOME_FT_AGAINST_MEAN_L10',
    'HOME_FT_MEAN',
    'HOME_FT_MEAN_L10',
    'HOME_FT_MEAN_ML10',
    'HOME_PLUS_MINUS_MEAN',
    'HOME_PLUS_MINUS_MEAN_L10',
    'HOME_PLUS_MINUS_MEAN_ML10',
    'HOME_PTS_AGAINST_MEAN',
    'HOME_PTS_AGAINST_MEAN_L10',
    'HOME_PTS_MEAN',
    'HOME_PTS_MEAN_L10',
    'HOME_PYTHAGOREAN_EXPECTATION',
    'HOME_PYTHAGOREAN_EXPECTATION_L10',
    'HOME_PYTHAGOREAN_EXPECTATION_ML10',
    'HOME_REB_AGAINST_MEAN',
    'HOME_REB_MEAN',
    'HOME_STL_AGAINST_MEAN',
    'HOME_STL_MEAN',
    'HOME_TOV_AGAINST_MEAN',
    'HOME_TOV_MEAN',
    'HOME_W_L_CUM',
    'PLUS_MINUS_MEAN_AWAY',
    'PLUS_MINUS_MEAN_HOME',
    'PLUS_MINUS_MEAN_L10_AWAY',
    'PLUS_MINUS_MEAN_L10_HOME',
    'PLUS_MINUS_MEAN_ML10_AWAY',
    'PLUS_MINUS_MEAN_ML10_HOME',
    'PTS_AGAINST_MEAN_AWAY',
    'PTS_AGAINST_MEAN_HOME',
    'PTS_AGAINST_MEAN_L10_AWAY',
    'PTS_AGAINST_MEAN_L10_HOME',
    'PTS_MEAN_AWAY',
    'PTS_MEAN_HOME',
    'PTS_MEAN_L10_AWAY',
    'PTS_MEAN_L10_HOME',
    'PYTHAGOREAN_EXPECTATION_AWAY',
    'PYTHAGOREAN_EXPECTATION_HOME',
    'PYTHAGOREAN_EXPECTATION_L10_AWAY',
    'PYTHAGOREAN_EXPECTATION_L10_HOME',
    'REB_AGAINST_MEAN_AWAY',
    'REB_AGAINST_MEAN_HOME',
    'REB_AGAINST_MEAN_L10_AWAY',
    'REB_AGAINST_MEAN_L10_HOME',
    'REB_MEAN_AWAY',
    'REB_MEAN_HOME',
    'REB_MEAN_L10_AWAY',
    'REB_MEAN_L10_HOME',
    'STL_AGAINST_MEAN_AWAY',
    'STL_AGAINST_MEAN_HOME',
    'STL_AGAINST_MEAN_L10_AWAY',
    'STL_AGAINST_MEAN_L10_HOME',
    'STL_MEAN_AWAY',
    'STL_MEAN_HOME',
    'STL_MEAN_L10_AWAY',
    'STL_MEAN_L10_HOME',
    'STL_MEAN_ML10_AWAY',
    'STL_MEAN_ML10_HOME',
    'TOV_AGAINST_MEAN_AWAY',
    'TOV_AGAINST_MEAN_HOME',
    'TOV_AGAINST_MEAN_L10_AWAY',
    'TOV_AGAINST_MEAN_L10_HOME',
    'TOV_MEAN_AWAY',
    'TOV_MEAN_HOME',
    'TOV_MEAN_L10_AWAY',
    'TOV_MEAN_L10_HOME',
    'TOV_MEAN_ML10_AWAY',
    'TOV_MEAN_ML10_HOME'
]

X_COLS = [
    'AST_AGAINST_MEAN_AWAY',
    'AST_AGAINST_MEAN_HOME',
    'AST_AGAINST_MEAN_L10_AWAY',
    'AST_AGAINST_MEAN_L10_HOME',
    'AST_AGAINST_MEAN_ML10_AWAY',
    'AST_AGAINST_MEAN_ML10_HOME',
    'AST_MEAN_AWAY',
    'AST_MEAN_HOME',
    'AST_MEAN_L10_AWAY',
    'AST_MEAN_L10_HOME',
    'AST_MEAN_ML10_AWAY',
    'AST_MEAN_ML10_HOME',
    'BLK_AGAINST_MEAN_AWAY',
    'BLK_AGAINST_MEAN_HOME',
    'BLK_AGAINST_MEAN_L10_AWAY',
    'BLK_AGAINST_MEAN_L10_HOME',
    'BLK_AGAINST_MEAN_ML10_AWAY',
    'BLK_AGAINST_MEAN_ML10_HOME',
    'BLK_MEAN_AWAY',
    'BLK_MEAN_HOME',
    'BLK_MEAN_L10_AWAY',
    'BLK_MEAN_L10_HOME',
    'BLK_MEAN_ML10_AWAY',
    'BLK_MEAN_ML10_HOME',
    'FG3_AGAINST_MEAN_AWAY',
    'FG3_AGAINST_MEAN_HOME',
    'FG3_AGAINST_MEAN_L10_AWAY',
    'FG3_AGAINST_MEAN_L10_HOME',
    'FG3_AGAINST_MEAN_ML10_AWAY',
    'FG3_AGAINST_MEAN_ML10_HOME',
    'FG3_MEAN_AWAY',
    'FG3_MEAN_HOME',
    'FG3_MEAN_L10_AWAY',
    'FG3_MEAN_L10_HOME',
    'FG3_MEAN_ML10_AWAY',
    'FG3_MEAN_ML10_HOME',
    'FG_AGAINST_MEAN_AWAY',
    'FG_AGAINST_MEAN_HOME',
    'FG_AGAINST_MEAN_L10_AWAY',
    'FG_AGAINST_MEAN_L10_HOME',
    'FG_AGAINST_MEAN_ML10_AWAY',
    'FG_AGAINST_MEAN_ML10_HOME',
    'FG_MEAN_AWAY',
    'FG_MEAN_HOME',
    'FG_MEAN_L10_AWAY',
    'FG_MEAN_L10_HOME',
    'FG_MEAN_ML10_AWAY',
    'FG_MEAN_ML10_HOME',
    'FT_AGAINST_MEAN_AWAY',
    'FT_AGAINST_MEAN_HOME',
    'FT_AGAINST_MEAN_L10_AWAY',
    'FT_AGAINST_MEAN_L10_HOME',
    'FT_AGAINST_MEAN_ML10_AWAY',
    'FT_AGAINST_MEAN_ML10_HOME',
    'FT_MEAN_AWAY',
    'FT_MEAN_HOME',
    'FT_MEAN_L10_AWAY',
    'FT_MEAN_L10_HOME',
    'FT_MEAN_ML10_AWAY',
    'FT_MEAN_ML10_HOME',
    'HOME_AST_AGAINST_MEAN',
    'HOME_AST_AGAINST_MEAN_L10',
    'HOME_AST_AGAINST_MEAN_ML10',
    'HOME_AST_MEAN',
    'HOME_AST_MEAN_L10',
    'HOME_AST_MEAN_ML10',
    'HOME_BLK_AGAINST_MEAN',
    'HOME_BLK_AGAINST_MEAN_L10',
    'HOME_BLK_AGAINST_MEAN_ML10',
    'HOME_BLK_MEAN',
    'HOME_BLK_MEAN_L10',
    'HOME_BLK_MEAN_ML10',
    'HOME_FG3_AGAINST_MEAN',
    'HOME_FG3_AGAINST_MEAN_L10',
    'HOME_FG3_AGAINST_MEAN_ML10',
    'HOME_FG3_MEAN',
    'HOME_FG3_MEAN_L10',
    'HOME_FG3_MEAN_ML10',
    'HOME_FG_AGAINST_MEAN',
    'HOME_FG_AGAINST_MEAN_L10',
    'HOME_FG_AGAINST_MEAN_ML10',
    'HOME_FG_MEAN',
    'HOME_FG_MEAN_L10',
    'HOME_FG_MEAN_ML10',
    'HOME_FT_AGAINST_MEAN',
    'HOME_FT_AGAINST_MEAN_L10',
    'HOME_FT_AGAINST_MEAN_ML10',
    'HOME_FT_MEAN',
    'HOME_FT_MEAN_L10',
    'HOME_FT_MEAN_ML10',
    'HOME_PLUS_MINUS_AGAINST_MEAN',
    'HOME_PLUS_MINUS_AGAINST_MEAN_L10',
    'HOME_PLUS_MINUS_MEAN',
    'HOME_PLUS_MINUS_MEAN_L10',
    'HOME_PLUS_MINUS_MEAN_ML10',
    'HOME_PTS_AGAINST_CUM',
    'HOME_PTS_AGAINST_CUM_L10',
    'HOME_PTS_AGAINST_CUM_ML10',
    'HOME_PTS_AGAINST_MEAN',
    'HOME_PTS_AGAINST_MEAN_L10',
    'HOME_PTS_AGAINST_MEAN_ML10',
    'HOME_PTS_MEAN',
    'HOME_PTS_MEAN_L10',
    'HOME_PTS_MEAN_ML10',
    'HOME_PYTHAGOREAN_EXPECTATION',
    'HOME_PYTHAGOREAN_EXPECTATION_L10',
    'HOME_PYTHAGOREAN_EXPECTATION_ML10',
    'HOME_REB_AGAINST_MEAN',
    'HOME_REB_AGAINST_MEAN_L10',
    'HOME_REB_AGAINST_MEAN_ML10',
    'HOME_REB_MEAN',
    'HOME_REB_MEAN_L10',
    'HOME_REB_MEAN_ML10',
    'HOME_STL_AGAINST_MEAN',
    'HOME_STL_AGAINST_MEAN_L10',
    'HOME_STL_AGAINST_MEAN_ML10',
    'HOME_STL_MEAN',
    'HOME_STL_MEAN_L10',
    'HOME_STL_MEAN_ML10',
    'HOME_TOV_AGAINST_MEAN',
    'HOME_TOV_AGAINST_MEAN_L10',
    'HOME_TOV_AGAINST_MEAN_ML10',
    'HOME_TOV_MEAN',
    'HOME_TOV_MEAN_L10',
    'HOME_TOV_MEAN_ML10',
    'HOME_W_L_CUM',
    'HOME_W_L_CUM_L10',
    'HOME_W_L_CUM_ML10',
    'PLUS_MINUS_AGAINST_MEAN_AWAY',
    'PLUS_MINUS_AGAINST_MEAN_HOME',
    'PLUS_MINUS_AGAINST_MEAN_L10_AWAY',
    'PLUS_MINUS_AGAINST_MEAN_L10_HOME',
    'PLUS_MINUS_MEAN_AWAY',
    'PLUS_MINUS_MEAN_HOME',
    'PLUS_MINUS_MEAN_L10_AWAY',
    'PLUS_MINUS_MEAN_L10_HOME',
    'PLUS_MINUS_MEAN_ML10_AWAY',
    'PLUS_MINUS_MEAN_ML10_HOME',
    'PTS_AGAINST_CUM_AWAY',
    'PTS_AGAINST_CUM_HOME',
    'PTS_AGAINST_CUM_L10_AWAY',
    'PTS_AGAINST_CUM_L10_HOME',
    'PTS_AGAINST_CUM_ML10_AWAY',
    'PTS_AGAINST_CUM_ML10_HOME',
    'PTS_AGAINST_MEAN_AWAY',
    'PTS_AGAINST_MEAN_HOME',
    'PTS_AGAINST_MEAN_L10_AWAY',
    'PTS_AGAINST_MEAN_L10_HOME',
    'PTS_AGAINST_MEAN_ML10_AWAY',
    'PTS_AGAINST_MEAN_ML10_HOME',
    'PTS_MEAN_AWAY',
    'PTS_MEAN_HOME',
    'PTS_MEAN_L10_AWAY',
    'PTS_MEAN_L10_HOME',
    'PTS_MEAN_ML10_AWAY',
    'PTS_MEAN_ML10_HOME',
    'PYTHAGOREAN_EXPECTATION_AWAY',
    'PYTHAGOREAN_EXPECTATION_HOME',
    'PYTHAGOREAN_EXPECTATION_L10_AWAY',
    'PYTHAGOREAN_EXPECTATION_L10_HOME',
    'PYTHAGOREAN_EXPECTATION_ML10_AWAY',
    'PYTHAGOREAN_EXPECTATION_ML10_HOME',
    'REB_AGAINST_MEAN_AWAY',
    'REB_AGAINST_MEAN_HOME',
    'REB_AGAINST_MEAN_L10_AWAY',
    'REB_AGAINST_MEAN_L10_HOME',
    'REB_AGAINST_MEAN_ML10_AWAY',
    'REB_AGAINST_MEAN_ML10_HOME',
    'REB_MEAN_AWAY',
    'REB_MEAN_HOME',
    'REB_MEAN_L10_AWAY',
    'REB_MEAN_L10_HOME',
    'REB_MEAN_ML10_AWAY',
    'REB_MEAN_ML10_HOME',
    'STL_AGAINST_MEAN_AWAY',
    'STL_AGAINST_MEAN_HOME',
    'STL_AGAINST_MEAN_L10_AWAY',
    'STL_AGAINST_MEAN_L10_HOME',
    'STL_AGAINST_MEAN_ML10_AWAY',
    'STL_AGAINST_MEAN_ML10_HOME',
    'STL_MEAN_AWAY',
    'STL_MEAN_HOME',
    'STL_MEAN_L10_AWAY',
    'STL_MEAN_L10_HOME',
    'STL_MEAN_ML10_AWAY',
    'STL_MEAN_ML10_HOME',
    'TOV_AGAINST_MEAN_AWAY',
    'TOV_AGAINST_MEAN_HOME',
    'TOV_AGAINST_MEAN_L10_AWAY',
    'TOV_AGAINST_MEAN_L10_HOME',
    'TOV_AGAINST_MEAN_ML10_AWAY',
    'TOV_AGAINST_MEAN_ML10_HOME',
    'TOV_MEAN_AWAY',
    'TOV_MEAN_HOME',
    'TOV_MEAN_L10_AWAY',
    'TOV_MEAN_L10_HOME',
    'TOV_MEAN_ML10_AWAY',
    'TOV_MEAN_ML10_HOME',
    'W_L_CUM_AWAY',
    'W_L_CUM_HOME',
    'W_L_CUM_L10_AWAY',
    'W_L_CUM_L10_HOME',
    'W_L_CUM_ML10_AWAY',
    'W_L_CUM_ML10_HOME'
]

Y_CLF_COL = [
    'HOME_WINS'
]

Y_REG_COL = [
    'HOME_POINT_SPREAD'
]

X_ORDINAL_COLS = [
    "SEASON",
    # "HT_RANK",
    # "HT_CLASS",
    # "VT_RANK",
    # "VT_CLASS"
]

X_NUM_COLS = [
    "HT_HW",
    # "HT_HL",
    "HT_VW",
    # "HT_VL",
    "HT_LAST10_W",
    # "HT_LAST10_L",
    "HT_LAST10_MATCHUP_W",
    # "HT_LAST10_MATCHUP_L",
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
    "VT_HW",
    # "VT_HL",
    "VT_VW",
    # "VT_VL",
    "VT_LAST10_W",
    # "VT_LAST10_L",
    "VT_LAST10_MATCHUP_W",
    # "VT_LAST10_MATCHUP_L",
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
    "VT_AWAY_REB"
]

X_COLUMNS = X_ORDINAL_COLS + X_NUM_COLS

Y_COLUMNS = [
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
