#!/usr/bin/env python
import model.config as model_config


def X_y_values(df, X_columns=model_config.X_COLUMNS, y_columns=model_config.Y_COLUMNS):
    X = df.loc[:, X_columns].values
    y = df.loc[:, y_columns].values
    return X, y


if __name__ == '__main__':
    pass
