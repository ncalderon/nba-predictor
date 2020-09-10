#!/usr/bin/env python
import model.config as model_config


def X_y_values(df, X_columns=model_config.X_columns, y_columns=model_config.y_columns):
    X = df.loc[:, X_columns].values
    y = df.loc[:, y_columns].values
    return X, y

if __name__ == '__main__':
    pass
