# In this file we will create a function: 
# that given a matchup (HomeTeam, AwayTeam) and a Season, will give the chance of each team to win


# We can make a rule based system or a ML model
'''
Funció de probabilitat de guanyar, donats dos equips (potser un ML per cada any, entrenat 
amb dades prèvies a aquell any i resultats de lliga del mateix any)
'''

# Necessitarem una taula amb equip de casa i equip de fora + 
# features dels equips + 
# Target (1 guanya el de casa, 0 guanya el visitant)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from xgboost import plot_importance



def read_data_season(season):
    # Load the data
    df = pd.read_csv('data/preprocessed/model_table.csv', sep=';')
    
    # for which season are we going to predict
    df = df[
        (df['Season'] <= season) & (df['LocalWin'] != -1)
    ]

    df['Season'] = df['Season'].astype('category')
    df['Round'] = df['Season'].astype('category')
    df['HomeTeam'] = df['HomeTeam'].astype('category')
    df['AwayTeam'] = df['AwayTeam'].astype('category')

    with open('data/preprocessed/label_encoders.json', 'r') as f:
        loaded_label_encoder_data = json.load(f)

        label_encoders = {}
        for col, classes in loaded_label_encoder_data.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            label_encoders[col] = le

    return df, label_encoders


def train_test_split(df, label_encoders, season):
    # Train, Test
    # Get the mapping of original values to encoded labels
    #season_le = label_encoders['SeasonEncoded']
    #season_mapping = dict(zip(season_le.classes_, range(len(season_le.classes_))))
    #train = df[~(df['Season'] == season_mapping[season])]
    #test = df[df['Season'] == season_mapping[season]]

    train = df[~(df['Season'] == season)]
    test = df[df['Season'] == season]

    #encoded_columns = ['Season', 'Round', 'HomeTeam', 'AwayTeam']

    # Define X (features) and y (target)
    X_train = train.drop(columns=['LocalWin'], axis=1)
    X_test = test.drop(columns=['LocalWin'], axis=1)
    y_train = train['LocalWin']
    y_test = test['LocalWin']

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    # Initialize and train the XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss', enable_categorical=True)
    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


def main(season='2020-2021'):
    print('Season:', season)
    df, label_encoders = read_data_season(season)
    X_train, X_test, y_train, y_test = train_test_split(df, label_encoders, season)
    print(X_train.columns)
    model = train(X_train, y_train)
    print('FEATURE IMPORTANCES')
    print(model.feature_importances_)

    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()
    plot_importance(model)

    evaluate(model, X_test, y_test)

    return model


if __name__ == "__main__":
    season = '2020-2021'
    #model = main(season)

    for season in [
        '2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', 
        '2020-2021', '2021-2022', '2022-2023'
    ]:
        model = main(season)

        # save model
        pickle.dump(model, open(f'models/xgb_classifier_{season}.pkl', "wb"))
    
