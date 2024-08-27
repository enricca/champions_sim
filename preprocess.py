import pandas as pd
import numpy as np
import json
import yaml
from sklearn.preprocessing import LabelEncoder


# In this file we will generate the necessary data to create the further probabilty funtion
# We will also generate the data to start the monte carlo simulations

'''
Resultats de vuitens en endavant a la champions
Resultats a fase de grups de champions (si puc, per partit. Si no, nombre de victòries, si pot ser a casa i fora)
Win Rate, gols a favor i gols en contra a la seva lliga (si pot ser, a casa i fora)
Cada equip, de quina lliga és
'''

# TODO tipar-ho tot, comentar i posar alguns asserts
# Treure comentaris innecessaris, ...

def clean_teams(df, column_name, team_map):
    df[column_name] = df[column_name].map(team_map).fillna(df[column_name])
    df[column_name] = df[column_name].str.replace(r"\s*\*\s*", "", regex=True)
    df[column_name] = df[column_name].str.replace('Ã©', 'é')

    return df


def rename_teams(
    df_champions_league,
    df_local,
    df_champions_elimination
):
    # Load dictionary
    with open('utils/team_renaming.yaml', 'r') as f:
        team_map = yaml.load(f, Loader=yaml.SafeLoader)
    
    df_champions_league = clean_teams(df_champions_league, "Team", team_map)
    df_local = clean_teams(df_local, "Team", team_map)
    df_champions_elimination = clean_teams(df_champions_elimination, "HomeTeam", team_map)
    df_champions_elimination = clean_teams(df_champions_elimination, "AwayTeam", team_map)

    return df_champions_league, df_local, df_champions_elimination


def preprocess_elimination_games(
        df: pd.DataFrame
):
    print('Processing Elimination games...')

    df['Score'].fillna('9–9', inplace=True)

    # Handle missing or empty 'Score' values by replacing them with a placeholder
    df.loc[df['Score'].str.strip() == "", "Score"] = "9–9"

    # Split the 'Score' column into 'HomeGoals' and 'AwayGoals' and convert to integers
    df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split('–', expand=True)
    #print(df['HomeGoals'].value_counts())

    df['HomeGoals'] = df['HomeGoals'].astype(int)
    df['AwayGoals'] = df['AwayGoals'].astype(int)

    # Determine the Winning and Losing teams
    df['WinningTeam'] = np.where(
        df['HomeGoals'] > df['AwayGoals'],
        df['HomeTeam'],
        np.where(df['HomeGoals'] < df['AwayGoals'], df['AwayTeam'], 'Draw')
    )

    df['LosingTeam'] = np.where(
        df['HomeGoals'] < df['AwayGoals'],
        df['HomeTeam'],
        np.where(df['HomeGoals'] > df['AwayGoals'], df['AwayTeam'], 'Draw')
    )
    
    return df


def calculate_season_weights(seasons):
    """Calculate weights for the seasons based on exponential decay."""
    decay_rate = 0.5  # Example decay rate; adjust as needed
    return np.exp(-decay_rate * (seasons.max() - seasons))


def local_league_hist(df_local):

    df_local['Year'] = df_local['Season'].str.split('-').str[0].astype(int)

    # Score d'equip per històric (lliga local, un històric per a cada season)
    df_local = df_local.sort_values(by=['Team', 'Season']).reset_index(drop=True)
    historical_points = []
    historical_positions = []
    for idx, row in df_local.iterrows():
        # Filter previous seasons for the same team
        past_seasons = df_local[(df_local['Team'] == row['Team']) & (df_local['Year'] < row['Year'])]
        
        if not past_seasons.empty:
            # Calculate weights
            weights = calculate_season_weights(past_seasons['Year'])
            
            # Weighted average of Points
            weighted_points = np.average(past_seasons['Points'], weights=weights)
            historical_points.append(weighted_points)
            
            # Weighted average of Position
            weighted_position = np.average(past_seasons['Position'], weights=weights)
            historical_positions.append(weighted_position)
        else:
            # No past seasons available, default to NaN or 0
            historical_points.append(row['Points'])
            historical_positions.append(row['Position'])
    
    df_local['HistoricalPoints'] = historical_points
    df_local['HistoricalPosition'] = historical_positions
    
    return df_local


def preprocess_local_leagues(df_local, league_weights):

    print('Processing local leagues...')
    # League weight
    df_local['LeagueWeight'] = df_local['League'].map(league_weights)
    df_local['LeagueWeight'].fillna(-2, inplace=True)

    for col in [
        'Wins', 'Draws', 'Losses', 'GoalsFavour', 'GoalsAgainst', 'GoalDifference', 'Points'
    ]:
        df_local[col] = df_local[col] / df_local['GamesPlayed']

    
    # Historical features
    df_local = local_league_hist(df_local)

    return df_local


def local_league_weights(
        df_champions,
        df_local
):
    print('Computing league weights...')

    team_league = df_local[['League', 'Team']].drop_duplicates()
    team_league_dict = team_league.set_index('Team')['League'].to_dict()

    df = df_champions[['WinningTeam', 'LosingTeam']]

    # Use the dictionary to map leagues to the WinningTeam and LosingTeam
    df['WinningTeamLeague'] = df['WinningTeam'].map(team_league_dict)
    df['LosingTeamLeague'] = df['LosingTeam'].map(team_league_dict)

    league_avgs = df.groupby('WinningTeamLeague', as_index=True).agg(
        weights=('WinningTeam', 'count'),
        #win_teams=('WinningTeam', 'nunique')
    )

    # normalize columns
    league_avgs['weights']=(league_avgs['weights']-league_avgs['weights'].mean())/league_avgs['weights'].std()
    #league_avgs['win_teams']=(league_avgs['win_teams']-league_avgs['win_teams'].mean())/league_avgs['win_teams'].std()

    league_weights = league_avgs['weights'].to_dict()

    return league_weights, team_league_dict


def build_model_dataset(df_champions, df_local):

    # our base will be the elimination games in champions league
    df_model = df_champions[
        ['Season', 'Round', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'WinningTeam']
    ]
    #    df_champions['WinningTeam'] != "Draw"

    # Target: did the local team win?
    df_model['LocalWin'] = np.where(
        df_model['HomeTeam'] == df_model['WinningTeam'], 
        1, 
        np.where(
            df_model['AwayTeam'] == df_model['WinningTeam'], 
            0,
            -1
        )
    )

    # Add local league features HomeTeam
    features_local_league = [
            'Season', 'Position', 'Team', 'GamesPlayed', 'Wins', 'Draws',
            'Losses', 'GoalsFavour', 'GoalsAgainst', 'GoalDifference', 'Points',
            'LeagueWeight', 'HistoricalPoints', 'HistoricalPosition'
        ]
    
    df_model = df_model.merge(
        df_local[features_local_league],
        how='left',
        left_on=['Season', 'HomeTeam'],
        right_on=['Season', 'Team'],
    )

    # Add local league features AwayTeam
    df_model = df_model.merge(
        df_local[features_local_league],
        how='left',
        left_on=['Season', 'AwayTeam'],
        right_on=['Season', 'Team'],
        suffixes=('Home', 'Away')
    )

    df_model['Year'] = df_model['Season'].str.split('-').str[0].astype(int)

    # delete unncessary columns
    df_model.drop(columns=['WinningTeam', 'HomeGoals', 'AwayGoals', 'TeamHome', 'TeamAway'], inplace=True)
    df_model.fillna(0, inplace=True)

    return df_model


def features_champions_elimination_team(df_model, df_champions_elimination, df_features, home_away):
    for round in ['sixteen', 'quarter', 'semis', 'final']:
        df_features['Count_'+round+home_away] = 0
        df_features['Count_'+round+home_away] = 0
    
    # Iterate through the DataFrame
    for index, row in df_features.iterrows():
        # Filter for previous seasons and the same HomeTeam
        previous_matches = df_champions_elimination[
            (df_champions_elimination['Season'] < row['Season']) & 
            (
                (df_champions_elimination['HomeTeam'] == row[home_away+'Team']) |
                (
                    (df_champions_elimination['AwayTeam'] == row[home_away+'Team']) & 
                    (df_champions_elimination['AwayTeam']=='final')
                )
            )
        ]

        # Rounds
        for round in ['sixteen', 'quarter', 'semis', 'final']:
            df_features.at[index, 'count_'+round+home_away] = previous_matches[
                previous_matches['Round']==round
            ].shape[0]

        # championships
        df_features.at[index, 'count_champions'+home_away] = previous_matches[
            (previous_matches['Round']=='final') & 
            ((previous_matches['HomeTeam']==row[home_away+'Team']))
        ].shape[0]

    df_model = df_model.merge(
        df_features[['Season', home_away+'Team', 'count_sixteen'+home_away, 'count_quarter'+home_away, 
                    'count_semis'+home_away, 'count_final'+home_away, 'count_champions'+home_away]],
        how='left',
        on=['Season', home_away+'Team']
    )

    # fillna
    df_model[
        ['count_sixteen'+home_away, 'count_quarter'+home_away, 'count_semis'+home_away, 
            'count_final'+home_away, 'count_champions'+home_away]
    ] = df_model[
        ['count_sixteen'+home_away, 'count_quarter'+home_away, 'count_semis'+home_away, 
            'count_final'+home_away, 'count_champions'+home_away]
    ].fillna(0)

    return df_model


def features_champions_elimination(df_model, df_champions_elimination):
    df_champions_elimination = df_champions_elimination.sort_values(by=['Season', 'Round'])
    df_features = df_champions_elimination.drop_duplicates(['Season', 'HomeTeam'])

    df_model = features_champions_elimination_team(df_model, df_champions_elimination, df_features, 'Home')
    df_model = features_champions_elimination_team(df_model, df_champions_elimination, df_features, 'Away')

    return df_model


def features_champions_league(df_model, df_champions_league):

    for col in [
        'Wins', 'Draws', 'Losses', 'GoalsFavour', 'GoalsAgainst', 'GoalDifference', 'Points'
    ]:
        df_champions_league[col] = df_champions_league[col] / df_champions_league['GamesPlayed']

    # rename columns
    df_champions_league.columns = df_champions_league.columns+'Champions'

    df_model = df_model.merge(
        df_champions_league,
        how='left',
        left_on=['Season', 'HomeTeam'],
        right_on=['SeasonChampions', 'TeamChampions']
    )

    df_model = df_model.merge(
        df_champions_league,
        how='left',
        left_on=['Season', 'AwayTeam'],
        right_on=['SeasonChampions', 'TeamChampions'],
        suffixes=('Home', 'Away')
    )

    df_model.drop(
        columns=[
            'SeasonChampionsHome', 'SeasonChampionsAway', 'LeagueChampionsHome', 'LeagueChampionsAway', 
            'TeamChampionsHome', 'TeamChampionsAway'
        ], 
        inplace=True
    )

    df_model.fillna(0, inplace=True)

    return df_model


def encode_categorical(df):
    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['Season', 'Round', 'HomeTeam', 'AwayTeam']

    for col in categorical_columns:
        new_col = col+'Encoded'
        le = LabelEncoder()
        df[new_col] = le.fit_transform(df[col])
        label_encoders[new_col] = le

    label_encoder_data = {}
    for col, le in label_encoders.items():
        label_encoder_data[col] = list(le.classes_)

    return df, label_encoder_data
    

def main(
    champions_elimination_path = 'data/scrapper/champions_elimination.csv', 
    local_leagues_path = 'data/scrapper/local_leagues_classifications.csv',
    champions_league_path = 'data/scrapper/local_leagues_classifications.csv'
):
    # Read files
    df_champions_elimination = pd.read_csv(champions_elimination_path, sep=';')
    df_local = pd.read_csv(local_leagues_path, sep=';')
    df_champions_league = pd.read_csv(champions_league_path, sep=';')

    # rename teams
    df_champions_league, df_local, df_champions_elimination = rename_teams(
        df_champions_league, 
        df_local, 
        df_champions_elimination
    )

    # Process eliminations
    df_champions_elimination = preprocess_elimination_games(df_champions_elimination)

    # Compute weights for each local league
    league_weights, team_league_dict= local_league_weights(df_champions_elimination, df_local)

    # Process local leagues
    df_local = preprocess_local_leagues(df_local, league_weights)

    # Build table for the model 
    df_model = build_model_dataset(df_champions_elimination, df_local)

    df_model = features_champions_elimination(df_model, df_champions_elimination)

    df_model = features_champions_league(df_model, df_champions_league)

    # df_model, label_encoder_data = encode_categorical(df_model)

    return df_champions_elimination, df_local, df_model, league_weights, team_league_dict#, label_encoder_data


if __name__ == "__main__":
    
    df_champions_elimination, df_local, df_model, league_weights, team_league_dict = main()

    # Save or further process the DataFrame `df` as needed
    df_champions_elimination.to_csv('data/preprocessed/champions_elimination_preprocessed.csv', sep=';', index=False)
    df_local.to_csv('data/preprocessed/local_leagues_preprocessed.csv', sep=';', index=False)
    df_model.to_csv('data/preprocessed/model_table.csv', sep=';', index=False)
    
    with open('data/preprocessed/league_weights.json', 'w') as fp:
        json.dump(league_weights, fp)
    with open('data/preprocessed/team_league_dict.json', 'w') as fp:
        json.dump(team_league_dict, fp)
    #with open('data/preprocessed/label_encoders.json', 'w') as fp:
    #    json.dump(label_encoder_data, fp)
        
        


# TODO Pensar com tracto els nulls. Lligues que no tinc, weight local i posició a la seva lliga
#       ### Segurament, posar-li molt mala nota a la lliga i bona a lo de la seva posició dins la lliga
#       ### Per exemple, el mateix que el 1r de la ligue-1

