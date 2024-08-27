# In this file we will perform MonteCarlo simulations to compute the probabily of a team to win the champions in a season
# we will also make an extra function to compute the compound probability of winning multiple champions

'''
Per a cada temporada:
-Pillar el bracket de vuitens de cada any de Champions (desde 2015)
-Simular els partits (i per tant, eliminatòries)
-Veure quants cops guanya el Madrid la simulació
-Això em dona una probabilitat que el Madrid guanyi aquella temporada la Champions

Fer probabilitat combinada que hagi guanyat tants cops com ha guanyat
'''
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt


def simulate_tournament(
    df_model,
    model, 
    teams, 
    season, 
    bracket_sim,
    start_round='sixteen'
):
    """
    Simulates a single tournament and returns the winner.
    
    Parameters:
        model: Trained ML model to predict match outcomes.
        teams: List of teams in the current round.
        season: The current season being simulated.
        start_round: The round at which the simulation starts (default: 'Round of 16').

    Returns:
        winner: The team that wins the championship.
    """
    rounds = ['sixteen', 'quarter', 'semis', 'final']
    current_round = start_round
    year =  int(season.split('-')[0])


    while len(teams) > 1:

        # print(current_round)
        # print(len(teams))
        next_round_teams = []
        
        # Iterate over match pairs
        for i in range(0, len(teams), 2):
            team1 = teams[i]
            team2 = teams[i + 1]

            # add to sim bracket
            if current_round != 'final':
                key_a = current_round+str(int(i/2)+1)+'A'
                key_b = current_round+str(int(i/2)+1)+'B'
                bracket_sim[key_a][team1] = bracket_sim[key_a][team1] +1
                bracket_sim[key_b][team2] = bracket_sim[key_b][team2] +1
            else:
                bracket_sim['finalA'][team1] = bracket_sim['finalA'][team1] +1
                bracket_sim['finalB'][team2] = bracket_sim['finalB'][team2] +1
            
            match_data = pd.DataFrame({
                'Season': [season], 
                'Year': [year], 
                'Round': [current_round], 
                'HomeTeam': [team1], 
                'AwayTeam': [team2]
            })

            # Prepare the input data for the ML model
            home_columns = [
                'PositionHome', 'GamesPlayedHome', 'WinsHome', 'DrawsHome', 'LossesHome', 
                'GoalsFavourHome', 'GoalsAgainstHome', 'GoalDifferenceHome', 'PointsHome', 
                'LeagueWeightHome', 'HistoricalPointsHome', 'HistoricalPositionHome', 
                'count_sixteenHome', 'count_championsHome', 'WinsChampionsHome', 'count_finalHome', 
                'GoalDifferenceChampionsHome', 'GoalsFavourChampionsHome', 'DrawsChampionsHome', 
                'count_semisHome', 'GoalsAgainstChampionsHome', 'count_quarterHome', 'PositionChampionsHome', 
                'GamesPlayedChampionsHome', 'LossesChampionsHome', 'PointsChampionsHome'
            ]

            away_columns = [
                'PositionAway', 'GamesPlayedAway', 'WinsAway', 'DrawsAway', 'LossesAway', 
                'GoalsFavourAway', 'GoalsAgainstAway', 'GoalDifferenceAway', 'PointsAway', 
                'LeagueWeightAway', 'HistoricalPointsAway', 'HistoricalPositionAway', 
                'count_sixteenAway', 'GoalsAgainstChampionsAway', 'count_semisAway', 
                'GamesPlayedChampionsAway', 'GoalsFavourChampionsAway', 'PositionChampionsAway', 
                'count_championsAway', 'PointsChampionsAway', 'WinsChampionsAway', 
                'count_quarterAway', 'GoalDifferenceChampionsAway', 'count_finalAway', 
                'DrawsChampionsAway', 'LossesChampionsAway' 
            ]

            # home features
            for col in home_columns:
                match_data[col] = df_model[
                    (df_model['Season']==season) & (df_model['HomeTeam']==team1)
                ][col].head(1)

            # away features
            for col in away_columns:
                match_data[col] = df_model[
                    (df_model['Season']==season) & (df_model['AwayTeam']==team2)
                ][col].head(1) 
            
            match_data.fillna(0, inplace=True)
            
            match_data['Season'] = match_data['Season'].astype('category')
            match_data['Round'] = match_data['Round'].astype('category')
            match_data['HomeTeam'] = match_data['HomeTeam'].astype('category')
            match_data['AwayTeam'] = match_data['AwayTeam'].astype('category')
            #match_data['HomeLeagueWeight'] = pd.to_numeric(match_data['HomeLeagueWeight'], errors='coerce') #.astype('float')
            #match_data['AwayLeagueWeight'] = pd.to_numeric(match_data['AwayLeagueWeight'], errors='coerce') #.astype('float')

            # Predict the probability of team1 winning
            match_data = match_data[model.get_booster().feature_names]
            win_prob = model.predict_proba(match_data)[:, 1][0]

            # Ensure win_prob is a valid probability
            if not 0 <= win_prob <= 1:
                raise ValueError("win_prob must be between 0 and 1")

            # Calculate and normalize the probabilities
            probabilities = [win_prob, 1 - win_prob]
            probabilities = np.array(probabilities) / np.sum(probabilities)

            # Determine the winner based on the predicted probability
            winner = np.random.choice(
                a=[team1, team2], 
                p=probabilities # p=[0.5, 0.5] 
            )
            next_round_teams.append(winner)

            if current_round=='final':
                bracket_sim['champion'][winner] = bracket_sim['champion'][winner] +1
        
        teams = next_round_teams

        if len(teams) > 1:
            current_round = rounds[rounds.index(current_round) + 1]  # Move to the next round

    return teams[0], bracket_sim  # The last remaining team is the winner


def monte_carlo_simulation(
    df_model,
    model, 
    teams, 
    season, 
    num_simulations=1000
):
    """
    Runs the Monte Carlo simulation to estimate the probability of each team winning the championship.
    
    Parameters:
        model: Trained ML model to predict match outcomes.
        teams: List of teams in the Round of 16.
        season: The current season being simulated.
        num_simulations: Number of simulations to run (default: 1000).
        
    Returns:
        win_counts: Dictionary containing the number of championships won by each team.
    """
    win_counts = defaultdict(int)

    bracket_sim = {
        'champion': {t: 0 for t in teams},
        'finalA': {t: 0 for t in teams},
        'finalB': {t: 0 for t in teams},
        'semis1A': {t: 0 for t in teams},
        'semis1B': {t: 0 for t in teams},
        'semis2A': {t: 0 for t in teams},
        'semis2B': {t: 0 for t in teams},
        'quarter1A': {t: 0 for t in teams},
        'quarter1B': {t: 0 for t in teams},
        'quarter2A': {t: 0 for t in teams},
        'quarter2B': {t: 0 for t in teams},
        'quarter3A': {t: 0 for t in teams},
        'quarter3B': {t: 0 for t in teams},
        'quarter4A': {t: 0 for t in teams},
        'quarter4B': {t: 0 for t in teams},
        'sixteen1A': {t: 0 for t in teams},
        'sixteen1B': {t: 0 for t in teams},
        'sixteen2A': {t: 0 for t in teams},
        'sixteen2B': {t: 0 for t in teams},
        'sixteen3A': {t: 0 for t in teams},
        'sixteen3B': {t: 0 for t in teams},
        'sixteen4A': {t: 0 for t in teams},
        'sixteen4B': {t: 0 for t in teams},
        'sixteen5A': {t: 0 for t in teams},
        'sixteen5B': {t: 0 for t in teams},
        'sixteen6A': {t: 0 for t in teams},
        'sixteen6B': {t: 0 for t in teams},
        'sixteen7A': {t: 0 for t in teams},
        'sixteen7B': {t: 0 for t in teams},
        'sixteen8A': {t: 0 for t in teams},
        'sixteen8B': {t: 0 for t in teams}
    }
    
    for i in range(num_simulations):
        print('Iteration:', i)
        winner, bracket_sim = simulate_tournament(df_model, model, teams, season, bracket_sim)
        win_counts[winner] += 1
    
    # Convert win counts to probabilities
    win_probabilities = {team: count / num_simulations for team, count in win_counts.items()}

    win_probabilities = {team: count / num_simulations for team, count in win_counts.items()}
    
    return win_probabilities, bracket_sim


def sixteen_draw(df_champions, season):

    df_season = df_champions[
        (df_champions['Season']==season) & (df_champions['Round']=='sixteen')
    ]

    teams_sixteen_draw = []

    # Iterate over each row in the DataFrame
    for _, row in df_season.iterrows():
        # Add the HomeTeam if it's not already in the list
        if row['HomeTeam'] not in teams_sixteen_draw:
            teams_sixteen_draw.append(row['HomeTeam'])
        
        # Add the AwayTeam if it's not already in the list
        if row['AwayTeam'] not in teams_sixteen_draw:
            teams_sixteen_draw.append(row['AwayTeam'])

    return teams_sixteen_draw


def read_data(
    season,
    path_elimination_games='data/scrapper/champions_elimination.csv',
    path_df_model='data/preprocessed/model_table.csv'
):
    df_champions = pd.read_csv(path_elimination_games, sep=';')
    df_model = pd.read_csv(path_df_model, sep=';')

    # Load the trained ML model from the pickle file
    with open(f'models/xgb_classifier_{season}.pkl', 'rb') as file:
        model = pickle.load(file)

    return df_champions, df_model, model


def plot_bracket(
    bracket_sim,
    plot_path = os.path.join('static', 'bracket_plot.png')
):

    # Step 1: Compute the Most Probable Team for Each Position
    most_probable_bracket = {}
    for position, probabilities in bracket_sim.items():
        # Get the team with the maximum probability
        most_probable_team = max(probabilities, key=probabilities.get)
        most_probable_bracket[position] = most_probable_team

    print(most_probable_bracket)

    # Step 2: Plotting the Most Probable Bracket
    fig, ax = plt.subplots(figsize=(12, 8))
    rounds = ['sixteen', 'quarter', 'semis', 'final', 'champion']
    
    # Define the positions for teams in each round
    pos = {
        'sixteen': [i for i in range(16)],
        'quarter': [i*2 for i in range(8)],
        'semis': [i*4 for i in range(4)],
        'final': [i*8 for i in range(2)],
        'champion': [8]
    }

    # Offset to adjust the vertical position in the plot
    offset = 0.4
    
    # Plot each round
    for i, rnd in enumerate(rounds[:-1]):
        # Get the team names for this round
        if rnd != 'final':
            teams = []
            for j in range(1, len(pos[rnd])//2 + 1):
                teams.append(most_probable_bracket[f'{rnd}{j}A'])
                teams.append(most_probable_bracket[f'{rnd}{j}B'])
        else:
            teams = [
                most_probable_bracket[f'{rnd}A'],
                most_probable_bracket[f'{rnd}B']
            ]
        
        for j, team in enumerate(teams):
            ax.text(i, pos[rnd][j] + offset, team, ha='center', va='center', fontsize=10)
            
            # Draw lines connecting the rounds
            if i < len(rounds) - 2:  # No line from the final round to the champion
                next_rnd = rounds[i+1]
                if j % 2 == 0:  # Connect the two teams that lead to the next round
                    ax.plot([i, i+1], [pos[rnd][j] + offset, pos[next_rnd][j//2] + offset], color='gray')
                else:
                    ax.plot([i, i+1], [pos[rnd][j] + offset, pos[next_rnd][j//2] + offset], color='gray')
    
    # Plot the champion in the final round
    champion_team = most_probable_bracket['champion']
    ax.text(len(rounds)-1, pos['champion'][0] + offset, champion_team, ha='center', va='center', fontsize=12, fontweight='bold')

    # Setting the plot limits and removing ticks
    ax.set_xlim(-0.5, len(rounds) - 0.5)
    ax.set_ylim(-1, 16)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(['Round of 16', 'Quarterfinals', 'Semifinals', 'Final', 'Champion'])
    ax.set_yticks([])
    
    # Enhance the plot's appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    #plt.show()


def main(
    season = '2020-2021',
    plot_path=os.path.join('static', 'bracket_plot.png')
):
    df_champions, df_model, model = read_data(season=season)

    teams_sixteen_draw = sixteen_draw(df_champions, season)

    win_probabilities, bracket_sim = monte_carlo_simulation(
        df_model,
        model, 
        teams_sixteen_draw, 
        season, 
        num_simulations=10 # 10
    )

    # Plot most probable bracket
    plot_bracket(
        bracket_sim, 
        plot_path
    )

    # Display the results
    sorted_win_probabilities = sorted(win_probabilities.items(), key=lambda x: x[1], reverse=True)

    for team, prob in sorted_win_probabilities:
        print(f"{team}: {prob:.2%} chance of winning the championship")

    return win_probabilities #sorted_win_probabilities

if __name__ == "__main__":
    sorted_win_probabilities = main()

{
    'champion': 'Juventus', 
    'finalA': 'Real Madrid', 'finalB': 'Juventus', 
    
    'semis1A': 'Manchester City', 'semis1B': 'Real Madrid', 
    'semis2A': 'Dortmund', 'semis2B': 'Juventus', 
    
    
    'quarter1A': 'Paris S-G', 'quarter1B': 'Manchester City', 
    'quarter2A': 'Bayern Munich', 'quarter2B': 'Real Madrid', 
    'quarter3A': 'Liverpool', 'quarter3B': 'Dortmund', 
    'quarter4A': 'Chelsea', 'quarter4B': 'Juventus', 
    
    
    'sixteen1A': 'Barcelona', 'sixteen1B': 'Paris S-G', 
    'sixteen2A': 'Gladbach', 'sixteen2B': 'Manchester City', 
    'sixteen3A': 'Lazio', 'sixteen3B': 'Bayern Munich', 
    'sixteen4A': 'Atalanta', 'sixteen4B': 'Real Madrid', 
    'sixteen5A': 'RB Leipzig', 'sixteen5B': 'Liverpool', 
    'sixteen6A': 'Sevilla', 'sixteen6B': 'Dortmund', 
    'sixteen7A': 'Atlético Madrid', 'sixteen7B': 'Chelsea', 
    'sixteen8A': 'Porto', 'sixteen8B': 'Juventus'
}