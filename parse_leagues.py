
import pandas as pd
from lxml import etree
import os


def prepare_parser(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    parser = etree.HTMLParser()
    tree = etree.fromstring(html_content, parser)

    return tree


def build_dataframe_row(
    season, 
    league,
    position, 
    all_rows,
    tree,
    team_xpath, 
    games_played_xpath, 
    wins_xpath, 
    draws_xpath, 
    losses_xpath, 
    goals_favour_xpath, 
    goals_against_xpath, 
    goal_difference_xpath, 
    points_xpath
):
    working = True

    # print('Extracting paths...')
    # print('team_xpath:', team_xpath)

    # Extract data using XPath
    team_elements = tree.xpath(team_xpath)
    games_playes_elements = tree.xpath(games_played_xpath)
    wins_elements = tree.xpath(wins_xpath)
    draws_elements = tree.xpath(draws_xpath)
    losses_elements = tree.xpath(losses_xpath)
    goals_favour_elements = tree.xpath(goals_favour_xpath)
    goals_against_elements = tree.xpath(goals_against_xpath)
    goal_difference_elements = tree.xpath(goal_difference_xpath)
    points_elements = tree.xpath(points_xpath)

    #print('team_elements:', team_elements)

    if not team_elements:
        print('NOT TEAM ELEMENTS')
        working=False

    if working:
        # Collect match data
        for team, games_played, wins, draws, losses, goals_favour, goals_against, goal_difference, points in zip(
            team_elements, games_playes_elements, wins_elements, draws_elements, losses_elements, 
            goals_favour_elements, goals_against_elements, goal_difference_elements, points_elements
        ):
            #print('team:', team.text.strip())
            #print('games_played:', games_played.text.strip())
            #print('wins:', wins.text.strip())
            row = {
                "Season": str(season)+'-'+str(season+1),
                "League": league,
                "Position": position,
                "Team": team.text.strip(),
                "GamesPlayed": games_played.text.strip(),
                "Wins": wins.text.strip(), 
                "Draws": draws.text.strip(), 
                "Losses": losses.text.strip(), 
                "GoalsFavour": goals_favour.text.strip(), 
                "GoalsAgainst": goals_against.text.strip(), 
                "GoalDifference": goal_difference.text.strip(), 
                "Points": points.text.strip(), 

            }
            all_rows.append(row)

    return all_rows, working


def parse_local_league_season(league, season):
    """
    Parses the Champions League HTML file for a given season and extracts match data.

    Args:
        season (str): The season year to parse (e.g., '2022-2023').

    Returns:
        pd.DataFrame: A DataFrame containing match data for the specified season.
    """

    # Construct the file path and read the HTML content
    file_path = f'htmls/leagues/{league}_{season}.html'

    tree = prepare_parser(file_path)

    # Initialize an empty DataFrame to store all match data
    all_rows = []

    # Iterate through each round and corresponding matches
    position = 1
    base_xpath = f'/html/body/div[5]/div/div[1]/div[3]/div[1]/div[2]/div/table/tbody/'
    working = True

    while working:
        print('Position:', position)
        try:
            # Construct XPaths
            team_xpath = base_xpath+f'tr[{position}]/td[2]/a'
            games_played_xpath = base_xpath+f'tr[{position}]/td[3]'
            wins_xpath = base_xpath+f'tr[{position}]/td[4]'
            draws_xpath = base_xpath+f'tr[{position}]/td[5]'
            losses_xpath = base_xpath+f'tr[{position}]/td[6]'
            goals_favour_xpath = base_xpath+f'tr[{position}]/td[7]'
            goals_against_xpath = base_xpath+f'tr[{position}]/td[8]'
            goal_difference_xpath = base_xpath+f'tr[{position}]/td[9]'
            points_xpath = base_xpath+f'tr[{position}]/td[10]'

            #print('Building dataframe row...')
            all_rows, working = build_dataframe_row(
                season, 
                league,
                position, 
                all_rows,
                tree,
                team_xpath, 
                games_played_xpath, 
                wins_xpath, 
                draws_xpath, 
                losses_xpath, 
                goals_favour_xpath, 
                goals_against_xpath, 
                goal_difference_xpath, 
                points_xpath
            )

            position = position+1
        except Exception as e:
            print(e)
            working = False

    # Convert the list of matches to a DataFrame
    df = pd.DataFrame(all_rows)

    return df


def parse_champions_season(season):
    """
    Parses the Champions League HTML file for a given season and extracts match data.

    Args:
        season (str): The season year to parse (e.g., '2022-2023').

    Returns:
        pd.DataFrame: A DataFrame containing match data for the specified season.
    """

    # Construct the file path and read the HTML content
    file_path = f'htmls/leagues/champions-league_{season}.html'

    tree = prepare_parser(file_path)

    # Initialize an empty DataFrame to store all match data
    all_rows = []

    # Iterate through each round and corresponding matches
    base_xpath = '/html/body/div[5]/div/div[1]/div[3]/div[1]/div[2]/'
    
    for group in range(8):
        for position in range(4):

            team_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[2]/a'
            games_played_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[3]'
            wins_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[4]'
            draws_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[5]'
            losses_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[6]'
            goals_favour_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[7]'
            goals_against_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[8]'
            goal_difference_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[9]'
            points_xpath = base_xpath+f'div[{group+1}]/table/tbody/tr[{position+1}]/td[10]'

            #print('team_xpath:', team_xpath)

            all_rows, working = build_dataframe_row(
                season, 
                'champions-league',
                position+1, 
                all_rows,
                tree,
                team_xpath, 
                games_played_xpath, 
                wins_xpath, 
                draws_xpath, 
                losses_xpath, 
                goals_favour_xpath, 
                goals_against_xpath, 
                goal_difference_xpath, 
                points_xpath
            )

            if not working:
                print('WARNING: Something went wrong!')

    # Convert the list of matches to a DataFrame
    df = pd.DataFrame(all_rows)

    return df



#################################################################################
################################### MAIN ########################################
#################################################################################

def main(
        csv_local_path='data/scrapper/local_leagues_classifications.csv', 
        csv_champions_path='data/scrapper/champions_league_classifications.csv', 
):
    """
    Main function to scrape and process Champions League data from 2000 to 2022.

    This function:
    1. Iterates through each season from 2000-2001 to 2022-2023.
    2. Collects match data for each season by parsing saved HTML files.
    3. Processes and cleans the collected data, including handling missing scores.
    4. Splits the 'Score' column into 'HomeGoals' and 'AwayGoals'.
    
    Returns:
        pd.DataFrame: A DataFrame containing the processed match data for all seasons.
    """
    
    all_seasons_local_data = []
    all_seasons_champions_data = []

    for league in [
        'la-liga', 'premier-league', 'serie-a', 'eredivise', 'ligue-1', 'bundesliga', 
        'champions-league'
    ]:
        for year in range(2013, 2024):  # Adjust the range as needed
            print(f"Scraping {league} , season {year}-{year + 1}...")

            season = f"{year}-{year + 1}"
            season = year

            try:
                # Parse season data
                if league == 'champions-league':
                    season_matches = parse_champions_season(season=season)
                    all_seasons_champions_data.append(season_matches)
                else:
                    season_matches = parse_local_league_season(league=league, season=season)
                    all_seasons_local_data.append(season_matches)

            except Exception as e:
                print(f"Error processing season {season}: {e}")

    # Concatenate all season data into a single DataFrame
    df_local = pd.concat(all_seasons_local_data, axis=0, ignore_index=True)
    df_champions = pd.concat(all_seasons_champions_data, axis=0, ignore_index=True)

    try:
        # Save to CSV
        df_local.to_csv(csv_local_path, index=False, sep=';')
        print(f"DataFrame saved to {csv_local_path}")

        df_champions.to_csv(csv_champions_path, index=False, sep=';')
        print(f"DataFrame saved to {csv_champions_path}")
    
    except Exception as e:
        print(f"An error occurred while saving files: {e}")

    return df_local, df_champions


if __name__ == "__main__":
    df = main()
    # Save or further process the DataFrame `df` as needed
