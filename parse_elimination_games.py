import pandas as pd
from lxml import etree


def parse_season(season):
    """
    Parses the Champions League HTML file for a given season and extracts match data.

    Args:
        season (str): The season year to parse (e.g., '2022-2023').

    Returns:
        pd.DataFrame: A DataFrame containing match data for the specified season.
    """

    # Construct the file path and read the HTML content
    file_path = f'htmls/champions_elimination/champions_elimination_{season}.html'

    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    parser = etree.HTMLParser()
    tree = etree.fromstring(html_content, parser)

    df = pd.DataFrame()

    # Define the round mapping
    round_mapping = {
        3: {'round': 'final', 'series': [1]},
        4: {'round': 'semis', 'series': [1, 2]},
        5: {'round': 'quarter', 'series': [1, 2, 3, 4]},
        6: {'round': 'sixteen', 'series': [1, 2, 3, 4, 5, 6, 7, 8]}
    }

    # Initialize an empty DataFrame to store all match data
    all_matches = []

    # Iterate through each round and corresponding matches
    for round_num, round_info in round_mapping.items():
        round_name = round_info['round']
        series_list = round_info['series']

        for series_num in series_list:
            matches = [1] if round_name=='final' else [1, 2]
            for match in matches:
                leg = 'first leg' if match==1 else 'second_leg'

                # Construct XPaths
                team1_xpath = (f'/html/body/div[3]/div[7]/div[3]/div/div[1]/div[2]/a'
                               if round_num == 3 else
                               f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[2]/div[{match}]/div[2]/small')
                team2_xpath = (f'/html/body/div[3]/div[7]/div[3]/div/div[1]/div[4]/a'
                               if round_num == 3 else
                               f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[2]/div[{match}]/div[4]/small')
                score_xpath = (f'/html/body/div[3]/div[7]/div[{round_num}]/div/div[1]/div[3]/a'
                            if round_num == 3 else
                            f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[2]/div[{match}]/div[3]/span/small/a')
                
                '''
                team1_xpath = f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[1]/div[2]/a'
                team2_xpath = f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[1]/div[4]/a'
                score_xpath = (f'/html/body/div[3]/div[7]/div[{round_num}]/div/div[1]/div[3]/a'
                            if round_num == 3 else
                            f'/html/body/div[3]/div[7]/div[{round_num}]/div[{series_num}]/div[1]/div[3]')
                '''

                # Extract data using XPath
                team1_elements = tree.xpath(team1_xpath)
                team2_elements = tree.xpath(team2_xpath)
                score_elements = tree.xpath(score_xpath)

                # Collect match data
                for team1, team2, score in zip(team1_elements, team2_elements, score_elements):
                    match = {
                        "Season": season,
                        "Round": round_name,
                        "Leg": leg,
                        "HomeTeam": team1.text.strip(),
                        "AwayTeam": team2.text.strip(),
                        "Score": score.text.strip()
                    }
                    all_matches.append(match)

    # Convert the list of matches to a DataFrame
    df = pd.DataFrame(all_matches)

    return df


#################################################################################
################################### MAIN ########################################
#################################################################################

def main(
    csv_path='data/scrapper/champions_elimination.csv', 
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
    
    all_seasons_data = []

    for year in range(2013, 2023):  # Adjust the range as needed
        print(f"Scraping season {year}-{year + 1}...")

        season = f"{year}-{year + 1}"

        try:
            # Parse season data
            season_matches = parse_season(season)
            all_seasons_data.append(season_matches)

        except Exception as e:
            print(f"Error processing season {season}: {e}")

    # Concatenate all season data into a single DataFrame
    df = pd.concat(all_seasons_data, axis=0, ignore_index=True)

    try:
        # Save to CSV
        df.to_csv(csv_path, index=False, sep=';')
        print(f"DataFrame saved to {csv_path}")
    
    except Exception as e:
        print(f"An error occurred while saving files: {e}")

    return df

if __name__ == "__main__":
    df = main()
    # Save or further process the DataFrame `df` as needed
