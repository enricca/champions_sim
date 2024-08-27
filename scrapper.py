from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree
import time
import openpyxl
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}


def get_html(
        league,
        year,
        driver
):
    """
    Fetches and saves the HTML content of the Champions League stats page for a given season.

    Args:
        season (str): The season year to fetch the stats for (e.g., '2022-2023').
        driver_path (str): Path to the WebDriver executable. Defaults to 'chromedriver'.

    Returns:
        None: The HTML content is saved to a file.
    """

    if league == 'champions_elimination':
        season = f"{year}-{year + 1}"
        base_url = 'https://fbref.com/en/comps/8/{season}/{season}-Champions-League-Stats'
        url = base_url.format(league=league, season=season)
        folder = 'champions_elimination'
        
    else:
        season = year
        base_url = 'https://www.skysports.com/{league}-table/{season}'
        url = base_url.format(league=league, season=season)
        folder = 'leagues'
        
    print(f'HTML {season}-{league}')
    print('URL:', url)

    try:
        # Open the web page
        print('Getting URL...')
        driver.get(url)

        # Wait until the main content table is loaded (using a specific element on the page)
        print('Waiting driver...')

        # Let the page load
        time.sleep(10)

        # Get the HTML content
        print('Getting HTML content...')
        html_content = driver.page_source

        # Define the output file name
        output_file = f'htmls/{folder}/{league}_{season}.html'

        # Save the HTML content to a text file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"HTML for season {season} saved as {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    

def main():
    # Initialize the Selenium WebDriver (ensure the WebDriver is in PATH or provide a full path)
    print('Obtaining driver...')
    driver = webdriver.Chrome() #service=service)

    for league in [
        #'la-liga', 'premier-league', 'serie-a', 'eredivise', 'ligue-1', 'bundesliga', 'champions-league', 
        'champions_elimination'
    ]:
        for year in range(2013, 2024):  # Adjust the range as needed
            print(f"Scraping season {year}-{year + 1}...")

            get_html(
                league=league,
                year=year,
                driver=driver
            )

    # Ensure the WebDriver is properly closed
    driver.quit()


if __name__ == "__main__":
    df = main()

