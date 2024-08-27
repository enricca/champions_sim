# Project Overview
This project serves as a learning exercise to explore various technologies, including web scraping and Flask applications. The project is lightweight and does not have significant hardware requirements. It was developed on an Acer Aspire laptop with an Intel i7 processor, 8 cores, running Windows 10, and was created in Barcelona, Spain.

## Objective
The main goal of this project is to simulate UEFA Champions League games to determine the most probable winner and predict the tournament bracket. To achieve this, I gathered data from past Champions League brackets and teams, using a web scraper to extract relevant HTML content, which I then parsed into usable tables. This data was further processed to create features for training a machine learning model for each season, leveraging both Champions League and local league data. Finally, I implemented a Monte Carlo simulation and developed a simple Flask application to visualize the results.

## Installation and Execution
To run this project:

- Clone the repository and create a virtual environment (e.g., using conda) with the dependencies listed in the requirements.txt file.

- In your terminal, navigate to the project directory and execute the following command:

    Copy code
    ```
    python app.py
    ```
- Follow the on-screen instructions to interact with the application.

## Project Structure
The project is organized into the following directories:

**htmls**: Contains the HTML files retrieved from the web.

**scrapper.py**: Code used to scrape the web and obtain these HTML files.
**parse_elimination_games.py** and **parse_leagues.py**: Code used to parse the HTML files into usable tables.
**data**: Holds both the raw data parsed from the HTML files and the processed data for further use.

**preprocess.py**: Script for processing and preparing the data.
**utils**: Contains a JSON file used for renaming teams, as there were multiple sources for web data.

**models**: Stores the machine learning models for each season.

**model.py**: Script used to train and manage the machine learning models.
**static** and **templates**: Contains necessary files for the Flask application.

**static**: Stores the results plot and the CSS file that defines the visual style of the Flask app.
**templates**: Contains the HTML files responsible for the layout and design of the Flask app.
The core application logic is in **app.py**, which builds the Flask app and integrates the simulation code from **montecarlo.py**.


## Future Development
This project has potential for further expansion and refinement. Possible next steps include:

Expanding Data Collection:

Incorporate data from additional years.
Gather data from all European domestic leagues.
Include detailed team information, such as starting lineups for each match or the most frequently played players throughout the season.
Collect data on coaches and managers, including their win percentages with each team.
Add referee data and their impact on game outcomes.
Enhance local league data with more granular details, such as home and away performance, and performance against top teams.
Improving Model Training:

Implement better handling of missing values.
Experiment with different machine learning models.
Refine feature selection processes.
Explore data augmentation techniques for upsampling.
Enhancing Aesthetics:

Improve the visual design and user interface of the application.
Advanced Probability Calculations:

Calculate the probability of a team winning multiple championships by using compound probability methods.


