
# This will be the app.py file for a flask service

'''
Posar una pantalla de control de paràmetres
-Equip a calcular
-Selector de temporada (pot haver-hi opció de ALL YEARS, amb probabilitat que hagi guanyat el que ha guanyat en realitat)
…
Quan fas submit a l’altre pantalla, vas a la nova, que treu gràfiques i nombres sobre el que ha fet:
-Si has seleccionat una temporada:
    -El bracket més probable simulat de la temporada
    -La probabilitat que guanyés el teu equip vs el que has posat tu
-Si has seleccionat el ALL YEARS:
    -Els guanyadors més probables per any
'''

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import montecarlo
import os


# Load teams data from CSV
print('Reading...')
df_champions = pd.read_csv('data/scrapper/champions_elimination.csv', sep=';')

teams_data = {}
for season in df_champions['Season'].unique():
    teams = montecarlo.sixteen_draw(df_champions, season)
    teams_data[season] = teams

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html', teams_data=teams_data)


@app.route('/get_teams/<season>')
def get_teams(season):
    teams = teams_data.get(season, [])
    return jsonify(teams)


@app.route('/result', methods=['POST'])
def result():
    # Get the form data
    team = request.form['team']
    season = request.form['season']

    # Run the Monte Carlo simulation
    plot_path = os.path.join('static', 'bracket_plot.png')
    montecarlo_result = montecarlo_run(team, season, plot_path)

    # Redirect to the results page
    return redirect(url_for('show_result', result=montecarlo_result, plot_file=plot_path))


@app.route('/result/show')
def show_result():
    result = request.args.get('result', type=float)

    # Render the result page
    return render_template('result.html', result=result)


def update_progress(progress):
    # Emit the progress to the client
    socketio.emit('progress_update', {'progress': progress})


def montecarlo_run(team, season, plot_path=os.path.join('static', 'bracket_plot.png')):
    # Call Monte Carlo simulation
    probs = montecarlo.main(
        season, 
        plot_path
    )
    # TODO fer millores:
    ### 1. Opció de ALL YEARS
    ### 2. RESULTS:
    ###     -Si has seleccionat una temporada:
    ###         -El bracket més probable simulat de la temporada
    ###         -La probabilitat que guanyés el teu equip vs el que has posat tu
    ###     -Si has seleccionat el ALL YEARS:
    ###         -Probabilitat convinada vs el teu guess
    ###         -Els guanyadors més probables per any

    # TODO tenir les simulacions i els resultats guardats en memòria
    # Quan executem simplement llegim i així el temps "d'inferència" s'escursa dràsticament

    if team in probs.keys():
        result = probs[team]
    else:
        result = 0

    return result

if __name__ == '__main__':
    #app.run(debug=True)
    socketio.run(app, debug=True)


