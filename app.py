from flask import Flask, render_template
from plotly.offline import plot
import plotly.graph_objects as go
from scripts.bootstrap_wave_history import bootstrap_wave_history  # added import

app = Flask(__name__)

def ensure_wave_history_exists():  # added ensure_wave_history_exists
    bootstrap_wave_history()

@app.route('/')
def index():
    ensure_wave_history_exists()  # call at startup
    wave_data = [
        {'wave_id': 1, 'values': [1, 2, 3]},
        {'wave_id': 2, 'values': [2, 3, 4]}
    ]  # Example dataset

    charts = []
    for wave in wave_data:
        fig = go.Figure(data=go.Scatter(y=wave['values']))
        chart = plot(fig, output_type='div', include_plotlyjs=False)
        charts.append({'chart': chart, 'key': f"{wave['wave_id']}-chart"})  # ensure unique keys

    return render_template('index.html', charts=charts)

if __name__ == "__main__":
    app.run(debug=True)