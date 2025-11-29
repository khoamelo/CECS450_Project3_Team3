import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# Load CSV Files
stephen = pd.read_csv('3_stephen_curry_shot_chart_2023.csv')
james   = pd.read_csv('2_james_harden_shot_chart_2023.csv')
lebron  = pd.read_csv('1_lebron_james_shot_chart_1_2023.csv')

# Add Player Names
stephen['Player'] = 'Stephen_Curry'
james['Player']   = 'James_Harden'
lebron['Player']  = 'Lebron_James'

# Combine All Data
df = pd.concat([stephen, james, lebron], axis=0)
df['date'] = pd.to_datetime(df['date']).dt.date

# Sunburst Builder
fig = px.sunburst(
    df,
    path=['Player', 'qtr', 'shot_type'],
    values='result',
    color='Player',
    color_discrete_map={
        'Lebron_James':  '#334668',
        'Stephen_Curry': '#6D83AA',
        'James_Harden':  '#C8D0DF'
    }
)

fig.update_layout(
    height=450,
    margin=dict(b=0, r=20, l=20),
    plot_bgcolor='#fafafa',
    paper_bgcolor='#fafafa',
    title_text="Shots Attempted In Each Quarter",
    title_font=dict(size=24, color='#8a8d93', family="Lato, sans-serif"),
    font=dict(color='#8a8d93'),
    hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
    showlegend=False
)

# Dash App
app = Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Graph(id='sunburst', figure=fig)
    ]
)

if __name__ == '__main__':
    app.run(debug=True, port=9050)