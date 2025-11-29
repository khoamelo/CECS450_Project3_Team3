import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

import warnings
warnings.filterwarnings('ignore')

# Load CSV Files
stephen = pd.read_csv('3_stephen_curry_shot_chart_2023.csv', low_memory=False)
james   = pd.read_csv('2_james_harden_shot_chart_2023.csv', low_memory=False)
lebron  = pd.read_csv('1_lebron_james_shot_chart_1_2023.csv', low_memory=False)

# # Add Player Names (keep your labels)
stephen['Player'] = 'Stephen_Curry'
james['Player']   = 'James_Harden'
lebron['Player']  = 'Lebron_James'

# Combine All Data
df = pd.concat([stephen, james, lebron], ignore_index=True)
if 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date']).dt.date
    except Exception:
        pass

# Shot distance
if 'distance_ft' in df.columns:
    bins = [-1, 3, 9, 16, 23, 26, 30, 99]
    labels = ['0–3 ft', '4–9 ft', '10–16 ft', '17–23 ft', '24–26 ft', '27–30 ft', '31+ ft']
    df['distance_bin'] = pd.cut(df['distance_ft'], bins=bins, labels=labels, right=True)
else:
    df['distance_bin'] = np.nan
# After creating df['distance_bin'] …
df['shot_type'] = df['shot_type'].astype(str)

# Keep only rows that have a distance bin for distance-based views
# and avoid categorical gotchas by using plain strings
df['distance_bin'] = df['distance_bin'].astype(str)
df.loc[df['distance_bin'].isin(['nan','NaN']), 'distance_bin'] = np.nan

# Made flag & simple FG% groups
# Map common encodings to 1/0 in a single line (simple approach)
df['Made'] = df['result'].astype(str).str.lower().isin(
    ['1','true','made','make','shot made','made shot','hit']
).astype(int)

# User friendly label for makes & misses to use for coloring dots in shot chart
df['MadeLabel'] = df['Made'].map({1: 'Make', 0: 'Miss'})

def fg_agg(group_cols, data):
    # drop rows missing any grouping key
    data2 = data.dropna(subset=[c for c in group_cols if c in data.columns]).copy()
    g = (
        data2.groupby(group_cols, as_index=False, observed=True)
            .agg(Attempts=('Made','count'),Made=('Made','sum'))
    )
    g['FG%'] = (g['Made'] / g['Attempts'] * 100).round(1)
    return g

CLUTCH_MINUTES = 5   # last N minutes of 4th quarter
CLUTCH_MARGIN  = 5   # within N points

def _parse_time_to_seconds(_s):
    try:
        m, s = str(_s).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan

# Add helper columns (safe if columns already exist)
if 'time_remaining' in df.columns and 'sec_left' not in df.columns:
    df['sec_left'] = df['time_remaining'].apply(_parse_time_to_seconds)

# Score margin (adjust column names if different)
if 'score_margin' not in df.columns:
    if {'lebron_team_score','opponent_team_score'}.issubset(df.columns):
        df['score_margin'] = (df['lebron_team_score'] - df['opponent_team_score']).abs()
    elif {'team_score','opponent_team_score'}.issubset(df.columns):
        df['score_margin'] = (df['team_score'] - df['opponent_team_score']).abs()
    else:
        df['score_margin'] = np.nan

# Filter for clutch moments (4th quarter, last 5 min, ≤5 point margin)
df_clutch_reg = df[
    (df['qtr'] == '4th Qtr') &
    (df['sec_left'].le(CLUTCH_MINUTES * 60)) &
    (df['score_margin'].le(CLUTCH_MARGIN))
].copy()

# Quarter filters (unchanged)
q_all = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr','1st OT','2nd OT']
q_reg = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr']
q_ot  = ['1st OT','2nd OT']

# Quarter views (ok to group directly)
fg_qtr_all = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_all)])
fg_qtr_reg = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_reg)])
fg_qtr_ot  = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_ot)])

# Distance views - filter first, then group
df_dist_all = df[df['distance_bin'].notna()]
df_dist_reg = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_reg))]
df_dist_ot  = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_ot))]

fg_dist_all = fg_agg(['Player','distance_bin','shot_type'], df_dist_all)
fg_dist_reg = fg_agg(['Player','distance_bin','shot_type'], df_dist_reg)
fg_dist_ot  = fg_agg(['Player','distance_bin','shot_type'], df_dist_ot)

# Clutch FG% aggregations
fg_clutch_qtr  = fg_agg(['Player','shot_type'], df_clutch_reg)
fg_clutch_dist = fg_agg(['Player','distance_bin','shot_type'],
                        df_clutch_reg[df_clutch_reg['distance_bin'].notna()])

# Sunburst builder (simple)
# - values = Attempts (size)
# - color  = FG%       (color)

def build(data, path, title_suffix):
    if data.empty:
        # keep layout stable if a view is empty
        data = pd.DataFrame({'Player':[], 'Attempts':[], 'FG%':[]})
        path = ['Player']
    fig_tmp = px.sunburst(
        data,
        path=path,
        values='Attempts',
        color='FG%',
        color_continuous_scale='RdYlGn',
        range_color=[0,100],
        branchvalues='total',
        maxdepth=-1
    )

    tr = fig_tmp.data[0]
    tr.name = title_suffix

    tr.marker = dict(
        colors=tr.marker.colors,
        colorscale='RdYlGn',
        cmin=0,
        cmax=100,
        colorbar=dict(
            title=dict(text='FG%', side='right'),
            thickness=15,
            len=0.7
        )
    )
    
    tr.hovertemplate = (
        "<b>%{label}</b><br>"+
        "Attempts: %{value}<br>"+
        "FG%: %{color:.1f}%<br>"
    )
    tr.textinfo = "label"
    return tr

# Build traces (use your same dropdown structure)
att_all = build(fg_qtr_all, ['Player','qtr','shot_type'], 'FG% — All (Quarter)')
att_reg = build(fg_qtr_reg, ['Player','qtr','shot_type'], 'FG% — Regulation')
att_ot  = build(fg_qtr_ot, ['Player','qtr','shot_type'], 'FG% — OT')

Dist_all = build(fg_dist_all, ['Player','distance_bin','shot_type'], 'FG% — Distance (All)')
Dist_reg = build(fg_dist_reg, ['Player','distance_bin','shot_type'], 'FG% — Distance (Reg)')
Dist_ot  = build(fg_dist_ot, ['Player','distance_bin','shot_type'], 'FG% — Distance (OT)')

# Clutch-time traces
Clutch_qtr  = build(fg_clutch_qtr,  ['Player','shot_type'], 'FG% — Clutch (Reg 4Q)')
Clutch_dist = build(fg_clutch_dist, ['Player','distance_bin','shot_type'], 'FG% — Clutch Distance (Reg 4Q)')

fig_addon = go.Figure(data=[att_all, att_reg, att_ot, Dist_all, Dist_reg, Dist_ot])
# Add clutch traces
fig_addon.add_traces([Clutch_qtr, Clutch_dist])

for i, tr in enumerate(fig_addon.data):
    tr.visible = (i==0)

def visible(idx):
    v = [False]*8
    v[idx] = True
    return v

# Layout and dropdown (text updated to reflect FG%)
fig_addon.update_layout(
    updatemenus=[dict(
        type="dropdown",
        x=0.92, xanchor="right",
        y=1.05, yanchor="top",
        showactive=True,
        buttons=[
            dict(label="FG% — All (Quarter)", method="update", args=[{"visible": visible(0)}, {"title":{"text":"FG% (color) & Attempts (size) — All (Quarter)","x":0.5}}]),
            dict(label="FG% — Regulation", method="update", args=[{"visible": visible(1)}, {"title":{"text":"FG% — Regulation","x":0.5}}]),
            dict(label="FG% — OT", method="update", args=[{"visible": visible(2)}, {"title":{"text":"FG% — OT","x":0.5}}]),
            dict(label="FG% — Distance (All)", method="update", args=[{"visible": visible(3)}, {"title":{"text":"FG% — Distance (All)","x":0.5}}]),
            dict(label="FG% — Distance (Reg)", method="update", args=[{"visible": visible(4)}, {"title":{"text":"FG% — Distance (Reg)","x":0.5}}]),
            dict(label="FG% — Distance (OT)", method="update", args=[{"visible": visible(5)}, {"title":{"text":"FG% — Distance (OT)","x":0.5}}]),
            dict(label=f"FG% — Clutch (Reg 4Q, last {CLUTCH_MINUTES}:00)", method="update", args=[{"visible": visible(6)}, {"title":{"text":f"FG% — Clutch (Reg 4Q, last {CLUTCH_MINUTES}:00)","x":0.5}}]),
            dict(label=f"FG% — Clutch Distance (Reg 4Q, last {CLUTCH_MINUTES}:00)", method="update", args=[{"visible": visible(7)}, {"title":{"text":f"FG% — Clutch Distance (Reg 4Q, last {CLUTCH_MINUTES}:00)","x":0.5}}])
        ]
    )]
)

# Make sunburst larger by default (height/width and margins)
# You can tweak `height` (px) or set `autosize`/`width` if needed for fixed sizing.
fig_addon.update_layout(height=700, margin=dict(t=80, l=20, r=20, b=20))


# Fixed axis ranges for shot chart
x_min, x_max = df['top'].min(), df['top'].max()
y_min, y_max = df['left'].min(), df['left'].max()

# # Function to draw an NBA half-court
# def draw_court(fig, court_color='black'):
#     shapes = []
#     # Hoop
#     shapes.append(dict(type='circle', xref='x', yref='y', x0=245, y0=30, x1=255, y1=40, line=dict(color=court_color)))
#     # Backboard
#     shapes.append(dict(type='line', xref='x', yref='y', x0=228, y0=29, x1=270, y1=29, line=dict(color=court_color)))
#     # Paint area
#     shapes.append(dict(type='rect', xref='x', yref='y', x0=194, y0=0, x1=306, y1=190, line=dict(color=court_color), fillcolor='rgba(0,0,0,0)'))
#     # Free throw circle
#     shapes.append(dict(type='circle', xref='x', yref='y', x0=194, y0=130, x1=306, y1=250, line=dict(color=court_color)))
#     # Three point arc
#     shapes.append(dict(type='path', xref='x', yref='y', path='M 30 150 Q 250 445 450 150', line=dict(color=court_color)))
#     shapes.append(dict(type='line', xref='x', yref='y', x0=30, y0=150, x1=30, y1=0, line=dict(color=court_color)))
#     shapes.append(dict(type='line', xref='x', yref='y', x0=450, y0=150, x1=450, y1=0, line=dict(color=court_color)))

#     fig.update_layout(shapes=shapes)
#     return fig

# # Initial scatter with court
# scatter_fig = px.scatter(
#     df, x='left', y='top',
#     color='MadeLabel',
#     color_discrete_map={'Make':'green', 'Miss':'red'},
#     labels={'left':'X','top':'Y'},
#     title='Shot Chart',
#     range_x=[y_min, y_max],
#     range_y=[x_max, x_min],
#     height=800
# )
# scatter_fig.update_traces(marker=dict(size=12))
# scatter_fig = draw_court(scatter_fig)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("NBA Last Minute Shot Analysis"),
    html.Div(style={'display': 'flex', 'width': '95%', 'margin': 'auto'}, children=[
        # html.Div(
        #     dcc.Graph(id='scatter', figure=scatter_fig),
        #     style={'flex': '1', 'margin-right': '10px', 'width': '70%'}
        # ),
        html.Div(
            dcc.Graph(id='sunburst', figure=fig_addon),
            style={'flex': '1', 'width': '100%', 'minWidth': '700px'}
        )
    ])
])

# # Callback to sync shot chart with sunburst click data
# @app.callback(
#     Output('scatter', 'figure'),
#     Input('sunburst', 'clickData')
# )
# def update_scatter(clickData):
#     dff = df.copy()
#     sunburst_filters = {
#         0: {'qtr': q_all,  'distance_bin': None},
#         1: {'qtr': q_reg,  'distance_bin': None},
#         2: {'qtr': q_ot,   'distance_bin': None},
#         3: {'qtr': q_all,  'distance_bin': 'any'},
#         4: {'qtr': q_reg,  'distance_bin': 'any'},
#         5: {'qtr': q_ot,   'distance_bin': 'any'},
#         6: {'qtr': ['4th Qtr'], 'clutch': True},
#         7: {'qtr': ['4th Qtr'], 'distance_bin': 'any', 'clutch': True}
#     }

#     if clickData:
#         mode = clickData['points'][0]['curveNumber']
#         filters = sunburst_filters.get(mode, {})

#         # Apply base filters
#         if 'qtr' in filters and filters['qtr'] is not None:
#             dff = dff[dff['qtr'].isin(filters['qtr'])]
#         if 'distance_bin' in filters and filters['distance_bin'] == 'any':
#             dff = dff[dff['distance_bin'].notna()]
#         if 'clutch' in filters and filters['clutch']:
#             dff = dff[
#                 (dff['qtr'] == '4th Qtr') &
#                 (dff['sec_left'].le(CLUTCH_MINUTES * 60)) &
#                 (dff['score_margin'].le(CLUTCH_MARGIN))
#             ]

#         # Apply hierarchical filters from sunburst
#         hierarchy = clickData['points'][0]['id'].split('/')
#         if len(hierarchy) >= 1:
#             dff = dff[dff['Player'] == hierarchy[0]]
#         if len(hierarchy) >= 2:
#             second = hierarchy[1]
#             if mode in [6, 7]:  
#                 if second in df['shot_type'].unique():
#                     dff = dff[dff['shot_type'] == second]
#                 elif second in df['distance_bin'].unique():
#                     dff = dff[dff['distance_bin'] == second]
#             else:
#                 if second in df['qtr'].unique():
#                     dff = dff[dff['qtr'] == second]
#                 elif second in df['distance_bin'].unique():
#                     dff = dff[dff['distance_bin'] == second]
#         if len(hierarchy) >= 3:
#             third = hierarchy[2]
#             if third in df['shot_type'].unique():
#                 dff = dff[dff['shot_type'] == third]

#     fig = px.scatter(
#         dff, x='left', y='top', color='MadeLabel',
#         color_discrete_map={'Make':'green', 'Miss':'red'},
#         labels={'left':'X','top':'Y'},
#         title='Shot Chart',
#         range_x=[y_min, y_max],
#         range_y=[x_max, x_min]
#     )
#     fig.update_traces(marker=dict(size=12))
#     fig = draw_court(fig)
#     return fig

if __name__ == "__main__":
    app.run(debug=True)