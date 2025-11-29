import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html

import warnings
warnings.filterwarnings('ignore')


# LOAD CSV FILES
stephen = pd.read_csv('3_stephen_curry_shot_chart_2023.csv', low_memory=False)
james   = pd.read_csv('2_james_harden_shot_chart_2023.csv', low_memory=False)
lebron  = pd.read_csv('1_lebron_james_shot_chart_1_2023.csv', low_memory=False)

# Add Player Names
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
df.loc[df['distance_bin'].isin(['nan', 'NaN']), 'distance_bin'] = np.nan

# Made flag & simple FG% groups
# Map common encodings to 1/0 in a single line (simple approach)
df['Made'] = df['result'].astype(str).str.lower().isin(
    ['1', 'true', 'made', 'make', 'shot made', 'made shot', 'hit']
).astype(int)

# User friendly label for makes & misses to use for coloring dots in shot chart
df['MadeLabel'] = df['Made'].map({1: 'Make', 0: 'Miss'})

def fg_agg(group_cols, data):
    # drop rows missing any grouping key
    data2 = data.dropna(subset=[c for c in group_cols if c in data.columns]).copy()
    g = (
        data2.groupby(group_cols, as_index=False, observed=True)
             .agg(Attempts=('Made', 'count'),
                  Made=('Made', 'sum'))
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
    if {'lebron_team_score', 'opponent_team_score'}.issubset(df.columns):
        df['score_margin'] = (df['lebron_team_score'] - df['opponent_team_score']).abs()
    elif {'team_score', 'opponent_team_score'}.issubset(df.columns):
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
q_all = ['1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr', '1st OT', '2nd OT']
q_reg = ['1st Qtr', '2nd Qtr', '3rd Qtr', '4th Qtr']
q_ot  = ['1st OT', '2nd OT']

# Quarter views (ok to group directly)
fg_qtr_all = fg_agg(['Player', 'qtr', 'shot_type'], df[df['qtr'].isin(q_all)])
fg_qtr_reg = fg_agg(['Player', 'qtr', 'shot_type'], df[df['qtr'].isin(q_reg)])
fg_qtr_ot  = fg_agg(['Player', 'qtr', 'shot_type'], df[df['qtr'].isin(q_ot)])

# Distance views - filter first, then group
df_dist_all = df[df['distance_bin'].notna()]
df_dist_reg = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_reg))]
df_dist_ot  = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_ot))]

fg_dist_all = fg_agg(['Player', 'distance_bin', 'shot_type'], df_dist_all)
fg_dist_reg = fg_agg(['Player', 'distance_bin', 'shot_type'], df_dist_reg)
fg_dist_ot  = fg_agg(['Player', 'distance_bin', 'shot_type'], df_dist_ot)

# Clutch FG% aggregations
fg_clutch_qtr  = fg_agg(['Player', 'shot_type'], df_clutch_reg)
fg_clutch_dist = fg_agg(['Player', 'distance_bin', 'shot_type'],
                        df_clutch_reg[df_clutch_reg['distance_bin'].notna()])


# Sunburst Builder
#   - color = Attempts
#   - tooltip shows aggregated Attempts + FG% at every level
def build(data, path, title_suffix):
    if data.empty:
        data = pd.DataFrame({'Player': [], 'Attempts': [], 'FG%': [], 'Made': []})
        path = ['Player']

    data = data.copy()
    data['Attempts'] = pd.to_numeric(data['Attempts'], errors='coerce').fillna(0).astype(int)
    data['Made']     = pd.to_numeric(data['Made'], errors='coerce').fillna(0).astype(int)
    data['FG%']      = pd.to_numeric(data['FG%'], errors='coerce')

    # Base sunburst 
    fig_tmp = px.sunburst(
        data,
        path=path,
        values='Attempts',
        color='Attempts',
        color_continuous_scale='RdYlGn',
        branchvalues='total',
        maxdepth=-1
    )

    tr = fig_tmp.data[0]
    tr.name = title_suffix

    ids = list(tr.ids)

    custom_data = []
    agg_attempts = []
    levels = []

    # Aggregate Attempts + FG% per node, and record depth level
    for node_id in ids:
        parts = node_id.split('/')
        filt = data.copy()

        # depth level: 0 = Player, 1 = Player+qtr, 2 = Player+qtr+shot_type, ...
        level = len(parts) - 1
        levels.append(level)

        for i, col in enumerate(path):
            if i < len(parts):
                filt = filt[filt[col] == parts[i]]

        total_attempts = int(filt['Attempts'].sum())
        total_made     = int(filt['Made'].sum())
        fg_pct = (100 * total_made / total_attempts) if total_attempts > 0 else 0.0

        custom_data.append([fg_pct, total_attempts])
        agg_attempts.append(total_attempts)

    tr.customdata = np.array(custom_data)

    # Normalize attempts per level so each ring uses its own min/max
    norm_colors = [0.0] * len(ids)   # one normalized value per node
    unique_levels = sorted(set(levels))

    for lv in unique_levels:
        # indices for nodes at this level
        idxs = [i for i, L in enumerate(levels) if L == lv]
        vals = [agg_attempts[i] for i in idxs]

        if len(vals) == 0:
            continue

        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            for i in idxs:
                norm_colors[i] = 0.5
        else:
            for i, v in zip(idxs, vals):
                norm_colors[i] = (v - vmin) / (vmax - vmin)

    # Attach normalized colors to marker, use 0–1 scale with Low/High labels
    tr.marker = dict(
        colors=norm_colors,           # per-level normalized 0–1
        colorscale='RdYlGn',
        cmin=0,
        cmax=1,
        colorbar=dict(
            title=dict(text='Attempts', side='right'),
            thickness=15,
            len=0.7,
            tickvals=[0, 1],
            ticktext=["Low", "High"]
        )
    )

    # Tooltip
    tr.hovertemplate = (
        "<b>%{label}</b><br>"
        "Attempts: %{customdata[1]:d}<br>"
        "FG%: %{customdata[0]:.1f}%<br>"
    )

    tr.textinfo = "label"
    return tr

# Build traces (use your same dropdown structure)
att_all   = build(fg_qtr_all,   ['Player','qtr','shot_type'], 'Attempts — All (Quarter)')
att_reg   = build(fg_qtr_reg,   ['Player','qtr','shot_type'], 'Attempts — Regulation')
att_ot    = build(fg_qtr_ot,    ['Player','qtr','shot_type'], 'Attempts — OT')

Dist_all  = build(fg_dist_all,  ['Player','distance_bin','shot_type'], 'Attempts — Distance (All)')
Dist_reg  = build(fg_dist_reg,  ['Player','distance_bin','shot_type'], 'Attempts — Distance (Reg)')
Dist_ot   = build(fg_dist_ot,   ['Player','distance_bin','shot_type'], 'Attempts — Distance (OT)')

# Clutch-time traces
Clutch_qtr  = build(fg_clutch_qtr,  ['Player','shot_type'], 'Attempts — Clutch (Reg 4Q)')
Clutch_dist = build(fg_clutch_dist, ['Player','distance_bin','shot_type'], 'Attempts — Clutch Distance (Reg 4Q)')

fig_addon = go.Figure(data=[att_all, att_reg, att_ot, Dist_all, Dist_reg, Dist_ot])
# Add clutch traces
fig_addon.add_traces([Clutch_qtr, Clutch_dist])

for i, tr in enumerate(fig_addon.data):
    tr.visible = (i == 0)

def visible(idx):
    v = [False] * 8
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
            dict(label="Attempts — All (Quarter)", method="update",
                 args=[{"visible": visible(0)},
                       {"title":{"text":"Attempts (color/size) — All (Quarter)","x":0.5}}]),
            dict(label="Attempts — Regulation", method="update",
                 args=[{"visible": visible(1)},
                       {"title":{"text":"Attempts — Regulation","x":0.5}}]),
            dict(label="Attempts — OT", method="update",
                 args=[{"visible": visible(2)},
                       {"title":{"text":"Attempts — OT","x":0.5}}]),
            dict(label="Attempts — Distance (All)", method="update",
                 args=[{"visible": visible(3)},
                       {"title":{"text":"Attempts — Distance (All)","x":0.5}}]),
            dict(label="Attempts — Distance (Reg)", method="update",
                 args=[{"visible": visible(4)},
                       {"title":{"text":"Attempts — Distance (Reg)","x":0.5}}]),
            dict(label="Attempts — Distance (OT)", method="update",
                 args=[{"visible": visible(5)},
                       {"title":{"text":"Attempts — Distance (OT)","x":0.5}}]),
            dict(label=f"Attempts — Clutch (Reg 4Q, last {CLUTCH_MINUTES}:00)", method="update",
                 args=[{"visible": visible(6)},
                       {"title":{"text":f"Attempts — Clutch (Reg 4Q, last {CLUTCH_MINUTES}:00)","x":0.5}}]),
            dict(label=f"Attempts — Clutch Distance (Reg 4Q, last {CLUTCH_MINUTES}:00)", method="update",
                 args=[{"visible": visible(7)},
                       {"title":{"text":f"Attempts — Clutch Distance (Reg 4Q, last {CLUTCH_MINUTES}:00)","x":0.5}}])
        ]
    )]
)

# Make sunburst larger by default (height/width and margins)
# You can tweak `height` (px) or set `autosize`/`width` if needed for fixed sizing.
fig_addon.update_layout(height=700, margin=dict(t=80, l=20, r=20, b=20))

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("NBA Last Minute Shot Analysis — Visual 2"),
    html.Div(
        style={'display': 'flex', 'width': '95%', 'margin': 'auto'},
        children=[
            html.Div(
                dcc.Graph(id='sunburst', figure=fig_addon),
                style={'flex': '1', 'width': '100%', 'minWidth': '700px'}
            )
        ]
    )
])

if __name__ == "__main__":
    app.run(debug=False)