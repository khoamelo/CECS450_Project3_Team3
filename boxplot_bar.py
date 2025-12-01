import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Read the Excel file
df = pd.read_excel("Tasks Result.xlsx")

# Column names
participant_col = "Participant"
visual_col      = "Visualization"
task_col        = "Task"
time_col        = "Time_Min"
accuracy_col    = "Accuracy"

# Ensure types are simple
df[visual_col]   = df[visual_col].astype(str)
df[task_col]     = df[task_col].astype(str)
df[time_col]     = df[time_col].astype(float)
df[accuracy_col] = df[accuracy_col].astype(float)


tasks = ["T1", "T2", "T3"]
visualizations = ["Baseline", "Improved"]

order = []
for t in tasks:
    for v in visualizations:
        order.append((v, t))

time_data = []   
acc_data = []    
labels = []      
colors = []      

for (v, t) in order:
    # select rows for visualization + task
    subset = df[(df[visual_col] == v) & (df[task_col] == t)]

    times = subset[time_col].values
    accs  = subset[accuracy_col].values

    time_data.append(times)
    acc_data.append(accs)

    labels.append(v + "\n" + t)

    if v.lower().startswith("base"):
        colors.append("#1f77b4")   # blue for Baseline
    else:
        colors.append("#2ca02c")   # green for Improved


# Efficiency Boxplot
fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(
    time_data,
    patch_artist=True,   
    widths=0.6,
    showfliers=True
)

# color each box
for i in range(len(bp["boxes"])):
    bp["boxes"][i].set_facecolor(colors[i])

ax.set_title("Efficiency Analysis")
ax.set_ylabel("Time on task (minutes)")
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)


all_times = []
for arr in time_data:
    for value in arr:
        all_times.append(value)

y_max = max(all_times) + 0.3
ax.set_ylim(0.0, y_max)
ax.set_xlim(0.2, len(labels) + 0.8)

# add median labels to each box
median_values = []
for median_line in bp["medians"]:
    y_val = median_line.get_ydata()[0]
    median_values.append(y_val)

label_offset = 0.35

for box_number in range(1, len(median_values) + 1):
    median_value = median_values[box_number - 1]
    x_position = box_number + label_offset

    ax.text(
        x_position,
        median_value,
        "{:.2f}".format(median_value),
        va="center",
        ha="left",
        fontsize=9,
        color="black"
    )

plt.tight_layout()
fig.savefig("efficiency_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()


# Effectiveness Bar Graph
# mean accuracy for each condition
mean_acc = []
for arr in acc_data:
    mean_value = sum(arr) / len(arr)
    mean_acc.append(mean_value)

x_positions = np.arange(1, len(labels) + 1)

fig2, ax2 = plt.subplots(figsize=(10, 6))

bars = ax2.bar(x_positions, mean_acc, width=0.6, color=colors)

ax2.set_title("Effectiveness Analysis")
ax2.set_ylabel("Accuracy on task")
ax2.set_xticks(x_positions)
ax2.set_xticklabels(labels, rotation=30, ha="right")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

# add text labels above bars
for i in range(len(bars)):
    height = bars[i].get_height()
    ax2.text(
        x_positions[i],
        height,
        "{:.2f}".format(height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
fig2.savefig("effectiveness_bar.png", dpi=300, bbox_inches="tight")
plt.show()