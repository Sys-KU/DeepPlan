#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import sys
import os
import csv

baseline = np.array([])
pipeswitch = np.array([])
deepplan_dha = np.array([])
deepplan_parallel = np.array([])
deepplan_all = np.array([])

target = sys.argv[1]
target = target.strip()
if target[0] != '/':
    target = os.path.join(os.getcwd(), target)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    for i, line in enumerate(rdr):
        baseline = np.append(baseline, float(line[0]))
        pipeswitch = np.append(pipeswitch, float(line[1]))
        deepplan_dha = np.append(deepplan_dha, float(line[2]))
        deepplan_parallel = np.append(deepplan_parallel, float(line[3]))
        deepplan_all = np.append(deepplan_all, float(line[4]))

pipeswitch = baseline / pipeswitch
deepplan_dha = baseline / deepplan_dha
deepplan_parallel = baseline / deepplan_parallel
deepplan_all = baseline / deepplan_all
baseline /= baseline

label_list = ["Baseline", "PipeSwitch", "DeepPlan (DHA)", "DeepPlan (PT)", "DeepPlan (PT+DHA)"]

color_list = ['#EAECEE', '#AEB6BF', '#85929E', '#5D6D7E', '#34495E', '#273746']
model_list = ["ResNet-50", "ResNet-101", "BERT-Base", "BERT-Large", "RoBERTa\nBase", "RoBERTa\nLarge", "GPT-2", "GPT-2 Medium"]

x_label = ""
y_label = "Inference speedup"

FONTSIZE_LABEL = 14
FONTSIZE_LEGEND = 14
WIDTH = 1.1
SIZE_FIGURE = (12, 3)


def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]

value_base = create_x(8, 1.3, 1, 8)
value_pipe = create_x(8, 1.3, 2, 8)
value_deep_dha = create_x(8, 1.3, 3, 8)
value_deep_parallel = create_x(8, 1.3, 4, 8)
value_deep_all = create_x(8, 1.3, 5, 8)

fig, ax = plt.subplots(1, 1, figsize=SIZE_FIGURE)

ax.bar(value_base, baseline, color=color_list[0], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_pipe, pipeswitch, color=color_list[1], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_dha, deepplan_dha, color=color_list[2], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_parallel, deepplan_parallel, color=color_list[3], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_all, deepplan_all, color=color_list[4], edgecolor="black", zorder=3, width=WIDTH)

fig.legend(labels=label_list, bbox_to_anchor=(0.52, 1.00), ncol=5, loc='center',
           fontsize=FONTSIZE_LEGEND, frameon=False)

plt.xticks([3.9 + i * 8 for i in range(0, 8)], model_list)
plt.tick_params(axis="x", direction="out", labelsize=FONTSIZE_LABEL, rotation=0)
plt.ylabel(y_label, fontsize=FONTSIZE_LABEL, labelpad=8)
plt.yticks(fontsize=FONTSIZE_LABEL)
plt.grid(linestyle='-', axis='y', zorder=-10)
plt.rcParams["font.family"] = "Helvetica"
plt.axhline(y=1.0, color='gray', linestyle='--')

plt.tight_layout()
#plt.show()
plt.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.0)
print("Saved graph to {}".format(sys.argv[2]))
