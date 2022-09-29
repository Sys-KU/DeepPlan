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

def read_file(file):
    baseline = np.array([])
    pipeswitch = np.array([])
    deepplan_dha = np.array([])
    deepplan_parallel = np.array([])
    deepplan_all = np.array([])

    with open(file, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            baseline = np.append(baseline, float(line[0]))
            pipeswitch = np.append(pipeswitch, float(line[1]))
            deepplan_dha = np.append(deepplan_dha, float(line[2]))
            deepplan_parallel = np.append(deepplan_parallel, float(line[3]))
            deepplan_all = np.append(deepplan_all, float(line[4]))

    return np.array([baseline, pipeswitch, deepplan_dha, deepplan_parallel, deepplan_all])

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

avg_ret = read_file(sys.argv[1])
min_ret = read_file(sys.argv[2])
max_ret = read_file(sys.argv[3])

base = avg_ret[0]
avg_ret = base / avg_ret
min_ret = base / min_ret
max_ret = base / max_ret

lower_err = abs(avg_ret - min_ret)
upper_err = abs(avg_ret - max_ret)

ax.bar(value_base, avg_ret[0], color=color_list[0], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_pipe, avg_ret[1], color=color_list[1], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_dha, avg_ret[2], color=color_list[2], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_parallel, avg_ret[3], color=color_list[3], edgecolor="black", zorder=3, width=WIDTH)
ax.bar(value_deep_all, avg_ret[4], color=color_list[4], edgecolor="black", zorder=3, width=WIDTH)

ax.errorbar(value_base, avg_ret[0], yerr=[lower_err[0], upper_err[0]], fmt='o', capsize=3, color="black", zorder=4, ms=1)
ax.errorbar(value_pipe, avg_ret[1], yerr=[lower_err[1], upper_err[1]], fmt='o', capsize=3, color="black", zorder=4, ms=1)
ax.errorbar(value_deep_dha, avg_ret[2], yerr=[lower_err[2], upper_err[2]], fmt='o', capsize=3, color="black", zorder=4, ms=1)
ax.errorbar(value_deep_parallel, avg_ret[3], yerr=[lower_err[3], upper_err[3]], fmt='o', capsize=3, color="black", zorder=4, ms=1)
ax.errorbar(value_deep_all, avg_ret[4], yerr=[lower_err[4], upper_err[4]], fmt='o', capsize=3, color="black", zorder=4, ms=1)

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
plt.savefig(sys.argv[4], bbox_inches="tight", pad_inches=0.0)
print("Saved graph to {}".format(sys.argv[4]))
