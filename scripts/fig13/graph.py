#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import csv

def get_data(model, engine):
    target = "{}/{}_{}.csv".format(sys.argv[1], model, engine)
    target = target.strip()
    if target[0] != '/':
        target = os.path.join(os.getcwd(), target)

    lat = np.array([])

    with open(target, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            lat = np.append(lat, float(line[0]))

    return lat


label_list = ["PipeSwitch", "DeepPlan (DHA)", "DeepPlan (PT+DHA)"]
#color_list = ['#AEB6BF', '#5D6D7E', '#34495E', '#273746', '#273746']
color_list = ['#AEB6BF', '#5D6D7E', '#273746']
model_list = ["bert_large", "gpt2"]
engine_list = ["pipeline", "deepplan", "deepplan+"]
marker_list = ['o', '^', 'P']
line_list = ['-', 'dotted', 'dashed']

x_value_list  = {
                "bert_large": [5 * i for i in range(1, 12)],
                "gpt2": [20 * i for i in range(1, 11)],
                }

ylim_list = { # graph 모양 확인하고 조절해야함.
            "bert_base": [],
            "bert_large": [0, 850],
            "roberta_base": [],
            "roberta_large": [],
            "gpt2": [0, 900],
            "gpt2_medium": []
            }


x_label = "# of model instances (concurrency)"
y_label = "99 % latency (ms)"

FONTSIZE_LABEL = 16
FONTSIZE_TICK = 15
FONTSIZE_LEGEND = 14
SIZE_FIGURE = (7, 7)
LINE_WIDTH = 3
MARKER_SIZE = 10

plt.figure(figsize=SIZE_FIGURE)

li_ax = []
for i in range(1, len(model_list) + 1):
    li_ax.append(plt.subplot(len(model_list), 1, i))

for i, model in enumerate(model_list):

    graph_title = ""
    if model_list[i] == "bert_large":
        graph_title = "BERT-Large"

    elif model_list[i] == "gpt2":
        graph_title = "GPT-2"

    for j, engine in enumerate(engine_list):
        result = get_data(model, engine)

        li_ax[i].plot(x_value_list[model_list[i]], result, linewidth = LINE_WIDTH, color=color_list[j], marker=marker_list[j], linestyle=line_list[j], markersize=MARKER_SIZE)
        
        li_ax[i].set_title(graph_title, fontsize=FONTSIZE_LABEL+2)
        li_ax[i].set_xticks(x_value_list[model_list[i]]) #, fontsize=FONTSIZE_TICK)
        li_ax[i].set_ylim(ylim_list[model_list[i]])

        li_ax[i].set_ylabel(y_label, fontsize=FONTSIZE_LABEL, labelpad=10)
        li_ax[i].tick_params(which="major", labelsize=FONTSIZE_TICK)
        li_ax[i].grid(alpha=0.5, linestyle='--')


plt.legend(labels=label_list, bbox_to_anchor=(0.45, 2.60), ncol=3, loc='center', columnspacing=0.5,
           fontsize=FONTSIZE_LEGEND, edgecolor="#FFFFFF")

plt.xlabel(x_label, fontsize=FONTSIZE_LABEL, labelpad=10)

plt.subplots_adjust(hspace=0.35)
plt.rcParams["font.family"] = "Helvetica"
plt.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.0)
print("Saved graph to {}".format(sys.argv[2]))
