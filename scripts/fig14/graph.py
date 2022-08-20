#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import sys
import csv

def get_data(target):

    target = "./data/v100/azure_bursty/{}".format(target)

    if "offered" in target:
        offered_load = np.array([])
    else :
        result = []

        latency = np.array([])
        goodput = np.array([])
        cold = np.array([])

    with open(target, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        if "offered" in target:
            for i, line in enumerate(rdr):
                line = line[0].split("\t")
                offered_load = np.append(offered_load, int(line[0]))
            return offered_load

        else:
            for i, line in enumerate(rdr):
                line = line[0].split("\t")
                latency = np.append(latency, float(line[0]))
                goodput = np.append(goodput, float(line[1]))
                cold = np.append(cold, float(line[2]))

            result.append(latency)
            result.append(goodput)
            result.append(cold)

        return result



x_value = [i for i in range(1, 181)]
x_ticks = [30 * i for i in range(0, 7)]


label_list = ["PipeSwitch", "DeepPlan (DHA)", "DeepPlan (PT+DHA)"]
#color_list = ['#EAECEE', '#AEB6BF', '#85929E', '#5D6D7E', '#34495E', '#273746']
csv_list = ["offeredload.csv", "lat.csv", "put.csv", "cold.csv"]

# Prepare these files
engine_list = ["pipeswitch.csv", "dha.csv", "dha_pt.csv"]

color_list = ['#AEB6BF', '#5D6D7E', '#273746']
line_list = ['solid', 'dotted', 'dashdot']

ylim_list = [(5500, 10001), (0, 550), (50, 103), (0, 22.5)]

x_label = "Time (minutes)"
y_label = ["Offered load\n (req./min.)", "99 % latency\n (ms)", "Goodput\n (%)", "Cold-start\n (%)"]

FONTSIZE_XLABEL = 16
FONTSIZE_YLABEL = 14
FONTSIZE_TICK = 13
FONTSIZE_LEGEND = 14
SIZE_FIGURE = (7, 7)
LINE_WIDTH = 1.5
ARKER_SIZE = 10
MARKER_SIZE = 10


plt.figure(figsize=SIZE_FIGURE)
gs = gridspec.GridSpec(nrows=4, # row 몇 개
                       ncols=1, # col 몇 개
                       height_ratios=[0.8, 1, 0.8, 0.8]
                      )

li_ax = []
for i in range(0, 4):
    li_ax.append(plt.subplot(gs[i]))

    if i == 0: # Offered Load graph
        offered_load = get_data("various_offeredload.csv")

        li_ax[i].plot(x_value, offered_load, linewidth = LINE_WIDTH, color='#000000', linestyle="solid")
        li_ax[i].set_ylim(ylim_list[i])
        li_ax[i].tick_params(axis="both", labelsize=FONTSIZE_TICK)
        li_ax[i].set_xticks(x_ticks)

        li_ax[i].axes.xaxis.set_ticklabels([])

        li_ax[i].set_ylabel(y_label[i], fontsize=FONTSIZE_YLABEL)
        li_ax[i].get_yaxis().set_label_coords(-0.13, 0.5)

        li_ax[i].set_xlim(0, 180)

        li_ax[i].grid(alpha=1, linestyle='--')


for i, engine in enumerate(engine_list):
    result = get_data(engine) # Read data
    for j, ax in enumerate(li_ax):
        if j > 0:
            ax.plot(x_value, result[j-1], linewidth = LINE_WIDTH, color=color_list[i], linestyle=line_list[i], markersize=MARKER_SIZE)

            ax.set_ylim(ylim_list[j])
            ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)

            ax.set_xticks(x_ticks)
            if j < 3:
                ax.axes.xaxis.set_ticklabels([])

            ax.set_ylabel(y_label[j-1], fontsize=FONTSIZE_YLABEL)
            ax.get_yaxis().set_label_coords(-0.13, 0.5)

            ax.set_xlim(0, 180)

            ax.grid(alpha=1, linestyle='--')

plt.legend(labels=label_list, bbox_to_anchor=(0.43, 4.7), ncol=3, loc='center', columnspacing=0.6,
           fontsize=FONTSIZE_LEGEND, edgecolor="#FFFFFF")

plt.xlabel(x_label, fontsize=FONTSIZE_XLABEL, labelpad=10)

plt.subplots_adjust(hspace=0.06)
plt.rcParams["font.family"] = "Helvetica"
plt.savefig(sys.argv[1], bbox_inches="tight", pad_inches=0.0)
