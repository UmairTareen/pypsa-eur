#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:25:57 2023

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import yaml
import os
with open("../../config/config.yaml") as file:
    config = yaml.safe_load(file)

scenario = 'ncdr'


capn = pd.read_csv("../../results/reff/csvs/capacities.csv", header=[0, 1, 2, 3], index_col=[0, 1])
capm = pd.read_csv("../../results/bau/csvs/capacities.csv", header=[0, 1, 2, 3], index_col=[0, 1])
capr = pd.read_csv("../../results/ncdr/csvs/capacities.csv", header=[0, 1, 2, 3], index_col=[0, 1])


nn = capn.groupby(level=1).sum().div(1e3)
mm = capm.groupby(level=1).sum().div(1e3)
rr = capr.groupby(level=1).sum().div(1e3)

def rename_techs(label):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        #"CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label

tech_colors = config["plotting"]["tech_colors"]
def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    # elif tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
    #     return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "process emissions CC", "solid biomass for industry CC", "gas for industry CC"]:
          return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
          return "biomass"
    elif tech in ["load", "process emissions", "uranium", "nuclear"]:
          return "nuclear"
    elif "battery" in tech:
        return "battery storage"
    elif "BEV charger" in tech:
        return "V2G"
    elif tech == "oil" or tech == "gas":
          return "fossil oil and gas"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech
colors = tech_colors 
colors["fossil oil and gas"] = colors["oil"]
colors["hydrogen storage"] = colors["H2 Store"]
colors["load shedding"] = 'black'
colors["gas-to-power/heat"] = 'darkred'
colors["LULUCF"] = 'greenyellow'
colors["methanation"] = 'gray'


nn = nn.groupby(nn.index.map(rename_techs_tyndp)).sum().T
mm = mm.groupby(mm.index.map(rename_techs_tyndp)).sum().T
rr = rr.groupby(rr.index.map(rename_techs_tyndp)).sum().T

#%%
nn=nn.T
nn.columns=nn.columns.droplevel(1)
nn.columns=nn.columns.droplevel(1)
nn.columns=nn.columns.droplevel(0)

mm=mm.T
mm.columns=mm.columns.droplevel(1)
mm.columns=mm.columns.droplevel(1)
mm.columns=mm.columns.droplevel(0)


rr=rr.T
rr.columns=rr.columns.droplevel(1)
rr.columns=rr.columns.droplevel(1)
rr.columns=rr.columns.droplevel(0)

mf1 = mm["2030"]
mf1 = pd.DataFrame(mf1)
mf2 = mm["2040"]
mf2 = pd.DataFrame(mf2)
mf3 = mm["2050"]
mf3 = pd.DataFrame(mf3)
rf1 = rr["2030"]
rf1 = pd.DataFrame(rf1)
rf2 = rr["2040"]
rf2 = pd.DataFrame(rf2)
rf3 = rr["2050"]
rf3 = pd.DataFrame(rf3)

#%%

params = {'legend.fontsize': 'x-large',
          }
pylab.rcParams.update(params)

fig, axs = plt.subplots(1, 4, figsize=(20, 6))
groups = [
    ["solar PV", "solar rooftop"],
    ["onshore wind", "offshore wind"],
    ["SMR", "SMR CC","H2 Electrolysis", "methanation", "methanolisation"],
    ["gas-to-power/heat","power-to-heat", "power-to-liquid"],
]
groupps = [
    ["solar PV", "solar rooftop"],
    ["onshore wind", "offshore wind"],
    ["SMR", "H2 Electrolysis"],
    ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
]


x= [-0.9,-0.73,-0.55,-0.37,-0.18,0,0.18]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050"]
ylims = [
    [0, 1700],
    [0, 1500],
    [0, 700],
    [0, 1000],
]
xlims = [
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
]

for ax, group,groupp, ylim, xlim in zip(axs, groups, groupps, ylims,xlims):
    nn.loc[groupp].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=8,width=0.12,legend=False)
    mf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=6.5,width=0.12, legend=False)
    mf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=5,width=0.12, legend=True)
    mf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=3.5,width=0.12, legend=False)
    rf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=2,width=0.12, legend=False)
    rf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0.5,width=0.12, legend=False)
    rf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-1,width=0.12, legend=False)
    
    ax.set_ylabel("Capacities [GW]", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(x, x_labels, rotation='vertical')
    
# fig.supxlabel('Total Costs', fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.tight_layout()
save_folder = f"../../results/{scenario}/plots"  # Specify the desired folder path for each country
if not os.path.exists(save_folder):
  os.makedirs(save_folder)

fn = "capacities_plot.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()

#%%
params = {'legend.fontsize': 'x-large',
          }
pylab.rcParams.update(params)

fig, axs = plt.subplots(1, 4, figsize=(20, 6))
groups = [
    ["transmission lines"],
    ["gas pipeline", "gas pipeline new"],
    ["H2 pipeline"],
    ["CCGT"],
]
groupps = [
    ["transmission lines"],
    ["gas pipeline", "gas pipeline new"],
    ["H2 Electrolysis"],
    ["CCGT"],
]

x= [-0.9,-0.73,-0.55,-0.37,-0.18,0,0.18]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050"]
ylims = [
    [0, 300],
    [0, 700],
    [0, 300],
    [0, 200],
]
xlims = [
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
]

for ax, group, groupp, ylim, xlim in zip(axs, groups,groupps, ylims,xlims):
    nn.loc[groupp].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=8,width=0.12,legend=False)
    mf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=6.5,width=0.12, legend=False)
    mf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=5,width=0.12, legend=True)
    mf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=3.5,width=0.12, legend=False)
    rf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=2,width=0.12, legend=False)
    rf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0.5,width=0.12, legend=False)
    rf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-1,width=0.12, legend=False)
    
    #ax.set_xlabel('grid expansion')
    #ax.legend(bbox_to_anchor=(1.02, 1.45))
    ax.set_ylabel("Capacities [GW]", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(x, x_labels, rotation='vertical')
    
# fig.supxlabel('Total Costs', fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.tight_layout()
fn = "capacities_plot_2.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()

#%%
params = {'legend.fontsize': 'x-large',
          }
pylab.rcParams.update(params)

fig, axs = plt.subplots(1, 5, figsize=(20, 6))
groups = [
    ["solar PV", "solar rooftop"],
    ["onshore wind", "offshore wind"],
    ["SMR", "SMR CC","H2 Electrolysis", "methanation", "methanolisation"],
    ["gas-to-power/heat","power-to-heat", "power-to-liquid"],
    ["transmission lines"],
]
groupps = [
    ["solar PV", "solar rooftop"],
    ["onshore wind", "offshore wind"],
    ["SMR", "H2 Electrolysis"],
    ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
    ["transmission lines"],
]


x= [-0.73,-0.3,0.13]
x_labels =["Reff", "BAU","SUff"]
ylims = [
    [0, 2000],
    [0, 1500],
    [0, 1000],
    [0, 1200],
    [0, 250],
]
xlims = [
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
    [-1, 0.5],
]

for ax, group,groupp, ylim, xlim in zip(axs, groups, groupps, ylims,xlims):
    nn.loc[groupp].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=3,width=0.3,legend=False)
    mf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=1.5,width=0.3, legend=True)
    rf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0,width=0.3, legend=False)
    
    #ax.set_xlabel('grid expansion')
    #ax.legend(bbox_to_anchor=(1.02, 1.45))
    ax.set_ylabel("Capacities [GW]", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(x, x_labels, rotation='vertical')
    
# fig.supxlabel('Total Costs', fontsize=20)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=20)
plt.tight_layout()
fn = "capacities_plot_scenario.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()