#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:08:29 2023

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import yaml
import os
with open("../../config/config.yaml") as file:
    config = yaml.safe_load(file)
    
scenario = 'ncdr'

costsn = pd.read_csv("../../results/reff/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsm = pd.read_csv("../../results/bau/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsr = pd.read_csv("../../results/ncdr/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])


nn = costsn.groupby(level=2).sum().div(1e9)
mm = costsm.groupby(level=2).sum().div(1e9)
rr = costsr.groupby(level=2).sum().div(1e9)

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
    elif tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif "solar" in tech:
        return "solar"
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


mf=pd.concat([mf1, mf2,mf3], axis=1)
mf=mf.T.sum()/3
mf=pd.DataFrame(mf)

rf=pd.concat([rf1, rf2,rf3], axis=1)
rf=rf.T.sum()/3
rf=pd.DataFrame(rf)


#%%

params = {'legend.fontsize': 'xx-large',
          }
pylab.rcParams.update(params)

fig, axs = plt.subplots(1, 5, figsize=(20, 6))
groups = [
    ["H2 pipeline", "H2 storage"],
    ["battery storage"],
    ["gas pipeline", "gas pipeline new", "transmission lines"],
    ["gas-to-power/heat", "power-to-gas", "power-to-heat", "power-to-liquid"],
    ["solar", "offshore wind", "onshore wind"],
]
groupps = [
    [ "H2 storage"],
    ["battery storage"],
    ["gas pipeline", "gas pipeline new"],
    ["gas-to-power/heat", "power-to-gas", "power-to-heat", "power-to-liquid"],
    ["solar", "offshore wind", "onshore wind"],
]

x= [-0.9,-0.68,-0.45,-0.22,0,0.22,0.44]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050"]
ylims = [
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 180],
    [0, 400],
]
xlims = [
    [-1, 0.6],
    [-1, 0.6],
    [-1, 0.6],
    [-1, 0.6],
    [-1, 0.6],
]

for ax, group,groupp, ylim, xlim in zip(axs, groups,groupps, ylims,xlims):
    nn.loc[groupp].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=6.5,width=0.15,legend=False)
    mf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=5,width=0.15, legend=False)
    mf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=3.5,width=0.15, legend=True)
    mf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=2,width=0.15, legend=False)
    rf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0.5,width=0.15, legend=False)
    rf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-1,width=0.15, legend=False)
    rf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-2.5,width=0.15, legend=False)
    
    #ax.set_xlabel('grid expansion')
    #ax.legend(bbox_to_anchor=(1.02, 1.45))
    ax.set_ylabel("Costs [Billion Euros]/year", fontsize=15)
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

fn = "costs_plot.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()

#%%

nn=nn.T
nn = nn.loc[:, (nn >0).any(axis=0)]
mf1=mf1.T
mf1 = mf1.loc[:, (mf1 >0).any(axis=0)]
mf2=mf2.T
mf2 = mf2.loc[:, (mf2 >0).any(axis=0)]
mf3=mf3.T
mf3 = mf3.loc[:, (mf3 >0).any(axis=0)]
rf1=rf1.T
rf1 = rf1.loc[:, (rf1 >0).any(axis=0)]
rf2=rf2.T
rf2 = rf2.loc[:, (rf2 >0).any(axis=0)]
rf3=rf3.T
rf3 = rf3.loc[:, (rf3 >0).any(axis=0)]

nn=nn.T
mf1=mf1.T
mf2=mf2.T
mf3=mf3.T
rf1=rf1.T
rf2=rf2.T
rf3=rf3.T
#%%

x= [-0.9,-0.725,-0.55,-0.37,-0.19,-0.01,0.17]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050"]

fig, ax = plt.subplots(figsize=(15, 8), )
pd.DataFrame(nn.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=9.5,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

pd.DataFrame(mf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=7.8,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(mf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
) 
pd.DataFrame(mf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=4.2,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
) 
pd.DataFrame(rf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=2.4,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=0.6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-1.2,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
ax.set_ylabel("Total Costs [Billion Euros]", fontsize=20)
ax.set_xlabel("")
ax.set_xlim(-1, 0.5)
ax.set_xticks(x, x_labels, rotation='vertical', fontsize=20)

ax = plt.gca() 
ax.bar_label(ax.containers[27], fmt='%.1f', label_type='edge', fontsize=15)
ax.bar_label(ax.containers[68],fmt='%.1f', label_type='edge', fontsize=15)
ax.bar_label(ax.containers[109], fmt='%.1f',label_type='edge', fontsize=15)
ax.bar_label(ax.containers[147], fmt='%.1f',label_type='edge', fontsize=15)
ax.bar_label(ax.containers[186], fmt='%.1f',label_type='edge', fontsize=15)   
ax.bar_label(ax.containers[225], fmt='%.1f',label_type='edge', fontsize=15)  
ax.bar_label(ax.containers[260], fmt='%.1f',label_type='edge', fontsize=15) 
                
a = ax.get_legend_handles_labels()  
b = {l:h for h,l in zip(*a)}        
c = [*zip(*b.items())]             
d = c[::-1]                        

plt.rc('ytick',labelsize=20)
plt.legend(*d, loc=(1, 0), ncol=2, fontsize=12)
plt.yticks(fontsize=20)
plt.tight_layout()
fn = "total_costs_plot.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()

#%%
nn=nn.T
nn = nn.loc[:, (nn >0).any(axis=0)]
mf=mf.T
mf = mf.loc[:, (mf >0).any(axis=0)]
rf=rf.T
rf = rf.loc[:, (rf >0).any(axis=0)]

nn=nn.T
mf=mf.T
rf=rf.T


x= [-0.74,-0.3,0.16]
x_labels =["Reff", "BAU", "SUff"]

fig, ax = plt.subplots(figsize=(15, 8), )
pd.DataFrame(nn.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=3,
    width=0.3,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

pd.DataFrame(mf.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=1.5,
    width=0.3,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

pd.DataFrame(rf.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=0,
    width=0.3,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

ax.set_ylabel("Total Costs [Billion Euros/year]", fontsize=20)
ax.set_xlabel("")
ax.set_ylim(0, 700)
ax.set_xlim(-1, 0.5)
ax.set_xticks(x, x_labels, rotation='vertical', fontsize=20)
ax = plt.gca() 
ax.bar_label(ax.containers[26], fmt='%.1f', label_type='edge', fontsize=20)
ax.bar_label(ax.containers[68],fmt='%.1f', label_type='edge', fontsize=20)
ax.bar_label(ax.containers[107], fmt='%.1f',label_type='edge', fontsize=20)
a = ax.get_legend_handles_labels()  
b = {l:h for h,l in zip(*a)}        
c = [*zip(*b.items())]            
d = c[::-1]
plt.rc('ytick',labelsize=20)
plt.legend(*d, loc=(1, 0), ncol=2, fontsize=15)
plt.yticks(fontsize=20)
plt.tight_layout()
fn = "total_scenario_costs_plot.png"
fn_path = os.path.join(save_folder, fn)
plt.savefig(fn_path, dpi=300, bbox_inches="tight")
plt.close()

