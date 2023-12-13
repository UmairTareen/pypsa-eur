#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:08:29 2023

@author: umair
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import yaml
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)

costsn = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resultsreff/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsm = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsbau/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsp = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultssuff/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsr = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsnocdr/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])


nn = costsn.groupby(level=2).sum().div(1e9)
mm = costsm.groupby(level=2).sum().div(1e9)
pp = costsp.groupby(level=2).sum().div(1e9)
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
# oo = oo.groupby(oo.index.map(rename_techs_tyndp)).sum().T
pp = pp.groupby(pp.index.map(rename_techs_tyndp)).sum().T
#qq = qq.groupby(qq.index.map(rename_techs_tyndp)).sum().T
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


pp=pp.T
pp.columns=pp.columns.droplevel(1)
pp.columns=pp.columns.droplevel(1)
pp.columns=pp.columns.droplevel(0)


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
pf1 = pp["2030"]
pf1 = pd.DataFrame(pf1)
pf2 = pp["2040"]
pf2 = pd.DataFrame(pf2)
pf3 = pp["2050"]
pf3 = pd.DataFrame(pf3)
rf1 = rr["2030"]
rf1 = pd.DataFrame(rf1)
rf2 = rr["2040"]
rf2 = pd.DataFrame(rf2)
rf3 = rr["2050"]
rf3 = pd.DataFrame(rf3)


mf=pd.concat([mf1, mf2,mf3], axis=1)
mf=mf.T.sum()/3
mf=pd.DataFrame(mf)

pf=pd.concat([pf1, pf2,pf3], axis=1)
pf=pf.T.sum()/3
pf=pd.DataFrame(pf)

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

x= [-0.9,-0.73,-0.55,-0.37,-0.18,0,0.18, 0.36,0.54, 0.72]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050",
                  "NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]
ylims = [
    [0, 10],
    [0, 10],
    [0, 10],
    [0, 180],
    [0, 400],
]
xlims = [
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
]

for ax, group,groupp, ylim, xlim in zip(axs, groups,groupps, ylims,xlims):
    nn.loc[groupp].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=8,width=0.12,legend=False)
    mf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=6.5,width=0.12, legend=False)
    mf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=5,width=0.12, legend=True)
    mf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=3.5,width=0.12, legend=False)
    pf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=2,width=0.12, legend=False)
    pf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0.5,width=0.12, legend=False)
    pf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-1,width=0.12, legend=False)
    rf1.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-2.5,width=0.12, legend=False)
    rf2.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-4,width=0.12, legend=False)
    rf3.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-5.5,width=0.12, legend=False)
    
    #ax.set_xlabel('grid expansion')
    #ax.legend(bbox_to_anchor=(1.02, 1.45))
    ax.set_ylabel("Costs [Billion Euros]", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(x, x_labels, rotation='vertical')
    
# fig.supxlabel('Total Costs', fontsize=20)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=20)
plt.tight_layout()

#%%
params = {'legend.fontsize': 'x-large',
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


x= [-0.6,0,0.6]
x_labels =["BAU", "SUff", "NO_CDR"]
ylims = [
    [0, 10],
    [0, 4],
    [0, 10],
    [0, 120],
    [0, 300],
]
xlims = [
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
]

for ax, group,groupp, ylim, xlim in zip(axs, groups,groupps, ylims,xlims):
    mf.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=2,width=0.4, legend=True)
    pf.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=0.5,width=0.4, legend=False)
    rf.loc[group].T.plot.bar(ax=ax, stacked=True, color=tech_colors,position=-1,width=0.4, legend=False)
    
    #ax.set_xlabel('grid expansion')
    #ax.legend(bbox_to_anchor=(1.02, 1.45))
    ax.set_ylabel("Costs [Billion Euros/year]", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(x, x_labels, rotation='vertical')
    
# fig.supxlabel('Total Costs', fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.tight_layout()

#%%

nn=nn.T
nn = nn.loc[:, (nn >0).any(axis=0)]
mf1=mf1.T
mf1 = mf1.loc[:, (mf1 >0).any(axis=0)]
mf2=mf2.T
mf2 = mf2.loc[:, (mf2 >0).any(axis=0)]
mf3=mf3.T
mf3 = mf3.loc[:, (mf3 >0).any(axis=0)]
pf1=pf1.T
pf1 = pf1.loc[:, (pf1 >0).any(axis=0)]
pf2=pf2.T
pf2 = pf2.loc[:, (pf2 >0).any(axis=0)]
pf3=pf3.T
pf3 = pf3.loc[:, (pf3 >0).any(axis=0)]
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
pf1=pf1.T
pf2=pf2.T
pf3=pf3.T
rf1=rf1.T
rf2=rf2.T
rf3=rf3.T
#%%

x= [-0.9,-0.73,-0.55,-0.37,-0.18,0,0.18, 0.36,0.54, 0.72]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050",
                  "NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]

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
pd.DataFrame(pf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=2.4,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=0.6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-1.2,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-3,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-4.8,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-6.6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
ax.set_ylabel("Total Costs [Billion Euros]", fontsize=20)
ax.set_xlabel("")
ax.set_ylim(0, 700)
ax.set_xlim(-1, 1)
ax.set_xticks(x, x_labels, rotation='vertical', fontsize=20)
# fig.supxlabel('Total Costs', fontsize=20)
ax = plt.gca() 
ax.bar_label(ax.containers[21], fmt='%.1f', label_type='edge', fontsize=15)
ax.bar_label(ax.containers[56],fmt='%.1f', label_type='edge', fontsize=15)
ax.bar_label(ax.containers[91], fmt='%.1f',label_type='edge', fontsize=15)
ax.bar_label(ax.containers[125], fmt='%.1f',label_type='edge', fontsize=15)
ax.bar_label(ax.containers[160], fmt='%.1f',label_type='edge', fontsize=15)   
ax.bar_label(ax.containers[195], fmt='%.1f',label_type='edge', fontsize=15)  
ax.bar_label(ax.containers[229], fmt='%.1f',label_type='edge', fontsize=15) 
ax.bar_label(ax.containers[262], fmt='%.1f',label_type='edge', fontsize=15)   
ax.bar_label(ax.containers[295], fmt='%.1f',label_type='edge', fontsize=15)  
ax.bar_label(ax.containers[327], fmt='%.1f',label_type='edge', fontsize=15)                   # Get the axes you need
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
#plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.legend(*d, loc=(1, 0), ncol=2, fontsize=12)
plt.yticks(fontsize=20)
plt.tight_layout()

#%%
nn=nn.T
nn = nn.loc[:, (nn >0).any(axis=0)]
mf=mf.T
mf = mf.loc[:, (mf >0).any(axis=0)]
pf=pf.T
pf = pf.loc[:, (pf >0).any(axis=0)]
rf=rf.T
rf = rf.loc[:, (rf >0).any(axis=0)]

nn=nn.T
mf=mf.T
pf=pf.T
rf=rf.T


x= [-0.74,-0.3,0.16,0.62]
x_labels =["Reff", "BAU", "SUff", "NO_CDR"]

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
pd.DataFrame(pf.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=0,
    width=0.3,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

pd.DataFrame(rf.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-1.5,
    width=0.3,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)

ax.set_ylabel("Total Costs [Billion Euros/year]", fontsize=20)
ax.set_xlabel("")
ax.set_ylim(0, 700)
ax.set_xlim(-1, 1)
ax.set_xticks(x, x_labels, rotation='vertical', fontsize=20)
# fig.supxlabel('Total Costs', fontsize=20)
ax = plt.gca() 
ax.bar_label(ax.containers[21], fmt='%.1f', label_type='edge', fontsize=20)
ax.bar_label(ax.containers[56],fmt='%.1f', label_type='edge', fontsize=20)
ax.bar_label(ax.containers[91], fmt='%.1f',label_type='edge', fontsize=20)
ax.bar_label(ax.containers[124], fmt='%.1f',label_type='edge', fontsize=20)                  # Get the axes you need
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
#plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.legend(*d, loc=(1, 0), ncol=2, fontsize=15)
plt.yticks(fontsize=20)
plt.tight_layout()
#%%
frames = [nn,rf1,rf2,rf3]
gf= pd.concat(frames, axis=1)
hf= pd.concat(frames, axis=1)

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(30, 10),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 4], "wspace": 0.05},
)
gf.T.plot.bar(ax=ax1,stacked= True, color=colors, legend =False)
hf.T.plot.area(ax=ax2,stacked= True, color=colors, legend =False)
ax1.set_ylabel("Costs [Billion Euros]", fontsize=20)
ax1.set_xlabel("")
ax1.set_xticks([0,1,2,3], labels=gf.columns, fontsize=20)
ax2.set_xticks([0,1,2,3], labels=hf.columns, fontsize=20)
ax1.set_yticks([0,100,200,300,400,500],fontsize=12)
ax2.set_yticks([0,100,200,300,400,500],fontsize=12)
ax2.set_ylabel("Costs [Billion Euros]", fontsize=12)
ax2.set_xlabel("")
plt.legend(loc='upper center', bbox_to_anchor=(0.32, -0.15),
          fancybox=True, shadow=True, ncol=7)
plt.tight_layout()

#%%
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import yaml
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)
tech_colors = config["plotting"]["tech_colors"]
colors = tech_colors 
colors["fossil oil and gas"] = colors["oil"]
colors["hydrogen storage"] = colors["H2 Store"]
colors["load shedding"] = 'black'
colors["gas-to-power/heat"] = 'darkred'
colors["Capital Costs"] = 'blue'
colors["Operational Costs"] = 'orange'
costsn = pd.read_csv("/home/umair/pypsa-eur_repository/results/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsm = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsbau/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsp = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultssuff/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])
costsr = pd.read_csv("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsnocdr/csvs/costs.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2])

nn = costsn.groupby(level=1).sum().div(1e9)
nn.columns=nn.columns.droplevel(1)
nn.columns=nn.columns.droplevel(1)
nn.columns=nn.columns.droplevel(0)
nn=nn.T
nn = nn.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
nn=nn.T

mm = costsm.groupby(level=1).sum().div(1e9)
mm.columns=mm.columns.droplevel(1)
mm.columns=mm.columns.droplevel(1)
mm.columns=mm.columns.droplevel(0)
mf1 = mm["2030"]
mf1 = pd.DataFrame(mf1)
mf2 = mm["2040"]
mf2 = pd.DataFrame(mf2)
mf3 = mm["2050"]
mf3 = pd.DataFrame(mf3)

mf1=mf1.T
mf1 = mf1.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
mf1=mf1.T
mf2=mf2.T
mf2 = mf2.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
mf2=mf2.T
mf3=mf3.T
mf3 = mf3.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
mf3=mf3.T

pp = costsp.groupby(level=1).sum().div(1e9)
pp.columns=pp.columns.droplevel(1)
pp.columns=pp.columns.droplevel(1)
pp.columns=pp.columns.droplevel(0)
pf1 = pp["2030"]
pf1 = pd.DataFrame(pf1)
pf2 = pp["2040"]
pf2 = pd.DataFrame(pf2)
pf3 = pp["2050"]
pf3 = pd.DataFrame(pf3)
pf1=pf1.T
pf1 = pf1.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
pf1=pf1.T
pf2=pf2.T
pf2 = pf2.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
pf2=pf2.T
pf3=pf3.T
pf3 = pf3.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
pf3=pf3.T

rr = costsr.groupby(level=1).sum().div(1e9)
rr.columns=rr.columns.droplevel(1)
rr.columns=rr.columns.droplevel(1)
rr.columns=rr.columns.droplevel(0)
rf1 = rr["2030"]
rf1 = pd.DataFrame(rf1)
rf2 = rr["2040"]
rf2 = pd.DataFrame(rf2)
rf3 = rr["2050"]
rf3 = pd.DataFrame(rf3)
rf1=rf1.T
rf1 = rf1.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
rf1=rf1.T
rf2=rf2.T
rf2 = rf2.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
rf2=rf2.T
rf3=rf3.T
rf3 = rf3.rename(columns={'capital': 'Capital Costs', 'marginal' : 'Operational Costs'})
rf3=rf3.T

x= [-0.9,-0.73,-0.55,-0.37,-0.18,0,0.18, 0.36,0.54, 0.72]
x_labels =["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050",
                  "NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]

fig, ax = plt.subplots(figsize=(10, 10), )
pd.DataFrame(nn.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=9.5,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=True,
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
pd.DataFrame(pf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=2.4,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=0.6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-1.2,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf1.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-3,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf2.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-4.8,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
pd.DataFrame(rf3.T.groupby(level=0).sum()).plot.bar(
    ax=ax,
    position=-6.6,
    width=0.1,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=False,
)
ax.set_ylabel("Costs [Billion Euros]", fontsize=15)
ax.set_xlabel("")
ax.set_ylim(0, 700)
ax.set_xlim(-1, 1)
ax.set_xticks(x, x_labels, rotation='vertical', fontsize=20)
fig.supxlabel('Total Costs')
ax = plt.gca()                   # Get the axes you need
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]   
plt.legend(*d, fontsize=20)                      # d = [(h1 h2) (l1 l2)]
plt.yticks(fontsize=20)
plt.tight_layout()

