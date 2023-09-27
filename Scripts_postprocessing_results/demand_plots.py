#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:50:52 2023

"""
import pypsa
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pypsa.descriptors import get_switchable_as_dense as as_dense
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)


n=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")

m_1=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc")
m_2=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc")
m_3=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")


p_1=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc")
p_2=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc")
p_3=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")

r_1=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc")
r_2=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc")
r_3=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")


def process_demand(scenario):
    demand = as_dense(scenario, "Load", "p_set").div(1e6)  # TWh
    demand_grouped = demand.groupby(
        [scenario.loads.carrier, scenario.loads.bus.map(scenario.buses.location)],
        axis=1
    ).sum()
    demand_by_region = (scenario.snapshot_weightings.generators @ demand_grouped).unstack(level=0)
    return demand_by_region

demandn_by_region = process_demand(n)
demandm1_by_region = process_demand(m_1)
demandm2_by_region = process_demand(m_2)
demandm3_by_region = process_demand(m_3)
demandp1_by_region = process_demand(p_1)
demandp2_by_region = process_demand(p_2)
demandp3_by_region = process_demand(p_3)
demandr1_by_region = process_demand(r_1)
demandr2_by_region = process_demand(r_2)
demandr3_by_region = process_demand(r_3)

mapping = {
    "H2 for industry": "hydrogen",
    "H2 for non-energy": "Non-energy demand",
    "H2 for shipping": "hydrogen",
    "shipping methanol": "methanol",
    "shipping methanol emissions": "methanol",
    "shipping oil": "oil",
    "shipping oil emissions": "emissions",
    "agriculture electricity": "electricity",
    "agriculture heat": "heat",
    "agriculture machinery oil": "oil",
    "agriculture machinery oil emissions": "emissions",
    "land transport oil emissions": "emissions",
    "electricity": "electricity",
    "gas for industry": "methane",
    "industry electricity": "electricity",
    "kerosene for aviation": "oil",
    "land transport EV": "electricity",
    "land transport fuel cell": "hydrogen",
    "land transport oil": "oil",
    "low-temperature heat for industry": "heat",
    "naphtha for industry": "Non-energy demand",
    "oil emissions": "emissions",
    "process emissions": "emissions",
    "residential and tertiary heat": "heat",
    "agriculture and industry heat": "heat",
    "residential rural heat": "heat",
    "residential urban decentral heat": "heat",
    "services rural heat": "heat",
    "services urban decentral heat": "heat",
    "solid biomass for industry": "solid biomass",
    "urban central heat": "heat",
    "NH3":"hydrogen",
    "residential and tertiary heat elec":"electricity"
}

mapping_sector = {
    "H2 for industry": ("hydrogen", "industry"),
    "H2 for shipping": ("hydrogen", "shipping"),
    "shipping methanol": ("methanol", "industry"),
    "shipping methanol emissions": ("emissions", "methanol"),
    "shipping oil": ("oil", "shipping"),
    "shipping oil emissions": ("emissions", "oil"),
    "agriculture electricity": ("electricity", "agriculture"),
    "agriculture heat": ("heat", "agriculture"),
    "agriculture machinery oil": ("oil", "agriculture"),
    "agriculture machinery oil emissions": ("emissions", "agriculture"),
    "electricity": ("electricity", "residential"),
    "gas for industry": ("methane", "industry"),
    "industry electricity": ("electricity", "industry"),
    "kerosene for aviation": ("oil", "aviation"),
    "land transport EV": ("electricity", "land transport"),
    "land transport fuel cell": ("hydrogen", "land transport"),
    "land transport oil": ("oil", "land transport"),
    "low-temperature heat for industry": ("heat", "industry"),
    "naphtha for industry": ("oil", "Non-energy demand"),
    "H2 for non-energy": ("hydrogen", "Non-energy demand"),
    "land transport oil emissions": ("emissions", "other"),
    "oil emissions": ("emissions", "other"),
    "process emissions": ("emissions", "process"),
    "residential and tertiary heat": ("heat", "residential rural"),
    "agriculture and industry heat": ("heat", "industry"),
    "residential rural heat": ("heat", "residential rural"),
    "residential urban decentral heat": ("heat", "residential urban"),
    "services rural heat": ("heat", "services rural"),
    "services urban decentral heat": ("heat", "services urban"),
    "solid biomass for industry": ("solid biomass", "industry"),
    "urban central heat": ("heat", "district heating"), 
    "methanol for shipping" : ("methane","shipping methanol"),
    "NH3":("hydrogen", "NH3 for industry"),
}




countries = ['BE', 'DE', 'FR', 'GB', 'NL']
years = ['2030', '2040', '2050']

hydrogen_consumption = {}

# Initialize dictionaries to store H2_nonenergy and H2_industry
H2_nonenergy = {}
H2_industry = {}

for year in years:
    df = pd.read_csv(f"../data/clever_Industry_{year}.csv", index_col=0).loc[countries].T
    H2_nonenergy[year] = df.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry[year] = df.loc["Total Final hydrogen consumption in industry"].sum()
    hydrogen_consumption[year] = {
        "H2_nonenergy": H2_nonenergy[year],
        "H2_industry": H2_industry[year]
    }

nf = demandn_by_region.sum()
mf1 = demandm1_by_region.sum()
mf2 = demandm2_by_region.sum()
mf3 = demandm3_by_region.sum()
pf1 = demandp1_by_region.sum()
pf2 = demandp2_by_region.sum()
pf3 = demandp3_by_region.sum()
rf1 = demandr1_by_region.sum()
rf2 = demandr2_by_region.sum()
rf3 = demandr3_by_region.sum()

csv_paths = [
    "/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resourcesreff/energy_totals.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesbau/energy_totals.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesbau/energy_totals.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesbau/energy_totals.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcessuff/energy_totals_s_6_2030.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcessuff/energy_totals_s_6_2040.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcessuff/energy_totals_s_6_2050.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesnocdr/energy_totals_s_6_2030.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesnocdr/energy_totals_s_6_2040.csv",
    "/home/umair/pypsa-eur_repository/simulations/myopic simulations/resourcesnocdr/energy_totals_s_6_2050.csv"
]

scenario_names = ["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050",
                  "NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]

residential_tertiary_heat = {}

for path, name in zip(csv_paths, scenario_names):
    df = pd.read_csv(path).T
    del df[5]
    residentialspace_elec = df.loc["electricity residential space"].sum()
    tertiaryspace_elec = df.loc["electricity services space"].sum()
    residentialwater_elec = df.loc["electricity residential water"].sum()
    tertiarywater_elec = df.loc["electricity services water"].sum()
    residentialcooking_elec = df.loc["electricity residential cooking"].sum()
    tertiarycooking_elec = df.loc["electricity services cooking"].sum()
    residential_tertiary_heat[name] = (
        residentialspace_elec + tertiaryspace_elec +
        residentialwater_elec + tertiarywater_elec +
        residentialcooking_elec + tertiarycooking_elec
    )

dataframes = [nf, mf1,mf2,mf3, pf1,pf2,pf3, rf1,rf2,rf3]


for df, name in zip(dataframes, scenario_names):
    df.loc["residential and tertiary heat elec"] = (
        residential_tertiary_heat[name]
    )
    df.loc["residential and tertiary heat"] = (
        df.loc["residential rural heat"] +
        df.loc["residential urban decentral heat"] +
        df.loc["services rural heat"] +
        df.loc["services urban decentral heat"] +
        df.loc["urban central heat"].sum() -
        residential_tertiary_heat[name]
    )
    df.loc["agriculture and industry heat"] = (
        df.loc["agriculture heat"] +
        df.loc["low-temperature heat for industry"].sum()
    )
    rows_to_drop = [
    "residential rural heat",
    "residential urban decentral heat",
    "services rural heat",
    "services urban decentral heat",
    "urban central heat",
    "agriculture heat",
    "low-temperature heat for industry",
    "shipping methanol emissions"
]

for df in dataframes:
    for row_label in rows_to_drop:
        if row_label in df.index:
            df.drop(row_label, inplace=True)

dataaframes = [pf1,pf2, pf3, rf1,rf2, rf3]
years = ['2040', '2050']

for df in dataaframes:
    for year in years:
        df.loc["H2 for industry"] = H2_industry[year]
        df.loc["H2 for non-energy"] = H2_nonenergy[year]
            
for df in dataframes:
    df.index = pd.MultiIndex.from_tuples([(mapping[i], i) for i in df.index])
    df.drop("emissions", inplace=True)


colors = config["plotting"]["tech_colors"]
colors["solid biomass"] = "greenyellow"
colors["methane"] = "orange"
colors["electricity"] = "midnightblue"
colors["Non-energy demand"] = "black"
colors["hydrogen"] = "violet"
colors["oil"] = "gray"



x= [-0.85,-0.58,-0.3, -0.02, 0.26, 0.54,0.82]
x_labels = ['Reff', 'BAU-2030','BAU-2040', 'BAU-2050','Suff-2030','Suff-2040','Suff-2050']
order = [
    "electricity",
    "heat",
    "oil",
    "solid biomass",
    "methane",
    "hydrogen",
    "Non-energy demand",
]

fig, ax = plt.subplots(figsize=(10, 10), )
pd.DataFrame(nf.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=4.8,
    width=0.2,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=True,
)
pd.DataFrame(mf1.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=3.4,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
  )
pd.DataFrame(mf2.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=2,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(mf3.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=0.6,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf1.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=-0.8,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(pf2.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=-2.2,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(pf3.groupby(level=0).sum().loc[order], columns=[""]).T.plot.bar(
    ax=ax,
    position=-3.6,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)

plt.ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)

plt.xlim(-1, 1)
plt.ylim(0, 8000)
plt.xticks(x, x_labels, rotation='vertical')
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                        
plt.legend(*d, ncol=1, fontsize=14)
plt.show()

#%%

colors = config["plotting"]["tech_colors"]
colors["H2 for industry"] = "cornflowerblue"
colors["agriculture electricity"] = "royalblue"
colors["agriculture and industry heat"] = "lightsteelblue"
colors["agriculture machinery oil"] = "darkorange"
colors["electricity"] = "navajowhite"
colors["gas for industry"] = "forestgreen"
colors["industry electricity"] = "limegreen"
colors["kerosene for aviation"] = "black"
colors["land transport EV"] = "lightcoral"
colors["land transport fuel cell"] = "mediumpurple"
colors["land transport oil"] = "thistle"
colors["low-temperature heat for industry"] = "sienna"
colors["naphta for industry"] = "sandybrown"
colors["residential and tertiary heat"] = "pink"
colors["residential urban decentral heat"] = "pink"
colors["services rural heat"] = "grey"
colors["services urban decentral heat"] = "lightgrey"
colors["shipping methanol"] = "lawngreen"
colors["shipping methanol emissions"] = "gold"
colors["shipping oil"] = "turquoise"
colors["solid biomass for industry"] = "paleturquoise"
colors["urban central heat"] = "cyan"
colors["residential and tertiary heat elec"] = "red"
colors["H2 for non-energy"] = "violet"
fig, ax = plt.subplots(figsize=(25, 20))

x= [0.075,0.19,0.3,0.43,0.53,0.64,0.75,1.075,1.19,1.3,1.43,1.53,1.64,1.75,2.075,2.19,2.3,2.43,2.53,2.64,2.75,3.075,3.19,3.3,3.43,3.53,3.64,3.75,4.075,4.19,4.3,4.43,4.53,4.64,4.75,5.075,5.19,5.3,5.43,5.53,5.64,5.75,6.075,6.19,6.3,6.43,6.53,6.64,6.75]
# x_labels = ['BAU','2030','electricity            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','heat             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','oil             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','biomass            ','2040','2050','2050 (NO-CCS)','BAU-2050', 'BAU','2030','methane            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','hydrogen            ','2040','2050','2050 (NO-CCS)','BAU-2050']
x_labels = ['Reff', 'BAU-2030','BAU-2040', 'Electricity       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Heat       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Oil       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Biomass       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Methane       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Hydrogen       /BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Non-energy       /BAU-2050','Suff-2030','Suff-2040','Suff-2050']
nf.unstack().loc[order].loc[order].plot.bar(
    ax=ax, stacked=True, edgecolor="k",position=-0.5, width=.075,legend=False,color=colors,
)
mf1.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-2, width=0.075,legend=True,color=colors,
)
mf2.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-3.5, width=0.075,legend=False,color=colors,
)
mf3.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-5, width=0.075,legend=False,color=colors,
)
pf1.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-6.5, width=0.075,legend=False,color=colors,
)
pf2.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-8, width=0.075,legend=False,color=colors,
)
pf3.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-9.5, width=0.075,legend=False,color=colors,
)
plt.ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)
plt.xlim(0, 7)
plt.ylim(0, 3500)


plt.xticks(x, x_labels, rotation='vertical', fontsize=12.5)
plt.rcParams['legend.fontsize'] = '20'
plt.rc('ytick', labelsize=20) 


plt.show()