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

folder = '/home/umair/pypsa-eur_repository/'

n=pypsa.Network(folder + "results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2020.nc")

m_1=pypsa.Network(folder + "results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2030.nc")
m_2=pypsa.Network(folder + "results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2040.nc")
m_3=pypsa.Network(folder + "results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2050.nc")


p_1=pypsa.Network(folder + "results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2030.nc")
p_2=pypsa.Network(folder + "results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2040.nc")
p_3=pypsa.Network(folder + "results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2050.nc")


#%%

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
    df = pd.read_csv(f"../data/clever_Industry_{year}.csv", index_col=0).loc['BE'].T
    H2_nonenergy[year] = df.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry[year] = df.loc["Total Final hydrogen consumption in industry"].sum()
    hydrogen_consumption[year] = {
        "H2_nonenergy": H2_nonenergy[year],
        "H2_industry": H2_industry[year]
    }


nf = demandn_by_region.loc['BE1 0'].copy().rename('2020')
mf1 = demandm1_by_region.loc['BE1 0'].copy().rename('2030')
mf2 = demandm2_by_region.loc['BE1 0'].copy().rename('2040')
mf3 = demandm3_by_region.loc['BE1 0'].copy().rename('2050')
pf1 = demandp1_by_region.loc['BE1 0'].copy().rename('2030')
pf2 = demandp2_by_region.loc['BE1 0'].copy().rename('2040')
pf3 = demandp3_by_region.loc['BE1 0'].copy().rename('2050')

csv_paths = [
    "/home/umair/pypsa-eur_repository/resources/reff/energy_totals_s_6_2020.csv",
    "/home/umair/pypsa-eur_repository/resources/bau/energy_totals_s_6_2030.csv",
    "/home/umair/pypsa-eur_repository/resources/bau/energy_totals_s_6_2040.csv",
    "/home/umair/pypsa-eur_repository/resources/bau/energy_totals_s_6_2050.csv",
    "/home/umair/pypsa-eur_repository/resources/ncdr/energy_totals_s_6_2030.csv",
    "/home/umair/pypsa-eur_repository/resources/ncdr/energy_totals_s_6_2040.csv",
    "/home/umair/pypsa-eur_repository/resources/ncdr/energy_totals_s_6_2050.csv"
]

scenario_names = ["Reff", "BAU-2030","BAU-2040","BAU-2050", "SUff-2030", "SUff-2040", "SUff-2050",
                  "NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]

residential_tertiary_heat = {}

for path, name in zip(csv_paths, scenario_names):
    df = pd.read_csv(path, index_col=0).T
    df = df['BE']

    # del df[5]
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

dataframes = [nf, mf1,mf2,mf3, pf1,pf2,pf3]


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

dataaframes = [pf1, pf2, pf3]
years = ['2030', '2040', '2050']

for df, year in zip(dataaframes, years):
        df.loc["H2 for industry"] = H2_industry[year]
        df.loc["H2 for non-energy"] = H2_nonenergy[year]


ind_2020 = pd.read_csv("../resources/reff/industrial_energy_demand_elec_s_6_2020.csv", index_col=0).loc['BE1 0']
ind_2030 = pd.read_csv("../resources/bau/industrial_energy_demand_elec_s_6_2030.csv", index_col=0).loc['BE1 0']
ind_2040 = pd.read_csv("../resources/bau/industrial_energy_demand_elec_s_6_2040.csv", index_col=0).loc['BE1 0']
ind_2050 = pd.read_csv("../resources/bau/industrial_energy_demand_elec_s_6_2050.csv", index_col=0).loc['BE1 0']
ind_2030_suff = pd.read_csv("../resources/ncdr/industrial_energy_demand_elec_s_6_2030.csv", index_col=0).loc['BE1 0']
ind_2040_suff = pd.read_csv("../resources/ncdr/industrial_energy_demand_elec_s_6_2040.csv", index_col=0).loc['BE1 0']
ind_2050_suff = pd.read_csv("../resources/ncdr/industrial_energy_demand_elec_s_6_2050.csv", index_col=0).loc['BE1 0']
nf.loc["naphtha for industry"] = ind_2020["naphtha"] 
nf.loc["solid biomass for industry"] = ind_2020["solid biomass"] 
nf.loc["NH3"] = ind_2020["ammonia"]  
mf1.loc["naphtha for industry"] = ind_2030["naphtha"]  
mf1.loc["NH3"] = ind_2030["ammonia"] 
mf2.loc["naphtha for industry"] = ind_2040["naphtha"]  
mf2.loc["NH3"] = ind_2040["ammonia"] 
mf3.loc["naphtha for industry"] = ind_2050["naphtha"]  
mf3.loc["NH3"] = ind_2050["ammonia"] 
pf1.loc["NH3"] = ind_2030["ammonia"] 
pf2.loc["NH3"] = ind_2040["ammonia"] 
pf3.loc["NH3"] = ind_2050["ammonia"] 
pf1.loc["naphtha for industry"] = ind_2030_suff["naphtha"] 
pf2.loc["naphtha for industry"] = ind_2040_suff["naphtha"] 
pf3.loc["naphtha for industry"] = ind_2050_suff["naphtha"] 


demands_2020 = n.loads_t.p_set.filter(like='BE')
land_transport_oil = demands_2020['BE1 0 land transport oil'].sum() / 1e6
nf.loc["land transport oil"] = land_transport_oil

# Load energy demand data
energy_demand = pd.read_csv("../resources/reff/energy_totals_s_6_2020.csv", index_col=0).T['BE']

# update values in the 'nf'
nf.loc["agriculture machinery oil"] = energy_demand.loc["total agriculture machinery"].sum()
nf.loc["kerosene for aviation"] = energy_demand.loc["total international aviation"].sum()
nf.loc["shipping oil"] = (energy_demand.loc["total domestic navigation"].sum() +
                          energy_demand.loc["total international navigation"].sum())

demands_2030_bau = m_1.loads_t.p_set.filter(like='BE')
land_transport_oil = demands_2030_bau['BE1 0 land transport oil'].sum() / 1e6
mf1.loc["land transport oil"] = land_transport_oil

demands_2040_bau = m_2.loads_t.p_set.filter(like='BE')
land_transport_oil = demands_2040_bau['BE1 0 land transport oil'].sum() / 1e6
mf2.loc["land transport oil"] = land_transport_oil

energy_demand_bau = pd.read_csv("../resources/bau/energy_totals_s_6_2030.csv", index_col=0).T
energy_demand_bau = energy_demand_bau['BE']
agriculture_machinery_oil_bau = energy_demand_bau.loc["total agriculture machinery"].sum()
aviation_p_bau = energy_demand_bau.loc["total international aviation"].sum()
navig_d_30 = energy_demand_bau.loc["total domestic navigation"].sum()
navig_i_30 = energy_demand_bau.loc["total international navigation"].sum()
navigation_30 = navig_d_30 + navig_i_30.sum()

mf1.loc["agriculture machinery oil"] = agriculture_machinery_oil_bau
mf1.loc["kerosene for aviation"] = aviation_p_bau
mf1.loc["shipping oil"] = navigation_30 * 0.7
mf1.loc["shipping methanol"] = navigation_30 * 0.3

mf2.loc["agriculture machinery oil"] = agriculture_machinery_oil_bau
mf2.loc["kerosene for aviation"] = aviation_p_bau
mf2.loc["shipping oil"] = navigation_30 * 0.3
mf2.loc["shipping methanol"] = navigation_30 * 0.7

mf3.loc["agriculture machinery oil"] = agriculture_machinery_oil_bau
mf3.loc["kerosene for aviation"] = aviation_p_bau
mf3.loc["shipping oil"] = navigation_30 * 0.0
mf3.loc["shipping methanol"] = navigation_30 * 1

demands_2030_suff=p_1.loads_t.p_set.filter(like='BE')
land_transport_oil =demands_2030_suff['BE1 0 land transport oil'].sum()/1e6
pf1.loc["land transport oil"] = land_transport_oil
demands_2040_suff=p_2.loads_t.p_set.filter(like='BE')
land_transport_oil =demands_2040_suff['BE1 0 land transport oil'].sum()/1e6
pf2.loc["land transport oil"] = land_transport_oil
energy_demand_suff_2030 = pd.read_csv("../resources/ncdr/energy_totals_s_6_2030.csv", index_col=0).T
energy_demand_suff_2030 = energy_demand_suff_2030['BE']
agriculture_machinery_oil_suff =energy_demand_suff_2030.loc["total agriculture machinery"].sum()
aviation_p_suff = energy_demand_suff_2030.loc["total international aviation"].sum() 
navig_d_30s = energy_demand_suff_2030.loc["total domestic navigation"].sum()
navig_i_30s =energy_demand_suff_2030.loc["total international navigation"].sum()
navigation_30s = navig_d_30s+ navig_i_30s.sum()
pf1.loc["agriculture machinery oil"] = agriculture_machinery_oil_suff
pf1.loc["kerosene for aviation"] = aviation_p_suff
pf1.loc["shipping oil"] = navigation_30s * 0.7
pf1.loc["shipping methanol"] = navigation_30s * 0.3

energy_demand_suff_2040 = pd.read_csv("../resources/ncdr/energy_totals_s_6_2040.csv", index_col=0).T
energy_demand_suff_2040 = energy_demand_suff_2040['BE']
agriculture_machinery_oil_suff =energy_demand_suff_2040.loc["total agriculture machinery"].sum()
aviation_p_suff = energy_demand_suff_2040.loc["total international aviation"].sum() 
navig_d_40s = energy_demand_suff_2040.loc["total domestic navigation"].sum()
navig_i_40s =energy_demand_suff_2040.loc["total international navigation"].sum()
navigation_40s = navig_d_40s+ navig_i_40s.sum()
pf2.loc["agriculture machinery oil"] = agriculture_machinery_oil_suff
pf2.loc["kerosene for aviation"] = aviation_p_suff
pf2.loc["shipping oil"] = navigation_40s * 0.3
pf2.loc["shipping methanol"] = navigation_40s * 0.7

energy_demand_suff_2050 = pd.read_csv("../resources/ncdr/energy_totals_s_6_2050.csv", index_col=0).T
energy_demand_suff_2050 = energy_demand_suff_2050['BE']
agriculture_machinery_oil_suff =energy_demand_suff_2050.loc["total agriculture machinery"].sum()
aviation_p_suff = energy_demand_suff_2050.loc["total international aviation"].sum() 
navig_d_50s = energy_demand_suff_2050.loc["total domestic navigation"].sum()
navig_i_50s =energy_demand_suff_2050.loc["total international navigation"].sum()
navigation_50s = navig_d_50s+ navig_i_50s.sum()
pf3.loc["agriculture machinery oil"] = agriculture_machinery_oil_suff
pf3.loc["kerosene for aviation"] = aviation_p_suff
pf3.loc["shipping oil"] = navigation_50s * 0.0
pf3.loc["shipping methanol"] = navigation_50s * 1
        
          
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
    # "methanol",
]

fig, ax = plt.subplots(figsize=(10, 13), )
pd.DataFrame(nf.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=4.8,
    width=0.2,
    stacked=True,
    color=colors,
    edgecolor="k",
    legend=True,
)
pd.DataFrame(mf1.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=3.4,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
  )
pd.DataFrame(mf2.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=2,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(mf3.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=0.6,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
)
pd.DataFrame(pf1.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=-0.8,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(pf2.groupby(level=0).sum().loc[order]).T.plot.bar(
    ax=ax,
    position=-2.2,
    width=0.2,
    stacked=True,
    color=colors,
    label= "Ref",
    edgecolor="k",
    legend=False,
    
)
pd.DataFrame(pf3.groupby(level=0).sum().loc[order]).T.plot.bar(
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
plt.ylim(0, 1000)
plt.xticks(x, x_labels, rotation='vertical')
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
# ax.bar_label(ax.containers[48], fmt='%.1f',label_type='edge', fontsize=15)
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                        
plt.legend(*d, ncol=1, fontsize=14)
plt.show()

#%%
frames = [nf, mf1,mf2,mf3]
gf= pd.concat(frames, axis=1)
gf = gf.groupby(level=0).sum()
gf = gf.rename(columns={0: '2020', 1: '2030', 2: '2040', 3: '2050'})
hf= pd.concat(frames, axis=1)
hf = hf.groupby(level=0).sum()
hf = hf.rename(columns={0: '2020', 1: '2030', 2: '2040', 3: '2050'})

fig, ax = plt.subplots(
    figsize=(20, 10),
)
# gf.T.plot.bar(ax=ax1,stacked= True, color=colors, legend =False)
hf.T.plot.area(ax=ax,stacked= True, color=colors, legend =False)
ax.set_ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)
ax.set_xlabel("")
ax.set_xticks([0,1,2,3], labels=gf.columns, fontsize=20)
ax.set_yticks([0,2000,4000,6000,8000],fontsize=12)
ax.set_ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)
ax.set_xlabel("")
plt.legend(ncol=1,bbox_to_anchor=(1.21,1),
          fancybox=False, shadow=False,fontsize=17 )
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
# plt.savefig('demandss.png')
plt.tight_layout()
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
fig, ax = plt.subplots(figsize=(35, 30))

x= [0.075,0.19,0.3,0.43,0.53,0.64,0.75,1.075,1.19,1.3,1.43,1.53,1.64,1.75,2.075,2.19,2.3,2.43,2.53,2.64,2.75,3.075,3.19,3.3,3.43,3.53,3.64,3.75,4.075,4.19,4.3,4.43,4.53,4.64,4.75,5.075,5.19,5.3,5.43,5.53,5.64,5.75,6.075,6.19,6.3,6.43,6.53,6.64,6.75]
# x_labels = ['BAU','2030','electricity            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','heat             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','oil             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','biomass            ','2040','2050','2050 (NO-CCS)','BAU-2050', 'BAU','2030','methane            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','hydrogen            ','2040','2050','2050 (NO-CCS)','BAU-2050']
x_labels = ['Reff', 'BAU-2030','BAU-2040', 'Electricity       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Heat       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Oil       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Biomass       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Methane       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Hydrogen       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Non-energy       BAU-2050','Suff-2030','Suff-2040','Suff-2050']
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
plt.ylim(0, 200)


plt.xticks(x, x_labels, rotation='vertical', fontsize=18)
plt.rcParams['legend.fontsize'] = '20'
plt.rc('ytick', labelsize=20) 


plt.show()

#%%
fig, ax = plt.subplots(figsize=(30, 15))
xx= [0.13,0.36,0.58,0.80,1.13,1.36,1.58,1.80,2.13,2.36,2.58,2.80,3.13,3.36,3.58,3.80,4.13,4.36,4.58,4.80,5.13,5.36,5.58,5.80,6.13,6.36,6.58,6.80]
# x_labels = ['BAU','2030','electricity            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','heat             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','oil             ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','biomass            ','2040','2050','2050 (NO-CCS)','BAU-2050', 'BAU','2030','methane            ','2040','2050','2050 (NO-CCS)','BAU-2050','BAU','2030','hydrogen            ','2040','2050','2050 (NO-CCS)','BAU-2050']
xx_labels = ['Reff', '2030','Electricity       2040', '2050','Reff', '2030','Heat       2040', '2050', 'Reff', '2030','Oil       2040', '2050', 'Reff', '2030','Biomass       2040', '2050', 'Reff', '2030','Methane       2040', '2050','Reff', '2030','Hydrogen       2040', '2050','Reff', '2030','Non-energy       2040', '2050']
nf.unstack().loc[order].loc[order].plot.bar(
    ax=ax, stacked=True, edgecolor="k",position=-0.3, width=0.15,legend=False,color=colors,
)
mf1.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-1.8, width=0.15,legend=True,color=colors,
)
mf2.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-3.3, width=0.15,legend=False,color=colors,
)
mf3.unstack().loc[order].plot.bar(
      ax=ax, stacked=True, edgecolor="k",position=-4.8, width=0.15,legend=False,color=colors,
)
# pf1.unstack().loc[order].plot.bar(
#       ax=ax, stacked=True, edgecolor="k",position=-6.5, width=0.075,legend=False,color=colors,
# )
# pf2.unstack().loc[order].plot.bar(
#       ax=ax, stacked=True, edgecolor="k",position=-8, width=0.075,legend=False,color=colors,
# )
# pf3.unstack().loc[order].plot.bar(
#       ax=ax, stacked=True, edgecolor="k",position=-9.5, width=0.075,legend=False,color=colors,
# )
plt.ylabel("Final energy and non-energy demand [TWh/a]", fontsize=30)
plt.xlim(0, 7)
plt.ylim(0, 2800)


plt.xticks(xx, xx_labels, rotation='vertical', fontsize=30)
# plt.rcParams['legend.fontsize'] = '20'
plt.rc('ytick', labelsize=30) 
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                        
plt.legend(*d, ncol=2, fontsize=25)
plt.show()

plt.show()