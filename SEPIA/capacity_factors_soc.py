#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:44:47 2024

@author: umair
"""

import pypsa
import pandas as pd
import yaml
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os 
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)
def rename_techs(label):
    prefix_to_remove = [
        # "residential ",
        # "services ",
        # "urban ",
        # "rural ",
        # "central ",
        # "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        # "gas boiler",
        "biogas",
        "solar thermal",
        # "air heat pump",
        # "ground heat pump",
        # "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
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
        "co2 Store": "DAC",
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

def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    # if "heat pump" in tech or "resistive heater" in tech:
    #     return "power-to-heat"
    if tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "hydrogen storage"
    elif tech in ["residential rural air heat pump", "residential rural ground heat pump", "residential urban decentral air heat pump", "services rural air heat pump", "services rural ground heat pump", "services urban decentral air heat pump"]:
        return "Residential and tertiary HP"
    # elif "solar rooftop" in tech:
    #     return "solar rooftop"
    elif tech in ["residential rural biomass boiler", "residential urban decentral biomass boiler", "services rural biomass boiler", "services urban decentral biomass boiler"]:
        return "Residential and tertiary Biomass Boilers"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif tech in ["residential rural gas boiler", "residential urban decentral gas boiler", "services rural gas boiler", "services urban decentral gas boiler"]:
        return "Residential and tertiary Gas Boilers"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    # elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
    #      return "biomass"
    elif tech in ["residential rural oil boiler", "residential urban decentral oil boiler", "services rural oil boiler", "services urban decentral oil boiler"]:
        return "Residential and tertiary Oil Boilers"
    elif "Li ion" in tech:
        return "battery storage"
    elif "EV charger" in tech:
        return "V2G"
    elif tech in ["residential rural resistive heater", "residential urban decentral resistive heater", "services rural resistive heater", "services urban decentral resistive heater"]:
        return "Residential and tertiary Electric Heaters"
    elif "load" in tech:
        return "load shedding"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech
    
country = 'BE'   
cf = pd.read_csv(f"../results/ncdr/csvs/nodal_capacities.csv", index_col=1)
df=pd.read_csv("../results/reff/csvs/nodal_capacities.csv", index_col=1)
df = df.iloc[:, 1:]
df = df.iloc[3:, :]
df.index = df.index.str[:2]
df = df[df.index == country]
df = df.rename(columns={'Unnamed: 2': 'tech', '6': '2020'})
columns_to_convert = ['2020']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df = df.groupby('tech').sum().reset_index()
df['tech'] = df['tech'].map(rename_techs_tyndp)
df = df.groupby('tech').sum().reset_index()
cf = cf.iloc[:, 1:]
cf = cf.iloc[3:, :]
cf.index = cf.index.str[:2]
cf = cf[cf.index == country]
cf = cf.rename(columns={'Unnamed: 2': 'tech', '6': '2030','6.1': '2040','6.2': '2050'})
columns_to_convert = ['2030', '2040', '2050']
cf[columns_to_convert] = cf[columns_to_convert].apply(pd.to_numeric, errors='coerce')
cf = cf.groupby('tech').sum().reset_index()
cf['tech'] = cf['tech'].map(rename_techs_tyndp)
cf = cf.groupby('tech').sum().reset_index()
result_df = pd.merge(df, cf, on='tech', how='outer')
result_df.fillna(0, inplace=True)
years = ['2020', '2030', '2040', '2050']
technologies = result_df['tech'].unique()
capacities = result_df.set_index('tech').loc[technologies, years]

tech_colors = config["plotting"]["tech_colors"]
colors = config["plotting"]["tech_colors"]
colors["Residential and tertiary HP"] = "blue"
colors["Residential and tertiary Biomass Boilers"] = "green"
colors["Residential and tertiary Gas Boilers"] = "brown"
colors["Residential and tertiary Oil Boilers"] = "black"
colors["Residential and tertiary Electric Heaters"] = "yellow"
colors["Gas Boilers"] = "brown"
colors["Oil Boilers"] = "black"
colors["Biomass Boilers"] = "green"
colors["Heat Pumps"] = "blue"
colors["Electric Heaters"] = "yellow"
groups = [
    ["Residential and tertiary HP"],
    ["Residential and tertiary Biomass Boilers"],
    ["Residential and tertiary Gas Boilers"],
    ["Residential and tertiary Oil Boilers"],
    ["Residential and tertiary Electric Heaters"],
]

fig_capacities = make_subplots(rows=1, cols=len(groups) // 1, subplot_titles=[
    f"{', '.join(tech_group)}" for tech_group in groups], shared_yaxes=True)

df = capacities
unit='Capacity [GW]'

for i, tech_group in enumerate(groups, start=1):
    row_idx = 1 if i <= len(groups) // 1 else 1
    col_idx = i if i <= len(groups) // 1 else i - len(groups) // 1

    for tech in tech_group:
        if tech in df.index:
            y_values = [val / 1000 for val in df.loc[tech, years]]
            trace = go.Bar(
                x=years,
                y=y_values,
                name=f"{tech}",
                marker_color=tech_colors.get(tech, 'gray')
            )
            fig_capacities.add_trace(trace, row=row_idx, col=col_idx)
            fig_capacities.update_yaxes(title_text=unit, row=2, col=1)

# Update layout
output_folder = f"test/"
fig_capacities.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities", yaxis_title=unit)
# Save plot as HTML
output_folder = "test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
capacities_html_file = os.path.join(output_folder, "capacities_chart.html")
fig_capacities.write_html(capacities_html_file)


#%%
gas_boilers = [
    "residential rural gas boiler",
    "residential urban decentral gas boiler",
    "services rural gas boiler",
    "services urban decentral gas boiler",
]
heat_pumps = [
    "residential rural air heat pump",
    "residential rural ground heat pump",
    "residential urban decentral air heat pump",
    "services rural air heat pump",
    "services rural ground heat pump",
    "services urban decentral air heat pump",
]
oil_boilers = [
    "residential rural oil boiler",
    "residential urban decentral oil boiler",
    "services rural oil boiler",
    "services urban decentral oil boiler",
]
biomass_boilers = [
    "residential rural biomass boiler",
    "residential urban decentral biomass boiler",
    "services rural biomass boiler",
    "services urban decentral biomass boiler",
]
electric_heaters = [
    "residential rural resistive heater",
    "residential urban decentral resistive heater",
    "services rural resistive heater",
    "services urban decentral resistive heater",
]
capacities_groups = [
    "Residential and tertiary HP",
    "Residential and tertiary Biomass Boilers",
    "Residential and tertiary Gas Boilers",
    "Residential and tertiary Oil Boilers",
    "Residential and tertiary Electric Heaters",
]
# heat_demands = [
#     "residential rural heat",
#     "residential urban decentral heat",
#     "services rural heat",
#     "services urban decentral heat",
# ]
planning_horizons = ['2020', '2030', '2040', '2050']

# Initialize dictionaries to hold the summed values for each planning horizon
gas_boiler_totals = {}
heat_pump_totals = {}
oil_boiler_totals = {}
biomass_boiler_totals = {}
electric_heater_totals = {}
# heat_demands_totals = {}
# Loop through each planning horizon
for planning_horizon in planning_horizons:
    # Load the network for the given planning horizon
    n = pypsa.Network(f"/home/umair/pypsa-eur_repository/results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc")
    
    # Initialize the total sums for this planning horizon
    gas_boiler_sum = 0
    heat_pump_sum = 0
    oil_boiler_sum = 0
    biomass_boiler_sum = 0
    electric_heater_sum = 0
    # heat_demands_sum = 0
    # Calculate values for each category
    for boiler in gas_boilers:
        gas_boiler_sum += (n.links_t.p1.filter(like="BE").filter(like=boiler).sum(axis=1).abs().sum()/1e6)
    
    for heat_pump in heat_pumps:
        heat_pump_sum += (n.links_t.p1.filter(like="BE").filter(like=heat_pump).sum(axis=1).abs().sum()/1e6) -(n.links_t.p0.filter(like="BE").filter(like=heat_pump).sum(axis=1).abs().sum()/1e6)
        
    for boiler in oil_boilers:
        oil_boiler_sum += (n.links_t.p1.filter(like="BE").filter(like=boiler).sum(axis=1).abs().sum()/1e6)
        
    for boiler in biomass_boilers:
        biomass_boiler_sum += (n.links_t.p1.filter(like="BE").filter(like=boiler).sum(axis=1).abs().sum()/1e6)
    
    for heater in electric_heaters:
        electric_heater_sum += (n.links_t.p1.filter(like="BE").filter(like=heater).sum(axis=1).abs().sum()/1e6)
        
    # for heat_demand in heat_demands:
    #     heat_demands_sum += n.loads_t.p.filter(like="BE").filter(like=heat_demand).sum(axis=1).sum()/1e6
    # Store the sums in the respective dictionaries
    gas_boiler_totals[planning_horizon] = gas_boiler_sum
    heat_pump_totals[planning_horizon] = heat_pump_sum
    oil_boiler_totals[planning_horizon] = oil_boiler_sum
    biomass_boiler_totals[planning_horizon] = biomass_boiler_sum
    electric_heater_totals[planning_horizon] = electric_heater_sum
    # heat_demands_totals[planning_horizon] = heat_demands_sum
# Convert the dictionaries to DataFrames
df_gas_boilers = pd.DataFrame.from_dict(gas_boiler_totals, orient='index', columns=['gas_boiler_total'])
df_heat_pumps = pd.DataFrame.from_dict(heat_pump_totals, orient='index', columns=['heat_pump_total'])
df_oil_boilers = pd.DataFrame.from_dict(oil_boiler_totals, orient='index', columns=['oil_boiler_total'])
df_biomass_boilers = pd.DataFrame.from_dict(biomass_boiler_totals, orient='index', columns=['biomass_boiler_total'])
df_electric_heaters = pd.DataFrame.from_dict(electric_heater_totals, orient='index', columns=['electric_heater_total'])
# df_heat_demands = pd.DataFrame.from_dict(heat_demands_totals, orient='index', columns=['heat_demands_totals'])

df_capacities = capacities.T
df_capacities = df_capacities[capacities_groups]
df_capacities = df_capacities * 8760/1e6
# Merge the DataFrames
df_totals = pd.concat([df_gas_boilers, df_heat_pumps, df_oil_boilers, df_biomass_boilers, df_electric_heaters], axis=1)
df_heat_totals = df_totals.sum(axis=1)
capacity_factors = {
    'Gas Boilers': df_totals['gas_boiler_total'] / df_capacities['Residential and tertiary Gas Boilers'],
    'Heat Pumps': df_totals['heat_pump_total'] / df_capacities['Residential and tertiary HP'],
    'Oil Boilers': df_totals['oil_boiler_total'] / df_capacities['Residential and tertiary Oil Boilers'],
    'Biomass Boilers' : df_totals['biomass_boiler_total'] / df_capacities['Residential and tertiary Biomass Boilers'],
    'Electric Heaters': df_totals['electric_heater_total'] / df_capacities['Residential and tertiary Electric Heaters'],
    # Add similar calculations for other technologies if needed
}

# Convert the dictionary to a DataFrame
df_CF = pd.DataFrame(capacity_factors)
df_CF = df_CF.sort_index()
df_CF =df_CF.fillna(0)

# Calculate the percentages
df_gas_boilers_percentage = (df_gas_boilers['gas_boiler_total'] / df_heat_totals) * 100
df_heat_pumps_percentage = (df_heat_pumps['heat_pump_total'] / df_heat_totals) * 100
df_oil_boilers_percentage = (df_oil_boilers['oil_boiler_total'] / df_heat_totals) * 100
df_biomass_boilers_percentage = (df_biomass_boilers['biomass_boiler_total'] / df_heat_totals) * 100
df_electric_heaters_percentage = (df_electric_heaters['electric_heater_total'] / df_heat_totals) * 100

# Combine into a single DataFrame
df_percentages = pd.DataFrame({
    'Gas Boilers': df_gas_boilers_percentage,
    'Heat Pumps': df_heat_pumps_percentage,
    'Oil Boilers': df_oil_boilers_percentage,
    'Biomass Boilers': df_biomass_boilers_percentage,
    'Electric Heaters': df_electric_heaters_percentage
})
# Plot Capacity Factors
fig_CF = go.Figure()
for column in df_CF.columns:
    fig_CF.add_trace(go.Scatter(x=df_CF.index, y=df_CF[column], mode='lines+markers', name=column,marker_color=tech_colors.get(column, 'gray')))
fig_CF.update_layout(height=600, width=800, showlegend=True, title="Capacity Factors", yaxis_title="Capacity Factor")

# Save the capacity factors plot as HTML
cf_html_file = os.path.join(output_folder, "capacity_factors_chart.html")
fig_CF.write_html(cf_html_file)

fig_percentages = go.Figure()
for column in df_percentages.columns:
    fig_percentages.add_trace(go.Scatter(x=df_percentages.index, y=df_percentages[column], mode='lines+markers', name=column, marker_color=tech_colors.get(column, 'gray')))
fig_percentages.update_layout(height=600, width=800, showlegend=True, title="Heating Technology Percentages", yaxis_title="Percentage production in total heat production")

# Save the percentages plot as HTML
percentages_html_file = os.path.join(output_folder, "percentages_chart.html")
fig_percentages.write_html(percentages_html_file)

# Combine the two HTML files into one
combined_html_file = os.path.join(output_folder, "combined_charts.html")
with open(combined_html_file, "w") as combined_file:
    combined_file.write("<html><head><title>Combined Plots</title></head><body>\n")
    for file in [capacities_html_file,cf_html_file, percentages_html_file]:
        with open(file, "r") as single_file:
            combined_file.write(single_file.read())
            combined_file.write("\n<hr>\n")
    combined_file.write("</body></html>")


    

#%%
elec_dist_totals = {}
elec_ind_totals = {}
elec_agr_totals = {}
for planning_horizon in planning_horizons:
    # Load the network for the given planning horizon
    n = pypsa.Network(f"/home/umair/pypsa-eur_repository/results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc")
    elec_dist = 0
    elec_ind = 0
    elec_agr = 0
    elec_dist = n.loads_t.p.filter(regex="BE1 0$")#.filter(like="electricity distribution grid").sum(axis=1).abs()
    elec_ind = n.loads_t.p.filter(like="BE").filter(regex="industry electricity$")
    elec_agr = n.loads_t.p.filter(like="BE").filter(regex="agriculture electricity$")
    elec_dist_totals[planning_horizon] =  elec_dist.squeeze()/1e3
    elec_ind_totals[planning_horizon] =  elec_ind.squeeze()/1e3
    elec_agr_totals[planning_horizon] =  elec_agr.squeeze()/1e3
df_elec_dist_totals = pd.DataFrame(elec_dist_totals) 
df_elec_ind_totals = pd.DataFrame(elec_ind_totals) 
df_elec_agr_totals = pd.DataFrame(elec_agr_totals)   
fig_elec_dist = go.Figure()

for planning_horizon in planning_horizons:
    fig_elec_dist.add_trace(go.Scatter(
        x=df_elec_dist_totals.index,
        y=df_elec_dist_totals[planning_horizon],
        mode='lines',
        name=f'{planning_horizon}'
    ))

fig_elec_dist.update_layout(
    title="Electricity Distribution Residential and Tertiary Demand",
    xaxis_title="Time",
    yaxis_title="Electricity Demand Res and Ter [GW]",
    height=600,
    width=1200,
    xaxis=dict(tickformat='%b')
)

# Save the electricity distribution totals plot as HTML
output_folder = "test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
elec_dist_html_file = os.path.join(output_folder, "elec_dist_totals_chart.html")
fig_elec_dist.write_html(elec_dist_html_file)

# Plot Electricity Industrial Demand Totals
fig_elec_ind = go.Figure()

for planning_horizon in planning_horizons:
    fig_elec_ind.add_trace(go.Scatter(
        x=df_elec_ind_totals.index,
        y=df_elec_ind_totals[planning_horizon],
        mode='lines',
        name=f'{planning_horizon}'
    ))

fig_elec_ind.update_layout(
    title="Electricity Industrial Demand",
    xaxis_title="Time",
    yaxis_title="Electricity Demand Industry [GW]",
    height=600,
    width=1200,
    xaxis=dict(tickformat='%b')
)

# Save the electricity industrial demand totals plot as HTML
elec_ind_html_file = os.path.join(output_folder, "elec_ind_totals_chart.html")
fig_elec_ind.write_html(elec_ind_html_file)

# Plot Electricity Agricultural Demand Totals
fig_elec_agr = go.Figure()

for planning_horizon in planning_horizons:
    fig_elec_agr.add_trace(go.Scatter(
        x=df_elec_agr_totals.index,
        y=df_elec_agr_totals[planning_horizon],
        mode='lines',
        name=f'{planning_horizon}'
    ))

fig_elec_agr.update_layout(
    title="Electricity Agricultural Demand",
    xaxis_title="Time",
    yaxis_title="Electricity Demand Agriculture [GW]",
    height=600,
    width=1200,
    xaxis=dict(tickformat='%b')
)

# Save the electricity agricultural demand totals plot as HTML
elec_agr_html_file = os.path.join(output_folder, "elec_agr_totals_chart.html")
fig_elec_agr.write_html(elec_agr_html_file)

# Combine the HTML files into one
combined_html_file = os.path.join(output_folder, "combined_charts_dem.html")
with open(combined_html_file, "w") as combined_file:
    combined_file.write("<html><head><title>Combined Plots</title></head><body>\n")
    for file in [elec_dist_html_file, elec_ind_html_file, elec_agr_html_file]:
        with open(file, "r") as single_file:
            combined_file.write(single_file.read())
            combined_file.write("\n<hr>\n")
    combined_file.write("</body></html>")
    
    
#%%
pump_hydro_totals = {}
battery_totals = {}
tes_totals = {}
h2_totals = {}
for planning_horizon in planning_horizons:
    # Load the network for the given planning horizon
    n = pypsa.Network(f"/home/umair/pypsa-eur_repository/results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc")
    pump_hydro = 0
    battery = 0
    tes = 0
    h2 = 0
    pump_hydro = n.storage_units_t.p.filter(like="BE").filter(like = "PHS").sum(axis=1)#.filter(like="electricity distribution grid").sum(axis=1).abs()
    battery = n.stores_t.e.filter(like="BE").filter(like="battery").sum(axis=1)
    tes = n.stores_t.e.filter(like="BE").filter(like="water tanks").sum(axis=1)
    h2 = n.stores_t.e.filter(like="BE").filter(like="H2 Store").sum(axis=1)
    pump_hydro_totals[planning_horizon] =  pump_hydro.squeeze()/1e3
    battery_totals[planning_horizon] =  battery.squeeze()/1e3
    tes_totals[planning_horizon] =  tes.squeeze()/1e3
    h2_totals[planning_horizon] =  h2.squeeze()/1e3
df_pump_hydro_totals = pd.DataFrame(pump_hydro_totals) 
df_battery_totals = pd.DataFrame(battery_totals) 
df_tes_totals = pd.DataFrame(tes_totals)
df_h2_totals = pd.DataFrame(h2_totals)

fig = make_subplots(rows=4, cols=1, subplot_titles=[
    "Pump Hydro Storage", "Battery Storage", "Thermal Energy Storage", "Hydrogen Storage"
])

# Plot Pump Hydro Storage Totals
for planning_horizon in planning_horizons:
    fig.add_trace(go.Scatter(
        x=df_pump_hydro_totals.index,
        y=df_pump_hydro_totals[planning_horizon],
        mode='lines',
        name=f'Pump Hydro {planning_horizon}'
    ), row=1, col=1)

# Plot Battery Storage Totals
for planning_horizon in planning_horizons:
    fig.add_trace(go.Scatter(
        x=df_battery_totals.index,
        y=df_battery_totals[planning_horizon],
        mode='lines',
        name=f'Battery {planning_horizon}'
    ), row=2, col=1)

# Plot Thermal Energy Storage Totals
for planning_horizon in planning_horizons:
    fig.add_trace(go.Scatter(
        x=df_tes_totals.index,
        y=df_tes_totals[planning_horizon],
        mode='lines',
        name=f'TES {planning_horizon}'
    ), row=3, col=1)

# Plot Hydrogen Storage Totals
for planning_horizon in planning_horizons:
    fig.add_trace(go.Scatter(
        x=df_h2_totals.index,
        y=df_h2_totals[planning_horizon],
        mode='lines',
        name=f'Hydrogen {planning_horizon}'
    ), row=4, col=1)

# Update layout to include common attributes and format x-axis to show only months
fig.update_layout(
    height=1200,
    width=1200,
    title="Energy Storage Over Time",
    xaxis=dict(tickformat='%b'),
    xaxis2=dict(tickformat='%b'),
    xaxis3=dict(tickformat='%b'),
    xaxis4=dict(tickformat='%b'),
    showlegend=True
)
fig.update_yaxes(title_text="[GW]", row=1, col=1)
fig.update_yaxes(title_text="[GW]", row=2, col=1)
fig.update_yaxes(title_text="[GW]", row=3, col=1)
fig.update_yaxes(title_text="[GW]", row=4, col=1)
# Save the combined plot as HTML
output_folder = "test"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
combined_html_file = os.path.join(output_folder, "combined_energy_storage_chart.html")
fig.write_html(combined_html_file)