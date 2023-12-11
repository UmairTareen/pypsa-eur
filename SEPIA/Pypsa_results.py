#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import pandas as pd # Read/analyse data
import numpy as np
import pypsa
import logging
import geopandas as gpd
import atlite
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import hvplot.pandas
from bokeh.io import output_file, save
import xarray as xr
from atlite.gis import shape_availability, ExclusionContainer
from pypsa.descriptors import get_switchable_as_dense as as_dense
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
import os
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import cartopy.crs as ccrs
import plotly.express as px
import plotly.subplots as sp
import yaml
import plotly.io as pyo
import matplotlib.pyplot as plt
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)

scenario = 'bau'

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


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "helmeth",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)
def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "hydrogen storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif "Li ion" in tech:
        return "battery storage"
    elif "BEV charger" in tech:
        return "V2G"
    elif "load" in tech:
        return "load shedding"
    elif tech == "oil" or tech == "gas":
         return "fossil oil and gas"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech
   
def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]
            
def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"
        
        
def build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon, prefix="../results/postnetworks/elec_"):

    filename = f"{prefix}s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"

    # Construct the absolute path directly
    return os.path.abspath(filename)

# Example usage:
planning_horizons = [2020, 2030, 2040, 2050]
filename = build_filename("", 6, "", "EQ0.7c-1H-T-H-B-I-A-dist1", "v1.5", planning_horizons[0])

# Check if the file exists
if os.path.exists(filename):
    logging.debug("Processing file: %s", filename)
    n = pypsa.Network(filename)
    logging.debug("Network loaded successfully.")
else:
    logging.error("File does not exist: %s", filename)

countries = ['BE', 'DE', 'FR', 'GB', 'NL']
def calculate_ac_transmission(lines, regex_pattern):
    transmission_ac = lines.s_nom_opt.filter(regex=regex_pattern).sum()

    # Add condition to check if transmission_ac is less than or equal to 0 for 2020
    if transmission_ac <= 0:
        transmission_ac = lines.s_nom.filter(regex=regex_pattern).sum()
        transmission = 0
    else:
        transmission = (lines.s_nom_opt.filter(regex=regex_pattern).sum() - lines.s_nom.filter(regex=regex_pattern).sum()) * (lines.capital_cost.filter(regex=regex_pattern).sum()) * 0.5

    return transmission_ac, transmission

def calculate_dc_transmission(links, regex_pattern):
    transmission_dc = links.p_nom_opt.filter(regex=regex_pattern).sum()

    # Add condition to check if transmission_ac is less than or equal to 0
    if transmission_dc <= 0:
        transmission_dc = links.p_nom.filter(regex=regex_pattern).sum()
        transmissionc = 0
    else:
        transmissionc = (links.p_nom_opt.filter(regex=regex_pattern).sum() - links.p_nom.filter(regex=regex_pattern).sum()) * (links.capital_cost.filter(regex=regex_pattern).sum()) * 0.5

    return transmission_dc, transmissionc

def calculate_transmission_values(simpl, cluster, opt, sector_opt, ll, planning_horizons):
    results_dict = {}

    for planning_horizon in planning_horizons:
        filename = build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon, prefix="../results/postnetworks/elec_")
        n = pypsa.Network(filename)

        cap_ac = pd.DataFrame(index=['BE', 'DE', 'FR', 'NL'])
        cos_ac = pd.DataFrame(index=['BE', 'DE', 'FR', 'NL'])
        cap_dc = pd.DataFrame(index=['BE', 'DE', 'FR', 'NL', 'GB'])
        cos_dc = pd.DataFrame(index=['BE', 'DE', 'FR', 'NL', 'GB'])

        # AC transmission calculations
        transmission_be_ac, transmission_be = calculate_ac_transmission(n.lines, '[012]')
        transmission_de_ac, transmission_de = calculate_ac_transmission(n.lines, '[034]')
        transmission_fr_ac, transmission_fr = calculate_ac_transmission(n.lines, '[13]')
        transmission_nl_ac, transmission_nl = calculate_ac_transmission(n.lines, '[24]')

        cap_ac['transmission_AC'] = [transmission_be_ac, transmission_de_ac, transmission_fr_ac, transmission_nl_ac]
        cos_ac['transmission_AC'] = [transmission_be, transmission_de, transmission_fr, transmission_nl]
        cos_ac.loc['GB', 'transmission_AC'] = 0
        cap_ac.loc['GB', 'transmission_AC'] = 0

        # DC transmission calculations
        transmission_be_dc, transmissionc_be = calculate_dc_transmission(n.links, '14801|T6')
        transmission_de_dc, transmissionc_de = calculate_dc_transmission(n.links, '14801|T22')
        transmission_fr_dc, transmissionc_fr = calculate_dc_transmission(n.links, '14826|T2|T12|T19|T21')
        transmission_nl_dc, transmissionc_nl = calculate_dc_transmission(n.links, '14814')
        transmission_gb_dc, transmissionc_gb = calculate_dc_transmission(n.links, '14814|14826|5581|5580|T2|T12|T19|T21|T6|T22')

        cap_dc['transmission_DC'] = [transmission_be_dc, transmission_de_dc, transmission_fr_dc, transmission_nl_dc, transmission_gb_dc]
        cos_dc['transmission_DC'] = [transmissionc_be, transmissionc_de, transmissionc_fr, transmissionc_nl, transmissionc_gb]

        # Create a dictionary for the planning horizon and store results
        results_dict[planning_horizon] = {
            'cap_ac': cap_ac,
            'cos_ac': cos_ac,
            'cap_dc': cap_dc,
            'cos_dc': cos_dc
        }

    return results_dict
results = calculate_transmission_values("", 6, "", "EQ0.7c-1H-T-H-B-I-A-dist1", "v1.5", planning_horizons)

def costs(countries):
    costs = {}
    for country in countries:
      df=pd.read_csv("../results/csvs/nodal_costs.csv", index_col=2)
      df = df.iloc[:, 2:]
      df = df.iloc[9:, :]
      df.index = df.index.str[:2]
      df = df[df.index == country]
      df = df.rename(columns={'Unnamed: 3': 'tech', '6': '2030','6.1': '2040','6.2': '2050',})
      df[['2030', '2040', '2050']] = df[['2030', '2040', '2050']].apply(pd.to_numeric, errors='coerce')
      df = df.groupby('tech').sum().reset_index()
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()

      cf = pd.read_csv("../resultsreff/csvs/nodal_costs.csv", index_col=2)
      cf = cf.iloc[:, 2:]
      cf = cf.iloc[4:, :]
      cf.index = cf.index.str[:2]
      cf = cf[cf.index == country]
      cf = cf.rename(columns={'Unnamed: 3': 'tech', '6': '2020'})
      cf[['2020']] = cf[['2020']].apply(pd.to_numeric, errors='coerce')
      cf = cf.groupby('tech').sum().reset_index()
      cf['tech'] = cf['tech'].map(rename_techs_tyndp)
      cf = cf.groupby('tech').sum().reset_index()

      result_df = pd.merge(cf, df, on='tech', how='outer')
      result_df.fillna(0, inplace=True)
      if not result_df.empty:
            years = ['2020', '2030', '2040', '2050']
            technologies = result_df['tech'].unique()

            costs[country] = result_df.set_index('tech').loc[technologies, years]

    return costs

costs = costs(countries)
for country in countries:
    # Initialize dataframes for the country in the costs dictionary if not already present

    planning_horizons = [2020, 2030, 2040, 2050]

    for planning_horizon in planning_horizons:
        # Convert planning_horizon to string for column name
        planning_horizon_str = str(planning_horizon)

        # Check if the planning horizon key exists in the results dictionary
        if planning_horizon in results:
            cos_ac_df = results[planning_horizon]['cos_ac']
            cos_dc_df = results[planning_horizon]['cos_dc']
            ac_transmission_values = cos_ac_df.loc[country, 'transmission_AC']
            dc_transmission_values = cos_dc_df.loc[country, 'transmission_DC']

            # Assign values to existing columns for each year
            costs[country].loc['AC Transmission', planning_horizon_str] = ac_transmission_values
            costs[country].loc['DC Transmission', planning_horizon_str] = dc_transmission_values

output_directory = "csvs"
# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for country, dataframe in costs.items():
    # Specify the file path within the output directory
    file_path = os.path.join(output_directory, f"{country}_costs_{scenario}.csv")
    
    # Save the DataFrame to a CSV file
    dataframe.to_csv(file_path, index=True)

    print(f"CSV file for {country} saved at: {file_path}")
    
def capacities(countries, results):
    capacities = {}
    for country in countries:
      df=pd.read_csv("../resultsreff/csvs/nodal_capacities.csv", index_col=1)
      cf = pd.read_csv("../results/csvs/nodal_capacities.csv", index_col=1)
      df = df.iloc[:, 1:]
      df = df.iloc[4:, :]
      df.index = df.index.str[:2]
      df = df[df.index == country]
      df = df.rename(columns={'Unnamed: 2': 'tech', '6': '2020'})
      columns_to_convert = ['2020']
      df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
      df = df.groupby('tech').sum().reset_index()
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()
      cf = cf.iloc[:, 1:]
      cf = cf.iloc[7:, :]
      cf.index = cf.index.str[:2]
      cf = cf[cf.index == country]
      cf = cf.rename(columns={'Unnamed: 2': 'tech', '6': '2030','6.1': '2040','6.2': '2050',})
      columns_to_convert = ['2030', '2040', '2050']
      cf[columns_to_convert] = cf[columns_to_convert].apply(pd.to_numeric, errors='coerce')
      cf = cf.groupby('tech').sum().reset_index()
      cf['tech'] = cf['tech'].map(rename_techs_tyndp)
      cf = cf.groupby('tech').sum().reset_index()
      result_df = pd.merge(df, cf, on='tech', how='outer')
      result_df.fillna(0, inplace=True)
      if not result_df.empty:
            years = ['2020', '2030', '2040', '2050']
            technologies = result_df['tech'].unique()

            capacities[country] = result_df.set_index('tech').loc[technologies, years]

    return capacities

capacities = capacities(countries, results)

for country in countries:
   
    planning_horizons = [2020, 2030, 2040, 2050]

    for planning_horizon in planning_horizons:
        # Convert planning_horizon to string for column name
        planning_horizon_str = str(planning_horizon)

        # Check if the planning horizon key exists in the results dictionary
        if planning_horizon in results:
            cap_ac_df = results[planning_horizon]['cap_ac']
            cap_dc_df = results[planning_horizon]['cap_dc']
            ac_transmission_values = cap_ac_df.loc[country, 'transmission_AC']
            dc_transmission_values = cap_dc_df.loc[country, 'transmission_DC']

            # Assign values to existing columns for each year
            capacities[country].loc['AC Transmission lines', planning_horizon_str] = ac_transmission_values
            capacities[country].loc['DC Transmission lines', planning_horizon_str] = dc_transmission_values
        
for country, dataframe in capacities.items():
    # Specify the file path where you want to save the CSV file
    file_path = f"csvs/{country}_capacities_{scenario}.csv"
    
    # Save the DataFrame to a CSV file
    dataframe.to_csv(file_path, index=True)

    print(f"CSV file for {country} saved at: {file_path}")  



def plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["fossil oil and gas"] = colors["oil"]
    colors["hydrogen storage"] = colors["H2 Store"]
    colors["load shedding"] = 'black'
    colors["gas-to-power/heat"] = 'darkred'
    colors["load"] = 'black'
    colors["Imports_Exports"] = colors["oil"]
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        filename = build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon, prefix="../results/postnetworks/elec_")
        n = pypsa.Network(filename)

        assign_location(n)
        assign_carriers(n)
        carrier = 'AC'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]].filter(like=country)
                        .groupby(c.df.carrier, axis=1)
                        .sum(),
                    ),
                    axis=1,
                )

        for c in n.iterate_components(n.one_port_components):
            comps = c.df.index[c.df.bus.isin(busesn)]
            supplyn = pd.concat(
                (
                    supplyn,
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"])).filter(like=country)
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
        if country == 'BE':
           ac_lines = n.lines_t.p1.filter(items=['0', '1', '2']).sum(axis=1)
           dc_lines = n.links_t.p0.filter(items=['14801','T6']).sum(axis=1)
           merged_series = pd.concat([ac_lines, dc_lines], axis=1)
           imp_exp = merged_series.sum(axis=1)
           imp_exp = imp_exp.rename('Imports_Exports')
           supplyn['Imports_Exports'] = imp_exp
         
        if country == 'DE':
           ac_lines = n.lines_t.p1.filter(items=['0', '3', '4']).sum(axis=1)
           dc_lines = n.links_t.p0.filter(items=['14801','T22']).sum(axis=1)
           merged_series = pd.concat([ac_lines, dc_lines], axis=1)
           imp_exp = merged_series.sum(axis=1)
           imp_exp = imp_exp.rename('Imports_Exports')
           supplyn['Imports_Exports'] = imp_exp
           
        if country == 'FR':
           ac_lines = n.lines_t.p0.filter(items=['1', '3']).sum(axis=1)
           dc_lines = n.links_t.p1.filter(items=['14826','T2', 'T12', 'T19', 'T21']).sum(axis=1)
           merged_series = pd.concat([ac_lines, dc_lines], axis=1)
           imp_exp = merged_series.sum(axis=1)
           imp_exp = imp_exp.rename('Imports_Exports')
           supplyn['Imports_Exports'] = imp_exp
           
        if country == 'GB':
           dc_lines = n.links_t.p1.filter(items=['14814','14826','T2','T6', 'T12', 'T19', 'T21','T22']).sum(axis=1)
           imp_exp = dc_lines.rename('Imports_Exports')
           supplyn['Imports_Exports'] = imp_exp
           
        if country == 'NL':
           ac_lines = n.lines_t.p0.filter(items=['2', '4']).sum(axis=1)
           dc_lines = n.links_t.p0.filter(items=['14814']).sum(axis=1)
           merged_series = pd.concat([ac_lines, dc_lines], axis=1)
           imp_exp = merged_series.sum(axis=1)
           imp_exp = imp_exp.rename('Imports_Exports')
           supplyn['Imports_Exports'] = imp_exp

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)

        start = "2013-02-01"
        stop = "2013-02-07"

        threshold = 0.1

        to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]

        if len(to_dropn) != 0:
            logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
            supplyn.drop(columns=to_dropn, inplace=True)

        supplyn.index.name = None

        supplyn = supplyn / 1e3


        supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()

        c_solarn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="solar", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        c_onwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="onwind", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        c_offwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="offwind", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        supplyn = supplyn.T
        supplyn.loc["solar"] = supplyn.loc["solar"] + c_solarn
        supplyn.loc["onshore wind"] = supplyn.loc["onshore wind"] + c_onwindn
        supplyn.loc["offshore wind"] = supplyn.loc["offshore wind"] + c_offwindn
        supplyn.loc["solar curtailment"] = -abs(c_solarn)
        supplyn.loc["onshore curtailment"] = -abs(c_onwindn)
        supplyn.loc["offshore curtailment"] = -abs(c_offwindn)
        supplyn = supplyn.T
        positive_supplyn = supplyn[supplyn >= 0].fillna(0)
        negative_supplyn = supplyn[supplyn < 0].fillna(0)

        

        
        positive_plot = positive_supplyn.loc[start:stop].hvplot.area(
           x='index', y=list(positive_supplyn.columns),
           color=[colors[tech] for tech in positive_supplyn.columns],
           line_dash='solid', line_width=0,
           xlabel='Time', ylabel='Power [GW]',
           title=f"Power Dispatch (Winter Week) - {country} - {planning_horizon}",
           width=1200, height=600,
           responsive=False,
           stacked=True,)

        negative_plot = negative_supplyn.loc[start:stop].hvplot.area(
           x='index', y=list(negative_supplyn.columns),
           color=[colors[tech] for tech in negative_supplyn.columns],
           line_dash='solid', line_width=0,
           xlabel='Time', ylabel='Power [GW]',
           width=1200, height=600,
           responsive=False,
           stacked=True,)

# Combine positive and negative plots using the + operator
        plot = positive_plot * negative_plot
    

            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", plot))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))
        
     html_filename = f"dispatch_plots_{country}.html"
     output_folder = 'output_charts' # Set your desired output folder
     os.makedirs(output_folder, exist_ok=True)
     html_filepath = os.path.join(output_folder, html_filename)
     tabs.save(html_filepath)

# Call the function
plot_series_power("", 6, "", "EQ0.7c-1H-T-H-B-I-A-dist1", "v1.5", planning_horizons)

def plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["agriculture heat"] = "grey"
    colors["gas-to-power/heat"] = "orange"
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        filename = build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon, prefix="../results/postnetworks/elec_")
        n = pypsa.Network(filename)

        assign_location(n)
        assign_carriers(n)
        carrier = 'heat'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]].filter(like=country)
                        .groupby(c.df.carrier, axis=1)
                        .sum(),
                    ),
                    axis=1,
                )

        for c in n.iterate_components(n.one_port_components):
            comps = c.df.index[c.df.bus.isin(busesn)]
            supplyn = pd.concat(
                (
                    supplyn,
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"])).filter(like=country)
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)

        start = "2013-02-01"
        stop = "2013-02-07"

        threshold = 0.1

        to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]

        if len(to_dropn) != 0:
            logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
            supplyn.drop(columns=to_dropn, inplace=True)

        supplyn.index.name = None

        supplyn = supplyn / 1e3
        supplyn.rename(
            columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
        )
        supplyn.columns = supplyn.columns.str.replace("residential ", "")
        supplyn.columns = supplyn.columns.str.replace("services ", "")
        supplyn.columns = supplyn.columns.str.replace("urban decentral ", "decentral ")


        supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()
        positive_supplyn = supplyn[supplyn >= 0].fillna(0)
        negative_supplyn = supplyn[supplyn < 0].fillna(0)

        

        
        positive_plot = positive_supplyn.loc[start:stop].hvplot.area(
           x='index', y=list(positive_supplyn.columns),
           color=[colors[tech] for tech in positive_supplyn.columns],
           line_dash='solid', line_width=0,
           xlabel='Time', ylabel='Heat [GW]',
           title=f"Heat Dispatch (Winter Week) - {country} - {planning_horizon}",
           width=1200, height=600,
           responsive=False,
           stacked=True,)

        negative_plot = negative_supplyn.loc[start:stop].hvplot.area(
           x='index', y=list(negative_supplyn.columns),
           color=[colors[tech] for tech in negative_supplyn.columns],
           line_dash='solid', line_width=0,
           xlabel='Time', ylabel='Heat [GW]',
           width=1200, height=600,
           responsive=False,
           stacked=True,)

# Combine positive and negative plots using the + operator
        plot = positive_plot * negative_plot
    

            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", plot))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))


        # Save the tabs as an HTML file
     html_filename = f"dispatch_plots_heat_{country}.html"
     output_folder = 'output_charts'  # Set your desired output folder
     os.makedirs(output_folder, exist_ok=True)
     html_filepath = os.path.join(output_folder, html_filename)
     tabs.save(html_filepath)

# Call the function
plot_series_heat("", 6, "", "EQ0.7c-1H-T-H-B-I-A-dist1", "v1.5", planning_horizons)

def create_bar_chart(costs, country, output_folder='output_charts', unit='Billion Euros/year'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"

    title = f"{country} - Results"
    df = costs[country]
    df = df.rename_axis(unit)
    df = df.reset_index()
    df.index = df.index.astype(str)

    # Create a bar chart using Plotly
    fig = go.Figure()
    df_transposed = df.set_index(unit).T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))

    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')

    # Save the HTML file for each country
    # output_file_path = os.path.join(output_folder, f"{country}_bar_chart.html")
    # fig.write_html(output_file_path)

    return fig

def create_capacity_chart(capacities, country, output_folder='output_charts', unit='Capacity [GW]'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    groups = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
        ["AC Transmission lines"],
        ["DC Transmission lines"],
        ["CCGT"],
        ["nuclear"],
    ]

    # Create a subplot for each technology
    years = ['2020', '2030', '2040', '2050']
    fig = make_subplots(rows=2, cols=len(groups) // 2, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in groups], shared_yaxes=True)

    df = capacities[country]

    for i, tech_group in enumerate(groups, start=1):
        row_idx = 1 if i <= len(groups) // 2 else 2
        col_idx = i if i <= len(groups) // 2 else i - len(groups) // 2

        for tech in tech_group:
            y_values = [val / 1000 for val in df.loc[tech, years]]
            trace = go.Bar(
                x=years,
                y=y_values,
                name=f"{tech}",
                marker_color=tech_colors.get(tech, 'gray')
            )
            fig.add_trace(trace, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=unit, row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {country}", yaxis_title=unit)

    # Save plot as HTML
    # html_file_path = os.path.join(output_folder, f"{country}_capacities_chart.html")
    # fig.write_html(html_file_path)

    return fig

def create_combined_chart_country(costs, capacities, country, output_folder='output_charts'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"

    # Create bar chart
    bar_chart = create_bar_chart(costs, country, output_folder)
    combined_html += f"<div><h2>{country} - Bar Chart</h2>{bar_chart.to_html()}</div>"

    # Create capacities chart
    capacities_chart = create_capacity_chart(capacities, country, output_folder)
    combined_html += f"<div><h2>{country} - Capacities Chart</h2>{capacities_chart.to_html()}</div>"

    # Save the Panel object to HTML
    plot_series_file_path = os.path.join(output_folder, f"dispatch_plots_{country}.html")
    plot_series_heat_file_path = os.path.join(output_folder, f"dispatch_plots_heat_{country}.html")

    # Include the saved HTML in the combined HTML
    with open(plot_series_heat_file_path, "r") as plot_series_heat_file:
        plot_series_heat_html = plot_series_heat_file.read()
        combined_html += f"<div><h2>{country} - Heat Dispatch</h2>{plot_series_heat_html}</div>"
        
    with open(plot_series_file_path, "r") as plot_series_file:
        plot_series_html = plot_series_file.read()
        combined_html += f"<div><h2>{country} - Power Dispatch</h2>{plot_series_html}</div>"

    combined_html += "</body></html>"
    # Save the combined HTML file
    combined_file_path = os.path.join(output_folder, f"{country}_combined_chart.html")
    with open(combined_file_path, "w") as combined_file:
        combined_file.write(combined_html)

# Example usage
for country in costs.keys():
    create_combined_chart_country(costs, capacities, country)
    
    
for country in costs.keys():
    combined_file_path = os.path.join('output_charts', f"dispatch_plots_heat_{country}.html")
    plot_series_file_path = os.path.join('output_charts', f"dispatch_plots_{country}.html")
    
    if os.path.exists(combined_file_path):
        os.remove(combined_file_path)
    if os.path.exists(plot_series_file_path):
        os.remove(plot_series_file_path)