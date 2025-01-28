#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import pypsa
import logging
import geopandas as gpd
import os
import sys
import panel as pn
import base64
import matplotlib.pyplot as plt
from pandas.plotting import table
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy.crs as ccrs
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
from jinja2 import Template
current_script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_script_dir, "../scripts/")
sys.path.append(scripts_path)
from plot_summary import rename_techs, preferred_order
from plot_power_network import assign_location, load_projection
from plot_power_network import add_legend_circles, add_legend_patches, add_legend_lines
from make_summary import assign_carriers
import yaml


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    # if "heat pump" in tech or "resistive heater" in tech:
    #     return "power-to-heat"
    if tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    # elif tech in ["H2 Store", "H2 storage"]:
    #     return "hydrogen storage"
    elif tech in [ "CHP", "H2 Fuel Cell"]:
        return "CHP"
    elif tech in [ "battery charger", "battery discharger"]:
        return "battery storage"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif "load" in tech:
        return "load shedding"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech

def logo():
    file = snakemake.input.sepia_config
    excel_file = pd.read_excel(file, ['MAIN_PARAMS'], index_col=0)
    excel_file = excel_file["MAIN_PARAMS"].drop('Description',axis=1).to_dict()['Value']
    logo = dict(source=excel_file['PROJECT_LOGO'],
        xref="paper",
        yref="paper",
        x=0.5,
        y=1,
        xanchor="center",
        yanchor="bottom",
        sizex=0.2,
        sizey=0.2,
        layer="below")
    return logo

def build_filename(simpl,cluster,opt,sector_opt,ll ,planning_horizon):
    prefix=f"results/{study}/postnetworks/elec_"
    return prefix+"s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc".format(
        simpl=simpl,
        cluster=cluster,
        opt=opt,
        sector_opt=sector_opt,
        ll=ll,
        planning_horizon=planning_horizon
    )

def load_file(filename):
    # Use pypsa.Network to load the network from the filename
    return pypsa.Network(filename)

def load_files(study, planning_horizons, simpl, cluster, opt, sector_opt, ll):
    files = {}
    for planning_horizon in planning_horizons:
        filename = build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon)
        files[planning_horizon] = load_file(filename)
    return files


def calculate_ac_transmission(lines, line_numbers):
    transmission_ac = lines.s_nom_opt[line_numbers].sum()

    # Add condition to check if transmission_ac is less than or equal to 0 for 2020
    if transmission_ac <= 0:
        transmission_ac = lines.s_nom[line_numbers].sum()
        transmission = 0
    else:
        transmission = (lines.s_nom_opt[line_numbers].sum() - lines.s_nom[line_numbers].sum()) * (lines.capital_cost[line_numbers].sum()) * 0.5

    return transmission_ac, transmission

def calculate_dc_transmission(links, link_numbers):
    transmission_dc = links.p_nom_opt[link_numbers].sum()

    # Add condition to check if transmission_ac is less than or equal to 0
    if transmission_dc <= 0:
        transmission_dc = links.p_nom[link_numbers].sum()
        transmissionc = 0
    else:
        transmissionc = (links.p_nom_opt[link_numbers].sum() - links.p_nom[link_numbers].sum()) * (links.capital_cost[link_numbers].sum()) * 0.5

    return transmission_dc, transmissionc

def calculate_transmission_values(simpl, cluster, opt, sector_opt, ll, planning_horizons):
    results_dict = {}

    for planning_horizon in planning_horizons:
        n = loaded_files[planning_horizon]

        cap_ac = pd.DataFrame(index=countries)
        cos_ac = pd.DataFrame(index=countries)
        cap_dc = pd.DataFrame(index=countries)
        cos_dc = pd.DataFrame(index=countries)
        
        for country in countries:
         filtered_ac = n.lines.bus0.str[:2] == country
         filtered_ac_r = n.lines.bus1.str[:2] == country
         combined_condition = filtered_ac | filtered_ac_r
         filtered_lines = n.lines[combined_condition]
         filtered_dc = (n.links.carrier == 'DC') & (n.links.bus0.str[:2] == country) & (~n.links.index.str.contains('reversed'))
         filtered_dc_r = (n.links.carrier == 'DC') & (n.links.bus1.str[:2] == country) & (~n.links.index.str.contains('reversed'))
         combined_condition_dc = filtered_dc | filtered_dc_r
         filtered_lines_dc = n.links[combined_condition_dc]
         transmission_ac, transmission = calculate_ac_transmission(filtered_lines, filtered_lines.index)
         transmission_dc, transmissionc = calculate_dc_transmission(filtered_lines_dc, filtered_lines_dc.index)
         
         
         cap_ac.loc[country, 'transmission_AC'] = transmission_ac
         cos_ac.loc[country, 'transmission_AC'] = transmission
         cap_dc.loc[country, 'transmission_DC'] = transmission_dc
         cos_dc.loc[country, 'transmission_DC'] = transmissionc


        # Create a dictionary for the planning horizon and store results
        results_dict[planning_horizon] = {
            'cap_ac': cap_ac,
            'cos_ac': cos_ac,
            'cap_dc': cap_dc,
            'cos_dc': cos_dc
        }

    return results_dict


def costs(countries, results):
    costs = {}
    fn = snakemake.input.costs
    options = pd.read_csv(fn, index_col=[0, 1]).sort_index()
    for country in countries:
      df=pd.read_csv(f"results/{study}/csvs/nodal_costs.csv", index_col=2)
      df = df.iloc[:, 2:]
      df = df.iloc[8:, :]
      df.index = df.index.str[:2]
      if country != 'EU':
       df = df[df.index == country]
      else:
       df = df
      df = df.rename(columns={'Unnamed: 3': 'tech', f'{cluster}': '2030',f'{cluster}.1': '2040',f'{cluster}.2': '2050'})
      df[['2030', '2040', '2050']] = df[['2030', '2040', '2050']].apply(pd.to_numeric, errors='coerce')
      df = df.groupby('tech').sum().reset_index()
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()

      cf = pd.read_csv("results/baseline/csvs/nodal_costs.csv", index_col=2)
      cf = cf.iloc[:, 2:]
      cf = cf.iloc[3:, :]
      cf.index = cf.index.str[:2]
      if country != 'EU':
       cf = cf[cf.index == country]
      else:
       cf=cf
      cf = cf.rename(columns={'Unnamed: 3': 'tech', f'{cluster}': '2020'})
      cf[['2020']] = cf[['2020']].apply(pd.to_numeric, errors='coerce')
      cf = cf.groupby('tech').sum().reset_index()
      cf['tech'] = cf['tech'].map(rename_techs_tyndp)
      cf = cf.groupby('tech').sum().reset_index()

      result_df = pd.merge(cf, df, on='tech', how='outer')
      result_df.fillna(0, inplace=True)
      mask = ~(result_df['tech'].isin(['load shedding']))
      result_df = result_df[mask]
      #calculationg gas costs for each country as in pypsa they are treated on EU level
      gas_val = pd.read_csv(f"results/{study}/country_csvs/natural_gas_imports_{country}.csv")
      gas_val = gas_val.set_index(gas_val.columns[0])
      gas_val = gas_val.iloc[2:]
      gas_val = gas_val.apply(pd.to_numeric, errors='coerce')
      gas_val.index = gas_val.index.astype(int)
      desired_years = [2020, 2030, 2040, 2050]
      if len(gas_val) >= len(desired_years):
        gas_val = gas_val.reset_index(drop=True).iloc[:len(desired_years)]
        gas_val.index = desired_years
      if country != 'EU':
        for year in planning_horizons:
         if year in gas_val.index:
          result_df.loc[result_df['tech'] == "gas", str(year)] = gas_val.loc[year] * options.loc[("gas", "fuel"), "value"] * 1e6
      else:
         result_df = result_df
      if not result_df.empty:
            years = ['2020', '2030', '2040', '2050']
            technologies = result_df['tech'].unique()
            
            costs[country] = result_df.set_index('tech').loc[technologies, years]

    for country in countries:

       for planning_horizon in planning_horizons:
        # Convert planning_horizon to string for column name
        planning_horizon_str = str(planning_horizon)

        # Check if the planning horizon key exists in the results dictionary
        if planning_horizon in results:
         if country != 'EU':
            cos_ac_df = results[planning_horizon]['cos_ac']
            cos_dc_df = results[planning_horizon]['cos_dc']
            ac_transmission_values = cos_ac_df.loc[country, 'transmission_AC']
            dc_transmission_values = cos_dc_df.loc[country, 'transmission_DC']

            # Assign values to existing columns for each year
            costs[country].loc['AC Transmission', planning_horizon_str] = ac_transmission_values
            costs[country].loc['DC Transmission', planning_horizon_str] = dc_transmission_values
      
       for country, dataframe in costs.items():
         # Specify the file path within the output directory
         file_path = f"results/{study}/country_csvs/{country}_costs.csv"
    
         # Save the DataFrame to a CSV file
         dataframe.to_csv(file_path, index=True)

         print(f"CSV file for {country} saved at: {file_path}")
        
    return costs

def Investment_costs(countries, results):
    investment_costs = {}
    for country in countries:
      df=pd.read_csv(f"results/{study}/csvs/nodal_costs.csv", index_col=2)
      df = df.iloc[:, 1:]
      df = df.iloc[6:, :]
      df.index = df.index.str[:2]
      if country != 'EU':
       df = df[df.index == country]
      else:
       df=df
      df = df.rename(columns={'Unnamed: 1': 'Costs','Unnamed: 3': 'tech', f'{cluster}': '2030',f'{cluster}.1': '2040',f'{cluster}.2': '2050'})
      df[['2030', '2040', '2050']] = df[['2030', '2040', '2050']].apply(pd.to_numeric, errors='coerce')
      df = df[df['Costs'] == 'capital']
      df = df.groupby('tech').sum().reset_index()
      df = df.drop(columns=['Costs'])
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()
      tech_mapping = {'oil': 'oil storage', 'gas': 'gas storage'}
      df['tech'] = df['tech'].replace(tech_mapping)
      condition = df[['2030', '2040', '2050']].eq(0).all(axis=1)
      df = df[~condition]

      cf = pd.read_csv("results/baseline/csvs/nodal_costs.csv", index_col=2)
      cf = cf.iloc[:, 1:]
      cf = cf.iloc[3:, :]
      cf.index = cf.index.str[:2]
      if country != 'EU':
       cf = cf[cf.index == country]
      else:
       cf=cf
      cf = cf.rename(columns={'Unnamed: 1': 'Costs','Unnamed: 3': 'tech', f'{cluster}': '2020'})
      cf[['2020']] = cf[['2020']].apply(pd.to_numeric, errors='coerce')
      cf = cf[cf['Costs'] == 'capital']
      cf = cf.groupby('tech').sum().reset_index()
      cf = cf.drop(columns=['Costs'])
      cf['tech'] = cf['tech'].map(rename_techs_tyndp)
      cf = cf.groupby('tech').sum().reset_index()
      tech_mapping = {'oil': 'oil storage', 'gas': 'gas storage'}
      cf['tech'] = cf['tech'].replace(tech_mapping)
      condition = cf[['2020']].eq(0).all(axis=1)
      cf = cf[~condition]
      
      result_df = pd.merge(cf, df, on='tech', how='outer')
      result_df.fillna(0, inplace=True)
      mask = ~(result_df['tech'].isin(['load shedding']))
      result_df = result_df[mask]
      if not result_df.empty:
            years = ['2020', '2030', '2040', '2050']
            technologies = result_df['tech'].unique()
            
            investment_costs[country] = result_df.set_index('tech').loc[technologies, years]

    for country in countries:

       for planning_horizon in planning_horizons:
        # Convert planning_horizon to string for column name
        planning_horizon_str = str(planning_horizon)

        # Check if the planning horizon key exists in the results dictionary
        if planning_horizon in results:
         if country != 'EU':
            cos_ac_df = results[planning_horizon]['cos_ac']
            cos_dc_df = results[planning_horizon]['cos_dc']
            ac_transmission_values = cos_ac_df.loc[country, 'transmission_AC']
            dc_transmission_values = cos_dc_df.loc[country, 'transmission_DC']

            # Assign values to existing columns for each year
            investment_costs[country].loc['AC Transmission', planning_horizon_str] = ac_transmission_values
            investment_costs[country].loc['DC Transmission', planning_horizon_str] = dc_transmission_values
      
       for country, dataframe in investment_costs.items():
         # Specify the file path within the output directory
         file_path = f"results/{study}/country_csvs/{country}_investment costs.csv"
    
         # Save the DataFrame to a CSV file
         dataframe.to_csv(file_path, index=True)

         print(f"CSV file for {country} saved at: {file_path}")
        
    return investment_costs 
   
def capacities(countries, results):
    capacities = {}
    for country in countries:
      df=pd.read_csv("results/baseline/csvs/nodal_capacities.csv", index_col=1)
      cf = pd.read_csv(f"results/{study}/csvs/nodal_capacities.csv", index_col=1)
      df = df.iloc[:, 1:]
      df = df.iloc[3:, :]
      df.index = df.index.str[:2]
      if country != 'EU':
       df = df[df.index == country]
      else:
       df=df
      df = df.rename(columns={'Unnamed: 2': 'tech', f'{cluster}': '2020'})
      columns_to_convert = ['2020']
      df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
      df = df.groupby('tech').sum().reset_index()
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()
      cf = cf.iloc[:, 1:]
      cf = cf.iloc[3:, :]
      cf.index = cf.index.str[:2]
      if country != 'EU':
       cf = cf[cf.index == country]
      else:
       cf=cf
      cf = cf.rename(columns={'Unnamed: 2': 'tech', f'{cluster}': '2030',f'{cluster}.1': '2040',f'{cluster}.2': '2050'})
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

    for country in countries:

       for planning_horizon in planning_horizons:
        # Convert planning_horizon to string for column name
        planning_horizon_str = str(planning_horizon)

        # Check if the planning horizon key exists in the results dictionary
        if planning_horizon in results:
         if country != 'EU':
            cap_ac_df = results[planning_horizon]['cap_ac']
            cap_dc_df = results[planning_horizon]['cap_dc']
            ac_transmission_values = cap_ac_df.loc[country, 'transmission_AC']
            dc_transmission_values = cap_dc_df.loc[country, 'transmission_DC']

            # Assign values to existing columns for each year
            capacities[country].loc['AC Transmission lines', planning_horizon_str] = ac_transmission_values
            capacities[country].loc['DC Transmission lines', planning_horizon_str] = dc_transmission_values
        
       for country, dataframe in capacities.items():
        # Specify the file path where you want to save the CSV file
        file_path = f"results/{study}/country_csvs/{country}_capacities.csv"
    
         # Save the DataFrame to a CSV file
        dataframe.to_csv(file_path, index=True)

        print(f"CSV file for {country} saved at: {file_path}")  

    return capacities

def storage_capacities(countries):
    s_capacities = {}
    for country in countries:
      df=pd.read_csv("results/baseline/csvs/nodal_capacities.csv", index_col=1)
      cf = pd.read_csv(f"results/{study}/csvs/nodal_capacities.csv", index_col=1)
      df = df[df['cluster'] == 'stores']
      df = df.iloc[2:, :]
      df.index = df.index.str[:2]
      if country != 'EU':
       df = df[df.index == country]
      else:
       df=df
      df = df.rename(columns={'Unnamed: 2': 'tech', f'{cluster}': '2020'})
      columns_to_convert = ['2020']
      df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
      df = df.groupby('tech').sum().reset_index()
      cf = cf[cf['cluster'] == 'stores']
      cf = cf.iloc[1:, :]
      cf.index = cf.index.str[:2]
      if country != 'EU':
       cf = cf[cf.index == country]
      else:
       cf=cf
      cf = cf.rename(columns={'Unnamed: 2': 'tech',  f'{cluster}': '2030',f'{cluster}.1': '2040',f'{cluster}.2': '2050'})
      columns_to_convert = ['2030', '2040', '2050']
      cf[columns_to_convert] = cf[columns_to_convert].apply(pd.to_numeric, errors='coerce')
      cf = cf.groupby('tech').sum().reset_index()
      result_df = pd.merge(df, cf, on='tech', how='outer')
      result_df.fillna(0, inplace=True)
      result_df['tech'] = result_df['tech'].replace({'urban central water tanks': 'Thermal Energy storage', 'battery':'Grid-scale battery', 'Li ion':'V2G'})
      if not result_df.empty:
            years = ['2020', '2030', '2040', '2050']
            technologies = result_df['tech'].unique()

            s_capacities[country] = result_df.set_index('tech').loc[technologies, years]
            for country, dataframe in s_capacities.items():
             # Specify the file path where you want to save the CSV file
             file_path = f"results/{study}/country_csvs/{country}_storage_capacities.csv"
         
              # Save the DataFrame to a CSV file
             dataframe.to_csv(file_path, index=True)

             print(f"CSV file for {country} saved at: {file_path}") 

    return s_capacities

def plot_demands(countries):
    colors = config["plotting"]["tech_colors"] 
    colors["methane"] = "orange"
    colors["Non-energy demand"] = "black" 
    colors["hydrogen for industry"] = "cornflowerblue"
    colors["agriculture electricity"] = "royalblue"
    colors["agriculture and industry heat"] = "lightsteelblue"
    colors["agriculture oil"] = "darkorange"
    colors["electricity demand of residential and tertairy"] = "navajowhite"
    colors["gas for Industry"] = "forestgreen"
    colors["electricity for Industry"] = "limegreen"
    colors["aviation oil demand"] = "black"
    colors["land transport EV"] = "lightcoral"
    colors["land transport hydrogen demand"] = "mediumpurple"
    colors["oil to transport demand"] = "thistle"
    colors["low-temperature heat for industry"] = "sienna"
    colors["naphtha for non-energy"] = "sandybrown"
    colors["shipping methanol"] = "lawngreen"
    colors["shipping hydrogen"] = "gold"
    colors["shipping oil"] = "turquoise"
    colors["solid biomass for Industry"] = "paleturquoise"
    colors["Residential and tertiary DH demand"] = "gray"
    colors["Residential and tertiary heat demand"] = "pink"
    colors["electricity demand for rail network"] = "blue"
    colors["H2 for non-energy"] = "violet" 
    
    mapping = {
        "hydrogen for industry": "hydrogen",
        "H2 for non-energy": "Non-energy demand",
        "shipping hydrogen": "hydrogen",
        "shipping oil": "oil",
        "agriculture electricity": "electricity",
        "agriculture heat": "heat",
        "agriculture oil": "oil",
        "electricity demand of residential and tertairy": "electricity",
        "gas for Industry": "methane",
        "electricity for Industry": "electricity",
        "aviation oil demand": "oil",
        "land transport EV": "electricity",
        "land transport hydrogen demand": "hydrogen",
        "oil to transport demand": "oil",
        "low-temperature heat for industry": "heat",
        "naphtha for non-energy": "Non-energy demand",
        "electricity demand for rail network": "electricity",
        "Residential and tertiary DH demand": "heat",
        "Residential and tertiary heat demand": "heat",
        "solid biomass for Industry": "solid biomass",
        "NH3":"hydrogen",
    }
    mapping_eu = {
            "preshydcfind": "hydrogen for industry",
            "preshydcfneind": "H2 for non-energy",
            "preshydwati": "shipping hydrogen",
            "preslqfcffrewati": "shipping oil",
            "preselccfagr": "agriculture electricity",
            "presvapcfagr": "agriculture heat",
            "prespetcfagr": "agriculture oil",
            "preselccfres": "electricity demand of residential and tertairy",
            "presgazcfind": "gas for Industry",
            "presgazcfindd": "gas for Industry",
            "preselccfind": "electricity for Industry",
            "preslqfcfavi": "aviation oil demand",
            "preselccftra": "land transport EV",
            "preshydcftra": "land transport hydrogen demand",
            "preslqfcftra": "oil to transport demand",
            "presvapcfind": "low-temperature heat for industry",
            "prespetcfneind": "naphtha for non-energy",
            "preserail": "electricity demand for rail network",
            "presvapcfdhs": "Residential and tertiary DH demand",
            "demandheat": "Residential and tertiary heat demand",
            "demandheata": "Residential and tertiary heat demand",
            "demandheatb": "Residential and tertiary heat demand",
            "demandheats": "Residential and tertiary heat demand",
            "presenccfind": "solid biomass for Industry",
            "presenccfindd": "solid biomass for Industry",
            "preammind":"NH3",
    }
    
    for country in countries:
        data = pd.read_excel(f"results/{study}/sepia/inputs{country}.xlsx", index_col=0)
        if country != 'EU':
         columns_to_drop = ['source', 'target']
         data = data.drop(columns=columns_to_drop)
         data = data.groupby(data.index).sum()
        else:
         columns_to_drop = ['source', 'target']
         data = data.drop(columns=columns_to_drop)
         data.rename(index=mapping_eu, inplace=True)
         data = data.groupby(data.index).sum()

        # Apply your mapping to the data
        data = data[data.index.isin(mapping.keys())]
        data.index = pd.MultiIndex.from_tuples([(mapping[i], i) for i in data.index])
        data = data.reset_index()
        data.rename(columns={'level_0': 'Demand'}, inplace=True)
        data.rename(columns={'level_1': 'Sectors'}, inplace=True)

        
        melted_data = pd.melt(data, id_vars=['Demand', 'Sectors'], var_name='year', value_name='value')
        melted_data['color'] = melted_data['Sectors'].map(colors)

        # Use plotly express to create a stacked bar plot
        fig = px.bar(
        melted_data,
        x='year',
        y='value',
        color='Sectors',
        color_discrete_map=dict(zip(melted_data['Sectors'].unique(), melted_data['color'].unique())),
        facet_col='Demand',
        labels={'year': '', 'value': 'Final energy and non-energy demand [TWh/a]'}
        )
        logo['y']=1.021
        fig.add_layout_image(logo)
        # Show the plot
        html_filename = f"{country}_sectoral_demands.html"
        output_folder = f'results/{study}/htmls/raw_html' # Set your desired output folder
        os.makedirs(output_folder, exist_ok=True)
        html_filepath = os.path.join(output_folder, html_filename)
        fig.write_html(html_filepath)
        file_path = f"results/{study}/country_csvs/{country}_sectordemands.csv"
        data.to_csv(file_path, index=True)
        
        
def plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons,start,stop,title):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["fossil oil and gas"] = colors["oil"]
    colors["hydrogen storage"] = colors["H2 Store"]
    colors["load shedding"] = 'black'
    colors["CHP"] = 'darkred'
    colors["load"] = 'black'
    colors["Imports_Exports"] = "dimgray"
    colors["EV charger"] = colors["V2G"]
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        n = loaded_files[planning_horizon]

        assign_location(n)
        assign_carriers(n)
        carrier = 'AC'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        if country != 'EU':
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
        else:
         for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]]
                        .groupby(c.df.carrier, axis=1)
                        .sum(),
                    ),
                    axis=1,
                )
        if country != 'EU':
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
        else:
         for c in n.iterate_components(n.one_port_components):
            comps = c.df.index[c.df.bus.isin(busesn)]
            supplyn = pd.concat(
                (
                    supplyn,
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            ) 

        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
        filtered_ac_lines = n.lines.bus0.str[:2] == country
        ac_lines = n.lines_t.p0.filter(items=filtered_ac_lines[filtered_ac_lines == True].index).sum(axis=1)
        filtered_ac_lines_r = n.lines.bus1.str[:2] == country
        ac_lines_r = n.lines_t.p1.filter(items=filtered_ac_lines_r[filtered_ac_lines_r == True].index).sum(axis=1)
        filtered_dc_lines = (n.links.carrier == 'DC') & (n.links.bus0.str[:2] == country)
        dc_lines = n.links_t.p0.filter(items=filtered_dc_lines[filtered_dc_lines == True].index).sum(axis=1)
        filtered_dc_lines_r = (n.links.carrier == 'DC') & (n.links.bus1.str[:2] == country)
        dc_lines_r = n.links_t.p1.filter(items=filtered_dc_lines_r[filtered_dc_lines_r == True].index).sum(axis=1)
        merged_series = pd.concat([ac_lines,ac_lines_r, dc_lines, dc_lines_r], axis=1)
        imp_exp = merged_series.sum(axis=1)
        imp_exp = imp_exp.rename('Imports_Exports')
        imp_exp=-imp_exp
        supplyn['Imports_Exports'] = imp_exp

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)



        threshold = 0.1

        to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]

        if len(to_dropn) != 0:
            logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
            supplyn.drop(columns=to_dropn, inplace=True)

        supplyn.index.name = None

        supplyn = supplyn / 1e3

        
        supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()

        if country != 'EU':
         c_solarn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="solar", axis=1
         ).filter(like=country).sum(axis=1) / 1e3
         c_onwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="onwind", axis=1
         ).filter(like=country).sum(axis=1) / 1e3
         c_offwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="offwind", axis=1
         ).filter(like=country).sum(axis=1) / 1e3
        else:
         c_solarn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="solar", axis=1
         ).sum(axis=1) / 1e3
         c_onwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="onwind", axis=1
         ).sum(axis=1) / 1e3
         c_offwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="offwind", axis=1
         ).sum(axis=1) / 1e3
        supplyn = supplyn.T
        if "solar" in supplyn.index:
         supplyn.loc["solar"] = supplyn.loc["solar"] + c_solarn
         supplyn.loc["solar curtailment"] = -abs(c_solarn)
        if "onshore wind" in supplyn.index:
         supplyn.loc["onshore wind"] = supplyn.loc["onshore wind"] + c_onwindn
         supplyn.loc["onshore curtailment"] = -abs(c_onwindn)
        if "offshore wind" in supplyn.index:
         supplyn.loc["offshore wind"] = supplyn.loc["offshore wind"] + c_offwindn
         supplyn.loc["offshore curtailment"] = -abs(c_offwindn)
        if "H2 pipeline" in supplyn.index:
           supplyn = supplyn.drop('H2 pipeline')
        supplyn = supplyn.T
        if "V2G" in n.carriers.index:
         if country != 'EU':
             v2g = n.links_t.p1.filter(like=country).filter(like="V2G").sum(axis=1)
             v2g = v2g.to_frame()
             v2g = v2g.rename(columns={v2g.columns[0]: 'V2G'})
             v2g = v2g/1e3
             supplyn['electricity distribution grid'] = supplyn['electricity distribution grid'] + v2g['V2G']
             supplyn['V2G'] = v2g['V2G'].abs()
         else:
             v2g = n.links_t.p1.filter(like="V2G").sum(axis=1)
             v2g = v2g.to_frame()
             v2g = v2g.rename(columns={v2g.columns[0]: 'V2G'})
             v2g = v2g/1e3
             supplyn['electricity distribution grid'] = supplyn['electricity distribution grid'] + v2g['V2G']
             supplyn['V2G'] = v2g['V2G'].abs()
        
        positive_supplyn = supplyn[supplyn >= 0].fillna(0)
        negative_supplyn = supplyn[supplyn < 0].fillna(0)
        
        positive_supplyn = positive_supplyn.loc[start:stop]
        negative_supplyn = negative_supplyn.loc[start:stop]
        positive_supplyn = positive_supplyn.applymap(lambda x: x if x >= 0.1 else 0)
        negative_supplyn = negative_supplyn.applymap(lambda x: x if x <= -0.1 else 0)
        positive_supplyn = positive_supplyn.loc[:, (positive_supplyn > 0).any()]
        negative_supplyn = negative_supplyn.loc[:, (negative_supplyn < 0).any()]
        

        
        fig = go.Figure()

        for col in positive_supplyn.columns:
            fig.add_trace(go.Scatter(
                x=positive_supplyn.index,
                y=positive_supplyn[col],
                mode='lines',
                line=dict(color=colors.get(col, 'black')),
                stackgroup='positive',
                showlegend=False,
                hovertemplate='%{y:.2f}',
                name=col
            ))

        for col in negative_supplyn.columns:
            fig.add_trace(go.Scatter(
                x=negative_supplyn.index,
                y=negative_supplyn[col],
                mode='lines',
                line=dict(color=colors.get(col, 'black')),
                stackgroup='negative',
                showlegend=False,
                hovertemplate='%{y:.2f}',
                name=col
            ))
         
        # Collect unique column names from both positive_supplyn and negative_supplyn
        unique_columns = set(positive_supplyn.columns).union(set(negative_supplyn.columns))
        # Add a dummy trace for each unique column name to the legend
        for col in unique_columns:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=colors.get(col, 'black'), width=4),
                legendgroup='supply',
                showlegend=True,
                name=col
            ))
        # Update layout to customize axes, title, etc.
        fig.update_layout(
         xaxis=dict(title='Time', tickformat="%m-%d"),
         yaxis=dict(title='Power [GW]',),
         title=title + " - " + country + ' - ' + str(planning_horizon),
         width=1200, height=600,
         hovermode='x',)
    
        fig.add_layout_image(logo)
            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", fig))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))
        
     html_filename = title + " - " + country + '.html'
     output_folder = f'results/{study}/htmls/raw_html' # Set your desired output folder
     os.makedirs(output_folder, exist_ok=True)
     html_filepath = os.path.join(output_folder, html_filename)
     tabs.save(html_filepath)


def plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons,start,stop,title):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["agriculture heat"] = "grey"
    colors["CHP"] = "orange"
    colors["centralised electric boiler"] = "#6488ea"
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        n = loaded_files[planning_horizon]

        assign_location(n)
        assign_carriers(n)
        carrier = 'heat'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        if country != 'EU':
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
        else:
         for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]]
                        .groupby(c.df.carrier, axis=1)
                        .sum(),
                    ),
                    axis=1,
                )
        if country != 'EU':
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
        else:
         for c in n.iterate_components(n.one_port_components):
            comps = c.df.index[c.df.bus.isin(busesn)]
            supplyn = pd.concat(
                (
                    supplyn,
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )
        supplyn = supplyn.rename(columns={"urban central resistive heater": "centralised electric boiler"})
        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)


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

        positive_supplyn = positive_supplyn.loc[start:stop]
        negative_supplyn = negative_supplyn.loc[start:stop]
        positive_supplyn = positive_supplyn.applymap(lambda x: x if x >= 0.1 else 0)
        negative_supplyn = negative_supplyn.applymap(lambda x: x if x <= -0.1 else 0)
        positive_supplyn = positive_supplyn.loc[:, (positive_supplyn > 0).any()]
        negative_supplyn = negative_supplyn.loc[:, (negative_supplyn < 0).any()]
        

        
        fig = go.Figure()

        for col in positive_supplyn.columns:
            fig.add_trace(go.Scatter(
                x=positive_supplyn.index,
                y=positive_supplyn[col],
                mode='lines',
                line=dict(color=colors.get(col, 'black')),
                stackgroup='positive',
                showlegend=False,
                hovertemplate='%{y:.2f}',
                name=col
            ))

        for col in negative_supplyn.columns:
            fig.add_trace(go.Scatter(
                x=negative_supplyn.index,
                y=negative_supplyn[col],
                mode='lines',
                line=dict(color=colors.get(col, 'black')),
                stackgroup='negative',
                showlegend=False,
                hovertemplate='%{y:.2f}',
                name=col
            ))
         
        # Collect unique column names from both positive_supplyn and negative_supplyn
        unique_columns = set(positive_supplyn.columns).union(set(negative_supplyn.columns))
        # Add a dummy trace for each unique column name to the legend
        for col in unique_columns:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=colors.get(col, 'black'), width=4),
                legendgroup='supply',
                showlegend=True,
                name=col
            ))
        # Update layout to customize axes, title, etc.
        fig.update_layout(
         xaxis=dict(title='Time', tickformat="%m-%d"),
         yaxis=dict(title='Heat [GW]',),
         title=title + " - " + country + ' - ' + str(planning_horizon),
         width=1200, height=600,
         hovermode='x',)
    
        fig.add_layout_image(logo)
            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", fig))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))


        # Save the tabs as an HTML file
     html_filename = title + " - " + country + '.html'
     output_folder = f'results/{study}/htmls/raw_html'  # Set your desired output folder
     os.makedirs(output_folder, exist_ok=True)
     html_filepath = os.path.join(output_folder, html_filename)
     tabs.save(html_filepath)

def plot_map(
    network,country,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=1.7e10,
    transmission=True,
    with_legend=True,
):
    tech_colors = snakemake.params.plotting["tech_colors"]
    n = network.copy()
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    costs = costs.T.groupby(costs.columns).sum().T

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()
    

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 750
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    title = "added grid"

    if ll == "v1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 1e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid capacity"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    line_widths = line_widths.replace(line_lower_threshold, 0)
    link_widths = link_widths.replace(line_lower_threshold, 0)
    
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})
    eu_location = config["plotting"].get("eu_node_location", dict(x=-50, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]
    
    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        # **map_opts,
    )

    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/year" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.001, 0.85),
        labelspacing=2.25,
        frameon=False,
        handletextpad=1,
        fontsize=15,
        title="Investment Costs",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5, 1]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]
    # if planning_horizons == 2020:
    #     value = "current grid"
    # else:
    #     value = "total grid capacity"
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.001, 0.45),
        frameon=False,
        labelspacing=1,
        handletextpad=1,
        fontsize=15,
        title=title,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="black"), legend_kw=legend_kw
    )

    legend_kw = dict(
        bbox_to_anchor=(1.35, 1),
        frameon=False,
        fontsize=15
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC line"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )
        
    if country != 'EU':
     lines = pd.DataFrame(n.lines)
     links = pd.DataFrame(n.links)

     # Filter rows for lines and links
     links=n.links[n.links['carrier'] == 'DC']
     lines = n.lines[(n.lines['bus0'].str.contains(country)) | (n.lines['bus1'].str.contains(country))]
     links = links[(links['bus0'].str.contains(country)) | (links['bus1'].str.contains(country))]
     links = links[~links.index.str.contains("reversed")]

     # Create 'BusCombination' column
     lines['BusCombination'] = (lines['bus0'].str.extract(r'([A-Z]+)').fillna('') +
                             ' - ' +
                             lines['bus1'].str.extract(r'([A-Z]+)').fillna(''))
     links['BusCombination'] = (links['bus0'].str.extract(r'([A-Z]+)').fillna('') +
                             ' - ' +
                             links['bus1'].str.extract(r'([A-Z]+)').fillna(''))

     # Collect data for the combined table
     table_data = pd.concat([
     pd.DataFrame({
         'Lines': lines['BusCombination'],
         'Capacity [GW]': lines['s_nom_opt'] / 1000,
         'Type': 'AC'
     }),
     pd.DataFrame({
         'Lines': links['BusCombination'],
         'Capacity [GW]': links['p_nom_opt'] / 1000,
         'Type': 'DC'
     })
     ], ignore_index=True)
     table_data['Capacity [GW]'] = table_data['Capacity [GW]'].round(1)
     table_data = table_data[table_data['Capacity [GW]'] != 0]

     # Plot the table on the same subplot as the map
     left, bottom, width, height = 0.05, 0.05, 0.4, 0.12
     tab = table(ax, table_data, bbox=[left, bottom, width, height],fontsize=20)
     type_col_index = table_data.columns.get_loc('Type')

     # Set text color based on 'Type' value
     for key, cell in tab.get_celld().items():
        if key[0] != 0:
            type_value = table_data.iloc[key[0] - 1, type_col_index]
            for idx, cell_text in enumerate(table_data.iloc[key[0] - 1]):
                cell.set_text_props(color='rosybrown' if type_value == 'AC' else 'darkseagreen', weight='bold' if idx == type_col_index else 'normal')

    # fig.tight_layout()
    logo = snakemake.input.logo
    img = plt.imread(logo)
    imagebox = OffsetImage(img, zoom=0.25)
    ab = AnnotationBbox(imagebox, xy=(0.5, 1.05), xycoords='axes fraction', boxcoords="axes fraction", frameon=False)
    ax.add_artist(ab)
    return fig

def create_map_plots(planning_horizons, country):
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    /* ... (existing styles) ... */
    </style>
    </head>
    <body>
    <div class="tab">
    """
    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
         n = loaded_files[planning_horizon]

        # Plot the map and get the figure
         fig = plot_map(
             n,country,
             components=["generators", "links", "stores", "storage_units"],
             bus_size_factor=90e9,
             transmission=True,
         )
         plt.rcParams['legend.title_fontsize'] = '20'
        # Save the map plot as an image
         output_image_path = f"results/{study}/htmls/raw_html/map_plot_{planning_horizon}_{country}.png"
         fig.savefig(output_image_path, bbox_inches="tight")
         plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
         with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Add tab content for each planning horizon with embedded image data
         html_content += f"""
        <button class="map-tablinks{' active' if i == 0 else ''}" onclick="openMapTab(event, 'map_{planning_horizon}_{i + 1}')">{planning_horizon}</button>
        """

    html_content += """
    </div>
    """

    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
         n = loaded_files[planning_horizon]
         fig = plot_map(
             n,country,
             components=["generators", "links", "stores", "storage_units"],
             bus_size_factor=90e9,
             transmission=True,
         )
         plt.rcParams['legend.title_fontsize'] = '20'
        # Save the map plot as an image
         output_image_path = f"results/{study}/htmls/raw_html/map_plot_{planning_horizon}_{country}.png"
         fig.savefig(output_image_path)
         plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
         with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Add tab content for each planning horizon with embedded image data
         html_content += f"""
        <div id="map_{planning_horizon}_{i + 1}" class="map-tabcontent" style="display: {'block' if i == 0 else 'none'};">
            <h2>Map Plot - {planning_horizon}</h2>
            <img src="data:image/png;base64,{encoded_image}" alt="Map Plot" width="1200" height="800">
        </div>
        """


    # Add JavaScript for tab functionality
    html_content += """
    <script>
    function openMapTab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("map-tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }
      tablinks = document.getElementsByClassName("map-tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    </script>
    </body>
    </html>
    """

    # Save the entire HTML content to a single file
    
    output_combined_html_path = f"results/{study}/htmls/raw_html/map_plots_{country}.html"
    with open(output_combined_html_path, "w") as html_file:
        html_file.write(html_content)
def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df["index_orig"] = df.index
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


def plot_h2_map(network):
    # if "H2 pipeline" not in n.links.carrier.unique():
    #     return
    n = network.copy()
    assign_location(n)
    h2_storage = n.stores.query("carrier == 'H2'")
    regions = gpd.read_file(f"resources/{study}/regions_onshore_elec_s_{cluster}.geojson").set_index("name")
    regions["H2"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e6)
    )  # TWh
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 1e5
    linewidth_factor = 7e3
    # MW below which not drawn
    line_lower_threshold = 750

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]

    if snakemake.params.foresight == "myopic":
        # sum capacitiy for pipelines from different investment periods
        h2_new = group_pipes(h2_new)

        if not h2_retro.empty:
            h2_retro = (
                group_pipes(h2_retro, drop_direction=True)
                .reindex(h2_new.index)
                .fillna(0)
            )

    if not h2_retro.empty:
        if snakemake.params.foresight != "myopic":
            positive_order = h2_retro.bus0 < h2_retro.bus1
            h2_retro_p = h2_retro[positive_order]
            swap_buses = {"bus0": "bus1", "bus1": "bus0"}
            h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
            h2_retro = pd.concat([h2_retro_p, h2_retro_n])

            h2_retro["index_orig"] = h2_retro.index
            h2_retro.index = h2_retro.apply(
                lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
                axis=1,
            )

        retro_w_new_i = h2_retro.index.intersection(h2_new.index)
        h2_retro_w_new = h2_retro.loc[retro_w_new_i]

        retro_wo_new_i = h2_retro.index.difference(h2_new.index)
        h2_retro_wo_new = h2_retro.loc[retro_wo_new_i]
        h2_retro_wo_new.index = h2_retro_wo_new.index_orig

        to_concat = [h2_new, h2_retro_w_new, h2_retro_wo_new]
        h2_total = pd.concat(to_concat).p_nom_opt.groupby(level=0).sum()

    else:
        h2_total = h2_new.p_nom_opt

    link_widths_total = h2_total / linewidth_factor

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).first()
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")
    regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"

    bus_colors = {"H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#805394"}
    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=6,
        vmin=0,
        legend_kwds={
            "label": "Hydrogen Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 1),
        labelspacing=1.2,
        handletextpad=0,
        frameon=False,
        fontsize=15,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.9),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        fontsize=15,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="black"),
        legend_kw=legend_kw,
    )

    colors = [bus_colors[c] for c in carriers] + [color_h2_pipe, color_retrofit]
    labels = carriers + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.2, 1),
        ncol=1,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)


    ax.set_facecolor("white")
    logo = snakemake.input.logo
    img = plt.imread(logo)
    imagebox = OffsetImage(img, zoom=0.25)
    ab = AnnotationBbox(imagebox, xy=(0.5, 1.05), xycoords='axes fraction', boxcoords="axes fraction", frameon=False)
    ax.add_artist(ab)
    return fig

def plot_ch4_map(network):
    # if "gas pipeline" not in n.links.carrier.unique():
    #     return
    n = network.copy()
    assign_location(n)

    bus_size_factor = 10e8
    linewidth_factor = 0.5e4
    # MW below which not drawn
    line_lower_threshold = 1e3

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    fossil_gas_i = n.generators[n.generators.carrier == "gas"].index
    fossil_gas = (
        n.generators_t.p.loc[:, fossil_gas_i]
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
        .groupby(n.generators.loc[fossil_gas_i, "bus"])
        .sum()
        / bus_size_factor
    )
    fossil_gas.rename(index=lambda x: x.replace(" gas", ""), inplace=True)
    fossil_gas = fossil_gas.reindex(n.buses.index).fillna(0)
    # make a fake MultiIndex so that area is correct for legend
    fossil_gas.index = pd.MultiIndex.from_product([fossil_gas.index, ["fossil gas"]])

    methanation_i = n.links.query("carrier == 'Sabatier'").index
    methanation = (
        abs(
            n.links_t.p1.loc[:, methanation_i].mul(
                n.snapshot_weightings.generators, axis=0
            )
        )
        .sum()
        .groupby(n.links.loc[methanation_i, "bus1"])
        .sum()
        / bus_size_factor
    )
    methanation = (
        methanation.groupby(methanation.index)
        .sum()
        .rename(index=lambda x: x.replace(" gas", ""))
    )
    # make a fake MultiIndex so that area is correct for legend
    methanation.index = pd.MultiIndex.from_product([methanation.index, ["methanation"]])

    biogas_i = n.stores[n.stores.carrier == "biogas"].index
    biogas = (
        n.stores_t.p.loc[:, biogas_i]
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
        .groupby(n.stores.loc[biogas_i, "bus"])
        .sum()
        / bus_size_factor
    )
    biogas = (
        biogas.groupby(biogas.index)
        .sum()
        .rename(index=lambda x: x.replace(" biogas", ""))
    )
    # make a fake MultiIndex so that area is correct for legend
    biogas.index = pd.MultiIndex.from_product([biogas.index, ["biogas"]])

    bus_sizes = pd.concat([fossil_gas, methanation, biogas])
    bus_sizes.sort_index(inplace=True)

    to_remove = n.links.index[~n.links.carrier.str.contains("gas pipeline")]
    n.links.drop(to_remove, inplace=True)

    link_widths_rem = n.links.p_nom_opt / linewidth_factor
    link_widths_rem[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_orig = n.links.p_nom / linewidth_factor
    link_widths_orig[n.links.p_nom < line_lower_threshold] = 0.0

    max_usage = n.links_t.p0[n.links.index].abs().max(axis=0)
    link_widths_used = max_usage / linewidth_factor
    link_widths_used[max_usage < line_lower_threshold] = 0.0

    tech_colors = snakemake.params.plotting["tech_colors"]

    pipe_colors = {
        "gas pipeline": "#f08080",
        "gas pipeline new": "#c46868",
        "gas pipeline retrofitted to H2": "#499a9c",
        "gas pipeline (available)": "#e8d1d1",
    }

    link_color_used = n.links.carrier.map(pipe_colors)

    n.links.bus0 = n.links.bus0.str.replace(" gas", "")
    n.links.bus1 = n.links.bus1.str.replace(" gas", "")

    bus_colors = {
        "fossil gas": tech_colors["fossil gas"],
        "methanation": tech_colors["methanation"],
        "biogas": "seagreen",
    }

    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})
    eu_location = config["plotting"].get("eu_node_location", dict(x=-50, y=46))
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]
    n.plot(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=pipe_colors["gas pipeline retrofitted to H2"],
        link_widths=link_widths_orig,
        branch_components=["Link"],
        ax=ax,
        # **map_opts,
    )

    n.plot(
        ax=ax,
        bus_sizes=0.0,
        link_colors=pipe_colors["gas pipeline (available)"],
        link_widths=link_widths_rem,
        branch_components=["Link"],
        color_geomap=False,
        # boundaries=map_opts["boundaries"],
    )

    n.plot(
        ax=ax,
        bus_sizes=0.0,
        link_colors=link_color_used,
        link_widths=link_widths_used,
        branch_components=["Link"],
        color_geomap=False,
        # boundaries=map_opts["boundaries"],
    )

    sizes = [100, 10]
    labels = [f"{s} TWh" for s in sizes]
    sizes = [s / bus_size_factor * 1e6 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.8),
        labelspacing=0.8,
        frameon=False,
        handletextpad=1,
        fontsize=15,
        title="gas sources",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.6),
        frameon=False,
        labelspacing=0.8,
        fontsize=15,
        handletextpad=1,
        title="gas pipeline",
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="black"),
        legend_kw=legend_kw,
    )

    colors = list(pipe_colors.values()) + list(bus_colors.values())
    labels = list(pipe_colors.keys()) + list(bus_colors.keys())

    # legend on the side
    # legend_kw = dict(
    #     bbox_to_anchor=(1.47, 1.04),
    #     frameon=False,
    # )

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(1, 0.9),
        ncol=1,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw=legend_kw,
    )
    logo = snakemake.input.logo
    img = plt.imread(logo)
    imagebox = OffsetImage(img, zoom=0.25)
    ab = AnnotationBbox(imagebox, xy=(0.5, 1.05), xycoords='axes fraction', boxcoords="axes fraction", frameon=False)
    ax.add_artist(ab)
    return fig



def create_H2_map_plots(planning_horizons):
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    /* ... (existing styles) ... */
    </style>
    </head>
    <body>
    <div class="tab">
    """
    planning_horizons = [2030, 2040, 2050]
    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
        n = loaded_files[planning_horizon]

        # Plot the H2 map and get the figure
        fig = plot_h2_map(network=n)
        plt.rcParams['legend.title_fontsize'] = '20'

        # Save the H2 map plot as an image
        output_image_path = f"results/{study}/htmls/raw_html/map_h2_plot_{planning_horizon}.png"
        fig.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Add tab content for each planning horizon with embedded image data
        html_content += f"""
       <button class="h2-tablinks{' active' if i == 0 else ''}" onclick="openH2Tab(event, 'h2_{planning_horizon}_{i + 1}')">{planning_horizon}</button>
       """

    html_content += """
    </div>
    """

    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
        n = loaded_files[planning_horizon]
        fig = plot_h2_map(network=n)
        plt.rcParams['legend.title_fontsize'] = '20'

        # Save the H2 map plot as an image
        output_image_path = f"results/{study}/htmls/raw_html/map_h2_plot_{planning_horizon}.png"
        fig.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        html_content += f"""
        <div id="h2_{planning_horizon}_{i + 1}" class="h2-tabcontent" style="display: {'block' if i == 0 else 'none'};">
            <h2>H2 Map Plot - {planning_horizon}</h2>
            <img src="data:image/png;base64,{encoded_image}" alt="H2 Map Plot" width="1200" height="800">
        </div>
        """
    html_content += """
    <script>
    function openH2Tab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("h2-tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }
      tablinks = document.getElementsByClassName("h2-tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    </script>
    </body>
    </html>
    """

    # Save the entire HTML content to a single file
    output_combined_html_path = f"results/{study}/htmls/raw_html/map_h2_plots.html"
    with open(output_combined_html_path, "w") as html_file:
        html_file.write(html_content)

def create_gas_map_plots(planning_horizons):
   
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    /* ... (existing styles) ... */
    </style>
    </head>
    <body>
    <div class="tab">
    """
    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
        n = loaded_files[planning_horizon]

        # Plot the H2 map and get the figure
        fig = plot_ch4_map(network=n)
        plt.rcParams['legend.title_fontsize'] = '20'

        # Save the H2 map plot as an image
        output_image_path = f"results/{study}/htmls/raw_html/map_ch4_plot_{planning_horizon}.png"
        fig.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Add tab content for each planning horizon with embedded image data
        html_content += f"""
        <button class="gas-tablinks{' active' if i == 0 else ''}" onclick="openGasTab(event, 'gas_{planning_horizon}_{i + 1}')">{planning_horizon}</button>
        """

    html_content += """
    </div>
    """
    
    for i, planning_horizon in enumerate(planning_horizons):
        # Load network for the current planning horizon
        n = loaded_files[planning_horizon]
        fig = plot_ch4_map(network=n)
        plt.rcParams['legend.title_fontsize'] = '20'

        # Save the H2 map plot as an image
        output_image_path = f"results/{study}/htmls/raw_html/map_ch4_plot_{planning_horizon}.png"
        fig.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to avoid displaying it in the notebook

        # Encode the image as base64
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Add tab content for each planning horizon with embedded image data
        html_content += f"""
        <div id="gas_{planning_horizon}_{i + 1}" class="gas-tabcontent" style="display: {'block' if i == 0 else 'none'};">
            <h2>Gas Map Plot - {planning_horizon}</h2>
            <img src="data:image/png;base64,{encoded_image}" alt="Gas Map Plot" width="1200" height="800">
        </div>
        """

    # Add JavaScript for tab functionality
    html_content += """
    <script>
    function openGasTab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("gas-tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }
      tablinks = document.getElementsByClassName("gas-tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    </script>
    </body>
    </html>
    """

    # Save the entire HTML content to a single file
    output_combined_html_path = f"results/{study}/htmls/raw_html/map_ch4_plots.html"
    with open(output_combined_html_path, "w") as html_file:
        html_file.write(html_content)
        
def create_bar_chart(costs, country,  unit='Billion Euros/year'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"

    title = f"{country} - Total Annual Costs"
    df = costs[country]
    df = df.rename_axis(unit)
    df = df.reset_index()
    df.index = df.index.astype(str)

    # Create a bar chart using Plotly
    fig = go.Figure()
    df_transposed = df.set_index(unit).T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    # Save the HTML file for each country
    # output_file_path = os.path.join(output_folder, f"{country}_bar_chart.html")
    # fig.write_html(output_file_path)

    return fig

def create_investment_costs(investment_costs, country,  unit='Billion Euros/year'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"

    title = f"{country} - Investment Costs"
    df = investment_costs[country]
    df = df.rename_axis(unit)
    df = df.reset_index()
    df.index = df.index.astype(str)

    # Create a bar chart using Plotly
    fig = go.Figure()
    df_transposed = df.set_index(unit).T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    return fig


def create_capacity_chart(capacities, country, unit='Capacity [GW]'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    groups = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["power-to-liquid"],
        ["AC Transmission lines"],
        ["DC Transmission lines"],
        ["CCGT"],
        ["nuclear"],
    ]
    
    groupss = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["power-to-liquid"],
        ["transmission lines"],
        ["gas pipeline","gas pipeline new"],
        ["CCGT"],
        ["nuclear"],
    ]

    # Create a subplot for each technology
    years = ['2020', '2030', '2040', '2050']
    if country != "EU":
        value = groups
    else:
        value = groupss
    fig = make_subplots(rows=2, cols=len(value) // 2, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in value], shared_yaxes=True)

    df = capacities[country]

    for i, tech_group in enumerate(value, start=1):
        row_idx = 1 if i <= len(value) // 2 else 2
        col_idx = i if i <= len(value) // 2 else i - len(value) // 2

        for tech in tech_group:
            if tech in df.index:
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
    logo['y']=1.03
    fig.add_layout_image(logo)
    # Save plot as HTML
    # html_file_path = os.path.join(output_folder, f"{country}_capacities_chart.html")
    # fig.write_html(html_file_path)

    return fig

def storage_capacity_chart(s_capacities, country, unit='Capacity [GWh]'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["Thermal Energy storage"] = colors["urban central water tanks"]
    colors["Grid-scale"] = 'green'
    colors["home battery"] = 'blue'
    groups = [
        ["Grid-scale battery", "home battery", "V2G"],
        ["H2"],
        ["Thermal Energy storage"],
        ["gas"],
    ]

    # Create a subplot for each technology
    years = ['2020', '2030', '2040', '2050']
    fig = make_subplots(rows=1, cols=len(groups) // 1, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in groups], shared_yaxes=False)

    df = s_capacities[country]

    for i, tech_group in enumerate(groups, start=1):
        row_idx = 1 if i <= len(groups) // 1 else 2
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
                fig.add_trace(trace, row=row_idx, col=col_idx)
                fig.update_yaxes(title_text=unit, row=2, col=1)
    
    # Update layout
    fig.update_layout(height=600, width=1400, showlegend=True, title=f" Storage Capacities for {country}", yaxis_title=unit)
    logo['y']=1.03
    fig.add_layout_image(logo)
    

    return fig

def create_combined_chart_country(costs,investment_costs, capacities, s_capacities, country):
    # Create output folder if it doesn't exist
    output_folder = f"results/{study}/htmls"
    raw_html = os.path.join(output_folder,'raw_html/')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(raw_html, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"
    
    #Load text descriptors
    def load_html_texts(file_path):
     html_texts = {}
     with open(file_path, 'r') as file:
        for line in file:
            # Split by ": " to get the key and the HTML content
            if ": " in line:
                key, html_content = line.split(": ", 1)
                html_texts[key.strip()] = html_content.strip()
     return html_texts
    # Load the HTML texts from file
    html_texts = load_html_texts(file_path)
    sectoral_demands_desc = html_texts.get('sectoral_demands_desc', '')
    annual_costs_desc = html_texts.get('annual_costs_desc', '')
    investment_costs_desc = html_texts.get('investment_costs_desc', '')
    capacities_desc = html_texts.get('capacities_desc', '')
    storage_capacities_desc = html_texts.get('storage_capacities_desc', '')
    heat_dispatch_win_desc = html_texts.get('heat_dispatch_win_desc', '')
    heat_dispatch_sum_desc = html_texts.get('heat_dispatch_sum_desc', '')
    power_dispatch_win_desc = html_texts.get('power_dispatch_win_desc', '')
    power_dispatch_sum_desc = html_texts.get('power_dispatch_sum_desc', '')
    map_plots_desc = html_texts.get('map_plots_desc', '')
    h2_map_plots_desc = html_texts.get('h2_map_plots_desc', '')
    gas_map_plots_desc = html_texts.get('gas_map_plots_desc', '')
    
    #load the html plot flags
    with open(snakemake.input.plots_html, 'r') as file:
     plots = yaml.safe_load(file)
    pypsa_plots = plots.get("Pypsa_plots", {})
    
    if pypsa_plots["Sectoral Demands"] == True:
     plot_demands_file_path = os.path.join(raw_html, f"{country}_sectoral_demands.html")
     with open(plot_demands_file_path, "r") as plot_demands_file:
        plot_demands_html = plot_demands_file.read()
        combined_html += f"<div><h2>{country} - Sectoral Demands</h2>{plot_demands_html}</div>"
    # Create bar chart
    if pypsa_plots["Annual Costs"] == True:
     bar_chart = create_bar_chart(costs, country)
     combined_html += f"<div><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    
    # Create Investment Costs
    if pypsa_plots["Annual Investment Costs"] == True:
     bar_chart_investment = create_investment_costs(investment_costs, country)
     combined_html += f"<div><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"

    # Create capacities chart
    if pypsa_plots["Capacities"] == True:
     capacities_chart = create_capacity_chart(capacities, country)
     combined_html += f"<div><h2>{country} - Capacities </h2>{capacities_chart.to_html()}</div>"
    
    # Create storage capacities chart
    if pypsa_plots["Storage Capacities"] == True:
     s_capacities_chart = storage_capacity_chart(s_capacities, country)
     combined_html += f"<div><h2>{country} - Storage Capacities </h2>{s_capacities_chart.to_html()}</div>"

    # Save the Panel object to HTML
    if pypsa_plots["Power Dispatch Winter"] == True:
     plot_series_file_path = os.path.join(raw_html, f"Power Dispatch (Winter Week) - {country}.html")
     with open(plot_series_file_path, "r") as plot_series_file:
         plot_series_html = plot_series_file.read()
         combined_html += f"<div><h2>{country} - Power Dispatch Winter</h2>{plot_series_html}</div>"
    if pypsa_plots["Power Dispatch Summer"] == True:
     plot_series_file_path_sum = os.path.join(raw_html, f"Power Dispatch (Summer Week) - {country}.html")
     with open(plot_series_file_path_sum, "r") as plot_series_file_sum:
         plot_series_html_w = plot_series_file_sum.read()
         combined_html += f"<div><h2>{country} - Power Dispatch Summer</h2>{plot_series_html_w}</div>"
    if pypsa_plots["Heat Dispatch Winter"] == True:    
     plot_series_heat_file_path = os.path.join(raw_html, f"Heat Dispatch (Winter Week) - {country}.html")
     with open(plot_series_heat_file_path, "r") as plot_series_heat_file:
         plot_series_heat_html = plot_series_heat_file.read()
         combined_html += f"<div><h2>{country} - Heat Dispatch Winter</h2>{plot_series_heat_html}</div>"
    if pypsa_plots["Heat Dispatch Summer"] == True:   
     plot_series_heat_file_path_sum = os.path.join(raw_html, f"Heat Dispatch (Summer Week) - {country}.html")
     with open(plot_series_heat_file_path_sum, "r") as plot_series_heat_file_sum:
         plot_series_heat_html_w = plot_series_heat_file_sum.read()
         combined_html += f"<div><h2>{country} - Heat Dispatch Summer</h2>{plot_series_heat_html_w}</div>"
    if pypsa_plots["Map Plots"] == True:
     plot_map_path = os.path.join(raw_html, f"map_plots_{country}.html")
     with open(plot_map_path, "r") as plot_map_path:
         plot_map_html = plot_map_path.read()
         combined_html += f"<div><h2>Map Plots</h2>{plot_map_html}</div>"
    if pypsa_plots["H2 Map Plots"] == True:
     plot_map_h2_path = os.path.join(raw_html, "map_h2_plots.html")
     with open(plot_map_h2_path, "r") as plot_map_h2_path:
         plot_map_h2_html = plot_map_h2_path.read()
         combined_html += f"<div><h2>H2 Map Plots</h2>{plot_map_h2_html}</div>"
    if pypsa_plots["Gas Map Plots"] == True:
     plot_map_ch4_path = os.path.join(raw_html, "map_ch4_plots.html")
     with open(plot_map_ch4_path, "r") as plot_map_ch4_path:
         plot_map_ch4_html = plot_map_ch4_path.read()
         combined_html += f"<div><h2>Gas Map Plots</h2>{plot_map_ch4_html}</div>"

    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content = ""
    if pypsa_plots["Sectoral Demands"] == True:
     table_of_contents_content += f"<a href='#{country} - Sectoral Demands'>Sectoral Demands</a><br>"
    if pypsa_plots["Annual Costs"] == True:
     table_of_contents_content += f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    if pypsa_plots["Annual Investment Costs"] == True:
     table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    if pypsa_plots["Capacities"] == True:
     table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    if pypsa_plots["Storage Capacities"] == True:
     table_of_contents_content += f"<a href='#{country} - Storage Capacities'>Storage Capacities</a><br>"
    if pypsa_plots["Heat Dispatch Winter"] == True:
     table_of_contents_content += f"<a href='#{country} - Heat Dispatch Winter'>Heat Dispatch Winter</a><br>"
    if pypsa_plots["Heat Dispatch Summer"] == True:
     table_of_contents_content += f"<a href='#{country} - Heat Dispatch Summer'>Heat Dispatch Summer</a><br>"
    if pypsa_plots["Power Dispatch Winter"] == True:
     table_of_contents_content += f"<a href='#{country} - Power Dispatch Winter'>Power Dispatch Winter</a><br>"
    if pypsa_plots["Power Dispatch Summer"] == True:
     table_of_contents_content += f"<a href='#{country} - Power Dispatch Summer'>Power Dispatch Summer</a><br>"
    if pypsa_plots["Map Plots"] == True:
     table_of_contents_content += "<a href='#Map Plots'>Map Plots</a><br>"
    if pypsa_plots["H2 Map Plots"] == True:
     table_of_contents_content += "<a href='#H2 Map Plots'>H2 Map Plots</a><br>"
    if pypsa_plots["Gas Map Plots"] == True:
     table_of_contents_content += "<a href='#Gas Map Plots'>Gas Map Plots</a><br>"
    
    # Add more links for other plots
    main_content = ""
    if pypsa_plots["Sectoral Demands"] == True:
     main_content += f"<div id='{country} - Sectoral Demands'><h2>{country} - Sectoral Demands</h2>{sectoral_demands_desc}{plot_demands_html}</div>"
    if pypsa_plots["Annual Costs"] == True:
     main_content += f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{annual_costs_desc}{bar_chart.to_html()}</div>"
    if pypsa_plots["Annual Investment Costs"] == True:
     main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{investment_costs_desc}{bar_chart_investment.to_html()}</div>"
    if pypsa_plots["Capacities"] == True:
     main_content += f"<div id='{country} - Capacities'><h2>{country} - Capacities</h2>{capacities_desc}{capacities_chart.to_html()}</div>"
    if pypsa_plots["Storage Capacities"] == True:
     main_content += f"<div id='{country} - Storage Capacities'><h2>{country} - Storage Capacities</h2>{storage_capacities_desc}{s_capacities_chart.to_html()}</div>"
    if pypsa_plots["Heat Dispatch Winter"] == True:
     main_content += f"<div id='{country} - Heat Dispatch Winter'><h2>{country} - Heat Dispatch Winter</h2>{heat_dispatch_win_desc}{plot_series_heat_html}</div>"
    if pypsa_plots["Heat Dispatch Summer"] == True:
     main_content += f"<div id='{country} - Heat Dispatch Summer'><h2>{country} - Heat Dispatch Summer</h2>{heat_dispatch_sum_desc}{plot_series_heat_html_w}</div>"
    if pypsa_plots["Power Dispatch Winter"] == True:
     main_content += f"<div id='{country} - Power Dispatch Winter'><h2>{country} - Power Dispatch Winter</h2>{power_dispatch_win_desc}{plot_series_html}</div>"
    if pypsa_plots["Power Dispatch Summer"] == True:
     main_content += f"<div id='{country} - Power Dispatch Summer'><h2>{country} - Power Dispatch Summer</h2>{power_dispatch_sum_desc}{plot_series_html_w}</div>"
    if pypsa_plots["Map Plots"] == True:
     main_content += f"<div id='Map Plots'><h2>Map Plots</h2>{map_plots_desc}{plot_map_html}</div>"
    if pypsa_plots["H2 Map Plots"] == True:
     main_content += f"<div id='H2 Map Plots'><h2>H2 Map Plots</h2>{h2_map_plots_desc}{plot_map_h2_html}</div>"
    if pypsa_plots["Gas Map Plots"] == True:
     main_content += f"<div id='Gas Map Plots'><h2>Gas Map Plots</h2>{gas_map_plots_desc}{plot_map_ch4_html}</div>"
    # Add more content for other plots
    
    template_path =  snakemake.input.template
    with open(template_path, "r") as template_file:
        template_content = template_file.read()
        template = Template(template_content)
        
    rendered_html = template.render(
    title=f"{country} - Combined Plots",
    country=country,
    TABLE_OF_CONTENTS=table_of_contents_content,
    MAIN=main_content,)
    
    combined_file_path = os.path.join(output_folder, f"{country}_combined_chart.html")
    with open(combined_file_path, "w") as combined_file:
     combined_file.write(rendered_html)

    
if __name__ == "__main__":
    if "snakemake" not in globals():
        #from _helpers import mock_snakemake
        #snakemake = mock_snakemake("prepare_results")

        import pickle
        with open("snakemake_dump.pkl", "rb") as f:
            snakemake = pickle.load(f)

        # Updating the configuration from the standard config file to run in standalone:
    #import pickle
    #with open("snakemake_dump.pkl", "wb") as f:
    #    pickle.dump(snakemake, f)

    simpl = snakemake.params.scenario["simpl"][0]
    cluster = snakemake.params.scenario["clusters"][0]
    opt = snakemake.params.scenario["opts"][0]
    sector_opt = snakemake.params.scenario["sector_opts"][0]
    ll = snakemake.params.scenario["ll"][0]
    planning_horizons = [2020, 2030, 2040, 2050]
    total_country = 'EU'
    countries = snakemake.params.countries 
    proj = load_projection(snakemake.params.plotting)
    map_opts = snakemake.params.plotting["map"]
    countries.append(total_country)
    logging.basicConfig(level=snakemake.config["logging"]["level"])
    config = snakemake.config
    study = snakemake.params.study
    logo = logo()
    file_path = snakemake.input.file_path
    loaded_files = load_files(study, planning_horizons, simpl, cluster, opt, sector_opt, ll)
    results = calculate_transmission_values(simpl, cluster, opt, sector_opt, ll, planning_horizons)
    costs = costs(countries, results)
    investment_costs = Investment_costs(countries, results)
    capacities = capacities(countries, results)
    s_capacities = storage_capacities(countries)

    with open(snakemake.input.plots_html, 'r') as file:
     plots = yaml.safe_load(file).get('Pypsa_plots')
    if plots['Power Dispatch Winter']:
        plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons,start = "2013-02-01",stop = "2013-02-07",title="Power Dispatch (Winter Week)")
    if plots['Power Dispatch Summer']:
        plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons,start = "2013-07-01",stop = "2013-07-07",title="Power Dispatch (Summer Week)")
    if plots['Heat Dispatch Winter']:
        plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons,start = "2013-02-01",stop = "2013-02-07",title="Heat Dispatch (Winter Week)")
    if plots['Heat Dispatch Summer']:
        plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons,start = "2013-07-01",stop = "2013-07-07",title="Heat Dispatch (Summer Week)")
    plot_demands(countries)
    if plots['Map Plots']:
        for country in countries:
            create_map_plots(planning_horizons, country)
    if plots['H2 Map Plots']:
        create_H2_map_plots(planning_horizons)
    if plots['Gas Map Plots']:
        create_gas_map_plots(planning_horizons)
    
    for country in countries:
        create_combined_chart_country(costs,investment_costs, capacities,s_capacities, country)
    
