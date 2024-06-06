#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import os
import shutil
from datetime import datetime
import plotly.express as px
from jinja2 import Template

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

def scenario_costs(country):
    costs_bau = pd.read_csv(f"results/bau/country_csvs/{country}_costs.csv")
    costs_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_costs.csv")
    # costs_ncdr = pd.read_csv(f"csvs/{country}_costs_ncdr.csv")
    # costs_reff = costs_bau[['tech', '2020']]
    costs_bau = costs_bau[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    # costs_ncdr = costs_ncdr[['tech', '2030', '2040', '2050']]
    
    # costs_reff = costs_reff.rename(columns={'2020': 'Reff'})
    
    costs_bau['Total'] = costs_bau[['2030', '2040', '2050']].sum(axis=1)
    costs_bau = costs_bau[['tech', 'Total']]
    costs_bau['Total'] = costs_bau['Total'] / 3
    costs_bau = costs_bau.rename(columns={'Total': 'BAU'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    # costs_ncdr['Total'] = costs_ncdr[['2030', '2040', '2050']].sum(axis=1)
    # costs_ncdr = costs_ncdr[['tech', 'Total']]
    # costs_ncdr['Total'] = costs_ncdr['Total'] / 3
    # costs_ncdr = costs_ncdr.rename(columns={'Total': 'Ncdr'})
    
    combined_df = pd.merge(costs_suff, costs_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    # combined_df = pd.merge(combined_df, costs_suff, on='tech', how='outer')
    # combined_df = pd.merge(combined_df, costs_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Billion Euros/year'
    title=f'Total Costs Comparison for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=1000,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    
    return fig

def scenario_investment_costs(country):
    costs_bau = pd.read_csv(f"results/bau/country_csvs/{country}_investment costs.csv")
    costs_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_investment costs.csv")
    # costs_ncdr = pd.read_csv(f"csvs/{country}_costs_ncdr.csv")
    # costs_reff = costs_bau[['tech', '2020']]
    costs_bau = costs_bau[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    # costs_ncdr = costs_ncdr[['tech', '2030', '2040', '2050']]
    
    # costs_reff = costs_reff.rename(columns={'2020': 'Reff'})
    
    costs_bau['Total'] = costs_bau[['2030', '2040', '2050']].sum(axis=1)
    costs_bau = costs_bau[['tech', 'Total']]
    costs_bau['Total'] = costs_bau['Total'] / 3
    costs_bau = costs_bau.rename(columns={'Total': 'BAU'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    # costs_ncdr['Total'] = costs_ncdr[['2030', '2040', '2050']].sum(axis=1)
    # costs_ncdr = costs_ncdr[['tech', 'Total']]
    # costs_ncdr['Total'] = costs_ncdr['Total'] / 3
    # costs_ncdr = costs_ncdr.rename(columns={'Total': 'Ncdr'})
    
    combined_df = pd.merge(costs_suff, costs_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    # combined_df = pd.merge(combined_df, costs_suff, on='tech', how='outer')
    # combined_df = pd.merge(combined_df, costs_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Billion Euros/year'
    title=f'Total Investment Costs Comparison for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=1000,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    
    return fig
    
def scenario_cumulative_costs(country):
    costs_bau = pd.read_csv(f"results/bau/country_csvs/{country}_investment costs.csv")
    costs_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_investment costs.csv")
    # costs_ncdr = pd.read_csv(f"csvs/{country}_costs_ncdr.csv")
    # costs_reff = costs_bau[['tech', '2020']]
    costs_bau = costs_bau[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    # costs_ncdr = costs_ncdr[['tech', '2030', '2040', '2050']]
    
    # costs_reff = costs_reff.rename(columns={'2020': 'Reff'})
    
    costs_bau['Total'] = costs_bau[['2030', '2040', '2050']].sum(axis=1)
    costs_bau = costs_bau[['tech', 'Total']]
    costs_bau['Total'] = costs_bau['Total'] / 3
    costs_bau['Total'] = costs_bau['Total'] * 27
    costs_bau = costs_bau.rename(columns={'Total': 'BAU'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff['Total'] = costs_suff['Total'] * 27
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    # costs_ncdr['Total'] = costs_ncdr[['2030', '2040', '2050']].sum(axis=1)
    # costs_ncdr = costs_ncdr[['tech', 'Total']]
    # costs_ncdr['Total'] = costs_ncdr['Total'] / 3
    # costs_ncdr = costs_ncdr.rename(columns={'Total': 'Ncdr'})
    
    combined_df = pd.merge(costs_suff, costs_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    # combined_df = pd.merge(combined_df, costs_suff, on='tech', how='outer')
    # combined_df = pd.merge(combined_df, costs_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Billion Euros'
    title=f'Total Comulative Costs (2023-2050) for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=1000, showlegend=True,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    
    return fig   
#%%
def scenario_capacities(country):
    caps_bau = pd.read_csv(f"results/bau/country_csvs/{country}_capacities.csv")
    caps_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_capacities.csv")
    # caps_ncdr = pd.read_csv(f"csvs/{country}_capacities_ncdr.csv")
    caps_reff = caps_bau[['tech', '2020']]
    caps_bau = caps_bau[['tech', '2030', '2040', '2050']]
    caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    # caps_ncdr = caps_ncdr[['tech', '2030', '2040', '2050']]
    
    # caps_reff = caps_reff.rename(columns={'2020': 'Reff'})
    
    caps_bau = caps_bau[['tech', '2050']]
    caps_bau = caps_bau.rename(columns={'2050': 'BAU'})
    
    caps_suff = caps_suff[['tech', '2050']]
    caps_suff = caps_suff.rename(columns={'2050': 'Suff'})
    
    # caps_suff['Total'] = caps_suff[['2030', '2040', '2050']].sum(axis=1)
    # caps_suff = caps_suff[['tech', 'Total']]
    # caps_suff['Total'] = caps_suff['Total'] / 3
    # caps_suff = caps_suff.rename(columns={'Total': 'Suff'})
    
    # caps_ncdr = caps_ncdr[['tech', '2050']]
    # caps_ncdr = caps_ncdr.rename(columns={'2050': 'Ncdr'})
    
    combined_df = pd.merge(caps_reff, caps_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    combined_df = pd.merge(combined_df, caps_suff, on='tech', how='outer')
    # combined_df = pd.merge(combined_df, caps_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Capacity [GW]'
    title=f"Capacities for {country}"
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    
    fig = go.Figure()
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
    groupss = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
        ["transmission lines"],
        ["gas pipeline","gas pipeline new"],
        ["CCGT"],
        ["nuclear"],
    ]
    if country != "EU":
        value = groups
    else:
        value = groupss

    fig = make_subplots(rows=2, cols=len(value) // 2, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in value], shared_yaxes=True)

    df = combined_df

    for i, tech_group in enumerate(value, start=1):
        row_idx = 1 if i <= len(value) // 2 else 2
        col_idx = i if i <= len(value) // 2 else i - len(value) // 2

        for tech in tech_group:
         if tech in df.index:
            y_values = [val / 1000 for val in df.loc[tech]]
            trace = go.Bar(
                x=df.columns,
                y=y_values,
                name=f"{tech}",
                marker_color=tech_colors.get(tech, 'gray')
            )
            fig.add_trace(trace, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=unit, row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {country}_2050 compared to 2020", yaxis_title=unit)
    logo['y']=1.021
    fig.add_layout_image(logo)
    return fig

def storage_capacities(country):
    caps_bau = pd.read_csv(f"results/bau/country_csvs/{country}_storage_capacities.csv")
    caps_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_storage_capacities.csv")
    # caps_ncdr = pd.read_csv(f"csvs/{country}_capacities_ncdr.csv")
    caps_reff = caps_bau[['tech', '2020']]
    caps_bau = caps_bau[['tech', '2030', '2040', '2050']]
    caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    # caps_ncdr = caps_ncdr[['tech', '2030', '2040', '2050']]
    
    # caps_reff = caps_reff.rename(columns={'2020': 'Reff'})
    
    caps_bau = caps_bau[['tech', '2050']]
    caps_bau = caps_bau.rename(columns={'2050': 'BAU'})
    
    caps_suff = caps_suff[['tech', '2050']]
    caps_suff = caps_suff.rename(columns={'2050': 'Suff'})
    
    # caps_suff['Total'] = caps_suff[['2030', '2040', '2050']].sum(axis=1)
    # caps_suff = caps_suff[['tech', 'Total']]
    # caps_suff['Total'] = caps_suff['Total'] / 3
    # caps_suff = caps_suff.rename(columns={'Total': 'Suff'})
    
    # caps_ncdr = caps_ncdr[['tech', '2050']]
    # caps_ncdr = caps_ncdr.rename(columns={'2050': 'Ncdr'})
    
    combined_df = pd.merge(caps_reff, caps_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    combined_df = pd.merge(combined_df, caps_suff, on='tech', how='outer')
    # combined_df = pd.merge(combined_df, caps_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Capacity [GWh]'
    title=f"Storage Capacities for {country}"
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["Thermal Energy storage"] = colors["urban central water tanks"]
    colors["Grid-scale"] = 'green'
    colors["home battery"] = 'blue'
    
    fig = go.Figure()
    groups = [
        ["Grid-scale battery", "home battery", "V2G"],
        ["H2"],
        ["Thermal Energy storage"],
        ["biogas"],
    ]

    fig = make_subplots(rows=1, cols=len(groups) // 1, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in groups], shared_yaxes=False)

    df = combined_df

    for i, tech_group in enumerate(groups, start=1):
        row_idx = 1 if i <= len(groups) // 1 else 2
        col_idx = i if i <= len(groups) // 1 else i - len(groups) // 1

        for tech in tech_group:
         if tech in df.index:
            y_values = [val / 1000 for val in df.loc[tech]]
            trace = go.Bar(
                x=df.columns,
                y=y_values,
                name=f"{tech}",
                marker_color=tech_colors.get(tech, 'gray')
            )
            fig.add_trace(trace, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=unit, row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=1400, showlegend=True, title=f"Capacities for {country}_2050 compared to 2020", yaxis_title=unit)
    logo['y']=1.021
    fig.add_layout_image(logo)
    return fig

def scenario_demands(country):
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
    
    data_ncdr = pd.read_csv(f"results/ncdr/country_csvs/{country}_sectordemands.csv", index_col=0)
    columns_to_drop = ['2020']
    data_ncdr = data_ncdr.drop(columns=columns_to_drop)
    data_bau = pd.read_csv(f"results/bau/country_csvs/{country}_sectordemands.csv", index_col=0)
    
    # Rename columns
    data_bau.rename(columns={'2020': 'reff', '2030': 'bau-2030', '2040': 'bau-2040', '2050': 'bau-2050'}, inplace=True)
    data_ncdr.rename(columns={'2030': 'suff-2030', '2040': 'suff-2040', '2050': 'suff-2050'}, inplace=True)
          
    # Melt dataframes
    melted_bau = pd.melt(data_bau, id_vars=['Demand', 'Sectors'], var_name='year', value_name='value')
    melted_ncdr = pd.melt(data_ncdr, id_vars=['Demand', 'Sectors'], var_name='year', value_name='value')
       
    # Add color information
    melted_bau['color'] = melted_bau['Sectors'].map(colors)
    melted_ncdr['color'] = melted_ncdr['Sectors'].map(colors)
    
    # Concatenate the dataframes
    melted_data = pd.concat([melted_bau,melted_ncdr])

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
    for col in fig.select_xaxes():
     col.update(tickangle=-45)
    logo['y']=1.025
    fig.add_layout_image(logo)
    return fig

def create_scenario_plots():
 scenarios=pd.read_csv("data/scenario_data.csv")

 capacities_ncdr=pd.read_csv("results/ncdr/country_csvs/BE_capacities.csv", index_col=0)
 capacities_ncdr_2050 = capacities_ncdr[['2050']]/1e3
 ac_transmission_ncdr = capacities_ncdr_2050.loc['AC Transmission lines', '2050']
 dc_transmission_ncdr = capacities_ncdr_2050.loc['DC Transmission lines', '2050']
 transmission_ncdr = ac_transmission_ncdr + dc_transmission_ncdr

 investment_ncdr=pd.read_csv("results/ncdr/country_csvs/BE_investment costs.csv", index_col=0)
 investment_ncdr_2050 = investment_ncdr[['2050']].sum().sum()/1e9
 demands_ncdr=pd.read_excel("results/ncdr/htmls/ChartData_BE.xlsx",  sheet_name="Chart 21", index_col=0)
 elec_demand_ncdr = demands_ncdr.loc[str(2050)].sum()
 total_costs_ncdr=pd.read_csv("results/ncdr/country_csvs/BE_costs.csv", index_col=0)
 total_costs_ncdr_2050 = total_costs_ncdr[['2050']].sum().sum()/1e9

 capacities_bau=pd.read_csv("results/bau/country_csvs/BE_capacities.csv", index_col=0)
 capacities_bau_2050 = capacities_bau[['2050']]/1e3
 ac_transmission_bau = capacities_bau_2050.loc['AC Transmission lines', '2050']
 dc_transmission_bau = capacities_bau_2050.loc['DC Transmission lines', '2050']
 transmission_bau = ac_transmission_bau + dc_transmission_bau

 investment_bau=pd.read_csv("results/bau/country_csvs/BE_investment costs.csv", index_col=0)
 investment_bau_2050 = investment_bau[['2050']].sum().sum()/1e9
 demands_bau=pd.read_excel("results/bau/htmls/ChartData_BE.xlsx",  sheet_name="Chart 21", index_col=0)
 elec_demand_bau = demands_bau.loc[str(2050)].sum()
 total_costs_bau=pd.read_csv("results/bau/country_csvs/BE_costs.csv", index_col=0)
 total_costs_bau_2050 = total_costs_bau[['2050']].sum().sum()/1e9


 scenarios['Pypsa-sufficiency'] = None
 scenarios.loc['Demand (TWh)', 'Pypsa-sufficiency'] = elec_demand_ncdr
 scenarios.loc['Investment Costs(Billion Euros/year)', 'Pypsa-sufficiency'] = investment_ncdr_2050
 scenarios.loc['Annual Costs(Billion Euros/year)', 'Pypsa-sufficiency'] = total_costs_ncdr_2050
 scenarios.loc['Emissions(%)', 'Pypsa-sufficiency'] = -100

 scenarios['Pypsa-BAU'] = None
 scenarios.loc['Demand (TWh)', 'Pypsa-BAU'] = elec_demand_bau
 scenarios.loc['Investment Costs(Billion Euros/year)', 'Pypsa-BAU'] = investment_bau_2050
 scenarios.loc['Annual Costs(Billion Euros/year)', 'Pypsa-BAU'] = total_costs_bau_2050
 scenarios.loc['Emissions(%)', 'Pypsa-BAU'] = -100

 techs = ['solar', 'onshore wind','offshore wind', 'nuclear']

 for tech in techs:
    scenarios.loc[tech, 'Pypsa-sufficiency'] = capacities_ncdr_2050.loc[tech, '2050']
    scenarios.loc[tech, 'Pypsa-BAU'] = capacities_bau_2050.loc[tech, '2050']

 scenarios.loc['Hydrogen Turbines or CHPs', 'Pypsa-sufficiency'] = capacities_ncdr_2050.loc['H2 turbine', '2050']
 scenarios.loc['CCGT/OCGT', 'Pypsa-sufficiency'] = capacities_ncdr_2050.loc['CCGT', '2050']
 scenarios.loc['Interconnections', 'Pypsa-sufficiency'] = transmission_ncdr
 scenarios.loc['Others', 'Pypsa-sufficiency'] = capacities_ncdr_2050.loc['hydroelectricity', '2050']

 scenarios.loc['Hydrogen Turbines or CHPs', 'Pypsa-BAU'] = capacities_bau_2050.loc['H2 turbine', '2050']
 scenarios.loc['CCGT/OCGT', 'Pypsa-BAU'] = capacities_bau_2050.loc['CCGT', '2050']
 scenarios.loc['Interconnections', 'Pypsa-BAU'] = transmission_bau
 scenarios.loc['Others', 'Pypsa-BAU'] = capacities_bau_2050.loc['hydroelectricity', '2050']

 scenarios = scenarios.apply(pd.to_numeric, errors='coerce').fillna(0)


 scenarios_transposed = scenarios.transpose()
 figures = {}
 # Plot the bar chart for Demand
 fig_demand = go.Figure()
 fig_demand.add_trace(go.Bar(name='Demand (TWh)', x=scenarios_transposed.index, y=scenarios_transposed['Demand (TWh)']))
 fig_demand.update_layout(
        title="Electricity demand comparison for scenarios in 2050",
        xaxis_title="Scenario",
        yaxis_title="Demand (TWh)"
    )
 figures['demand'] = fig_demand

 # Plot the bar chart for VRE capacities
 columns_to_plot_vre = ['solar', 'onshore wind', 'offshore wind']
 colors_vre = {
        'solar': '#f9d002',
        'onshore wind': '#235ebc',
        'offshore wind': '#6895dd',
    }
 fig_vre = go.Figure()
 for column in columns_to_plot_vre:
        color = colors_vre.get(column, None)
        fig_vre.add_trace(go.Bar(name=column, x=scenarios_transposed.index, y=scenarios_transposed[column], marker_color=color))
 fig_vre.update_layout(
        title="VRE capacities for scenarios in 2050",
        xaxis_title="Scenario",
        yaxis_title="Capacities (GW)"
    )
 figures['vre'] = fig_vre

 # Plot the bar chart for Flexibility options
 columns_to_plot_flex = ['nuclear', 'Hydrogen Turbines or CHPs', 'CCGT/OCGT', 'Interconnections', 'Others']
 colors_flex = {
        'nuclear': '#ff8c00',
        'Hydrogen Turbines or CHPs': '#bf13a0',
        'CCGT/OCGT': '#a85522',
        'Interconnections': '#6c9459',
        'Others': '#cc1f1f'
    }
 fig_flex = go.Figure()
 for column in columns_to_plot_flex:
        color = colors_flex.get(column, None)
        fig_flex.add_trace(go.Bar(name=column, x=scenarios_transposed.index, y=scenarios_transposed[column], marker_color=color))
 fig_flex.update_layout(
        title="Flexibility options in electricity grid for scenarios in 2050",
        xaxis_title="Scenario",
        yaxis_title="Capacities (GW)"
    )
 figures['flexibility'] = fig_flex

 # Plot the bar chart for Costs
 columns_to_plot_costs = ['Annual Costs(Billion Euros/year)', 'Investment Costs(Billion Euros/year)']
 colors_costs = {
        'Annual Costs(Billion Euros/year)': 'blue',
        'Investment Costs(Billion Euros/year)': 'green'
    }
 fig_costs = go.Figure()
 for column in columns_to_plot_costs:
        color = colors_costs.get(column, None)
        fig_costs.add_trace(go.Bar(name=column, x=scenarios_transposed.index, y=scenarios_transposed[column], marker_color=color))
 fig_costs.update_layout(
        title="Costs comparison for scenarios",
        xaxis_title="Scenario",
        yaxis_title="Costs (Billion Euros/year)"
    )
 figures['costs'] = fig_costs

 # Plot the bar chart for Emissions
 fig_emissions = go.Figure()
 fig_emissions.add_trace(go.Bar(name='Emissions(%)', x=scenarios_transposed.index, y=scenarios_transposed['Emissions(%)'], marker_color='red'))
 fig_emissions.update_layout(
        title="Emissions comparison for scenarios compared to 1990",
        xaxis_title="Scenario",
        yaxis_title="%"
    )
 figures['emissions'] = fig_emissions

 return figures
    
def create_combined_scenario_chart_country(country, output_folder='results/scenario_results/'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"

    # Create bar chart
    bar_chart = scenario_costs(country)
    combined_html += f"<div><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    
    bar_chart_investment = scenario_investment_costs(country)
    combined_html += f"<div><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    
    bar_chart_cumulative = scenario_cumulative_costs(country)
    combined_html += f"<div><h2>{country} - Cummulative Investment Costs (2023-2050)</h2>{bar_chart_cumulative.to_html()}</div>"
    
    # Create capacities chart
    capacities_chart = scenario_capacities(country)
    combined_html += f"<div><h2>{country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    
    # Create capacities chart
    storage_capacities_chart = storage_capacities(country)
    combined_html += f"<div><h2>{country} -  Storage Capacities</h2>{storage_capacities_chart.to_html()}</div>"
    
    # Create demands chart
    demands_chart = scenario_demands(country)
    combined_html += f"<div><h2>{country} - Sectoral Demands</h2>{demands_chart.to_html()}</div>"
    
    #Create scenario comparison plots
    if country == 'BE':
        scenario_figures = create_scenario_plots()
        demand_comparison = scenario_figures['demand']
        combined_html += f"<div><h2>{country} - Scenarios Demands Comparison</h2>{demand_comparison.to_html()}</div>"
        vre_comparison = scenario_figures['vre']
        combined_html += f"<div><h2>{country} - Scenarios VRE Capacities Comparison</h2>{vre_comparison.to_html()}</div>"
        flexibility_comparison = scenario_figures['flexibility']
        combined_html += f"<div><h2>{country} - Scenario Flexibility Capacities in Electricity Grid Comparison</h2>{flexibility_comparison.to_html()}</div>"
        costs_comparison = scenario_figures['costs']
        combined_html += f"<div><h2>{country} - Scenarios Costs Comparison</h2>{costs_comparison.to_html()}</div>"
        emission_comparison = scenario_figures['emissions']
        combined_html += f"<div><h2>{country} - Scenarios Emissions Comparison</h2>{emission_comparison.to_html()}</div>"

    combined_html += "</body></html>"
    table_of_contents_content = ""
    main_content = ""
    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content += f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Cummulative Investment Costs (2023-2050)'>Cummulative Investment Costs (2023-2050)</a><br>"
    table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{country} - Storage Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{country} - Sectoral Demands'>Sectoral Demands</a><br>"
    if country == 'BE':
        table_of_contents_content += f"<a href='#{country} - Scenarios Demands Comparison'>Scenarios Demands Comparison</a><br>"
        table_of_contents_content += f"<a href='#{country} - Scenarios VRE Capacities Comparison'>Scenarios VRE Capacities Comparison</a><br>"
        table_of_contents_content += f"<a href='#{country} - Scenario Flexibility Capacities in Electricity Grid Comparison'>Scenario Flexibility Capacities in Electricity Grid Comparison</a><br>"
        table_of_contents_content += f"<a href='#{country} - Scenarios Costs Comparison'>Scenarios Costs Comparison</a><br>"
        table_of_contents_content += f"<a href='#{country} - Scenarios Emissions Comparison'>Scenarios Emissions Comparison</a><br>"

    # Add more links for other plots
    main_content += f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    main_content += f"<div id='{country} - Cummulative Investment Costs (2023-2050)'><h2>{country} - Cummulative Investment Costs (2023-2050)</h2>{bar_chart_cumulative.to_html()}</div>"
    main_content += f"<div id='{country} - Capacities'><h2>{country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Storage Capacities'><h2>{country} - Storage Capacities</h2>{storage_capacities_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Sectoral Demands'><h2>{country} - Sectoral Demands</h2>{demands_chart.to_html()}</div>"
    if country == 'BE':
        main_content += f"<div id='{country} - Scenarios Demands Comparison'><h2>{country} - Scenarios Demands Comparison</h2>{demand_comparison.to_html()}</div>"
        main_content += f"<div id='{country} - Scenarios VRE Capacities Comparison'><h2>{country} - Scenarios VRE Capacities Comparison</h2>{vre_comparison.to_html()}</div>"
        main_content += f"<div id='{country} - Scenario Flexibility Capacities in Electricity Grid Comparison'><h2>{country} - Scenario Flexibility Capacities in Electricity Grid Comparison</h2>{flexibility_comparison.to_html()}</div>"
        main_content += f"<div id='{country} - Scenarios Costs Comparison'><h2>{country} - Scenarios Costs Comparison</h2>{costs_comparison.to_html()}</div>"
        main_content += f"<div id='{country} - Scenarios Emissions Comparison'><h2>{country} - Scenarios Emissions Comparison</h2>{emission_comparison.to_html()}</div>"
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
    
    combined_file_path = os.path.join(output_folder, f"{country}_combined_scenario_chart.html")
    with open(combined_file_path, "w") as combined_file:
     combined_file.write(rendered_html)




if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_scenarios")
        
    total_country = 'EU'
    countries = snakemake.params.countries 
    countries.append(total_country) 
    config = snakemake.config
    logo = logo()
    for country in countries:
        create_combined_scenario_chart_country(country)
        
    
 
