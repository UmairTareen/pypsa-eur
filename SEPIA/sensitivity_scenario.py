#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:57:11 2024

@author: umair
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import os
from jinja2 import Template
import plotly.io as pio

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
    # costs_sensitivity_1 = pd.read_csv(f"results/sensitivity_analysis_pypsa/country_csvs/{country}_costs.csv")
    # costs_sensitivity_2 = pd.read_csv(f"results/sensitivity_analysis_fps/country_csvs/{country}_costs.csv")
    # costs_sensitivity_3 = pd.read_csv(f"results/sensitivity_analysis_mackinze/country_csvs/{country}_costs.csv")
    # costs_sensitivity_4 = pd.read_csv(f"results/sensitivity_analysis_energyville/country_csvs/{country}_costs.csv")
    
    # costs_sensitivity_1 = costs_sensitivity_1[['tech', '2030', '2040', '2050']]
    # costs_sensitivity_2 = costs_sensitivity_2[['tech', '2030', '2040', '2050']]
    # costs_sensitivity_3 = costs_sensitivity_3[['tech', '2030', '2040', '2050']]
    # costs_sensitivity_4 = costs_sensitivity_4[['tech', '2030', '2040', '2050']]
    
    # costs_sensitivity_1['Total'] = costs_sensitivity_1[['2030', '2040', '2050']].sum(axis=1)
    # costs_sensitivity_1 = costs_sensitivity_1[['tech', 'Total']]
    # costs_sensitivity_1['Total'] = costs_sensitivity_1['Total'] / 3
    # costs_sensitivity_1 = costs_sensitivity_1.rename(columns={'Total': 'PyPSA'})
    
    # costs_sensitivity_2['Total'] = costs_sensitivity_2[['2030', '2040', '2050']].sum(axis=1)
    # costs_sensitivity_2 = costs_sensitivity_2[['tech', 'Total']]
    # costs_sensitivity_2['Total'] = costs_sensitivity_2['Total'] / 3
    # costs_sensitivity_2 = costs_sensitivity_2.rename(columns={'Total': 'FPS'})
    
    # costs_sensitivity_3['Total'] = costs_sensitivity_3[['2030', '2040', '2050']].sum(axis=1)
    # costs_sensitivity_3 = costs_sensitivity_3[['tech', 'Total']]
    # costs_sensitivity_3['Total'] = costs_sensitivity_3['Total'] / 3
    # costs_sensitivity_3 = costs_sensitivity_3.rename(columns={'Total': 'Mackinze'})
    
    # costs_sensitivity_4['Total'] = costs_sensitivity_4[['2030', '2040', '2050']].sum(axis=1)
    # costs_sensitivity_4 = costs_sensitivity_4[['tech', 'Total']]
    # costs_sensitivity_4['Total'] = costs_sensitivity_4['Total'] / 3
    # costs_sensitivity_4 = costs_sensitivity_4.rename(columns={'Total': 'Energyville'})
    print(f"Config run name: {config['run']['name']}")
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_costs.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_costs.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_costs.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_costs.csv")]
     
    elif "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_costs.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_costs.csv")]
    # Dictionary to store the processed dataframes
    elif "sensitivity_analysis_seq" in config["run"]["name"]:
     sensitivity_analyses = [
     ("No-seq", f"results/sensitivity_analysis_seq_0/country_csvs/{country}_costs.csv"),
     ("2 Mtons/year", f"results/sensitivity_analysis_seq_2/country_csvs/{country}_costs.csv"),
     ("5 Mtons/year", f"results/sensitivity_analysis_seq_5/country_csvs/{country}_costs.csv"),
     ("10 Mtons/year", f"results/sensitivity_analysis_seq_10/country_csvs/{country}_costs.csv"),
     ("No-limit", f"results/sensitivity_analysis_seq_nolim/country_csvs/{country}_costs.csv")]
        
    print(f"Selected sensitivity analyses: {sensitivity_analyses}")
    costs_sensitivity = {}

    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech', '2030', '2040', '2050']]
      df['Total'] = df[['2030', '2040', '2050']].sum(axis=1)
      df = df[['tech', 'Total']]
      df['Total'] = df['Total'] / 3
      df = df.rename(columns={'Total': name})
      # Store the processed dataframe in the dictionary
      costs_sensitivity[name] = df
    
    combined_df = list(costs_sensitivity.values())[0]
    for df in list(costs_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
    
    unit='Euros/year'
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    title=f'Average Costs Per Year Comparison For {title_country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    # colors["AC Transmission"] = "#FF3030"
    # colors["DC Transmission"] = "#104E8B"
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
    
    # if "sensitivity_analysis_nuclear" in config["run"]["name"]:
    #  Scenario_1 = "8594 EUR/kW_nuclear"
    #  Scenario_2 = "7000 EUR/kW_nuclear"
    #  Scenario_3 = "6000 EUR/kW_nuclear"
    #  Scenario_4 =  "4500 EUR/kW_nuclear"
    #  names_values = [
    #  (f"Scenario_1 = {Scenario_1}", Scenario_1),
    #  (f"Scenario_2 = {Scenario_2}", Scenario_2),
    #  (f"Scenario_3 = {Scenario_3}", Scenario_3),
    #  (f"Scenario_4 = {Scenario_4}", Scenario_4)]
    # if "sensitivity_analysis_offshore" in config["run"]["name"]:
    #  pypsa_value = "+ 0 GW"
    #  nothsea_value = "+ 16 GW"
    #  names_values = [
    #  (f"PyPSA value = {pypsa_value}", pypsa_value),
    #  (f"Nortsea_value = {nothsea_value}", nothsea_value)]
    # if "sensitivity_analysis_seq" in config["run"]["name"]:
    #  a_value = "No-seq"
    #  b_value = "2 Mtons/year"
    #  c_value = "5 Mtons/year"
    #  d_value = "10 Mtons/year"
    #  e_value = "No-limit"
    #  names_values = [
    #  (f"No-seq = {a_value}", a_value),
    #  (f"2 Mtons/year = {b_value}", b_value),
    #  (f"5 Mtons/year = {c_value}", c_value),
    #  (f"10 Mtons/year = {d_value}", d_value),
    #  (f"No-limit = {e_value}", e_value)]
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    fig = go.Figure()
    df_transposed = combined_df.T
    df_transposed.index = [scenario_label_map.get(idx, idx) for idx in df_transposed.index]

    for tech in df_transposed.columns:
      y = df_transposed[tech]
      color = tech_colors.get(tech, 'lightgrey')
      fig.add_trace(go.Bar(
        x=df_transposed.index,
        y=y.where(y > 0, 0),
        name=tech,
        marker_color=color
    ))
      fig.add_trace(go.Bar(
        x=df_transposed.index,
        y=y.where(y < 0, 0),
        name=tech,
        marker_color=color,
        showlegend=False
    ))
    # for name, value in names_values:
    #  fig.add_trace(go.Scatter(
    #     x=[None],
    #     y=[None],
    #     mode='markers',
    #     name=name,
    #     marker=dict(color='rgba(0,0,0,0)')
    # ))
    layout_common = dict(
    title=title,
    barmode='relative',  # Changed from 'stack'
    yaxis=dict(title=unit, title_font=dict(size=15), tickfont=dict(size=15)),
    xaxis=dict(tickfont=dict(size=15)),
    legend=dict(font=dict(size=15)),
    hovermode='y'
)
    # Configure layout and labels
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
      fig.update_layout(height=1000, width=1000, **layout_common)
    else:
      fig.update_layout(height=800, width=600, **layout_common)
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} total costs.pdf", format='pdf')
    # fig.add_layout_image(logo)
    
    return fig

def scenario_investment_costs(country):
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_investment costs.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_investment costs.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_investment costs.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_investment costs.csv")]
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_investment costs.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_investment costs.csv")]
    elif "sensitivity_analysis_seq" in config["run"]["name"]:
     sensitivity_analyses = [
     ("No-seq", f"results/sensitivity_analysis_seq_0/country_csvs/{country}_investment costs.csv"),
     ("2 Mtons/year", f"results/sensitivity_analysis_seq_2/country_csvs/{country}_investment costs.csv"),
     ("5 Mtons/year", f"results/sensitivity_analysis_seq_5/country_csvs/{country}_investment costs.csv"),
     ("10 Mtons/year", f"results/sensitivity_analysis_seq_10/country_csvs/{country}_investment costs.csv"),
     ("No-limit", f"results/sensitivity_analysis_seq_nolim/country_csvs/{country}_investment costs.csv")]
    # Dictionary to store the processed dataframes
    costs_sensitivity = {}

    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech', '2030', '2040', '2050']]
      df['Total'] = df[['2030', '2040', '2050']].sum(axis=1)
      df = df[['tech', 'Total']]
      df['Total'] = df['Total'] / 3
      df = df.rename(columns={'Total': name})
      # Store the processed dataframe in the dictionary
      costs_sensitivity[name] = df
    
    combined_df = list(costs_sensitivity.values())[0]
    for df in list(costs_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
    
    unit='Euros/year'
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    title=f'Average Investment Costs Per Year Comparison For {title_country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    tech_colors["Biogas Plants"] = tech_colors["Biomass"]
    # colors["AC Transmission"] = "#FF3030"
    # colors["DC Transmission"] = "#104E8B"
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
    
    # if "sensitivity_analysis_nuclear" in config["run"]["name"]:
    #  Scenario_1 = "8594 EUR/kW_nuclear"
    #  Scenario_2 = "7000 EUR/kW_nuclear"
    #  Scenario_3 = "6000 EUR/kW_nuclear"
    #  Scenario_4 =  "4500 EUR/kW_nuclear"
    #  names_values = [
    #  (f"Scenario_1 = {Scenario_1}", Scenario_1),
    #  (f"Scenario_2 = {Scenario_2}", Scenario_2),
    #  (f"Scenario_3 = {Scenario_3}", Scenario_3),
    #  (f"Scenario_4 = {Scenario_4}", Scenario_4)]
    # if "sensitivity_analysis_offshore" in config["run"]["name"]:
    #  pypsa_value = "+ 0 GW"
    #  nothsea_value = "+ 16 GW"
    #  names_values = [
    #  (f"PyPSA value = {pypsa_value}", pypsa_value),
    #  (f"Nortsea_value = {nothsea_value}", nothsea_value)]
    # if "sensitivity_analysis_seq" in config["run"]["name"]:
    #  a_value = "No-seq"
    #  b_value = "2 Mtons/year"
    #  c_value = "5 Mtons/year"
    #  d_value = "10 Mtons/year"
    #  e_value = "No-limit"
    #  names_values = [
    #  (f"No-seq = {a_value}", a_value),
    #  (f"2 Mtons/year = {b_value}", b_value),
    #  (f"5 Mtons/year = {c_value}", c_value),
    #  (f"10 Mtons/year = {d_value}", d_value),
    #  (f"No-limit = {e_value}", e_value)]
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    fig = go.Figure()
    df_transposed = combined_df.T
    df_transposed.index = [scenario_label_map.get(idx, idx) for idx in df_transposed.index]

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    # for name, value in names_values:
    #  fig.add_trace(go.Scatter(
    #     x=[None], 
    #     y=[None], 
    #     mode='markers', 
    #     name=name,
    #     marker=dict(color='rgba(0,0,0,0)')
    # ))
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     fig.update_layout(height=1000, width=1000,title=title, barmode='stack', yaxis=dict(title=unit,title_font=dict(size=15),tickfont=dict(size=15)),xaxis=dict(tickfont=dict(size=15)),legend=dict(font=dict(size=15)))
    else:
     fig.update_layout(height=800, width=600,title=title, barmode='stack', yaxis=dict(title=unit,title_font=dict(size=15),tickfont=dict(size=15)),xaxis=dict(tickfont=dict(size=15)),legend=dict(font=dict(size=15)))
    fig.update_layout(hovermode='y')
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} investment costs.pdf", format='pdf')
    # fig.add_layout_image(logo)
    
    return fig

def scenario_operational_costs(country):
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_operational costs.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_operational costs.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_operational costs.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_operational costs.csv")]
    elif "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_operational costs.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_operational costs.csv")]
    
    costs_sensitivity = {}

    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech', '2030', '2040', '2050']]
      df['Total'] = df[['2030', '2040', '2050']].sum(axis=1)
      df = df[['tech', 'Total']]
      df['Total'] = df['Total'] / 3
      df = df.rename(columns={'Total': name})
      # Store the processed dataframe in the dictionary
      costs_sensitivity[name] = df
    
    combined_df = list(costs_sensitivity.values())[0]
    for df in list(costs_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
    
    unit='Euros/year'
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    title=f'Average Operational Costs Per Year Comparison For {title_country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    # colors["AC Transmission"] = "#FF3030"
    # colors["DC Transmission"] = "#104E8B"
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
    
    # if "sensitivity_analysis_nuclear" in config["run"]["name"]:
    #  Scenario_1 = "8594 EUR/kW_nuclear"
    #  Scenario_2 = "7000 EUR/kW_nuclear"
    #  Scenario_3 = "6000 EUR/kW_nuclear"
    #  Scenario_4 =  "4500 EUR/kW_nuclear"
    #  names_values = [
    #  (f"Scenario_1 = {Scenario_1}", Scenario_1),
    #  (f"Scenario_2 = {Scenario_2}", Scenario_2),
    #  (f"Scenario_3 = {Scenario_3}", Scenario_3),
    #  (f"Scenario_4 = {Scenario_4}", Scenario_4)]
    # if "sensitivity_analysis_offshore" in config["run"]["name"]:
    #  pypsa_value = "+ 0 GW"
    #  nothsea_value = "+ 16 GW"
    #  names_values = [
    #  (f"PyPSA value = {pypsa_value}", pypsa_value),
    #  (f"Nortsea_value = {nothsea_value}", nothsea_value)]
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    fig = go.Figure()
    df_transposed = combined_df.T
    df_transposed.index = [scenario_label_map.get(idx, idx) for idx in df_transposed.index]

    for tech in df_transposed.columns:
      y = df_transposed[tech]
      color = tech_colors.get(tech, 'lightgrey')
      fig.add_trace(go.Bar(
        x=df_transposed.index,
        y=y.where(y > 0, 0),
        name=tech,
        marker_color=color
    ))
      fig.add_trace(go.Bar(
        x=df_transposed.index,
        y=y.where(y < 0, 0),
        name=tech,
        marker_color=color,
        showlegend=False
    ))
    # for name, value in names_values:
    #  fig.add_trace(go.Scatter(
    #     x=[None],
    #     y=[None],
    #     mode='markers',
    #     name=name,
    #     marker=dict(color='rgba(0,0,0,0)')
    # ))
    layout_common = dict(
    title=title,
    barmode='relative',  # changed from 'stack'
    yaxis=dict(title=unit, title_font=dict(size=15), tickfont=dict(size=15)),
    xaxis=dict(tickfont=dict(size=15)),
    legend=dict(font=dict(size=15)),
    hovermode='y'
)
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
      fig.update_layout(height=1000, width=1000, **layout_common)
    else:
      fig.update_layout(height=800, width=600, **layout_common)
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} operational costs.pdf", format='pdf')
    # fig.add_layout_image(logo)
    
    return fig
    
def scenario_cumulative_costs(country):
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_investment costs.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_investment costs.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_investment costs.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_investment costs.csv")]
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_investment costs.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_investment costs.csv")]
    
    elif "sensitivity_analysis_seq" in config["run"]["name"]:
     sensitivity_analyses = [
     ("No-seq", f"results/sensitivity_analysis_seq_0/country_csvs/{country}_investment costs.csv"),
     ("2 Mtons/year", f"results/sensitivity_analysis_seq_2/country_csvs/{country}_investment costs.csv"),
     ("5 Mtons/year", f"results/sensitivity_analysis_seq_5/country_csvs/{country}_investment costs.csv"),
     ("10 Mtons/year", f"results/sensitivity_analysis_seq_10/country_csvs/{country}_investment costs.csv"),
     ("No-limit", f"results/sensitivity_analysis_seq_nolim/country_csvs/{country}_investment costs.csv")]
    # Dictionary to store the processed dataframes
    costs_sensitivity = {}

    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech', '2030', '2040', '2050']]
      df['Total'] = df[['2030', '2040', '2050']].sum(axis=1)
      df = df[['tech', 'Total']]
      df['Total'] = df['Total'] / 3
      df['Total'] = df['Total'] * 30
      df = df.rename(columns={'Total': name})
      # Store the processed dataframe in the dictionary
      costs_sensitivity[name] = df
    
    combined_df = list(costs_sensitivity.values())[0]
    for df in list(costs_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
    
    unit='Euros'
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    title=f'Total Comulative Costs (2020-2050) for {title_country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    # colors["AC Transmission"] = "#FF3030"
    # colors["DC Transmission"] = "#104E8B"
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
    
    # if "sensitivity_analysis_nuclear" in config["run"]["name"]:
    #  Scenario_1 = "8594 EUR/kW_nuclear"
    #  Scenario_2 = "7000 EUR/kW_nuclear"
    #  Scenario_3 = "6000 EUR/kW_nuclear"
    #  Scenario_4 =  "4500 EUR/kW_nuclear"
    #  names_values = [
    #  (f"Scenario_1 = {Scenario_1}", Scenario_1),
    #  (f"Scenario_2 = {Scenario_2}", Scenario_2),
    #  (f"Scenario_3 = {Scenario_3}", Scenario_3),
    #  (f"Scenario_4 = {Scenario_4}", Scenario_4)]
    # if "sensitivity_analysis_offshore" in config["run"]["name"]:
    #  pypsa_value = "+ 0 GW"
    #  nothsea_value = "+ 16 GW"
    #  names_values = [
    #  (f"PyPSA value = {pypsa_value}", pypsa_value),
    #  (f"Nortsea_value = {nothsea_value}", nothsea_value)]
    # if "sensitivity_analysis_seq" in config["run"]["name"]:
    #  a_value = "No-seq"
    #  b_value = "2 Mtons/year"
    #  c_value = "5 Mtons/year"
    #  d_value = "10 Mtons/year"
    #  e_value = "No-limit"
    #  names_values = [
    #  (f"No-seq = {a_value}", a_value),
    #  (f"2 Mtons/year = {b_value}", b_value),
    #  (f"5 Mtons/year = {c_value}", c_value),
    #  (f"10 Mtons/year = {d_value}", d_value),
    #  (f"No-limit = {e_value}", e_value)]
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    fig = go.Figure()
    df_transposed = combined_df.T
    df_transposed.index = [scenario_label_map.get(idx, idx) for idx in df_transposed.index]

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    # for name, value in names_values:
    #  fig.add_trace(go.Scatter(
    #     x=[None], 
    #     y=[None], 
    #     mode='markers', 
    #     name=name,
    #     marker=dict(color='rgba(0,0,0,0)')
    # ))
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     fig.update_layout(height=1000, width=1000,title=title, barmode='stack', yaxis=dict(title=unit,title_font=dict(size=15),tickfont=dict(size=15)),xaxis=dict(tickfont=dict(size=15)),legend=dict(font=dict(size=15)))
    else:
     fig.update_layout(height=800, width=600,title=title, barmode='stack', yaxis=dict(title=unit,title_font=dict(size=15),tickfont=dict(size=15)),xaxis=dict(tickfont=dict(size=15)),legend=dict(font=dict(size=15)))
    fig.update_layout(hovermode='y')
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} cumulative investment costs.pdf", format='pdf')
    # fig.add_layout_image(logo)
    
    return fig   
#%%
def scenario_capacities(country):
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_capacities.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_capacities.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_capacities.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_capacities.csv")]
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_capacities.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_capacities.csv")] 
    elif "sensitivity_analysis_seq" in config["run"]["name"]:
     sensitivity_analyses = [
     ("No-seq", f"results/sensitivity_analysis_seq_0/country_csvs/{country}_capacities.csv"),
     ("2 Mtons/year", f"results/sensitivity_analysis_seq_2/country_csvs/{country}_capacities.csv"),
     ("5 Mtons/year", f"results/sensitivity_analysis_seq_5/country_csvs/{country}_capacities.csv"),
     ("10 Mtons/year", f"results/sensitivity_analysis_seq_10/country_csvs/{country}_capacities.csv"),
     ("No-limit", f"results/sensitivity_analysis_seq_nolim/country_csvs/{country}_capacities.csv")]
    # Dictionary to store the processed dataframes
    capacity_sensitivity = {}
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech','2050']]
      df = df.rename(columns={'2050': name})
      # Store the processed dataframe in the dictionary
      capacity_sensitivity[name] = df
    
    combined_df = list(capacity_sensitivity.values())[0]
    for df in list(capacity_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
     combined_df.columns = [scenario_label_map.get(col, col) for col in combined_df.columns]
    
    unit='Capacity [GW]'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    # colors["AC Transmission"] = "#FF3030"
    # colors["DC Transmission"] = "#104E8B"
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
    
    fig = go.Figure()
    fig = go.Figure()
    groups = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["power-to-heat"],
        ["power-to-gas"],
        ["AC Transmission lines"],
        ["DC Transmission lines"],
        ["CCGT"],
        ["nuclear"],
    ]
    groupss = [
        ["solar"],
        ["onshore wind", "offshore wind"],
        ["power-to-heat"],
        ["power-to-gas"],
        ["transmission lines"],
        ["gas pipeline","gas pipeline new"],
        ["CCGT"],
        ["nuclear"],
    ]
    if country != "EU":
        value = groups
    else:
        value = groupss
    def smart_capitalize(phrase):
     return phrase[0].upper() + phrase[1:] if phrase and not phrase[0].isupper() else phrase

    fig = make_subplots(rows=2, cols=len(value) // 2, subplot_titles=[", ".join(smart_capitalize(t) for t in tech_group) for tech_group in value], shared_yaxes=True)

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
                name=smart_capitalize(tech),
                marker_color=tech_colors.get(tech, 'gray')
            )
            fig.add_trace(trace, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=unit, row=2, col=1)

    # Update layout
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {title_country} in 2050 compared for all scenarios", yaxis_title=unit,legend=dict(font=dict(size=15)))
    else:
     fig.update_layout(height=800, width=1000, showlegend=True, title=f"Capacities for {title_country} in 2050 compared for all scenarios", yaxis_title=unit,legend=dict(font=dict(size=15)))
    logo['y']=1.021
    num_cols = len(value) // 2 + (len(value) % 2 > 0)
    for row in [1, 2]:
     for col in range(1, num_cols + 1):
        fig.update_xaxes(tickfont=dict(size=13), row=row, col=col)
        fig.update_yaxes(
            tickfont=dict(size=13),
            title_font=dict(size=15),
            row=row,
            col=col
        )
    # fig.add_layout_image(logo)
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} capacities.pdf", format='pdf')
    return fig

def storage_capacities(country):
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     sensitivity_analyses = [
     ("Scenario_1", f"results/sensitivity_analysis_nuclear_pypsa/country_csvs/{country}_storage_capacities.csv"),
     ("Scenario_2", f"results/sensitivity_analysis_nuclear_fps/country_csvs/{country}_storage_capacities.csv"),
     ("Scenario_3", f"results/sensitivity_analysis_nuclear_mackinze/country_csvs/{country}_storage_capacities.csv"),
     ("Scenario_4", f"results/sensitivity_analysis_nuclear_energyville/country_csvs/{country}_storage_capacities.csv")]
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     sensitivity_analyses = [
     ("PyPSA_Suff", f"results/sensitivity_analysis_offshore_pypsa/country_csvs/{country}_storage_capacities.csv"),
     ("Northsea_Capacity", f"results/sensitivity_analysis_offshore_northsea/country_csvs/{country}_storage_capacities.csv")]
    elif "sensitivity_analysis_seq" in config["run"]["name"]:
     sensitivity_analyses = [
     ("No-seq", f"results/sensitivity_analysis_seq_0/country_csvs/{country}_storage_capacities.csv"),
     ("2 Mtons/year", f"results/sensitivity_analysis_seq_2/country_csvs/{country}_storage_capacities.csv"),
     ("5 Mtons/year", f"results/sensitivity_analysis_seq_5/country_csvs/{country}_storage_capacities.csv"),
     ("10 Mtons/year", f"results/sensitivity_analysis_seq_10/country_csvs/{country}_storage_capacities.csv"),
     ("No-limit", f"results/sensitivity_analysis_seq_nolim/country_csvs/{country}_storage_capacities.csv")]
    
    # Dictionary to store the processed dataframes
    storeage_capacity_sensitivity = {}
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     scenario_label_map = {
        "Scenario_1": "8594 EUR/kW_nuclear",
        "Scenario_2": "7000 EUR/kW_nuclear",
        "Scenario_3": "6000 EUR/kW_nuclear",
        "Scenario_4": "4500 EUR/kW_nuclear"}
    if "sensitivity_analysis_offshore" in config["run"]["name"]:
     scenario_label_map = {
        "PyPSA_Suff": "+ 0 GW",
        "Northsea_Capacity": "+ 16 GW"}
    # Process each sensitivity analysis
    for name, file_path in sensitivity_analyses:
      # Read the CSV file
      df = pd.read_csv(file_path)
      df = df[['tech','2050']]
      df = df.rename(columns={'2050': name})
      # Store the processed dataframe in the dictionary
      storeage_capacity_sensitivity[name] = df
    
    combined_df = list(storeage_capacity_sensitivity.values())[0]
    for df in list(storeage_capacity_sensitivity.values())[1:]:
     combined_df = pd.merge(combined_df, df, on='tech', how='outer')
     combined_df = combined_df.fillna(0)
     combined_df = combined_df.set_index('tech')
     combined_df.columns = [scenario_label_map.get(col, col) for col in combined_df.columns]
    
    unit='Capacity [GWh]'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["Thermal Energy storage"] = colors["urban central water tanks"]
    colors["Grid-scale"] = 'green'
    colors["home battery"] = 'blue'
    colors["Gas storage"] = colors["gas"]
    
    fig = go.Figure()
    groups = [
        ["Grid-scale battery"],
        ["Thermal Energy storage"],
        ["Gas storage"],
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
    if country == "EU":
     title_country = "5 Countries"
    else:
     title_country = country
    if "sensitivity_analysis_nuclear" in config["run"]["name"]:
     fig.update_layout(height=600, width=1400, showlegend=True, title=f"Storage capacities for {title_country} in 2050 compared for all scenarios", yaxis_title=unit,legend=dict(font=dict(size=15)))
    else:
     fig.update_layout(height=500, width=1200, showlegend=True, title=f"Storage capacities for {title_country} in 2050 compared for all scenarios", yaxis_title=unit,legend=dict(font=dict(size=15)))
    logo['y']=1.021
    num_cols = len(groups)
    for col in range(1, num_cols + 1):
     fig.update_xaxes(tickfont=dict(size=13), row=1, col=col)
     fig.update_yaxes(
        tickfont=dict(size=13),
        title_font=dict(size=15),
        row=1,
        col=col
    )
    if country == 'BE':
     pio.write_image(fig, f"results/pdf/{study} storage capacities.pdf", format='pdf')
    # fig.add_layout_image(logo)
    return fig

    
def create_combined_scenario_chart_country(country):
    def load_html_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        return html_content
    # Create output folder if it doesn't exist
    if "offshore" in config["run"]["name"]:
        output_folder='results/sensitivity_results_offshore/'
    else:
        output_folder='results/sensitivity_results/'
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"
    display_country = "5 Countries" if country == "EU" else country
    # Create bar chart
    bar_chart = scenario_costs(country)
    combined_html += f"<div><h2>{display_country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    
    bar_chart_investment = scenario_investment_costs(country)
    combined_html += f"<div><h2>{display_country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    
    bar_chart_operational = scenario_operational_costs(country)
    combined_html += f"<div><h2>{display_country} - Annual Operational Costs</h2>{bar_chart_operational.to_html()}</div>"
    
    bar_chart_cumulative = scenario_cumulative_costs(country)
    combined_html += f"<div><h2>{display_country} - Cummulative Investment Costs (2020-2050)</h2>{bar_chart_cumulative.to_html()}</div>"
    
    # Create capacities chart
    capacities_chart = scenario_capacities(country)
    combined_html += f"<div><h2>{display_country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    
    # Create capacities chart
    storage_capacities_chart = storage_capacities(country)
    combined_html += f"<div><h2>{display_country} -  Storage Capacities</h2>{storage_capacities_chart.to_html()}</div>"

    combined_html += "</body></html>"
    table_of_contents_content = ""
    if "offshore" in config["run"]["name"]:
     html_filnemae = 'SEPIA/html_sensitivity_offshore_' + country + '.html'
    else:
     html_filnemae = 'SEPIA/html_sensitivity_nuclear_' + country + '.html' 
    if os.path.exists(html_filnemae):
        main_content = load_html_file(html_filnemae)
    else:
        main_content = ''
    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content += f"<a href='#{display_country} - Annual Costs'>Annual Costs</a><br>"
    table_of_contents_content += f"<a href='#{display_country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    table_of_contents_content += f"<a href='#{display_country} - Annual Operational Costs'>Annual Operational Costs</a><br>"
    table_of_contents_content += f"<a href='#{display_country} - Cummulative Investment Costs (2020-2050)'>Cummulative Investment Costs (2020-2050)</a><br>"
    table_of_contents_content += f"<a href='#{display_country} - Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{display_country} - Storage Capacities'>Storage Capacities</a><br>"

    # Add more links for other plots
    main_content += f"<div id='{display_country} - Annual Costs'><h2>{display_country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    main_content += f"<div id='{display_country} - Annual Investment Costs'><h2>{display_country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    main_content += f"<div id='{display_country} - Annual Operational Costs'><h2>{display_country} - Annual Operational Costs</h2>{bar_chart_operational.to_html()}</div>"
    main_content += f"<div id='{display_country} - Cummulative Investment Costs (2020-2050)'><h2>{display_country} - Cummulative Investment Costs (2020-2050)</h2>{bar_chart_cumulative.to_html()}</div>"
    main_content += f"<div id='{display_country} - Capacities'><h2>{display_country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    main_content += f"<div id='{display_country} - Storage Capacities'><h2>{display_country} - Storage Capacities</h2>{storage_capacities_chart.to_html()}</div>"
    # Add more content for other plots
    
    template_path =  snakemake.input.template
    with open(template_path, "r") as template_file:
        template_content = template_file.read()
        template = Template(template_content)
        
    rendered_html = template.render(
    title=f"{country} - Sensitivity Scenario Plots",
    country=country,
    TABLE_OF_CONTENTS=table_of_contents_content,
    MAIN=main_content,)
    if "offshore" in config["run"]["name"]:
     combined_file_path = os.path.join(output_folder, f"sensitivity_offshore_{country}.html")
    else:
     combined_file_path = os.path.join(output_folder, f"sensitivity_nuclear_{country}.html")
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
    study = snakemake.config["run"]["name"]
    logo = logo()
    for country in countries:
        create_combined_scenario_chart_country(country)