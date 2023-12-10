#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:59:23 2023

@author: umair
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import cartopy.crs as ccrs
import plotly.express as px
import plotly.subplots as sp
import yaml
import os

with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)
  
countries = ['BE', 'DE', 'FR', 'GB', 'NL']
tech_colors = config["plotting"]["tech_colors"]

def scenario_costs():
 for country in countries:
    costs_bau = pd.read_csv(f"csvs/{country}_costs_bau.csv")
    # costs_suff = pd.read_csv(f"csvs/{country}_costs_suff.csv")
    costs_ncdr = pd.read_csv(f"csvs/{country}_costs_suff.csv")
    costs_reff = costs_bau[['tech', '2020']]
    costs_bau = costs_bau[['tech', '2030', '2040', '2050']]
    # costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    costs_ncdr = costs_ncdr[['tech', '2030', '2040', '2050']]
    
    costs_reff = costs_reff.rename(columns={'2020': 'Reference'})
    
    costs_bau['Total'] = costs_bau[['2030', '2040', '2050']].sum(axis=1)
    costs_bau = costs_bau[['tech', 'Total']]
    costs_bau['Total'] = costs_bau['Total'] / 3
    costs_bau = costs_bau.rename(columns={'Total': 'BAU'})
    
    # costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    # costs_suff = costs_suff[['tech', 'Total']]
    # costs_suff['Total'] = costs_suff['Total'] / 3
    # costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    costs_ncdr['Total'] = costs_ncdr[['2030', '2040', '2050']].sum(axis=1)
    costs_ncdr = costs_ncdr[['tech', 'Total']]
    costs_ncdr['Total'] = costs_ncdr['Total'] / 3
    costs_ncdr = costs_ncdr.rename(columns={'Total': 'Sufficienty'})
    
    combined_df = pd.merge(costs_reff, costs_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    # combined_df = pd.merge(combined_df, costs_suff, on='tech', how='outer')
    combined_df = pd.merge(combined_df, costs_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Billion Euros/year'
    title=f'Total Costs Comparison for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))

    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    
    return fig
    
    
#%%
def scenario_capacities():
 for country in countries:
    caps_bau = pd.read_csv(f"csvs/{country}_capacities_bau.csv")
    # caps_suff = pd.read_csv(f"csvs/{country}_capacities_suff.csv")
    caps_ncdr = pd.read_csv(f"csvs/{country}_capacities_suff.csv")
    caps_reff = caps_bau[['tech', '2020']]
    caps_bau = caps_bau[['tech', '2030', '2040', '2050']]
    # caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    caps_ncdr = caps_ncdr[['tech', '2030', '2040', '2050']]
    
    caps_reff = caps_reff.rename(columns={'2020': 'Reff'})
    
    caps_bau = caps_bau[['tech', '2050']]
    caps_bau = caps_bau.rename(columns={'2050': 'BAU'})
    
    # caps_suff['Total'] = caps_suff[['2030', '2040', '2050']].sum(axis=1)
    # caps_suff = caps_suff[['tech', 'Total']]
    # caps_suff['Total'] = caps_suff['Total'] / 3
    # caps_suff = caps_suff.rename(columns={'Total': 'Suff'})
    
    caps_ncdr = caps_ncdr[['tech', '2050']]
    caps_ncdr = caps_ncdr.rename(columns={'2050': 'Ncdr'})
    
    combined_df = pd.merge(caps_reff, caps_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    # combined_df = pd.merge(combined_df, caps_suff, on='tech', how='outer')
    combined_df = pd.merge(combined_df, caps_ncdr, on='tech', how='outer', suffixes=('_suff', '_ncdr'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Capacity [GW]'
    title=f"Capacities for {country}"
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    
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

    fig = make_subplots(rows=2, cols=len(groups) // 2, subplot_titles=[
        f"{', '.join(tech_group)}" for tech_group in groups], shared_yaxes=True)

    df = combined_df

    for i, tech_group in enumerate(groups, start=1):
        row_idx = 1 if i <= len(groups) // 2 else 2
        col_idx = i if i <= len(groups) // 2 else i - len(groups) // 2

        for tech in tech_group:
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
    fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {country}_2050", yaxis_title=unit)
    
    return fig

def create_combined_scenario_chart_country(country, output_folder='scenario_charts'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"

    # Create bar chart
    bar_chart = scenario_costs()
    combined_html += f"<div><h2>{country} - Bar Chart</h2>{bar_chart.to_html()}</div>"

    # Create capacities chart
    capacities_chart = scenario_capacities()
    combined_html += f"<div><h2>{country} - Capacities Chart</h2>{capacities_chart.to_html()}</div>"

    combined_html += "</body></html>"

    # Save the combined HTML file
    combined_file_path = os.path.join(output_folder, f"{country}_combined_scenario_chart.html")
    with open(combined_file_path, "w") as combined_file:
        combined_file.write(combined_html)

# Example usage
for country in countries:
    create_combined_scenario_chart_country(country)

