#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:59:23 2023

@author: umair
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import os
import shutil
from datetime import datetime

prepare_folder_website = True

def scenario_costs(country):
    costs_bau = pd.read_csv(f"../results/csvs/{country}_costs_bau.csv")
    costs_suff = pd.read_csv(f"../results/csvs/{country}_costs_ncdr.csv")
    # costs_ncdr = pd.read_csv(f"csvs/{country}_costs_ncdr.csv")
    costs_reff = costs_bau[['tech', '2020']]
    costs_bau = costs_bau[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    # costs_ncdr = costs_ncdr[['tech', '2030', '2040', '2050']]
    
    costs_reff = costs_reff.rename(columns={'2020': 'Reff'})
    
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
    
    combined_df = pd.merge(costs_reff, costs_bau, on='tech', how='outer', suffixes=('_reff', '_bau'))
    combined_df = pd.merge(combined_df, costs_suff, on='tech', how='outer')
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

    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    
    return fig
    
    
#%%
def scenario_capacities(country):
    caps_bau = pd.read_csv(f"../results/csvs/{country}_capacities_bau.csv")
    caps_suff = pd.read_csv(f"../results/csvs/{country}_capacities_ncdr.csv")
    # caps_ncdr = pd.read_csv(f"csvs/{country}_capacities_ncdr.csv")
    caps_reff = caps_bau[['tech', '2020']]
    caps_bau = caps_bau[['tech', '2030', '2040', '2050']]
    caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    # caps_ncdr = caps_ncdr[['tech', '2030', '2040', '2050']]
    
    caps_reff = caps_reff.rename(columns={'2020': 'Reff'})
    
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
    fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {country}_2050 compared to 2020", yaxis_title=unit)
    
    return fig

def create_combined_scenario_chart_country(country, output_folder='../results/pypsa_results/scenario'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>Combined Plots</title></head><body>"

    # Create bar chart
    bar_chart = scenario_costs(country)
    combined_html += f"<div><h2>{country} - Bar Chart</h2>{bar_chart.to_html()}</div>"

    # Create capacities chart
    capacities_chart = scenario_capacities(country)
    combined_html += f"<div><h2>{country} - Capacities Chart</h2>{capacities_chart.to_html()}</div>"

    combined_html += "</body></html>"

    # Save the combined HTML file
    combined_file_path = os.path.join(output_folder, f"{country}_combined_scenario_chart.html")
    with open(combined_file_path, "w") as combined_file:
        combined_file.write(combined_html)




if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_scenarios")
        
    countries = snakemake.params.countries 
    config = snakemake.config
    
    for country in countries:
        create_combined_scenario_chart_country(country)
        
#%%

if prepare_folder_website == True:  
    
 def create_website_files(source_directories, target_directory, new_names, files_to_delete):
    # Get the current date
    current_date = datetime.now()

    # Format the date in the "year_month_day" format
    folder_name = current_date.strftime("%Y%m%d")

    # Create a new folder with the formatted name in the target directory
    new_folder_path = os.path.join(target_directory, folder_name)
    try:
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully in '{target_directory}'.")
    except OSError as e:
        print(f"Error creating folder: {e}")
        return

    # Copy, rename, and delete files in each source folder
    for source_folder, new_name in zip(source_directories, new_names):
        try:
            source_folder_name = os.path.basename(source_folder)
            destination_folder_name = f"{new_name}"
            destination_folder_path = os.path.join(new_folder_path, destination_folder_name)

            shutil.copytree(source_folder, destination_folder_path)
            print(f"Folder '{source_folder_name}' copied and renamed to '{destination_folder_name}' in '{new_folder_path}'.")

            # Delete specified files inside the copied folder
            for file_to_delete in files_to_delete:
                file_path = os.path.join(destination_folder_path, file_to_delete)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"File '{file_to_delete}' deleted from '{destination_folder_name}'.")
                else:
                    print(f"File '{file_to_delete}' not found in '{destination_folder_name}'.")
        except shutil.Error as e:
            print(f"Error copying folder '{source_folder}': {e}")
 

 if __name__ == "__main__":
    # Specify the list of source directories, the target directory, new names, and files to delete
    source_directories = ["../results/pypsa_results", "../results/bau/htmls", "../results/ncdr/htmls"]
    target_directory = "../results/" 
    new_names = ["Pypsa_results", "bau", "suff"]  # Replace with the desired new names
    files_to_delete = ["ChartData_BE.xlsx","ChartData_DE.xlsx","ChartData_FR.xlsx","ChartData_GB.xlsx", "ChartData_NL.xlsx","ChartData_EU.xlsx"]  # Specify the names of files to delete

    create_website_files(source_directories, target_directory, new_names, files_to_delete)

    
 