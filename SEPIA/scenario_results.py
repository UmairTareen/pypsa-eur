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
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    fig.add_layout_image(logo)
    
    return fig

def scenario_investment_costs(country):
    costs_bau = pd.read_csv(f"results/bau/country_csvs/{country}_investment costs.csv")
    costs_suff = pd.read_csv(f"results/ncdr/country_csvs/{country}_investment costs.csv")
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
    fig.update_layout(title=title, barmode='stack', yaxis=dict(title=unit))
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

    # Create capacities chart
    capacities_chart = scenario_capacities(country)
    combined_html += f"<div><h2>{country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    
    # Create demands chart
    demands_chart = scenario_demands(country)
    combined_html += f"<div><h2>{country} - Sectoral Demands</h2>{demands_chart.to_html()}</div>"

    combined_html += "</body></html>"
    table_of_contents_content = ""
    main_content = ""
    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content += f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{country} - Sectoral Demands'>Sectoral Demands</a><br>"

    # Add more links for other plots
    main_content += f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    main_content += f"<div id='{country} - Capacities'><h2>{country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Sectoral Demands'><h2>{country} - Sectoral Demands</h2>{demands_chart.to_html()}</div>"
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
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_scenarios")
        
    countries = snakemake.params.countries 
    config = snakemake.config
    logo = logo()
    for country in countries:
        create_combined_scenario_chart_country(country)
        
    
 
