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
import yaml
import json
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

def Cumulative_emissions_sector(country):
 # Load configuration data
 file = snakemake.input.sepia_config
 config = pd.read_excel(file, sheet_name="NODES", index_col=0)
 config = config[config['Type'] == 'GHG_SECTORS']
 code_to_label = config['Label'].to_dict()

 # Load and preprocess data
 ref = pd.read_csv(f"results/ref/country_csvs/ghg_sector_cum_{country}.csv", index_col=0)
 suff = pd.read_csv(f"results/suff/country_csvs/ghg_sector_cum_{country}.csv", index_col=0)

 # Rename and preprocess
 ref.rename(columns=code_to_label, inplace=True)
 ref['Industry'] = ref[['Other energy industry', 'Industrial processes', 'Fuel usage - industry']].sum(axis=1)
 ref = ref.drop(columns=['Other energy industry', 'Industrial processes', 'Fuel usage - industry'])
 ref = ref.rename(columns={"Fuel combustion - agriculture": "Agriculture", "Fuel combustion - transport": "Transport", "Fuel combustion - aviation bunkers": "Aviation bunkers", "DAC":"DACCS","Fuel combustion - maritime bunkers":"Maritime bunkers","biogas to gas": "Biogas","Fuel combustion – residential and tertiary":"Residential and tertiary sectors"})
 ref = ref.loc[:, (ref != 0).any(axis=0)]
 ref['Total'] = ref.sum(axis=1)
 ref_cumulative = ref.drop(columns='Total').cumsum()
 ref_cumulative_positive = ref_cumulative.where(ref_cumulative > 0, 0)
 ref_cumulative_negative = ref_cumulative.where(ref_cumulative < 0, 0)
 ref_cumulative_positive = ref_cumulative_positive.loc[:, (ref_cumulative_positive != 0).any()]
 ref_cumulative_negative = ref_cumulative_negative.loc[:, (ref_cumulative_negative != 0).any()]
 ref_total = ref['Total'].cumsum()

 # Rename columns and preprocess ncdr
 suff.rename(columns=code_to_label, inplace=True)
 suff['Industry'] =suff[['Other energy industry', 'Industrial processes', 'Fuel usage - industry']].sum(axis=1)
 suff = suff.drop(columns=['Other energy industry', 'Industrial processes', 'Fuel usage - industry'])
 suff = suff.rename(columns={"Fuel combustion - agriculture": "Agriculture", "Fuel combustion - transport": "Transport", "Fuel combustion - aviation bunkers": "Aviation bunkers", "DAC":"DACCS","Fuel combustion - maritime bunkers":"Maritime bunkers","biogas to gas": "Biogas","Fuel combustion – residential and tertiary":"Residential and tertiary sectors"})
 suff = suff.loc[:, (suff != 0).any(axis=0)]
 suff['Total'] = suff.sum(axis=1)
 suff_cumulative = suff.drop(columns='Total').cumsum()
 suff_cumulative_positive = suff_cumulative.where(suff_cumulative > 0, 0)
 suff_cumulative_negative = suff_cumulative.where(suff_cumulative < 0, 0)
 suff_cumulative_positive = suff_cumulative_positive.loc[:, (suff_cumulative_positive != 0).any()]
 suff_cumulative_negative = suff_cumulative_negative.loc[:, (suff_cumulative_negative != 0).any()]
 suff_total = suff['Total'].cumsum() 
 

 colors = {
        "Agriculture": "#008556",
        "Industry": "#feda47",
        "Transport": "#a26643",
        "Residential and tertiary sectors": "#d60a51",
        "Maritime bunkers": "#f18959",
        "Aviation bunkers": "#ff4d00",
        "BECCS": "#889717",
        "Biogas": "#dfeac2",
        "Biomass": "green",
        "DACCS": "#b1d1fc",
        "Heat and power production ": "#75519c",
        "Land use and forestry": "#befdb7",
    }

    # Create subplots
 fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Reference Scenario",
            "Sufficiency Scenario"
        ]
    )

    # Add Reference scenario area plot
 for col in ref_cumulative_positive.columns:
        fig.add_trace(
            go.Scatter(
                x=ref_cumulative_positive.index,
                y=ref_cumulative_positive[col],  # Positive values
                mode='lines',
                fill='tonexty',
                name=col,
                line=dict(color=colors.get(col, "#000000")),
                stackgroup='positive_ref',  # Unique stackgroup for positive values
                legendgroup=col,
                showlegend=True  # Show legend for the first instance
            ),
            row=1, col=1
        )
 for col in ref_cumulative_negative.columns:
        fig.add_trace(
            go.Scatter(
                x=ref_cumulative_negative.index,
                y=ref_cumulative_negative[col],  # Negative values
                mode='lines',
                fill='tonexty',
                name=col,
                line=dict(color=colors.get(col, "#000000")),
                stackgroup='negative_ref',  # Unique stackgroup for negative values
                legendgroup=col,
                showlegend=True  # Avoid duplicate legends
            ),
            row=1, col=1
        )
 fig.add_trace(
        go.Scatter(
            x=ref_total.index,
            y=ref_total,
            mode='lines',
            name="Total",
            line=dict(color='black', width=2),
            legendgroup="Total",
            showlegend=True
        ),
        row=1, col=1
    )
    # Add Sufficiency scenario area plot
 for col in suff_cumulative_positive.columns:
        fig.add_trace(
            go.Scatter(
                x=suff_cumulative_positive.index,
                y=suff_cumulative_positive[col],  # Positive values
                mode='lines',
                fill='tonexty',
                name=col,
                line=dict(color=colors.get(col, "#000000")),
                stackgroup='positive_suff',  # Unique stackgroup for positive values
                legendgroup=col,
                showlegend=False  # Avoid duplicate legends
            ),
            row=1, col=2
        )
 for col in suff_cumulative_negative.columns:
        fig.add_trace(
            go.Scatter(
                x=suff_cumulative_negative.index,
                y=suff_cumulative_negative[col],  # Negative values
                mode='lines',
                fill='tonexty',
                name=col,
                line=dict(color=colors.get(col, "#000000")),
                stackgroup='negative_suff',  # Unique stackgroup for negative values
                legendgroup=col,
                showlegend=False  # Avoid duplicate legends
            ),
            row=1, col=2
        )
 fig.add_trace(
        go.Scatter(
            x=suff_total.index,
            y=suff_total,
            mode='lines',
            name="Total",
            line=dict(color='black', width=2),
            legendgroup="Total",
            showlegend=False
        ),
        row=1, col=2
    )
    # Update layout
 fig.update_layout(
        title="Cumulative Emissions by Sector",
        xaxis=dict(
        title="Year",
        tickvals=[2020, 2030, 2040, 2050],),
        xaxis2=dict(
        title="Year",
        tickvals=[2020, 2030, 2040, 2050],),
        yaxis=dict(
        title="Cumulative Emissions (MtCO2 eq)"),
        yaxis2=dict(
        matches='y',  # Synchronize y-axis2 with y-axis
        title="Cumulative Emissions (MtCO2 eq)"), # Optional: Add this if needed),
        height=700,
        width=1400,
        legend=dict(orientation="h", y=-0.2),  # Position legend below
        template="plotly_white"  # Optional: Clean aesthetic
    )
 if country == 'BE':
  pio.write_image(fig, "results/pdf/Cumulative Emissions by Sector.pdf", format='pdf')
 return fig


def scenario_costs(country):
    costs_ref = pd.read_csv(f"results/ref/country_csvs/{country}_costs.csv")
    costs_suff = pd.read_csv(f"results/suff/country_csvs/{country}_costs.csv")
    
    costs_ref = costs_ref[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    
    costs_ref['Total'] = costs_ref[['2030', '2040', '2050']].sum(axis=1)
    costs_ref = costs_ref[['tech', 'Total']]
    costs_ref['Total'] = costs_ref['Total'] / 3
    costs_ref = costs_ref.rename(columns={'Total': 'Ref'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    combined_df = pd.merge(costs_suff, costs_ref, on='tech', how='outer', suffixes=('_Suff', '_Ref'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Euros/year'
    title=f'Total Costs Comparison for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=800,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Total Costs.pdf", format='pdf')
    fig.add_layout_image(logo)
    
    return fig

def scenario_investment_costs(country):
    costs_ref = pd.read_csv(f"results/ref/country_csvs/{country}_investment costs.csv")
    costs_suff = pd.read_csv(f"results/suff/country_csvs/{country}_investment costs.csv")
    
    costs_ref = costs_ref[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    
    costs_ref['Total'] = costs_ref[['2030', '2040', '2050']].sum(axis=1)
    costs_ref = costs_ref[['tech', 'Total']]
    costs_ref['Total'] = costs_ref['Total'] / 3
    costs_ref = costs_ref.rename(columns={'Total': 'Ref'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    combined_df = pd.merge(costs_suff, costs_ref, on='tech', how='outer', suffixes=('_Suff', '_ref'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Euros/year'
    title=f'Total Investment Costs Comparison for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=800,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Investment Costs.pdf", format='pdf')
    fig.add_layout_image(logo)
    
    return fig
    
def scenario_cumulative_costs(country):
    costs_ref = pd.read_csv(f"results/ref/country_csvs/{country}_investment costs.csv")
    costs_suff = pd.read_csv(f"results/suff/country_csvs/{country}_investment costs.csv")
    
    costs_ref = costs_ref[['tech', '2030', '2040', '2050']]
    costs_suff = costs_suff[['tech', '2030', '2040', '2050']]
    
    costs_ref['Total'] = costs_ref[['2030', '2040', '2050']].sum(axis=1)
    costs_ref = costs_ref[['tech', 'Total']]
    costs_ref['Total'] = costs_ref['Total'] / 3
    costs_ref['Total'] = costs_ref['Total'] * 30
    costs_ref = costs_ref.rename(columns={'Total': 'Ref'})
    
    costs_suff['Total'] = costs_suff[['2030', '2040', '2050']].sum(axis=1)
    costs_suff = costs_suff[['tech', 'Total']]
    costs_suff['Total'] = costs_suff['Total'] / 3
    costs_suff['Total'] = costs_suff['Total'] * 30
    costs_suff = costs_suff.rename(columns={'Total': 'Suff'})
    
    combined_df = pd.merge(costs_suff, costs_ref, on='tech', how='outer', suffixes=('_Suff', '_ref'))
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Euros'
    title=f'Total Comulative Costs (2020-2050) for {country}'
    tech_colors = config["plotting"]["tech_colors"]
    
    fig = go.Figure()
    df_transposed = combined_df.T

    for tech in df_transposed.columns:
        fig.add_trace(go.Bar(x=df_transposed.index, y=df_transposed[tech], name=tech, marker_color=tech_colors.get(tech, 'lightgrey')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Euro reference value = 2020', marker=dict(color='rgba(0,0,0,0)')))
    # Configure layout and labels
    fig.update_layout(height=1000, width=800, showlegend=True,title=title, barmode='stack', yaxis=dict(title=unit))
    fig.update_layout(hovermode='y')
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Cumulative Costs.pdf", format='pdf')
    fig.add_layout_image(logo)
    
    return fig   
#%%
def scenario_capacities(country):
    caps_ref = pd.read_csv(f"results/ref/country_csvs/{country}_capacities.csv")
    caps_suff = pd.read_csv(f"results/suff/country_csvs/{country}_capacities.csv")
    
    caps_baseline = caps_ref[['tech', '2020']]
    caps_ref = caps_ref[['tech', '2030', '2040', '2050']]
    caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    
    caps_ref = caps_ref[['tech', '2050']]
    caps_ref = caps_ref.rename(columns={'2050': 'Ref'})
    
    caps_suff = caps_suff[['tech', '2050']]
    caps_suff = caps_suff.rename(columns={'2050': 'Suff'})
    
    combined_df = pd.merge(caps_baseline, caps_ref, on='tech', how='outer', suffixes=('_baseline', '_ref'))
    combined_df = pd.merge(combined_df, caps_suff, on='tech', how='outer')
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Capacity [GW]'
    title=f"Capacities for {country}"
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission lines"] = "#FF3030"
    colors["DC Transmission lines"] = "#104E8B"
    
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
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Capacities.pdf", format='pdf')
    fig.add_layout_image(logo)
    return fig

def storage_capacities(country):
    caps_ref = pd.read_csv(f"results/ref/country_csvs/{country}_storage_capacities.csv")
    caps_suff = pd.read_csv(f"results/suff/country_csvs/{country}_storage_capacities.csv")
    caps_baseline = caps_ref[['tech', '2020']]
    caps_ref = caps_ref[['tech', '2030', '2040', '2050']]
    caps_suff = caps_suff[['tech', '2030', '2040', '2050']]
    
    caps_ref = caps_ref[['tech', '2050']]
    caps_ref = caps_ref.rename(columns={'2050': 'Ref'})
    
    caps_suff = caps_suff[['tech', '2050']]
    caps_suff = caps_suff.rename(columns={'2050': 'Suff'})
    
    combined_df = pd.merge(caps_baseline, caps_ref, on='tech', how='outer', suffixes=('_baseline', '_ref'))
    combined_df = pd.merge(combined_df, caps_suff, on='tech', how='outer')
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.set_index('tech')
    
    unit='Capacity [GWh]'
    title=f"Storage Capacities for {country}"
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["Thermal Energy storage"] = colors["urban central water tanks"]
    colors["Grid-scale"] = 'green'
    
    fig = go.Figure()
    groups = [
        ["Grid-scale battery", "V2G"],
        ["Thermal Energy storage"],
        ["gas"],
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
    logo['y']=1.05
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Storage Capacities.pdf", format='pdf')
    fig.add_layout_image(logo)
    return fig

def crabon_capture_techs(country):
    carbon_ref = pd.read_excel(f"results/ref/sepia/inputs{country}.xlsx", sheet_name="Inputs_co2", index_col=2)
    carbon_suff = pd.read_excel(f"results/suff/sepia/inputs{country}.xlsx", sheet_name="Inputs_co2", index_col=2)
    carbon_ref.drop(columns=['label', 'source','2030', '2040'], inplace=True)
    carbon_suff.drop(columns=['label', 'source','2020', '2030', '2040'], inplace=True)
    rename_dict = {
        'emmprocessccst': 'CC',
        'emmdac': 'DAC',
        'emmindbeccs': 'CC',
        'emmbmchpcc': 'CC',
        'emmbiogasatcc': 'CC',
        'emmgasccx': 'CC',
        'emmseq': 'CCS',
        'emmfischer': 'CCU',
        'emmmet': 'CCU',
        'emmsaba': 'CCU',
        'emmsmrcc': 'CC',
        'emmbmsngccca': 'CC',
        'emmgaschpatmcc': 'CC'
    }

    carbon_ref.rename(index=rename_dict, inplace=True)
    carbon_suff.rename(index=rename_dict, inplace=True)
    carbon_ref = carbon_ref[carbon_ref.index.isin(rename_dict.values())]
    carbon_suff = carbon_suff[carbon_suff.index.isin(rename_dict.values())]
    carbon_ref = carbon_ref.groupby(level=0).sum()
    carbon_suff = carbon_suff.groupby(level=0).sum()
    carbon_ref.rename(columns={'2050': 'Ref'}, inplace=True)
    carbon_suff.rename(columns={'2050': 'Suff'}, inplace=True)
    carbon_combined = pd.concat([carbon_ref, carbon_suff], axis=1)
    carbon_combined.fillna(0, inplace=True)
    carbon_combined = carbon_combined.round(1)
    desired_order = ['CCU', 'CC', 'DAC', 'CCS', ]  # Define your preferred order
    carbon_combined = carbon_combined.reindex(desired_order)

    color_palette = {
        'CC': '#f2385a',
        'DAC': '#ff5270',
        'CCS': '#f29dae',
        'CCU': '#B98F76',
    }
    num_pies = len(carbon_combined.columns)
    fig = make_subplots(
    rows=1, 
    cols=1 + num_pies,
    column_widths=[0.5] + [0.5/num_pies]*num_pies,
    specs=[[{"type": "bar"}]+[{"type": "pie"}]*num_pies],
    horizontal_spacing=0.01)
    for category in carbon_combined.index:
     fig.add_trace(
        go.Bar(
            name=category, 
            x=carbon_combined.columns,
            y=carbon_combined.loc[category],
            marker_color=color_palette[category],
            width=0.15
        ),
        row=1, col=1
    )

    for idx, column in enumerate(carbon_combined.columns, start=2):  # Start at col 2
     fig.add_trace(
        go.Pie(
            labels=carbon_combined.index,
            values=carbon_combined[column],
            marker=dict(colors=[color_palette[cat] for cat in carbon_combined.index]),
            name=column,
            textinfo="label+percent",
            hole=0.3,
            scalegroup='group1',
            showlegend=False,
        ),
        row=1, col=idx
    )

    # Update layout
    fig.update_layout(
     yaxis_title="Mtons CO2 eq/year",
     plot_bgcolor='white',
     xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5,
        tickfont=dict(size=15)
     ),
     yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5,
        title_font=dict(size=15),
        tickfont=dict(size=15)
     ),
    # annotations=[
    #     dict(
    #         text=column,
    #         x=(2.2 + idx)/(1.2 + num_pies),  # Position based on column
    #         y=0.19,
    #         xref="paper",
    #         yref="paper",
    #         showarrow=False,
    #         font=dict(size=15)
    #     ) for idx, column in enumerate(carbon_combined.columns)
    # ]
     )
    fig.update_layout(height=800, width=1400, title=f"Required carbon capture capacities for {country}_2050 compared to 2020")
    logo['y']=1.05
    if country == 'BE':
     pio.write_image(fig, "results/pdf/Carbon capture capacities.pdf", format='pdf')
    fig.add_layout_image(logo)
    return fig

def belgium_energy_independence():
 with open("data/europe.geojson") as f:
    europe_geojson = json.load(f)
 visible_countries = {
    "Belgium": 3,
    "Germany": 1,
    "France": 1,
    "Netherlands": 1,
    "United Kingdom": 1,}

 country_names = [feature["properties"]["NAME"] for feature in europe_geojson["features"]]
 z_vals = [visible_countries.get(name, None) for name in country_names]
 imports_ref = pd.read_csv("results/ref/country_csvs/total_imports_BE.csv",index_col=0).clip(lower=0)
 imports_suff = pd.read_csv("results/suff/country_csvs/total_imports_BE.csv",index_col=0).clip(lower=0)
 local_ref = pd.read_csv("results/ref/country_csvs/local_product_BE.csv",index_col=0).clip(lower=0)
 local_suff = pd.read_csv("results/suff/country_csvs/local_product_BE.csv",index_col=0).clip(lower=0)
 base_cols = {
    'imp_gaz_pe': 'Natural gas',
    'imp_pet_pe': 'Petroleum',
    'imp_elc_se': 'Electricity',
    'imp_cms_pe': 'Coal',
    'imp_hyd_se': 'Hydrogen',
    'imp_enc_pe': 'Solid biomass',
    'imp_amm_fe': 'Ammonia',
    'imp_met_fe': 'Methanol'}
 carrier_colors = {
    'Natural gas': '#e05b09',
    'Petroleum': '#c9c9c9',
    'Electricity': '#110d63',
    'Coal': '#545454',
    'Hydrogen': '#bf13a0',
    'Solid biomass': '#baa741',
    'Ammonia': '#46caf0',
    'Methanol': '#468c8b',
    'Local production': 'black',
    'Imports': 'whitesmoke'}
 ref_renamed = {k: f"{v}" for k, v in base_cols.items()}
 suff_renamed = {k: f"{v}" for k, v in base_cols.items()}
 imports_ref.rename(columns=ref_renamed, inplace=True)
 imports_suff.rename(columns=suff_renamed, inplace=True)
 imports_ref_2050 = imports_ref.loc[2050]
 imports_suff_2050 = imports_suff.loc[2050]
 imports_2020 = imports_ref.loc[2020]

 imports_ref_2050_sum = imports_ref.loc[2050].sum()
 imports_suff_2050_sum = imports_suff.loc[2050].sum()
 imports_2020_sum = imports_ref.loc[2020].sum()
 local_ref_2050_sum = local_ref.loc[2050].sum()
 local_suff_2050_sum = local_suff.loc[2050].sum()
 local_2020_sum = local_ref.loc[2020].sum()

 num_pies= 3
 fig = make_subplots(
    rows=2, cols=4,
    specs=[
        [{"type": "choropleth", "rowspan": 2}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
        [None, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}]
    ],
    column_widths=[0.5] + [0.5/num_pies]*num_pies,
    row_heights=[0.5] + [0.5],
    vertical_spacing=0.01,
    horizontal_spacing=0.01,
    subplot_titles=(
        "", "2020", "Ref (2050)", "Suff (2050)",
        "", "", "", ""
    )
)

 fig.add_trace(go.Choropleth(
    geojson=europe_geojson,
    locations=country_names,
    z=z_vals,
    featureidkey="properties.NAME",
    colorscale=[
        [0, "gray"],
        [0.5, "whitesmoke"],
        [1, "black"]
    ],
    zmin=0,
    zmax=2,
    showscale=False,
    marker_line_color='gray',
    hoverinfo="location"
), row=1, col=1)

 fig.update_geos(
    scope="europe",
    center=dict(lat=50.85, lon=4.35),
    projection_scale=5,
    showland=True,
    landcolor="whitesmoke",
    showcountries=True,
    # fitbounds="locations"
)
 labels = imports_2020.index
 values = imports_2020.values
 colors = [carrier_colors[label] for label in labels]
 fig.add_trace(go.Pie(
    labels=labels,
    values=values,
    hole=0.5,
    name="2020",
    scalegroup='group1',
    marker=dict(colors=colors)
), row=1, col=2)
 labels = imports_ref_2050.index
 values = imports_ref_2050.values
 colors = [carrier_colors[label] for label in labels]
 fig.add_trace(go.Pie(
    labels=labels,
    values=values,
    hole=0.5,
    name="Ref 2050",
    scalegroup='group1',
    marker=dict(colors=colors)
), row=1, col=3)
 labels = imports_suff_2050.index
 values = imports_suff_2050.values
 colors = [carrier_colors[label] for label in labels]
 fig.add_trace(go.Pie(
    labels=labels,
    values=values,
    hole=0.5,
    name="Suff 2050",
    scalegroup='group1',
    marker=dict(colors=colors)
), row=1, col=4)

 fig.add_trace(go.Pie(
    labels=["Imports", "Local production"],
    values=[imports_2020_sum,local_2020_sum],
    hole=0.5,
    name="Total",
    # textinfo='percent+value',
    scalegroup='group2',
    marker=dict(colors=[carrier_colors["Imports"], carrier_colors["Local production"]])
), row=2, col=2)

 fig.add_trace(go.Pie(
    labels=["Imports", "Local production"],
    values=[imports_ref_2050_sum,local_ref_2050_sum],
    hole=0.5,
    name="Total",
    # textinfo='label+value',
    scalegroup='group2',
    marker=dict(colors=[carrier_colors["Imports"], carrier_colors["Local production"]])
), row=2, col=3)

 fig.add_trace(go.Pie(
    labels=["Imports", "Local production"],
    values=[imports_suff_2050_sum,local_suff_2050_sum],
    hole=0.5,
    name="Total",
    # textinfo='label+value',
    scalegroup='group2',
    marker=dict(colors=[carrier_colors["Imports"], carrier_colors["Local production"]])
), row=2, col=4)
 
 fig.update_layout(height=800, width=1400, title="Belgium energy independence compared to 2020")
 logo['y']=1.05
 if country == 'BE':
  pio.write_image(fig, "results/pdf/Belgium energy independence.pdf", format='pdf')
 fig.add_layout_image(logo)
 return fig

def create_scenario_plots():
 scenarios=pd.read_csv("data/scenario_data.csv")

 capacities_suff=pd.read_csv("results/suff/country_csvs/BE_capacities.csv", index_col=0)
 capacities_suff_2050 = capacities_suff[['2050']]/1e3
 ac_transmission_suff = capacities_suff_2050.loc['AC Transmission lines', '2050']
 dc_transmission_suff = capacities_suff_2050.loc['DC Transmission lines', '2050']
 transmission_suff = ac_transmission_suff + dc_transmission_suff

 investment_suff=pd.read_csv("results/suff/country_csvs/BE_investment costs.csv", index_col=0)
 investment_suff_2050 = investment_suff[['2030', '2040', '2050']].sum(axis=1)
 investment_suff_2050 = investment_suff_2050.sum()/3
 investment_suff_2050 = investment_suff_2050/1e9
 demands_suff=pd.read_excel("results/suff/htmls/ChartData_BE.xlsx",  sheet_name="Chart 7",  header=None)
 new_header = demands_suff.iloc[2]
 demands_suff= demands_suff[3:]
 demands_suff.columns = new_header
 demands_suff.set_index(new_header[0], inplace=True)
 demands_suff = demands_suff.drop(['Non-energy', 'Aviation bunkers', 'Maritime bunkers'], axis=1)
 elec_demand_suff = demands_suff.loc[str(2050)].sum()
 total_costs_suff=pd.read_csv("results/suff/country_csvs/BE_costs.csv", index_col=0)
 total_costs_suff_2050 = total_costs_suff[['2030', '2040', '2050']].sum(axis=1)
 total_costs_suff_2050 = total_costs_suff_2050.sum()/3
 total_costs_suff_2050 = total_costs_suff_2050/1e9

 capacities_ref=pd.read_csv("results/ref/country_csvs/BE_capacities.csv", index_col=0)
 capacities_ref_2050 = capacities_ref[['2050']]/1e3
 ac_transmission_ref = capacities_ref_2050.loc['AC Transmission lines', '2050']
 dc_transmission_ref = capacities_ref_2050.loc['DC Transmission lines', '2050']
 transmission_ref = ac_transmission_ref + dc_transmission_ref

 investment_ref=pd.read_csv("results/ref/country_csvs/BE_investment costs.csv", index_col=0)
 investment_ref_2050 = investment_ref[['2030', '2040', '2050']].sum(axis=1)
 investment_ref_2050 = investment_ref_2050.sum()/3
 investment_ref_2050 = investment_ref_2050/1e9
 demands_ref=pd.read_excel("results/ref/htmls/ChartData_BE.xlsx",  sheet_name="Chart 7",  header=None)
 new_header = demands_ref.iloc[2]  # third row (index 2 in 0-based index)
 demands_ref = demands_ref[3:]  # drop all rows above the third row
 demands_ref.columns = new_header  # set the third row as the header
 demands_ref.set_index(new_header[0], inplace=True)
 demands_ref = demands_ref.drop(['Non-energy', 'Aviation bunkers', 'Maritime bunkers'], axis=1)
 elec_demand_ref = demands_ref.loc[str(2050)].sum()
 total_costs_ref=pd.read_csv("results/ref/country_csvs/BE_costs.csv", index_col=0)
 total_costs_ref_2050 = total_costs_ref[['2030', '2040', '2050']].sum(axis=1)
 total_costs_ref_2050 = total_costs_ref_2050.sum()/3
 total_costs_ref_2050 = total_costs_ref_2050/1e9
 
 jrc_historic=pd.read_csv("data/Historic_power_generation_jrc.csv", index_col=0)
 pypsa = pd.read_excel("results/suff/htmls/ChartData_BE.xlsx", sheet_name="Chart 22", skiprows=2)
 pypsa.set_index(pypsa.columns[0], inplace=True)
 pypsa=pypsa.loc[2020]
 pypsa = pd.DataFrame(pypsa).T
  # Drop columns where all values are zero, except for 'Imports'
 pypsa_tot = pypsa.loc[:, (pypsa != 0).any(axis=0)]
 if 'Imports' in pypsa.columns:
      if 'Imports' not in pypsa_tot.columns:
          pypsa_tot['Imports'] = pypsa['Imports']       
 pypsa_tot['Wind'] = pypsa_tot['Onshore wind'] + pypsa_tot['Offshore wind']
 pypsa_tot = pypsa_tot.drop(columns=['Onshore wind', 'Offshore wind'])
 pypsa_tot = pypsa_tot.T
 common_index = jrc_historic.index.intersection(pypsa_tot.index)
 jrc_historic = jrc_historic.loc[common_index]
 pypsa_tot = pypsa_tot.loc[common_index]
  # Rename columns
 jrc_historic.columns = ['JRC-Historic-2020']
 pypsa_tot.columns = ['PyPSA-2020']
  # Concatenate the dataframes
 combined_df = pd.concat([jrc_historic, pypsa_tot], axis=1)
 rename_dict = {'Uranium': 'Nuclear', 'Gas grid': 'Natural gas'}
 combined_df.rename(index=rename_dict, inplace=True)


 scenarios['Pypsa-sufficiency'] = None
 scenarios.loc['Final Energy Demand (Twh)', 'Pypsa-sufficiency'] = elec_demand_suff
 scenarios.loc['Average Investment Costs(Billion Euros/year)', 'Pypsa-sufficiency'] = investment_suff_2050
 scenarios.loc['Average Annual Costs(Billion Euros/year)', 'Pypsa-sufficiency'] = total_costs_suff_2050
 # scenarios.loc['Emissions(%)', 'Pypsa-sufficiency'] = -95

 scenarios['Pypsa-Ref'] = None
 scenarios.loc['Final Energy Demand (Twh)', 'Pypsa-Ref'] = elec_demand_ref
 scenarios.loc['Average Investment Costs(Billion Euros/year)', 'Pypsa-Ref'] = investment_ref_2050
 scenarios.loc['Average Annual Costs(Billion Euros/year)', 'Pypsa-Ref'] = total_costs_ref_2050
 # scenarios.loc['Emissions(%)', 'Pypsa-BAU'] = -95

 techs = ['solar', 'onshore wind','offshore wind', 'nuclear']

 for tech in techs:
    scenarios.loc[tech, 'Pypsa-sufficiency'] = capacities_suff_2050.loc[tech, '2050']
    scenarios.loc[tech, 'Pypsa-Ref'] = capacities_ref_2050.loc[tech, '2050']

 scenarios.loc['Hydrogen Turbines or CHPs', 'Pypsa-sufficiency'] = capacities_suff_2050.loc['H2 turbine', '2050']
 scenarios.loc['CCGT/OCGT', 'Pypsa-sufficiency'] = capacities_suff_2050.loc['CCGT', '2050']
 scenarios.loc['Interconnections', 'Pypsa-sufficiency'] = transmission_suff
 scenarios.loc['Others', 'Pypsa-sufficiency'] = capacities_suff_2050.loc['hydroelectricity', '2050']

 scenarios.loc['Hydrogen Turbines or CHPs', 'Pypsa-Ref'] = capacities_ref_2050.loc['H2 turbine', '2050']
 scenarios.loc['CCGT/OCGT', 'Pypsa-Ref'] = capacities_ref_2050.loc['CCGT', '2050']
 scenarios.loc['Interconnections', 'Pypsa-Ref'] = transmission_ref
 scenarios.loc['Others', 'Pypsa-Ref'] = capacities_ref_2050.loc['hydroelectricity', '2050']

 scenarios = scenarios.apply(pd.to_numeric, errors='coerce').fillna(0)
 
 scenarios_data=pd.read_csv("data/scenario_data.csv").T
 scenarios_data.drop(columns=scenarios_data.columns, inplace=True)
 scenarios_data["Power"] = None
 scenarios_data["Industry"] = None 
 scenarios_data["Domestic Transport"] = None 
 scenarios_data["Maritime Transport (Domestic)"] = None 
 scenarios_data["Maritime Transport (International)"] = None 
 scenarios_data["Aviation Transport"] = None 
 scenarios_data["Agriculture"] = None 
 scenarios_data["Residential & Tertiary"] = None 
 scenarios_data["LULUCF"] = None 

 new_rows = pd.DataFrame(columns=scenarios_data.columns, index=["Climact PAC2.0", "PyPSA-Sufficiency", "PyPSA-Ref"])
 scenarios_data = pd.concat([scenarios_data, new_rows])
 scenarios_data.loc["McKinsey", ["Power", "Industry", "Domestic Transport", 
                            "Agriculture", "LULUCF", "Residential & Tertiary"]] = "✔"
 scenarios_data.loc["McKinsey", ["Maritime Transport (Domestic)", "Maritime Transport (International)","Aviation Transport"]] = "X"
 scenarios_data.loc["PyPSA-Sufficiency", :] = "✔"
 scenarios_data.loc["PyPSA-Ref", :] = "✔"
 scenarios_data.loc["PyPSA-Ref", ["Maritime Transport (International)"]] = "X"
 scenarios_data.loc["PyPSA-Sufficiency", ["Maritime Transport (International)"]] = "X"
 scenarios_data.loc["CLEVER", :] = "✔"
 scenarios_data.loc["Climact PAC2.0", :] = "✔"
 scenarios_data.loc["Climact PAC2.0", ["Maritime Transport (International)"]] = "X"
 scenarios_data.loc["Elia Blue Print(Elec)", :] = "✔"
 scenarios_data.loc["Elia Blue Print(GA)", :] = "✔"
 scenarios_data.loc["Elia Blue Print(DE)", :] = "✔"
 scenarios_data.loc["Elia Blue Print(Elec)", ["Maritime Transport (International)"]] = "50%"
 scenarios_data.loc["Elia Blue Print(GA)", ["Maritime Transport (International)"]] = "50%"
 scenarios_data.loc["Elia Blue Print(DE)", ["Maritime Transport (International)"]] = "50%"
 scenarios_data.loc["Energyville(Shift)", ["Power", "Industry", "Domestic Transport", 
                            "Residential & Tertiary", "Agriculture", "Maritime Transport (Domestic)"]] = "✔"
 scenarios_data.loc["Energyville(Shift)", ["Maritime Transport (International)", "LULUCF", "Aviation Transport"]] = "X"
 scenarios_data.loc["Energyville(Central)", ["Power", "Industry", "Domestic Transport", 
                            "Residential & Tertiary", "Agriculture", "Maritime Transport (Domestic)"]] = "✔"
 scenarios_data.loc["Energyville(Central)", ["Maritime Transport (International)", "LULUCF", "Aviation Transport"]] = "X"

 scenarios_transposed = scenarios.transpose()
 figures = {}
 # Plot the bar chart for Demand
 fig_demand = go.Figure()
 fig_demand.add_trace(go.Bar(name='Demand (TWh)', x=scenarios_transposed.index, y=scenarios_transposed['Final Energy Demand (Twh)']))
 fig_demand.update_layout(
        title="Final energy demand comparison for scenarios in 2050",
        xaxis_title="Scenario",
        height=700, width=1200,
        yaxis_title="Demand (TWh)"
    )
 if country == 'BE':
  pio.write_image(fig_demand, "results/pdf/Scenario demands comparison.pdf", format='pdf')
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
        height=700, width=1200,
        yaxis_title="Capacities (GW)"
    )
 if country == 'BE':
  pio.write_image(fig_vre, "results/pdf/Scenario VRE comparison.pdf", format='pdf')
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
        height=700, width=1400,
        yaxis_title="Capacities (GW)"
    )
 if country == 'BE':
  pio.write_image(fig_flex, "results/pdf/Scenario flexibility comparison.pdf", format='pdf')
 figures['flexibility'] = fig_flex

 # Plot the bar chart for Costs
 columns_to_plot_costs = ['Average Annual Costs(Billion Euros/year)', 'Average Investment Costs(Billion Euros/year)']
 colors_costs = {
        'Average Annual Costs(Billion Euros/year)': 'blue',
        'Average Investment Costs(Billion Euros/year)': 'green'
    }
 fig_costs = go.Figure()
 for column in columns_to_plot_costs:
        color = colors_costs.get(column, None)
        fig_costs.add_trace(go.Bar(name=column, x=scenarios_transposed.index, y=scenarios_transposed[column], marker_color=color))
 fig_costs.update_layout(
        title="Costs comparison for scenarios",
        xaxis_title="Scenario",
        height=700, width=1400,
        yaxis_title="Average Annual Costs (Billion Euros/year)"
    )
 if country == 'BE':
  pio.write_image(fig_costs, "results/pdf/Scenario costs comparison.pdf", format='pdf')
 figures['costs'] = fig_costs

  #Plot the bar chart for Emissions
 fig_emissions = go.Figure()
 fig_emissions.add_trace(go.Bar(name='Emissions(%)', x=scenarios_transposed.index, y=scenarios_transposed['Emissions(%)'], marker_color='red'))
 fig_emissions.update_layout(
         title="Emissions comparison for scenarios compared to 1990",
         xaxis_title="Scenario",
         height=700, width=1200,
         yaxis_title="%"
     )
 figures['emissions'] = fig_emissions
 
 fig_historic = go.Figure()
 fig_historic.add_trace(go.Bar(
      x=combined_df.index,
      y=combined_df['JRC-Historic-2020'],
      name='JRC-Historic-2020',
      marker_color='blue'))
 fig_historic.add_trace(go.Bar(
      x=combined_df.index,
      y=combined_df['PyPSA-2020'],
      name='PyPSA-2020',
      marker_color='red'))
 fig_historic.update_layout(
      title="2020 scenario comparison with JRC historic data",
      yaxis_title='Energy Production (TWh)',
      height=700, width=1200,
      # barmode='group',
      # font=dict(size=15),
  )
 if country == 'BE':
  pio.write_image(fig_historic, "results/pdf/Comparison with JRC data.pdf", format='pdf')
 figures['historic'] = fig_historic
 
 fig_scenarios_data = go.Figure(data=[go.Table(
     header=dict(
         values=["Scenario"] + list(scenarios_data.columns),  # Add the 'Scenario' column header at the start
         fill_color='paleturquoise',
         align='center'
     ),
     cells=dict(
         values=[scenarios_data.index] + [scenarios_data[col] for col in scenarios_data.columns],  # Include the index as a column
         fill_color='lavender',
         align='center'
     )
 )])
 
 figures['scenarios_data'] = fig_scenarios_data
 
 return figures
    
def create_combined_scenario_chart_country(country, output_folder='results/scenario_results/'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create combined HTML
    combined_html = "<html><head><title>The Negawatt Belgium sufficiency scenario</title></head><body>"
    combined_html_be = "<html><head><title>Scenario Comparison</title></head><body>"
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

    def load_html_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        return html_content
    def get_country_specific_desc(country, html_texts):
    # Extract descriptions specific to the selected country
     country_desc = {}
     if country == 'EU':
        country_desc = {key: content for key, content in html_texts.items() if key.startswith("EU_")}
     elif country == 'BE':
        country_desc = {key: content for key, content in html_texts.items() if key.startswith("BE_")}
     else:
        # For all other countries, load the generic descriptors
        country_desc = {key: content for key, content in html_texts.items() if not key.startswith(("EU_", "BE_"))}

     return country_desc
    # Load the HTML texts from file
    html_texts = load_html_texts(file_path)
    country_desc = get_country_specific_desc(country, html_texts)

    cummulative_emm_desc = country_desc.get(f"{country}_cumulative_emissions_sce", '')
    annual_costs_desc = country_desc.get(f"{country}_annual_costs_sce", '')
    investment_costs_desc = country_desc.get(f"{country}_investment_costs_sce", '')
    cumu_investment_costs_desc = country_desc.get(f"{country}_cumu_investment_costs_sce", '')
    capacities_desc = country_desc.get(f"{country}_capacities_sce", '')
    storage_capacities_desc = country_desc.get(f"{country}_storage_capacities_sce", '')
    cc_capacities_desc = country_desc.get(f"{country}_cc_capacities_sce", '')
    be_energy_ind_desc = country_desc.get(f"{country}_energy_ind_sce", '')
    scenario_dem_comp_desc = country_desc.get(f"{country}_scenario_dem_comp_sce", '')
    scenario_vre_desc = country_desc.get(f"{country}_scenario_vre_sce", '')
    scenario_flex_desc = country_desc.get(f"{country}_scenario_flex_sce", '')
    scenario_cost_desc = country_desc.get(f"{country}_scenario_cost_sce", '')
    hist_desc = country_desc.get(f"{country}_hist_sce", '')
    sectoral_desc = country_desc.get(f"{country}_sectoral_sce", '')
    
    #load the html plot flags
    with open(snakemake.input.plots_html, 'r') as file:
     plots = yaml.safe_load(file)
    scenario_plots = plots.get("Scenario_plots", {})
    
    if scenario_plots["Cummulative Emissions"] == True:
     emm_chart = Cumulative_emissions_sector(country)
     combined_html += f"<div><h2>{country} - Cumulative Emissions</h2>{emm_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    # Create bar chart
    if scenario_plots["Annual Costs"] == True:
     bar_chart = scenario_costs(country)
     combined_html += f"<div><h2>{country} - Annual Costs</h2>{bar_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    if scenario_plots["Annual Investment Costs"] == True:
     bar_chart_investment = scenario_investment_costs(country)
     combined_html += f"<div><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    if scenario_plots["Cummulative Investment Costs"] == True:
     bar_chart_cumulative = scenario_cumulative_costs(country)
     combined_html += f"<div><h2>{country} - Cumulative Investment Costs (2023-2050)</h2>{bar_chart_cumulative.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    # Create capacities chart
    if scenario_plots["Capacities"] == True:
     capacities_chart = scenario_capacities(country)
     combined_html += f"<div><h2>{country} - Capacities</h2>{capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    # Create storage capacities chart
    if scenario_plots["Storage Capacities"] == True:
     storage_capacities_chart = storage_capacities(country)
     combined_html += f"<div><h2>{country} -  Storage Capacities</h2>{storage_capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
     
     if scenario_plots["CC Capacities"] == True:
      cc_capacities_chart = crabon_capture_techs(country)
      combined_html += f"<div><h2>{country} - Required Carbon Capture Capacities</h2>{cc_capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
      
     if country == 'BE':
      if scenario_plots["Energy Independence"] == True:
       energy_ind_chart = belgium_energy_independence()
       combined_html += f"<div><h2>{country} - Energy Independence</h2>{energy_ind_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    
    #Create scenario comparison plots
    if country == 'BE':
        scenario_figures = create_scenario_plots()
        if scenario_plots["Scenarios Demands Comparison"] == True:
         demand_comparison = scenario_figures['demand']
         combined_html_be += f"<div><h2>{country} - Scenarios Demands Comparison</h2>{demand_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios VRE Capacities Comparison"] == True:
         vre_comparison = scenario_figures['vre']
         combined_html_be += f"<div><h2>{country} - Scenarios VRE Capacities Comparison</h2>{vre_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenario Flexibility Capacities"] == True:
         flexibility_comparison = scenario_figures['flexibility']
         combined_html_be += f"<div><h2>{country} - Scenario Flexibility Capacities in Electricity Grid Comparison</h2>{flexibility_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios Costs Comparison"] == True:
         costs_comparison = scenario_figures['costs']
         combined_html_be += f"<div><h2>{country} - Scenarios Costs Comparison</h2>{costs_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Historic Generation Comparison"] == True:
         historic_comparison = scenario_figures['historic']
         combined_html_be += f"<div><h2>{country} - Historic Generation Comparison from JRC data with Pypsa 2020 Baseline Scenario</h2>{historic_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios Sectors Comparison"] == True:
         scenarios_data = scenario_figures['scenarios_data']
         combined_html_be += f"<div><h2>{country} - Comparison of modelled sectors considered in different recent Scenarios</h2>{scenarios_data.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

    combined_html += "</body></html>"
    combined_html_be += "</body></html>"
    table_of_contents_content = ""
    table_of_contents_content_be = ""
    main_content_be = ''
    html_filnemae = 'SEPIA/html_scenarios_' + country + '.html'
    if os.path.exists(html_filnemae):
        main_content = load_html_file(html_filnemae)
    else:
        main_content = ''
    # Create the content for the "Table of Contents" and "Main" sections
    if scenario_plots["Cummulative Emissions"] == True:
     table_of_contents_content += f"<a href='#{country} - Cumulative Emissions'>Cumulative Emissions</a><br>"
    if scenario_plots["Annual Costs"] == True:
     table_of_contents_content += f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    if scenario_plots["Annual Investment Costs"] == True:
     table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    if scenario_plots["Cummulative Investment Costs"] == True:
     table_of_contents_content += f"<a href='#{country} - Cumulative Investment Costs (2020-2050)'>Cumulative Investment Costs (2020-2050)</a><br>"
    if scenario_plots["Capacities"] == True:
     table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    if scenario_plots["Storage Capacities"] == True:
     table_of_contents_content += f"<a href='#{country} - Storage Capacities'>Storage Capacities</a><br>"
    if scenario_plots["CC Capacities"] == True:
     table_of_contents_content += f"<a href='#{country} - Required Carbon Capture Capacities'>Carbon Capture Capacities</a><br>"
    if country == 'BE':
     if scenario_plots["Energy Independence"] == True:
      table_of_contents_content += f"<a href='#{country} - Energy Independence'>Energy Independence</a><br>"
    if country == 'BE':
        if scenario_plots["Scenarios Demands Comparison"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Scenarios Demands Comparison'>Scenarios Demands Comparison</a><br>"
        if scenario_plots["Scenarios VRE Capacities Comparison"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Scenarios VRE Capacities Comparison'>Scenarios VRE Capacities Comparison</a><br>"
        if scenario_plots["Scenario Flexibility Capacities"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Scenario Flexibility Capacities in Electricity Grid Comparison'>Scenario Flexibility Capacities in Electricity Grid Comparison</a><br>"
        if scenario_plots["Scenarios Costs Comparison"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Scenarios Costs Comparison'>Scenarios Costs Comparison</a><br>"
        if scenario_plots["Historic Generation Comparison"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Historic Generation Comparison from JRC data with Pypsa 2020 Baseline Scenario'>Historic Generation Coparison from JRC data with Pypsa 2020 Baseline Scenario</a><br>"
        if scenario_plots["Scenarios Sectors Comparison"] == True:
         table_of_contents_content_be += f"<a href='#{country} - Comparison of Sectors in Belgium's Energy Scenarios'>Comparison of different sectors considered for Belgium scenarios</a><br>"

    # Add more links for other plots
    if scenario_plots["Cummulative Emissions"] == True:
     main_content += f"<div id='{country} - Cumulative Emissions'><h2>{country} - Cumulative Emissions</h2>{cummulative_emm_desc}{emm_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["Annual Costs"] == True:
     main_content += f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{annual_costs_desc}{bar_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["Annual Investment Costs"] == True:
     main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{investment_costs_desc}{bar_chart_investment.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["Cummulative Investment Costs"] == True:
     main_content += f"<div id='{country} - Cumulative Investment Costs (2020-2050)'><h2>{country} - Cumulative Investment Costs (2020-2050)</h2>{cumu_investment_costs_desc}{bar_chart_cumulative.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["Capacities"] == True:
     main_content += f"<div id='{country} - Capacities'><h2>{country} - Capacities</h2>{capacities_desc}{capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["Storage Capacities"] == True:
     main_content += f"<div id='{country} - Storage Capacities'><h2>{country} - Storage Capacities</h2>{storage_capacities_desc}{storage_capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if scenario_plots["CC Capacities"] == True:
     main_content += f"<div id='{country} - Required Carbon Capture Capacities'><h2>{country} - Required Carbon Capture Capacities</h2>{cc_capacities_desc}{cc_capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if country == 'BE':
     if scenario_plots["Energy Independence"] == True:
      main_content += f"<div id='{country} - Energy Independence'><h2>{country} - Energy Independence</h2>{be_energy_ind_desc}{energy_ind_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    if country == 'BE':
        if scenario_plots["Scenarios Demands Comparison"] == True:
         main_content_be += f"<div id='{country} - Scenarios Demands Comparison'><h2>{country} - Scenarios Demands Comparison</h2>{scenario_dem_comp_desc}{demand_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios VRE Capacities Comparison"] == True:
         main_content_be += f"<div id='{country} - Scenarios VRE Capacities Comparison'><h2>{country} - Scenarios VRE Capacities Comparison</h2>{ scenario_vre_desc}{vre_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenario Flexibility Capacities"] == True:
         main_content_be += f"<div id='{country} - Scenario Flexibility Capacities in Electricity Grid Comparison'><h2>{country} - Scenario Flexibility Capacities in Electricity Grid Comparison</h2>{scenario_flex_desc}{flexibility_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios Costs Comparison"] == True:
         main_content_be += f"<div id='{country} - Scenarios Costs Comparison'><h2>{country} - Scenarios Costs Comparison</h2>{scenario_cost_desc}{costs_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Historic Generation Comparison"] == True:
         main_content_be += f"<div id='{country} - Historic Generation Comparison from JRC data with Pypsa 2020 Baseline Scenario'><h2>{country} - Historic Generation Comparison from JRC data with Pypsa 2020 Baseline Scenario</h2>{hist_desc}{historic_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        if scenario_plots["Scenarios Sectors Comparison"] == True:
         main_content_be += f"<div id='{country} - Comparison of Sectors in Belgium's Energy Scenarios'><h2>{country} - Comparison of Sectors in Belgium's Energy Scenarios</h2>{sectoral_desc}{scenarios_data.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
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
     
    # Render the second HTML file (Belgium-Specific Comparisons)
    rendered_html_be = ""
    if country == 'BE':
     rendered_html_be = template.render(
        title=f"{country} - Scenario Comparisons",
        country=country,
        TABLE_OF_CONTENTS=table_of_contents_content_be,
        MAIN=main_content_be
    )
    be_file_path = os.path.join(output_folder, f"{country}_scenario_comparison.html")
    if rendered_html_be:
     with open(be_file_path, "w") as be_file:
        be_file.write(rendered_html_be)




if __name__ == "__main__":
    if "snakemake" not in globals():
        #from _helpers import mock_snakemake

        #snakemake = mock_snakemake("prepare_scenarios")
        import pickle
        with open("snakemake_dump.pkl", "rb") as f:
            snakemake = pickle.load(f)

        
    total_country = 'EU'
    countries = snakemake.params.countries 
    file_path = snakemake.input.file_path
    countries.append(total_country) 
    config = snakemake.config
    logo = logo()
    for country in countries:
        create_combined_scenario_chart_country(country)
        
    
 
