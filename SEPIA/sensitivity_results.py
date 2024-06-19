#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import pypsa
import logging
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from jinja2 import Template
current_script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_script_dir, "../scripts/")
sys.path.append(scripts_path)
from plot_summary import rename_techs



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
    elif "solar rooftop" in tech:
        return "solar rooftop"
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
    elif "Li ion" in tech:
        return "battery storage"
    elif "EV charger" in tech:
        return "V2G"
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

      mask = ~(df['tech'].isin(['load shedding']))
      result_df = df[mask]
      if not result_df.empty:
            years = ['2030', '2040', '2050']
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
         directory = os.path.dirname(file_path)
         if not os.path.exists(directory):
          os.makedirs(directory)
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
      
      mask = ~(df['tech'].isin(['load shedding']))
      result_df = df[mask]
      if not result_df.empty:
            years = ['2030', '2040', '2050']
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
      cf = pd.read_csv(f"results/{study}/csvs/nodal_capacities.csv", index_col=1)
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
      mask = ~(cf['tech'].isin(['load shedding']))
      result_df = cf[mask]
      if not result_df.empty:
            years = ['2030', '2040', '2050']
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
      cf = pd.read_csv(f"results/{study}/csvs/nodal_capacities.csv", index_col=1)
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
      mask = ~(cf['tech'].isin(['load shedding']))
      result_df = cf[mask]
      result_df['tech'] = result_df['tech'].replace({'urban central water tanks': 'Thermal Energy storage', 'battery':'Grid-scale battery', 'battery storage':'V2G'})
      if not result_df.empty:
            years = ['2030', '2040', '2050']
            technologies = result_df['tech'].unique()

            s_capacities[country] = result_df.set_index('tech').loc[technologies, years]
            for country, dataframe in s_capacities.items():
             # Specify the file path where you want to save the CSV file
             file_path = f"results/{study}/country_csvs/{country}_storage_capacities.csv"
         
              # Save the DataFrame to a CSV file
             dataframe.to_csv(file_path, index=True)

             print(f"CSV file for {country} saved at: {file_path}") 

    return s_capacities

        
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
        ["solar","solar rooftop"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
        ["AC Transmission lines"],
        ["DC Transmission lines"],
        ["CCGT"],
        ["nuclear"],
    ]
    
    groupss = [
        ["solar","solar rooftop"],
        ["onshore wind", "offshore wind"],
        ["SMR"],
        ["gas-to-power/heat", "power-to-heat", "power-to-liquid"],
        ["transmission lines"],
        ["gas pipeline","gas pipeline new"],
        ["CCGT"],
        ["nuclear"],
    ]

    # Create a subplot for each technology
    years = ['2030', '2040', '2050']
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
        ["biogas"],
    ]

    # Create a subplot for each technology
    years = ['2030', '2040', '2050']
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
    
    # Create bar chart
    bar_chart = create_bar_chart(costs, country)
    combined_html += f"<div><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    
    # Create Investment Costs
    bar_chart_investment = create_investment_costs(investment_costs, country)
    combined_html += f"<div><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"

    # Create capacities chart
    capacities_chart = create_capacity_chart(capacities, country)
    combined_html += f"<div><h2>{country} - Capacities </h2>{capacities_chart.to_html()}</div>"
    
    # Create storage capacities chart
    s_capacities_chart = storage_capacity_chart(s_capacities, country)
    combined_html += f"<div><h2>{country} - Storage Capacities </h2>{s_capacities_chart.to_html()}</div>"


    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content = f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{country} - Storage Capacities'>Storage Capacities</a><br>"
    
    # Add more links for other plots

    main_content = f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    main_content += f"<div id='{country} - Capacities'><h2>{country} - Capacities</h2>{capacities_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Storage Capacities'><h2>{country} - Storage Capacities</h2>{s_capacities_chart.to_html()}</div>"

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
    
    combined_file_path = os.path.join(output_folder, f"{country}_sensitivity_chart.html")
    with open(combined_file_path, "w") as combined_file:
     combined_file.write(rendered_html)

    
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_results")
        

        # Updating the configuration from the standard config file to run in standalone:
    simpl = snakemake.params.scenario["simpl"][0]
    cluster = snakemake.params.scenario["clusters"][0]
    opt = snakemake.params.scenario["opts"][0]
    sector_opt = snakemake.params.scenario["sector_opts"][0]
    ll = snakemake.params.scenario["ll"][0]
    planning_horizons = [2030, 2040, 2050]
    total_country = 'EU'
    countries = snakemake.params.countries 
    map_opts = snakemake.params.plotting["map"]
    countries.append(total_country)
    logging.basicConfig(level=snakemake.config["logging"]["level"])
    config = snakemake.config
    study = snakemake.params.study
    logo = logo()
    loaded_files = load_files(study, planning_horizons, simpl, cluster, opt, sector_opt, ll)
    results = calculate_transmission_values(simpl, cluster, opt, sector_opt, ll, planning_horizons)
    costs = costs(countries, results)
    investment_costs = Investment_costs(countries, results)
    capacities = capacities(countries, results)
    s_capacities = storage_capacities(countries)
    
    
    for country in countries:
        create_combined_chart_country(costs,investment_costs, capacities,s_capacities, country)
    


