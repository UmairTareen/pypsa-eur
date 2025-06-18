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
from add_electricity import calculate_annuity

def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if tech in ["H2 Electrolysis", "methanation","helmeth", "H2 liquefaction","heat pump","resistive heater","Fischer-Tropsch", "air heat pump","air-sourced heat pump","ground heat pump"]:
        return "Power-to-X"
    elif tech in ["electricity distribution grid"]:
        return "Distribution Network"
    elif tech in [ "CHP", "H2 Fuel Cell","CCGT","OCGT","H2 turbine","solid biomass powerplants","coal powerplants", "oil powerplants"]:
        return "CHP & Powerplants"
    elif tech in [ "battery charger", "battery discharger","battery", "Li ion", "EV charger", "V2G","hot water storage", "H2", "H2 storage"]:
        return "TES & Battery & H2 storage"
    elif tech in [ "biomass boiler", "oil boiler","gas boiler"]:
        return "Boilers"
    elif "solar" in tech:
        return "Solar PV"
    elif "wind" in tech:
        return "Wind"
    elif tech in ["co2 sequestered","CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC","DAC"]:
        # if study == "suff":
        #     return "CCU"
        # else:
            return "CCU (Suff) & CCUS (Ref)"
    elif tech in ["biomass", "solid biomass", "solid biomass for industry", "biogas", "solid biomass transport", "biomass exports", "biogas exports"]:
          return "Biomass"
    elif tech in ["shipping oil", "naphtha for industry", "land transport oil", "kerosene for aviation", "agriculture machinery oil", "oil","gas", "coal for industry","gas for industry","coal","lignite","coal fuel","gas fuel"]:
          return "Fossil Fuels"
    elif "load" in tech:
        return "load shedding"
    elif "hydroelectricity" in tech:
        return "Hydro-electricity"
    elif "electricity imports/exports" in tech:
        return "Electricity Imports/Exports"
    elif "hydrogen imports/exports" in tech:
        return "Hydrogen Imports/Exports"
    elif tech in ["SMR", "ammonia cracker", "Haber-Bosch", "BioSNG", "biomass to liquid","methanol","ammonia", "methanol exports", "ammonia exports","methanolisation","shipping methanol"]:
          return "Synthetic Fuels & Techs"
    elif tech in ["uranium", "nuclear", "nuclear fuel"]:
          return "Nuclear"
    else:
        return tech
    
def rename_techs_tyndp_EU(tech):
    tech = rename_techs(tech)
    if tech in ["H2 pipeline", "gas pipeline","gas pipeline new","H2 pipeline retrofitted","transmission lines","Transmission Lines"]:
          return "Transmission Lines & Pipelines"
    else:
        return tech
# def rename_techs_tyndp(tech):
#     tech = rename_techs(tech)
#     if "heat pump" in tech or "resistive heater" in tech:
#         return "power-to-heat"
#     elif tech in ["H2 Electrolysis", "methanation","helmeth", "H2 liquefaction"]:
#         return "power-to-gas"
#     elif tech in ["electricity distribution grid"]:
#         return "distribution network"
#     elif tech in [ "CHP", "H2 Fuel Cell","CCGT","OCGT","H2 turbine","solid biomass powerplants","coal powerplants", "oil powerplants"]:
#         return "CHP & powerplants"
#     elif tech in [ "battery charger", "battery discharger","battery", "Li ion", "EV charger", "V2G"]:
#         return "battery storage"
#     elif tech in [ "biomass boiler", "oil boiler","gas boiler"]:
#         return "boilers"
#     elif "solar" in tech:
#         return "solar"
#     elif tech == "Fischer-Tropsch":
#         return "power-to-liquid"
#     elif "offshore wind" in tech:
#         return "offshore wind"
#     elif tech in ["co2 sequestered","CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC"]:
#          return "CCU"
#     elif tech in ["biomass", "solid biomass", "solid biomass for industry", "biogas", "solid biomass transport", "biomass exports", "biogas exports"]:
#          return "biomass"
#     elif tech in ["shipping oil", "naphtha for industry", "land transport oil", "kerosene for aviation", "agriculture machinery oil", "oil","gas", "coal for industry","gas for industry","coal","lignite","coal fuel","gas fuel"]:
#          return "fossil fuels"
#     elif tech in ["hot water storage", "H2", "H2 storage"]:
#         return "TES & H2 storage"
#     elif "load" in tech:
#         return "load shedding"
#     elif tech in ["SMR", "ammonia cracker", "Haber-Bosch", "BioSNG", "biomass to liquid","methanol","ammonia", "methanol exports", "ammonia exports","methanolisation","shipping methanol"]:
#          return "synthetic fuels & techs"
#     elif tech in ["uranium", "nuclear", "nuclear fuel"]:
#          return "nuclear"
#     elif tech in ["H2 pipeline", "gas pipeline","gas pipeline new","H2 pipeline retrofitted"]:
#          return "H2 & gas pipelines"
#     else:
#         return tech

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

def annuity_factor(v):
    return calculate_annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100

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
    length_ac = lines.length[line_numbers].sum()
    
    options = pd.read_csv(fn ,index_col=[0, 1]).sort_index()
    options.loc[options.unit.str.contains("/kW"), "value"] *= 1e3
    options = (
      options.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1))
    options = options.fillna(config["costs"]["fill_values"])
    options["fixed"] = [
        annuity_factor(v) * v["investment"] for i, v in options.iterrows()
    ]
    ac_cost = options.loc[("HVAC overhead", "fixed")]
    
    transmission = ((lines.s_nom_opt[line_numbers].sum()) * ac_cost * length_ac)

    return transmission_ac, transmission

def calculate_dc_transmission(links, link_numbers):
    transmission_dc = links.p_nom_opt[link_numbers].sum()
    length_dc = links.length[link_numbers].sum()
    
    options = pd.read_csv(fn ,index_col=[0, 1]).sort_index()
    options = pd.read_csv(fn ,index_col=[0, 1]).sort_index()
    options.loc[options.unit.str.contains("/kW"), "value"] *= 1e3
    options = (
      options.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1))
    options = options.fillna(config["costs"]["fill_values"])
    options["fixed"] = [
        annuity_factor(v) * v["investment"] for i, v in options.iterrows()
    ]
    dc_cost = options.loc[("HVDC overhead", "fixed")]
    
    transmissionc = ((links.p_nom_opt[link_numbers].sum()) * dc_cost * length_dc)

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

def calculate_elec_import_export_costs(country, planning_horizons):
    def calculate_import_export_separate(flows, direction, network, marginal_prices):
        import_cost = pd.DataFrame(index=flows.index, columns=flows.columns)
        export_revenue = pd.DataFrame(index=flows.index, columns=flows.columns)

        for line in flows.columns:
            # Identify buses
            if direction == 'ac0' or direction == 'dc0':
                bus0 = network.lines.loc[line, "bus0"] if 'ac' in direction else network.links.loc[line, "bus0"]
                bus1 = network.lines.loc[line, "bus1"] if 'ac' in direction else network.links.loc[line, "bus1"]
            else:
                bus0 = network.lines.loc[line, "bus1"] if 'ac' in direction else network.links.loc[line, "bus1"]
                bus1 = network.lines.loc[line, "bus0"] if 'ac' in direction else network.links.loc[line, "bus0"]

            if bus0 not in marginal_prices.columns or bus1 not in marginal_prices.columns:
                continue

            price_bus0 = marginal_prices[bus0]
            price_bus1 = marginal_prices[bus1]

            for t in flows.index:
                flow = flows.at[t, line]
                if flow < 0:  # Import
                    import_cost.at[t, line] = -flow * price_bus1[t]
                    export_revenue.at[t, line] = 0
                else:  # Export
                    import_cost.at[t, line] = 0
                    export_revenue.at[t, line] = flow * price_bus0[t]

        return import_cost, export_revenue

    results = {}

    for planning_horizon in planning_horizons:
        # Load network
        n = loaded_files[planning_horizon]

        # Marginal prices
        marginal_price_filter = n.buses.carrier == "AC"
        marginal_price = n.buses_t.marginal_price.filter(items=marginal_price_filter[marginal_price_filter].index)
        marginal_price = marginal_price.drop(columns=["GB2 0"], errors='ignore')

        # AC lines
        filtered_ac_lines = n.lines.bus0.str[:2] == country
        ac_lines = n.lines_t.p0.filter(items=filtered_ac_lines[filtered_ac_lines].index)

        filtered_ac_lines_r = n.lines.bus1.str[:2] == country
        ac_lines_r = n.lines_t.p1.filter(items=filtered_ac_lines_r[filtered_ac_lines_r].index)

        # DC links
        filtered_dc_lines = (n.links.carrier == 'DC') & (n.links.bus0.str[:2] == country)
        dc_lines = n.links_t.p0.filter(items=filtered_dc_lines[filtered_dc_lines].index)
        dc_lines = dc_lines.drop(columns=["5580", "5581"], errors='ignore')

        filtered_dc_lines_r = (n.links.carrier == 'DC') & (n.links.bus1.str[:2] == country)
        dc_lines_r = n.links_t.p1.filter(items=filtered_dc_lines_r[filtered_dc_lines_r].index)
        dc_lines_r = dc_lines_r.drop(columns=["5580", "5581"], errors='ignore')

        # Calculate import/export
        ac_import, ac_export = calculate_import_export_separate(ac_lines, "ac0", n, marginal_price)
        ac_r_import, ac_r_export = calculate_import_export_separate(ac_lines_r, "acr", n, marginal_price)
        dc_import, dc_export = calculate_import_export_separate(dc_lines, "dc0", n, marginal_price)
        dc_r_import, dc_r_export = calculate_import_export_separate(dc_lines_r, "dcr", n, marginal_price)

        # Total import/export
        total_import_cost = ac_import.add(ac_r_import, fill_value=0).add(dc_import, fill_value=0).add(dc_r_import, fill_value=0)
        total_export_revenue = ac_export.add(ac_r_export, fill_value=0).add(dc_export, fill_value=0).add(dc_r_export, fill_value=0)

        # Net cost
        total_import_cost_sum = total_import_cost.sum().sum()
        total_export_revenue_sum = total_export_revenue.sum().sum()
        net_cost = total_import_cost_sum - total_export_revenue_sum

        results[planning_horizon] = net_cost

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['net_cost'])
    results_df.index.name = 'planning_horizon'

    return results_df

def calculate_h2_import_export_costs(country, planning_horizons):
    def calculate_import_export_separate(flows, direction, network, marginal_prices):
     import_cost = pd.DataFrame(index=flows.index, columns=flows.columns)
     export_revenue = pd.DataFrame(index=flows.index, columns=flows.columns)

     for pipeline in flows.columns:
        # Always use network.links for pipelines
        if direction == 'forward':
            bus0 = network.links.loc[pipeline, "bus0"]
            bus1 = network.links.loc[pipeline, "bus1"]
        else:  # reverse
            bus0 = network.links.loc[pipeline, "bus1"]
            bus1 = network.links.loc[pipeline, "bus0"]

        if bus0 not in marginal_prices.columns or bus1 not in marginal_prices.columns:
            continue

        price_bus0 = marginal_prices[bus0]
        price_bus1 = marginal_prices[bus1]

        for t in flows.index:
            flow = flows.at[t, pipeline]
            if flow < 0:  # Import
                import_cost.at[t, pipeline] = -flow * price_bus1[t]
                export_revenue.at[t, pipeline] = 0
            else:  # Export
                import_cost.at[t, pipeline] = 0
                export_revenue.at[t, pipeline] = flow * price_bus0[t]

     return import_cost, export_revenue

    results = {}

    for planning_horizon in planning_horizons:
        # Load network
        n = loaded_files[planning_horizon]

        # Marginal prices
        marginal_price_filter = n.buses.carrier == "H2"
        marginal_price = n.buses_t.marginal_price.filter(items=marginal_price_filter[marginal_price_filter].index)
        marginal_price = marginal_price.drop(columns=["GB2 0"], errors='ignore')

        # AC lines
        filtered_h2_pipelines = (n.links.carrier == 'H2 pipeline') & (n.links.bus0.str[:2] == country)
        h2_pipelines = n.links_t.p0.filter(items=filtered_h2_pipelines[filtered_h2_pipelines].index)

        filtered_h2_pipelines_r = (n.links.carrier == 'H2 pipeline') & (n.links.bus1.str[:2] == country)
        h2_pipelines_r = n.links_t.p1.filter(items=filtered_h2_pipelines_r[filtered_h2_pipelines_r].index)

        # DC links
        filtered_h2_retro_pipelines = (n.links.carrier == 'H2 pipeline retrofitted') & (n.links.bus0.str[:2] == country)
        h2_retro_pipelines = n.links_t.p0.filter(items=filtered_h2_retro_pipelines[filtered_h2_retro_pipelines].index)


        filtered_h2_retro_pipelines_r = (n.links.carrier == 'H2 pipeline retrofitted') & (n.links.bus1.str[:2] == country)
        h2_retro_pipelines_r = n.links_t.p1.filter(items=filtered_h2_retro_pipelines_r[filtered_h2_retro_pipelines_r].index)

        # Calculate import/export
        h2_import, h2_export = calculate_import_export_separate(h2_pipelines, "ac0", n, marginal_price)
        h2_r_import, h2_r_export = calculate_import_export_separate(h2_pipelines_r, "acr", n, marginal_price)
        h2_retro_import, h2_retro_export = calculate_import_export_separate(h2_retro_pipelines, "dc0", n, marginal_price)
        h2_retro_r_import, h2_retro_r_export = calculate_import_export_separate(h2_retro_pipelines_r, "dcr", n, marginal_price)

        # Total import/export
        total_import_cost = h2_import.add(h2_r_import, fill_value=0).add(h2_retro_import, fill_value=0).add(h2_retro_r_import, fill_value=0)
        total_export_revenue = h2_export.add(h2_r_export, fill_value=0).add(h2_retro_export, fill_value=0).add(h2_retro_r_export, fill_value=0)

        # Net cost
        total_import_cost_sum = total_import_cost.sum().sum()
        total_export_revenue_sum = total_export_revenue.sum().sum()
        net_cost = total_import_cost_sum - total_export_revenue_sum

        results[planning_horizon] = net_cost

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['net_cost'])
    results_df.index.name = 'planning_horizon'

    return results_df


def costs(countries, results):
    costs = {}
    fn = snakemake.input.costs
    options = pd.read_csv(fn, index_col=[0, 1]).sort_index()
    for country in countries:
      net_costs_elc = calculate_elec_import_export_costs(country, planning_horizons)
      net_costs_h2 = calculate_h2_import_export_costs(country, planning_horizons)
      uranium = pd.read_excel(f"results/{study}/htmls/ChartData_{country}.xlsx",sheet_name="Chart 22", index_col=0,skiprows=2).drop(2020, axis=0)
      uranium = uranium["Uranium"]
      coal = pd.read_csv(f"results/{study}/country_csvs/total_imports_{country}.csv", index_col=0).drop(2020, axis=0)
      coal = coal["imp_cms_pe"]
      gas_val = pd.read_excel(f"results/{study}/htmls/ChartData_{country}.xlsx", sheet_name="Chart 24", index_col=0,skiprows=2).drop(2020, axis=0)
      gas_val = gas_val["Natural gas"]
      exports = pd.read_csv(f"results/{study}/country_csvs/exports_{country}.csv", index_col=0).drop(2020, axis=0)
      exports = exports.clip(lower=0)
      exports = exports.where(exports <= 0, -exports)
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
      if country != 'EU':
       elc_row = {"tech": "electricity imports/exports"}
       for year in net_costs_elc.index:
          elc_row[str(year)] = net_costs_elc.loc[year, 'net_cost']
       df = pd.concat([df, pd.DataFrame([elc_row])], ignore_index=True)
       h2_row = {"tech": "hydrogen imports/exports"}
       for year in net_costs_h2.index:
          h2_row[str(year)] = net_costs_h2.loc[year, 'net_cost']
       df = pd.concat([df, pd.DataFrame([h2_row])], ignore_index=True)
       ura_row = {"tech": "nuclear fuel"}
       for year in uranium.index:
           ura_row[str(year)] = uranium.loc[year] * options.loc[("uranium", "fuel"), "value"] * 1e6  
       df = pd.concat([df, pd.DataFrame([ura_row])], ignore_index=True)
       coal_row = {"tech": "coal fuel"}
       for year in coal.index:
           coal_row[str(year)] = coal.loc[year] * options.loc[("coal", "fuel"), "value"] * 1e6  
       df = pd.concat([df, pd.DataFrame([coal_row])], ignore_index=True)
       gas_row = {"tech": "gas fuel"}
       for year in gas_val.index:
           val = gas_val.loc[year]
           val = max(val, 0)
           gas_row[str(year)] = val * options.loc[("gas", "fuel"), "value"] * 1e6  
       df = pd.concat([df, pd.DataFrame([gas_row])], ignore_index=True)
       biomass_exports = exports["enc_pe_exp"]
       bm_row = {"tech": "biomass exports"}
       for year in biomass_exports.index:
           bm_row[str(year)] = biomass_exports.loc[year] * options.loc[("biomass", "fuel"), "value"] * 1e6
       df = pd.concat([df, pd.DataFrame([bm_row])], ignore_index=True)

       biogas_exports = exports["gaz_se_exp"]
       biogas_row = {"tech": "biogas exports"}
       for year in biogas_exports.index:
           biogas_row[str(year)] = biogas_exports.loc[year] * options.loc[("biogas", "fuel"), "value"] * 1e6
       df = pd.concat([df, pd.DataFrame([biogas_row])], ignore_index=True)

       meth_exports = exports["met_fe_exp"]
       meth_row = {"tech": "methanol exports"}
       for year in meth_exports.index:
           meth_row[str(year)] = meth_exports.loc[year] * methanol_fuel * 1e6
       df = pd.concat([df, pd.DataFrame([meth_row])], ignore_index=True)

       amm_exports = exports["amm_fe_exp"]
       amm_row = {"tech": "ammonia exports"}
       for year in amm_exports.index:
           amm_row[str(year)] = amm_exports.loc[year] * ammonia_fuel * 1e6
       df = pd.concat([df, pd.DataFrame([amm_row])], ignore_index=True)
      df['tech'] = df['tech'].map(rename_techs_tyndp)
      df = df.groupby('tech').sum().reset_index()

      mask = ~(df['tech'].isin(['load shedding']))
      result_df = df[mask]
      
      if not result_df.empty:
            years = ['2030', '2040', '2050']
            technologies = result_df['tech'].unique()
            
            costs[country] = result_df.set_index('tech').loc[technologies, years]

    for planning_horizon in planning_horizons:
      planning_horizon_str = str(planning_horizon)

      if planning_horizon in results:
        cos_ac_df = results[planning_horizon]['cos_ac']
        cos_dc_df = results[planning_horizon]['cos_dc']

        for country in countries:
            if country != 'EU':
                ac_transmission_values = cos_ac_df.loc[country, 'transmission_AC']
                dc_transmission_values = cos_dc_df.loc[country, 'transmission_DC']
                total_cost = ac_transmission_values + dc_transmission_values
                costs[country].loc['Transmission Lines', planning_horizon_str] = total_cost
            else:
                # Sum over all non-EU countries
                total_ac = cos_ac_df.loc[cos_ac_df.index != 'EU', 'transmission_AC'].sum()
                total_dc = cos_dc_df.loc[cos_dc_df.index != 'EU', 'transmission_DC'].sum()
                costs['EU'].loc['Transmission Lines', planning_horizon_str] = total_ac + total_dc
    if country == 'EU':
     costs['EU'].index = costs['EU'].index.map(rename_techs_tyndp_EU)
     costs['EU'] = costs['EU'].groupby(costs['EU'].index).sum()
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
      tech_mapping = {'Fossil Fuels': 'Oil & Gas Storage', 'Biomass': 'Biogas Plants'}
      df['tech'] = df['tech'].replace(tech_mapping)
      condition = df[['2030', '2040', '2050']].eq(0).all(axis=1)
      df = df[~condition]
      
      mask = ~(df['tech'].isin(['load shedding']))
      result_df = df[mask]
      if not result_df.empty:
            years = ['2030', '2040', '2050']
            technologies = result_df['tech'].unique()
            
            investment_costs[country] = result_df.set_index('tech').loc[technologies, years]

    for planning_horizon in planning_horizons:
      planning_horizon_str = str(planning_horizon)

      if planning_horizon in results:
        cos_ac_df = results[planning_horizon]['cos_ac']
        cos_dc_df = results[planning_horizon]['cos_dc']

        for country in countries:
            if country != 'EU':
                ac_transmission_values = cos_ac_df.loc[country, 'transmission_AC']
                dc_transmission_values = cos_dc_df.loc[country, 'transmission_DC']
                total_cost = ac_transmission_values + dc_transmission_values
                investment_costs[country].loc['Transmission Lines', planning_horizon_str] = total_cost
            else:
                # Sum over all non-EU countries
                total_ac = cos_ac_df.loc[cos_ac_df.index != 'EU', 'transmission_AC'].sum()
                total_dc = cos_dc_df.loc[cos_dc_df.index != 'EU', 'transmission_DC'].sum()
                investment_costs['EU'].loc['Transmission Lines', planning_horizon_str] = total_ac + total_dc
    if country == 'EU':
     investment_costs['EU'].index = investment_costs['EU'].index.map(rename_techs_tyndp_EU)
     investment_costs['EU'] = investment_costs['EU'].groupby(investment_costs['EU'].index).sum()
    for country, dataframe in investment_costs.items():
         # Specify the file path within the output directory
         file_path = f"results/{study}/country_csvs/{country}_investment costs.csv"
    
         # Save the DataFrame to a CSV file
         dataframe.to_csv(file_path, index=True)

         print(f"CSV file for {country} saved at: {file_path}")
        
    return investment_costs 
def rename_techs_tynd(tech):
    tech = rename_techs(tech)
    if tech in ["H2 Electrolysis", "methanation","helmeth", "H2 liquefaction","heat pump","resistive heater","Fischer-Tropsch",
                "electricity distribution grid","CHP", "H2 Fuel Cell","CCGT","OCGT","H2 turbine","solid biomass powerplants","coal powerplants", "oil powerplants",
                "battery charger", "battery discharger","battery", "Li ion", "EV charger", "V2G","hot water storage", "H2", "H2 storage",
                "biomass boiler", "oil boiler","gas boiler","solar","Wind","co2 sequestered","CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC","DAC",
                "hydroelectricity","SMR", "ammonia cracker", "Haber-Bosch", "BioSNG", "biomass to liquid","methanol","ammonia","methanolisation","shipping methanol",
                "air heat pump","air-sourced heat pump","ground heat pump","solar PV","solar rooftop", "offshore wind","offshore wind (AC)", "offshore wind (DC)",
                "onshore wind", "solar thermal","H2 pipeline", "gas pipeline","gas pipeline new","H2 pipeline retrofitted","transmission lines","Transmission Lines"]:
        return "VOM of Technologies"
    elif tech in ["biomass", "solid biomass", "solid biomass for industry", "biogas", "solid biomass transport", "biomass exports", "biogas exports"]:
          return "Biomass"
    elif tech in ["shipping oil", "naphtha for industry", "land transport oil", "kerosene for aviation", "agriculture machinery oil", "oil","gas", "coal for industry","gas for industry","coal","lignite","coal fuel","gas fuel"]:
          return "Fossil Fuels"
    elif "load" in tech:
        return "load shedding"
    elif "electricity imports/exports" in tech:
        return "Electricity Imports/Exports"
    elif "hydrogen imports/exports" in tech:
        return "Hydrogen Imports/Exports"
    elif tech in ["uranium", "nuclear", "nuclear fuel"]:
          return "Nuclear"
    elif tech in ["methanol exports", "ammonia exports"]:
          return "Synthetic Fuels"
    else:
        return tech
def operational_costs(countries, results):
    operational_costs = {}
    fn = snakemake.input.costs
    options = pd.read_csv(fn, index_col=[0, 1]).sort_index()
    for country in countries:
      net_costs_elc = calculate_elec_import_export_costs(country, planning_horizons)
      net_costs_h2 = calculate_h2_import_export_costs(country, planning_horizons)
      uranium = pd.read_excel(f"results/{study}/htmls/ChartData_{country}.xlsx",sheet_name="Chart 22", index_col=0,skiprows=2).drop(2020, axis=0)
      uranium = uranium["Uranium"]
      coal = pd.read_csv(f"results/{study}/country_csvs/total_imports_{country}.csv", index_col=0).drop(2020, axis=0)
      coal = coal["imp_cms_pe"]
      gas_val = pd.read_excel(f"results/{study}/htmls/ChartData_{country}.xlsx", sheet_name="Chart 24", index_col=0,skiprows=2).drop(2020, axis=0)
      gas_val = gas_val["Natural gas"]
      exports = pd.read_csv(f"results/{study}/country_csvs/exports_{country}.csv", index_col=0).drop(2020, axis=0)
      exports = exports.clip(lower=0)
      exports = exports.where(exports <= 0, -exports)
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
      df = df[df['Costs'] == 'marginal']
      df = df.groupby('tech').sum().reset_index()
      df = df.drop(columns=['Costs'])
      if country != 'EU':
       elc_row = {"tech": "electricity imports/exports"}
       for year in net_costs_elc.index:
          elc_row[str(year)] = net_costs_elc.loc[year, 'net_cost']
       df = pd.concat([df, pd.DataFrame([elc_row])], ignore_index=True)
       h2_row = {"tech": "hydrogen imports/exports"}
       for year in net_costs_h2.index:
          h2_row[str(year)] = net_costs_h2.loc[year, 'net_cost']
       df = pd.concat([df, pd.DataFrame([h2_row])], ignore_index=True)
       ura_row = {"tech": "nuclear fuel"}
       for year in uranium.index:
           ura_row[str(year)] = uranium.loc[year] * options.loc[("uranium", "fuel"), "value"] * 1e6  
       df = pd.concat([df, pd.DataFrame([ura_row])], ignore_index=True)
       coal_row = {"tech": "coal fuel"}
       for year in coal.index:
           coal_row[str(year)] = coal.loc[year] * options.loc[("coal", "fuel"), "value"] * 1e6  
       df = pd.concat([df, pd.DataFrame([coal_row])], ignore_index=True)
       gas_row = {"tech": "gas fuel"}
       for year in gas_val.index:
           val = gas_val.loc[year]
           val = max(val, 0)
           gas_row[str(year)] = val * options.loc[("gas", "fuel"), "value"] * 1e6 
       df = pd.concat([df, pd.DataFrame([gas_row])], ignore_index=True)
       biomass_exports = exports["enc_pe_exp"]
       bm_row = {"tech": "biomass exports"}
       for year in biomass_exports.index:
           bm_row[str(year)] = biomass_exports.loc[year] * options.loc[("biomass", "fuel"), "value"] * 1e6
       df = pd.concat([df, pd.DataFrame([bm_row])], ignore_index=True)

       biogas_exports = exports["gaz_se_exp"]
       biogas_row = {"tech": "biogas exports"}
       for year in biogas_exports.index:
           biogas_row[str(year)] = biogas_exports.loc[year] * options.loc[("biogas", "fuel"), "value"] * 1e6
       df = pd.concat([df, pd.DataFrame([biogas_row])], ignore_index=True)

       meth_exports = exports["met_fe_exp"]
       meth_row = {"tech": "methanol exports"}
       for year in meth_exports.index:
           meth_row[str(year)] = meth_exports.loc[year] * methanol_fuel * 1e6
       df = pd.concat([df, pd.DataFrame([meth_row])], ignore_index=True)

       amm_exports = exports["amm_fe_exp"]
       amm_row = {"tech": "ammonia exports"}
       for year in amm_exports.index:
           amm_row[str(year)] = amm_exports.loc[year] * ammonia_fuel * 1e6
       df = pd.concat([df, pd.DataFrame([amm_row])], ignore_index=True)
      df['tech'] = df['tech'].map(rename_techs_tynd)
      df = df.groupby('tech').sum().reset_index()
      condition = df[['2030', '2040', '2050']].eq(0).all(axis=1)
      df = df[~condition]
      
      mask = ~(df['tech'].isin(['load shedding']))
      result_df = df[mask]
      if not result_df.empty:
            years = ['2030', '2040', '2050']
            technologies = result_df['tech'].unique()
            
            operational_costs[country] = result_df.set_index('tech').loc[technologies, years]

      
    for country, dataframe in operational_costs.items():
         # Specify the file path within the output directory
         file_path = f"results/{study}/country_csvs/{country}_operational costs.csv"
    
         # Save the DataFrame to a CSV file
         dataframe.to_csv(file_path, index=True)

         print(f"CSV file for {country} saved at: {file_path}")
    return operational_costs 

def rename_techs_tyndpp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction","Haber-Bosch"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "hydrogen storage"
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
    elif tech in ["co2 sequestered","CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif tech in ["electricity distribution grid"]:
        return "distribution network"
    elif "hot water storage" in tech:
        return "thermal energy storage"
    elif "load" in tech:
        return "load shedding"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech     
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
      cf['tech'] = cf['tech'].map(rename_techs_tyndpp)
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
      result_df['tech'] = result_df['tech'].replace({'urban central water tanks': 'Thermal Energy storage', 'battery':'Grid-scale battery', 'Li ion':'EV battery','gas':'Gas storage'})
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

        
def create_bar_chart(costs, country, unit='Euros/year'):
    tech_colors = config["plotting"]["tech_colors"]
    tech_colors["AC Transmission"] = "#FF3030"
    tech_colors["DC Transmission"] = "#104E8B"

    title = f"{country} - Total Annual Costs"
    df = costs[country]
    df = df.rename_axis(unit)
    df = df.reset_index()
    df.index = df.index.astype(str)

    fig = go.Figure()
    df_transposed = df.set_index(unit).T

    for tech in df_transposed.columns:
        y = df_transposed[tech]
        color = tech_colors.get(tech, 'lightgrey')

        # Positive values
        fig.add_trace(go.Bar(
            x=df_transposed.index,
            y=y.where(y > 0, 0),
            name=tech,
            marker_color=color,
            width=0.62
        ))

        # Negative values (plotted separately, but still under the same name)
        fig.add_trace(go.Bar(
            x=df_transposed.index,
            y=y.where(y < 0, 0),
            name=tech,
            marker_color=color,
            width=0.62,
            showlegend=False  # avoid duplicate legend
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Euro reference value = 2020',
        marker=dict(color='rgba(0,0,0,0)')
    ))

    fig.update_layout(
        height=900, width=1000,
        title=title,
        barmode='relative',  # important for splitting positive and negative
        yaxis=dict(title=unit, title_font=dict(size=15), tickfont=dict(size=15)),
        xaxis=dict(tickfont=dict(size=15)),
        legend=dict(font=dict(size=15)),
        hovermode='y'
    )

    return fig

def create_investment_costs(investment_costs, country,  unit='Euros/year'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    colors["AC Transmission"] = "#FF3030"
    colors["DC Transmission"] = "#104E8B"
    tech_colors["Biogas Plants"] = tech_colors["Biomass"]

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
    fig.update_layout(height=900, width=1000,title=title, barmode='stack', yaxis=dict(title=unit,title_font=dict(size=15),tickfont=dict(size=15)),xaxis=dict(tickfont=dict(size=15)),legend=dict(font=dict(size=15)))
    fig.update_layout(hovermode='y')
    # fig.add_layout_image(logo)
    return fig

def create_operational_costs(operational_costs, country, unit='Euros/year'):
    tech_colors = config["plotting"]["tech_colors"]
    tech_colors["AC Transmission"] = "#FF3030"
    tech_colors["DC Transmission"] = "#104E8B"
    

    title = f"{country} - Operational Costs"
    df = operational_costs[country]
    df = df.rename_axis(unit)
    df = df.reset_index()
    df.index = df.index.astype(str)

    fig = go.Figure()
    df_transposed = df.set_index(unit).T

    for tech in df_transposed.columns:
        y = df_transposed[tech]
        color = tech_colors.get(tech, 'lightgrey')

        # Positive values
        fig.add_trace(go.Bar(
            x=df_transposed.index,
            y=y.where(y > 0, 0),
            name=tech,
            marker_color=color,
            width=0.62
        ))

        # Negative values
        fig.add_trace(go.Bar(
            x=df_transposed.index,
            y=y.where(y < 0, 0),
            name=tech,
            marker_color=color,
            width=0.62,
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Euro reference value = 2020',
        marker=dict(color='rgba(0,0,0,0)')
    ))

    fig.update_layout(
        height=900, width=1000,
        title=title,
        barmode='relative',
        yaxis=dict(title=unit, title_font=dict(size=15), tickfont=dict(size=15)),
        xaxis=dict(tickfont=dict(size=15)),
        legend=dict(font=dict(size=15)),
        hovermode='y'
    )

    return fig

def create_capacity_chart(capacities, country, unit='Capacity [GW]'):
    tech_colors = config["plotting"]["tech_colors"]
    colors = config["plotting"]["tech_colors"]
    # colors["AC Transmission lines"] = "#FF3030"
    # colors["DC Transmission lines"] = "#104E8B"
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

    # Create a subplot for each technology
    years = ['2030', '2040', '2050']
    if country != "EU":
        value = groups
    else:
        value = groupss
    def smart_capitalize(phrase):
     return phrase[0].upper() + phrase[1:] if phrase and not phrase[0].isupper() else phrase
    fig = make_subplots(rows=2, cols=len(value) // 2, subplot_titles=[", ".join(smart_capitalize(t) for t in tech_group) for tech_group in value], shared_yaxes=True)

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
                    name=smart_capitalize(tech),
                    marker_color=tech_colors.get(tech, 'gray')
                )
                fig.add_trace(trace, row=row_idx, col=col_idx)
                fig.update_yaxes(title_text=unit, row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, showlegend=True, title=f"Capacities for {country}", yaxis_title=unit,legend=dict(font=dict(size=15)))
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
    logo['y']=1.03
    # fig.add_layout_image(logo)
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
    colors["Gas storage"] = colors["gas"]
    groups = [
        ["Grid-scale battery"],
        ["Thermal Energy storage"],
        ["Gas storage"],
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
    fig.update_layout(height=500, width=1200, showlegend=True, title=f" Storage Capacities for {country}", yaxis_title=unit,legend=dict(font=dict(size=15)))
    num_cols = len(groups)
    for col in range(1, num_cols + 1):
     fig.update_xaxes(tickfont=dict(size=13), row=1, col=col)
     fig.update_yaxes(
        tickfont=dict(size=13),
        title_font=dict(size=15),
        row=1,
        col=col
    )
    logo['y']=1.03
    # fig.add_layout_image(logo)
    

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
    combined_html += f"<div><h2>{country} - Annual Costs</h2>{bar_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    # Create Investment Costs
    bar_chart_investment = create_investment_costs(investment_costs, country)
    combined_html += f"<div><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    bar_chart_operational = create_operational_costs(operational_costs, country)
    combined_html += f"<div><h2>{country} - Annual Operational Costs</h2>{bar_chart_operational.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    # Create capacities chart
    capacities_chart = create_capacity_chart(capacities, country)
    combined_html += f"<div><h2>{country} - Capacities </h2>{capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    
    # Create storage capacities chart
    s_capacities_chart = storage_capacity_chart(s_capacities, country)
    combined_html += f"<div><h2>{country} - Storage Capacities </h2>{s_capacities_chart.to_html(full_html=False, include_plotlyjs='cdn')}</div>"


    # Create the content for the "Table of Contents" and "Main" sections
    table_of_contents_content = f"<a href='#{country} - Annual Costs'>Annual Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Annual Investment Costs'>Annual Investment Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Annual Operational Costs'>Annual Operational Costs</a><br>"
    table_of_contents_content += f"<a href='#{country} - Capacities'>Capacities</a><br>"
    table_of_contents_content += f"<a href='#{country} - Storage Capacities'>Storage Capacities</a><br>"
    
    # Add more links for other plots

    main_content = f"<div id='{country} - Annual Costs'><h2>{country} - Annual Costs</h2>{bar_chart.to_html()}</div>"
    main_content += f"<div id='{country} - Annual Investment Costs'><h2>{country} - Annual Investment Costs</h2>{bar_chart_investment.to_html()}</div>"
    main_content += f"<div id='{country} - Annual Operational Costs'><h2>{country} - Annual Operational Costs</h2>{bar_chart_operational.to_html()}</div>"
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
    methanol_fuel = 119 #https://www.methanol.org/wp-content/uploads/2023/05/Marine_Methanol_Report_Methanol_Institute_May_2023.pdf
    ammonia_fuel = 92 #https://www.iee.fraunhofer.de/en/presse-infothek/press-media/2022/green-ammonia-for-climate-protection.html
    total_country = 'EU'
    countries = snakemake.params.countries 
    fn = snakemake.input.costs
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
    operational_costs = operational_costs(countries, results)
    capacities = capacities(countries, results)
    s_capacities = storage_capacities(countries)
    
    
    for country in countries:
        create_combined_chart_country(costs,investment_costs, capacities,s_capacities, country)
    


