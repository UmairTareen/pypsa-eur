#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:07:32 2025

@author: umair
"""

import pandas as pd
import os
import pypsa
import sys
import logging
current_script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_script_dir, "../scripts/")
sys.path.append(scripts_path)
from add_electricity import calculate_annuity


def capacities_from_JRC(country):    
   year = 2019
   JRC_year = 2021
   conversion_factor = 11630  #ktoe to Mwh
   full_load_hours = 2500  #assumption on full load hours to get boiler capacities from FEC
   full_load_hours_elc = 3500

   fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
   fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
   fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"

   df_elc = pd.read_excel(fn_power, "Cap", index_col=0)[year]
   df_chp = pd.read_excel(fn_power, "Cap_CHP", index_col=0)[year]
   df_res = pd.read_excel(fn_residential, "RES_hh_fec", index_col=0)[year]
   df_ter = pd.read_excel(fn_tertiary, "SER_hh_fec", index_col=0)[year]
   df_res_tot = pd.read_excel(fn_residential, "RES_summary", index_col=0)[year]
   df_ter_tot = pd.read_excel(fn_tertiary, "SER_summary", index_col=0)[year]

   #Residential boiler capacities
   gas_boiler_val = df_res.loc["Natural gas"].sum() * conversion_factor
   gas_boiler_cap = gas_boiler_val / full_load_hours
   oil_boiler_val = df_res.loc[["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"]].sum() * conversion_factor
   oil_boiler_cap = oil_boiler_val / full_load_hours
   bm_boiler_val = df_res.loc["Biomass"].sum() * conversion_factor
   bm_boiler_cap = bm_boiler_val / full_load_hours
   elc_boiler_val = df_res.loc[["Conventional electric heating","Electricity"]].sum() * conversion_factor
   elc_boiler_cap = elc_boiler_val / full_load_hours
   hp_val = df_res.loc["Advanced electric heating"].sum() * conversion_factor
   hp_cap = hp_val / full_load_hours

   #tertiary boiler capacities
   gas_boiler_ter = df_ter.loc[["Conventional gas heaters", "Natural gas"]].sum() * conversion_factor
   gas_boiler_cap_ter = gas_boiler_ter / full_load_hours
   oil_boiler_ter = df_ter.loc[["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"]].sum() * conversion_factor
   oil_boiler_cap_ter = oil_boiler_ter / full_load_hours
   bm_boiler_ter = df_ter.loc["Biomass"].sum() * conversion_factor
   bm_boiler_cap_ter = bm_boiler_ter / full_load_hours
   elc_boiler_ter = df_ter.loc[["Conventional electric heating","Electricity"]].sum() * conversion_factor
   elc_boiler_cap_ter = elc_boiler_ter / full_load_hours
   hp_ter = df_ter.loc["Advanced electric heating"].sum() * conversion_factor
   hp_cap_ter = hp_ter / full_load_hours

   #Getting electricity distribution grid cap
   elc_res = df_res_tot.loc["Electricity"].sum() * conversion_factor
   elc_ter = df_ter_tot.loc["Electricity"].sum() * conversion_factor
   dist_grid = (elc_res + elc_ter) / full_load_hours_elc

   #Extracting transmission lines values from TYNDP2020
   ac_lines = config["TYNDP_values"]
   dc_lines = config["DC_transmission_line"]
   country_lower = country.lower()  # convert "BE" → "be"
   ac_total = sum(value for key, value in ac_lines.items() if country_lower in key)
   dc_total = sum(value for key, value in dc_lines.items() if country_lower in key)

   #Considering gas storage values from gasgrid data
   if country == "BE":
     gas_storage = 7595598.6
   elif country == "FR":
     gas_storage = 108092625.4
   elif country == "DE":
     gas_storage = 191327414.5
   elif country == "NL":
     gas_storage = 96227488.0
   def make_row(cluster, country, tech, types, df):
     return {
        "cluster": cluster,
        "country": country,
        "tech": tech,
        "2020": df.loc[types].sum() if isinstance(types, list) else df.loc[types].sum()
    }

   ct_totals = [
    make_row("links", country, "nuclear", "Nuclear power plants", df_elc),
    make_row("links", country, "coal powerplants", "Coal power plants", df_elc),
    make_row("links", country, "CCGT", "Gas turbine combined cycle", df_elc),
    make_row("links", country, "OCGT", ["Gas turbine ","Steam turbine","Internal combustion engine","Biogas power plants (dedicated)","Derived gas power plants","Refinery gas power plants"], df_elc),
    make_row("links", country, "oil powerplants", ["Diesel oil power plants","Fuel oil power plants"], df_elc),
    make_row("links", country, "solid biomass powerplants", ["Solid biomass power plants (incl. waste cofiring)","Waste power plants (dedicated)"], df_elc),
    make_row("generators", country, "onwind", "Onshore", df_elc),
    make_row("generators", country, "offwind", "Offshore", df_elc),
    make_row("generators", country, "solar", "Solar PV power plants", df_elc),
    make_row("generators", country, "ror", "Run-of-river", df_elc),
    make_row("storage_units", country, "hydro", "Reservoirs (dams)", df_elc),
    make_row("storage_units", country, "PHS", "Pumped storage", df_elc),
    make_row("links", country, "urban central coal CHP", ["Coal power plants","Lignite power plants"], df_chp),
    make_row("links", country, "urban central gas CHP", ["Gas turbine ","Steam turbine","Internal combustion engine","Biogas power plants (dedicated)","Derived gas power plants","Refinery gas power plants"], df_chp),
    make_row("links", country, "urban central oil CHP", ["Diesel oil power plants","Fuel oil power plants"], df_chp),
    make_row("links", country, "urban central solid biomass CHP",["Solid biomass power plants (incl. waste cofiring)","Waste power plants (dedicated)"], df_chp),
    
    
]
   for tech, cap in [
    ("residential urban decentral gas boiler", gas_boiler_cap),
    ("residential urban decentral oil boiler", oil_boiler_cap),
    ("residential urban decentral biomass boiler", bm_boiler_cap),
    ("residential urban decentral resistive heater", elc_boiler_cap),
    ("residential urban decentral air heat pump", hp_cap),
    ("services urban decentral gas boiler", gas_boiler_cap_ter),
    ("services urban decentral oil boiler", oil_boiler_cap_ter),
    ("services urban decentral biomass boiler", bm_boiler_cap_ter),
    ("services urban decentral resistive heater", elc_boiler_cap_ter),
    ("services urban decentral air heat pump", hp_cap_ter),
    ("electricity distribution grid", dist_grid),
    ("DC Transmission lines", dc_total),
]:
    ct_totals.append({
        "cluster": "links",
        "country": country,
        "tech": tech,
        "2020": cap
    })
   ct_totals.append({
    "cluster": "lines",
    "country": country,
    "tech": "AC Transmission lines",
    "2020": ac_total
})
   ct_totals.append({
    "cluster": "stores",
    "country": country,
    "tech": "gas",
    "2020": gas_storage
})
   
   totals_df = pd.DataFrame(ct_totals)
   return totals_df
#%%

def capacities_GB_from_JRC(country):    
   year = 2015
   JRC_year = 2015
   conversion_factor = 11630  #ktoe to Mwh
   full_load_hours = 2500  #assumption on full load hours to get boiler capacities from FEC
   full_load_hours_elc = 3500

   fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
   fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
   fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"

   df_elc = pd.read_excel(fn_power, "Cap", index_col=0)[year]
   df_chp = pd.read_excel(fn_power, "Cap_CHP", index_col=0)[year]
   df_res = pd.read_excel(fn_residential, "RES_hh_fec", index_col=0)[year]
   df_ter = pd.read_excel(fn_tertiary, "SER_hh_fec", index_col=0)[year]
   df_res_tot = pd.read_excel(fn_residential, "RES_summary", index_col=0)[year]
   df_ter_tot = pd.read_excel(fn_tertiary, "SER_summary", index_col=0)[year]

   #Residential boiler capacities
   gas_boiler_val = df_res.loc["Gases incl. biogas"].sum() * conversion_factor
   gas_boiler_cap = gas_boiler_val / full_load_hours
   oil_boiler_val = df_res.loc[["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"]].sum() * conversion_factor
   oil_boiler_cap = oil_boiler_val / full_load_hours
   bm_boiler_val = df_res.loc["Biomass and wastes"].sum() * conversion_factor
   bm_boiler_cap = bm_boiler_val / full_load_hours
   elc_boiler_val = df_res.loc[["Conventional electric heating","Electricity"]].sum() * conversion_factor
   elc_boiler_cap = elc_boiler_val / full_load_hours
   hp_val = df_res.loc["Advanced electric heating"].sum() * conversion_factor
   hp_cap = hp_val / full_load_hours

   #tertiary boiler capacities
   gas_boiler_ter = df_ter.loc[["Conventional gas heaters", "Gases incl. biogas"]].sum() * conversion_factor
   gas_boiler_cap_ter = gas_boiler_ter / full_load_hours
   oil_boiler_ter = df_ter.loc[["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"]].sum() * conversion_factor
   oil_boiler_cap_ter = oil_boiler_ter / full_load_hours
   bm_boiler_ter = df_ter.loc["Biomass and wastes"].sum() * conversion_factor
   bm_boiler_cap_ter = bm_boiler_ter / full_load_hours
   elc_boiler_ter = df_ter.loc[["Conventional electric heating","Electricity"]].sum() * conversion_factor
   elc_boiler_cap_ter = elc_boiler_ter / full_load_hours
   hp_ter = df_ter.loc["Advanced electric heating"].sum() * conversion_factor
   hp_cap_ter = hp_ter / full_load_hours

   #Getting electricity distribution grid cap
   elc_res = df_res_tot.loc["Electricity"].sum() * conversion_factor
   elc_ter = df_ter_tot.loc["Electricity"].sum() * conversion_factor
   dist_grid = (elc_res + elc_ter) / full_load_hours_elc

   #Extracting transmission lines values from TYNDP2020
   ac_lines = config["TYNDP_values"]
   dc_lines = config["DC_transmission_line"]
   country_lower = country.lower()  # convert "BE" → "be"
   ac_total = sum(value for key, value in ac_lines.items() if country_lower in key)
   dc_total = sum(value for key, value in dc_lines.items() if country_lower in key)

   #Considering gas storage values from gasgrid data
   if country == "GB":
     gas_storage = 43927350.7 + 3184393.4

   def make_row(cluster, country, tech, types, df):
     return {
        "cluster": cluster,
        "country": country,
        "tech": tech,
        "2020": df.loc[types].sum() if isinstance(types, list) else df.loc[types].sum()
    }

   ct_totals = [
    make_row("links", country, "nuclear", "Nuclear power plants", df_elc),
    make_row("links", country, "coal powerplants", "Coal fired power plants", df_elc),
    make_row("links", country, "CCGT", "Gas turbine combined cycle", df_elc),
    make_row("links", country, "OCGT", ["Gas turbine ","Steam turbine","Internal combustion engine","Derived gas fired power plants","Refinery gas fired power plants"], df_elc),
    make_row("links", country, "oil powerplants", ["Diesel oil fired power plants","Fuel oil fired power plants"], df_elc),
    make_row("links", country, "solid biomass powerplants", "Biomass and waste fired power plants", df_elc),
    make_row("generators", country, "onwind", "Onshore", df_elc),
    make_row("generators", country, "offwind", "Offshore", df_elc),
    make_row("generators", country, "solar", "Solar PV power plants", df_elc),
    make_row("generators", country, "ror", "Run-of-river", df_elc),
    make_row("storage_units", country, "hydro", "Reservoirs (dams)", df_elc),
    make_row("storage_units", country, "PHS", "Pump storage", df_elc),
    make_row("links", country, "urban central coal CHP", ["Coal fired power plants","Lignite fired power plants"], df_chp),
    make_row("links", country, "urban central gas CHP", ["Gas turbine ","Steam turbine","Internal combustion engine","Derived gas fired power plants","Refinery gas fired power plants"], df_chp),
    make_row("links", country, "urban central oil CHP", ["Diesel oil fired power plants","Fuel oil fired power plants"], df_chp),
    make_row("links", country, "urban central solid biomass CHP","Biomass and waste fired power plants", df_chp),
    
    
]
   for tech, cap in [
    ("residential urban decentral gas boiler", gas_boiler_cap),
    ("residential urban decentral oil boiler", oil_boiler_cap),
    ("residential urban decentral biomass boiler", bm_boiler_cap),
    ("residential urban decentral resistive heater", elc_boiler_cap),
    ("residential urban decentral air heat pump", hp_cap),
    ("services urban decentral gas boiler", gas_boiler_cap_ter),
    ("services urban decentral oil boiler", oil_boiler_cap_ter),
    ("services urban decentral biomass boiler", bm_boiler_cap_ter),
    ("services urban decentral resistive heater", elc_boiler_cap_ter),
    ("services urban decentral air heat pump", hp_cap_ter),
    ("electricity distribution grid", dist_grid),
    ("DC Transmission lines", dc_total),
]:
    ct_totals.append({
        "cluster": "links",
        "country": country,
        "tech": tech,
        "2020": cap
    })
   ct_totals.append({
    "cluster": "lines",
    "country": country,
    "tech": "AC Transmission lines",
    "2020": ac_total
})
   ct_totals.append({
    "cluster": "stores",
    "country": country,
    "tech": "gas",
    "2020": gas_storage
})
   
   totals_df = pd.DataFrame(ct_totals)
   return totals_df

def prepare_costs_2020(country):
  n=pypsa.Network(f"resources/{study}/networks/elec_s_6_ec_lvopt_.nc")   
  costs = pd.read_csv("data/costs_2020.csv", index_col=[0, 1]).sort_index()
  costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
  costs = (
    costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1))
  costs = costs.fillna(config["costs"]["fill_values"])
  def annuity_factor(v):
    return calculate_annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100

  costs["fixed"] = [
    annuity_factor(v) * v["investment"] for i, v in costs.iterrows()]

  capacities = pd.read_csv(f"results/{study}/country_csvs/capacities_{country}.csv", index_col=2)
  ac_lines_cap = capacities.loc["AC Transmission lines", "2020"].sum()
  dc_lines_cap = capacities.loc["DC Transmission lines", "2020"].sum()
  ac_length = n.lines.loc[
    n.lines["bus0"].str.contains(country, case=False) | 
    n.lines["bus1"].str.contains(country, case=False),
    "length"].sum()
  dc_length = n.links.loc[
    (n.links["carrier"] == "DC") &
    (n.links["bus0"].str.contains(country, case=False) |
     n.links["bus1"].str.contains(country, case=False)),
    "length"].sum()

  ac_lines_inv = ac_lines_cap * costs.at["HVAC overhead", "fixed"] * ac_length
  dc_lines_inv = dc_lines_cap * costs.at["HVDC overhead", "fixed"] * dc_length
  transmission_lines = ac_lines_inv + dc_lines_inv
  techs = {
    "CCGT":         {"cluster": "links",      "cost_name": "CCGT"},
    "offwind":      {"cluster": "generators", "cost_name": "offwind"},
    "onwind":       {"cluster": "generators", "cost_name": "onwind"},
    "solar":        {"cluster": "generators", "cost_name": "solar"},
    "OCGT":         {"cluster": "links",      "cost_name": "OCGT"},
    "ror":          {"cluster": "generators", "cost_name": "ror"},
    "nuclear":      {"cluster": "links",      "cost_name": "nuclear"},
    "coal powerplants":      {"cluster": "links",      "cost_name": "coal"},
    "oil powerplants":      {"cluster": "links",      "cost_name": "oil"},
    "solid biomass powerplants":      {"cluster": "links",      "cost_name": "biomass"},
    "hydro":         {"cluster": "storage_units",      "cost_name": "hydro"},
    "PHS":         {"cluster": "storage_units",      "cost_name": "PHS"},
    "urban central coal CHP":         {"cluster": "links",      "cost_name": "central coal CHP"},
    "urban central gas CHP":         {"cluster": "links",      "cost_name": "central gas CHP"},
    "urban central oil CHP":         {"cluster": "links",      "cost_name": "central gas CHP"},
    "urban central solid biomass CHP":         {"cluster": "links",      "cost_name": "biomass CHP"},
    "residential urban decentral gas boiler":         {"cluster": "links",      "cost_name": "decentral gas boiler"},
    "residential urban decentral oil boiler":         {"cluster": "links",      "cost_name": "decentral oil boiler"},
    "residential urban decentral biomass boiler":         {"cluster": "links",      "cost_name": "biomass boiler"},
    "residential urban decentral resistive heater":         {"cluster": "links",      "cost_name": "decentral resistive heater"},
    "residential urban decentral air heat pump":         {"cluster": "links",      "cost_name": "decentral air-sourced heat pump"},
    "services urban decentral gas boiler":         {"cluster": "links",      "cost_name": "decentral gas boiler"},
    "services urban decentral oil boiler":         {"cluster": "links",      "cost_name": "decentral oil boiler"},
    "services urban decentral biomass boiler":         {"cluster": "links",      "cost_name": "biomass boiler"},
    "services urban decentral resistive heater":         {"cluster": "links",      "cost_name": "decentral resistive heater"},
    "services urban decentral air heat pump":         {"cluster": "links",      "cost_name": "decentral air-sourced heat pump"},
    "electricity distribution grid":         {"cluster": "links",      "cost_name": "electricity distribution grid"},
    "gas":         {"cluster": "stores",      "cost_name": "gas storage"},
}

  ct_totals = []

  for tech, meta in techs.items():
    cluster = meta["cluster"]
    cost_name = meta["cost_name"]

    cap = capacities.loc[tech, "2020"].sum()
    inv = cap * costs.at[cost_name, "fixed"] * costs.at[cost_name, "efficiency"]

    ct_totals.append({
        "cluster": cluster,
        "country": country,
        "costss": "capital",
        "tech": tech,
        "2020": inv
    })
    
  ct_totals.append({
    "cluster": "lines",
    "country": country,
    "costss": "capital",
    "tech": "Transmission Lines",
    "2020": transmission_lines
})

  prod = pd.read_excel(f"results/{study}/sepia/inputs_{country}.xlsx", sheet_name="Inputs", index_col="label")
  prod = prod.groupby(level=0).sum(numeric_only=True) * 1e6
  techs_marginal = {
    "Gas-fired power generation":         {"cluster": "links",      "cost_name": "CCGT"},
    "wind-generated electricity":      {"cluster": "generators", "cost_name": "offwind"},
    "Solar photovoltaic Production":        {"cluster": "generators", "cost_name": "solar"},
    "Total ror production":          {"cluster": "generators", "cost_name": "ror"},
    "Nuclear production":      {"cluster": "links",      "cost_name": "nuclear"},
    "Coal-fired power generation":      {"cluster": "links",      "cost_name": "coal"},
    "Oil generation":      {"cluster": "links",      "cost_name": "oil"},
    "solid biomass power plants":      {"cluster": "links",      "cost_name": "biomass"},
    "Total hydropower production":         {"cluster": "storage_units",      "cost_name": "PHS"},
    "Power output from coal-fired CHP plants":         {"cluster": "links",      "cost_name": "central coal CHP"},
    "Power output from methane-fired CHP plants":         {"cluster": "links",      "cost_name": "central gas CHP"},
    "Power output from oil-fired CHP plants":         {"cluster": "links",      "cost_name": "central gas CHP"},
    "Power output from solid biomass CHP plants":         {"cluster": "links",      "cost_name": "biomass CHP"},
    "Residential and tertiary gas for heating":         {"cluster": "links",      "cost_name": "decentral gas boiler"},
    "Residential and tertiary oil for heating":         {"cluster": "links",      "cost_name": "decentral oil boiler"},
    "Residential and tertiary biomass for heating":         {"cluster": "links",      "cost_name": "biomass boiler"},
    "Residential and tertairy heat from electric heaters":         {"cluster": "links",      "cost_name": "decentral resistive heater"},
    "Residential and tertiary HP for heating":         {"cluster": "links",      "cost_name": "decentral air-sourced heat pump"},
}
  for tech, meta in techs_marginal.items():
    cluster = meta["cluster"]
    cost_name = meta["cost_name"]

    supply = prod.loc[tech, "2020"].sum()
    marginal = supply * costs.at[cost_name, "VOM"]

    ct_totals.append({
        "cluster": cluster,
        "country": country,
        "costss": "marginal",
        "tech": cost_name,
        "2020": marginal
    })
  
  imports = pd.read_csv(f"results/{study}/country_csvs/total_imports_{country}.csv", index_col=0)
  local = pd.read_csv(f"results/{study}/country_csvs/local_product_{country}.csv", index_col=0)

  gas_fuel = (imports.loc[2020, "imp_gaz_pe"] + local.loc[2020, "prod_gaz_pe"]) * costs.loc[("gas", "fuel")] * 1e6
  oil_fuel = (imports.loc[2020, "imp_pet_pe"] + local.loc[2020, "prod_pet_pe"]) * costs.loc[("oil", "fuel")] * 1e6
  bm_fuel = (imports.loc[2020, "imp_enc_pe"] + local.loc[2020, "prod_enc_pe"]) * costs.loc[("biomass", "fuel")] * 1e6
  coal_fuel = (imports.loc[2020, "imp_cms_pe"] + local.loc[2020, "prod_cms_pe"]) * costs.loc[("coal", "fuel")] * 1e6
  ura_fuel = local.loc[2020, "prod_ura_pe"] * costs.loc[("uranium", "fuel")] * 1e6
  fuel_costs = {
    "gas": gas_fuel,
    "oil": oil_fuel,
    "biomass": bm_fuel,
    "coal": coal_fuel,
    "uranium": ura_fuel
}

  for tech, marginal in fuel_costs.items():
    ct_totals.append({
        "cluster": "links",  # or "generators" depending on your use case
        "country": country,
        "costss": "marginal",
        "tech": tech,
        "2020": marginal
    })
  
  costs_df = pd.DataFrame(ct_totals)
  return costs_df

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "generate_capacties_costs")

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    config = snakemake.config
    countries = ['BE', 'DE', 'FR', 'NL']
    countriess = ['GB']
    study = snakemake.params.study
    os.makedirs(f"results/{study}/country_csvs", exist_ok=True)
    all_totals = []
    # Create separate files for each country
    for country in countries:
     totals_df = capacities_from_JRC(country)
     totals_df.to_csv(f"results/{study}/country_csvs/capacities_{country}.csv", index=False)
     all_totals.append(totals_df)

    for country in countriess:
     totals_df = capacities_GB_from_JRC(country)
     totals_df.to_csv(f"results/{study}/country_csvs/capacities_{country}.csv", index=False)
     all_totals.append(totals_df)

    eu_df = pd.concat(all_totals, ignore_index=True)
    eu_df.to_csv(f"results/{study}/country_csvs/capacities_EU.csv", index=False)
     
    countriesss=snakemake.params.countries
    all_costs = []
    for country in countriesss:
      costs_df = prepare_costs_2020(country)
      all_costs.append(costs_df)
      costs_df.to_csv(f"results/{study}/country_csvs/costs_{country}.csv", index=False)
    eu_costs_df = pd.concat(all_costs, ignore_index=True)
    eu_costs_df.to_csv(f"results/{study}/country_csvs/costs_EU.csv", index=False)