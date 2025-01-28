# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build industrial energy demand per model region.
"""

import pandas as pd
from _helpers import set_scenario_config

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industrial_energy_demand_per_node",
            simpl="",
            clusters=48,
            planning_horizons=2030,
        )
    set_scenario_config(snakemake)

    # import ratios
    fn = snakemake.input.industry_sector_ratios
    sector_ratios = pd.read_csv(fn, header=[0, 1], index_col=0)

    # material demand per node and industry (Mton/a)
    fn = snakemake.input.industrial_production_per_node
    nodal_production = pd.read_csv(fn, index_col=0) / 1e3
    countrries = snakemake.config['countries']
    config=snakemake.config 
    if config["run"]["name"] == "suff" or "sensitivity_analysis" in config["run"]["name"]:
      def clever_industry_data():
        fn = snakemake.input.clever_industry
        df= pd.read_csv(fn ,index_col=0)/1e3
        return df
      clever_Industry = clever_industry_data()   
      for country in countrries:
          # Filter rows in nodal_production DataFrame where index starts with the country code
          country_production = nodal_production[nodal_production.index.str.startswith(country)]
          #country_production = country_production[~country_production.index.isin(['DK2 0','ES4 0', 'GB5 0', 'IT3 0'])] (for 28 countries)
          country_production = country_production[~country_production.index.isin([f'{country}2 0'])]
          # Update values in nodal_production DataFrame using clever_Industry DataFrame
          nodal_production.loc[country_production.index, 'Electric arc'] = clever_Industry.loc[country, 'Production of primary steel']
          nodal_production.loc[country_production.index, 'DRI + Electric arc'] = clever_Industry.loc[country, 'Production of recycled steel']
          nodal_production.loc[country_production.index, 'Cement'] = clever_Industry.loc[country, 'Production of cement']
          nodal_production.loc[country_production.index, 'Glass production'] = clever_Industry.loc[country, 'Production of glass']
          nodal_production.loc[country_production.index, 'Other chemicals'] = clever_Industry.loc[country, 'Production of high value chemicals ']
          nodal_production.loc[country_production.index, 'Paper production'] = clever_Industry.loc[country, 'Production of paper']
    else:
      nodal_production = nodal_production
    # energy demand today to get current electricity
    fn = snakemake.input.industrial_energy_demand_per_node_today
    nodal_today = pd.read_csv(fn, index_col=0)

    nodal_sector_ratios = pd.concat(
        {node: sector_ratios[node[:2]] for node in nodal_production.index}, axis=1
    )

    nodal_production_stacked = nodal_production.stack()
    nodal_production_stacked.index.names = [None, None]

    # final energy consumption per node and industry (TWh/a)
    nodal_df = (
        (nodal_sector_ratios.multiply(nodal_production_stacked))
        .T.groupby(level=0)
        .sum()
    )

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
        "heat": "low-temperature heat",
    }
    nodal_df.rename(columns=rename_sectors, inplace=True)

    nodal_df["current electricity"] = nodal_today["electricity"]

    nodal_df.index.name = "TWh/a (MtCO2/a)"
    
    def clever_industry_data():
        fn = snakemake.input.clever_industry
        df= pd.read_csv(fn ,index_col=0)
        return df
    clever_Industry = clever_industry_data() 
       
    for country in countrries:
     if config["run"]["name"] == "suff" or "sensitivity_analysis" in config["run"]["name"]: 
        country_energy = nodal_df[nodal_df.index.str.startswith(country)]
        country_energy = country_energy[~country_energy.index.isin([f'{country}2 0'])]
        #country_energy = country_energy[~country_energy.index.isin(['DK2 0','ES4 0', 'GB5 0', 'IT3 0'])] (for 28 countries)
        # nodal_df.loc[country_energy.index, 'ammonia'] = clever_Industry.loc[country, 'Total Final Energy Consumption of the ammonia industry']
        nodal_df.loc[country_energy.index, 'electricity'] = clever_Industry.loc[country, 'Total Final electricity consumption in industry']
        nodal_df.loc[country_energy.index, 'coal'] = clever_Industry.loc[country, 'Total Final energy consumption from solid fossil fuels (coal ...) in industry']
        nodal_df.loc[country_energy.index, 'solid biomass'] = clever_Industry.loc[country, 'Total Final energy consumption from solid biomass in industry']
        nodal_df.loc[country_energy.index, 'methane'] = clever_Industry.loc[country, 'Total Final energy consumption from gas grid / gas consumed locally in industry']
        nodal_df.loc[country_energy.index, 'low-temperature heat'] = clever_Industry.loc[country, 'Total Final heat consumption in industry']
        nodal_df.loc[country_energy.index, 'hydrogen'] = clever_Industry.loc[country, 'Total Final hydrogen consumption in industry'] + clever_Industry.loc[country, 'Non-energy consumption of hydrogen for the feedstock production'].sum()
        nodal_df.loc[country_energy.index, 'naphtha'] = clever_Industry.loc[country, 'Non-energy consumption of oil for the feedstock production']
    
     else:
      nodal_df = nodal_df
     
    if config["run"]["name"] == "baseline":
      nodal_df.loc['BE1 0', 'naphtha']  =  84.4
      nodal_df.loc['BE1 0', 'coke']  =  15
      
      
    if config["run"]["name"] == "ref":
      nodal_df.loc['BE1 0', 'naphtha']  =  84.4

    fn = snakemake.output.industrial_energy_demand_per_node
    nodal_df.to_csv(fn, float_format="%.2f")
