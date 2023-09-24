# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build industrial energy demand per model region.
"""

import pandas as pd
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
paths = os.path.join(os.path.dirname(__file__), '../data/')
def clever_industry_data():
    years = int(snakemake.wildcards.planning_horizons)
    df= pd.read_csv(f'{paths}/clever_Industry_{years}.csv',index_col=0)
    return df

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industrial_energy_demand_per_node",
            simpl="",
            clusters=48, #48
            planning_horizons=2030, #2030
        )
    
    params = snakemake.params.planning_horizons
    # import EU ratios df as csv
    fn = snakemake.input.industry_sector_ratios
    industry_sector_ratios = pd.read_csv(fn, index_col=0)
    
    countrries = ['BE', 'DE', 'FR', 'GB', 'NL']
    clever_Industry = clever_industry_data()

    # material demand per node and industry (kton/a)
    fn = snakemake.input.industrial_production_per_node
    nodal_production = pd.read_csv(fn, index_col=0)
    new_row_names = {
    'BE1 0': 'BE',
    'DE1 0': 'DE',
    'FR1 0': 'FR',
    'GB0 0': 'GB',
    'NL1 0': 'NL'}
    nodal_production.rename(index=new_row_names, inplace=True)
    for country in countrries:
        nodal_production.loc[country, 'Electric arc'] = clever_Industry.loc[country, 'Production of primary steel']
        nodal_production.loc[country, 'DRI + Electric arc'] = clever_Industry.loc[country, 'Production of recycled steel']
        nodal_production.loc[country, 'Cement'] = clever_Industry.loc[country, 'Production of cement']
        nodal_production.loc[country, 'Glass production'] = clever_Industry.loc[country, 'Production of glass']
        nodal_production.loc[country, 'Other chemicals'] = clever_Industry.loc[country, 'Production of high value chemicals ']
        nodal_production.loc[country, 'Paper production'] = clever_Industry.loc[country, 'Production of paper']
    new_row_names = {
    'BE': 'BE1 0',
    'DE': 'DE1 0',
    'FR': 'FR1 0',
    'GB': 'GB0 0',
    'NL': 'NL1 0'}
    nodal_production.rename(index=new_row_names, inplace=True)
    # energy demand today to get current electricity
    fn = snakemake.input.industrial_energy_demand_per_node_today
    nodal_today = pd.read_csv(fn, index_col=0)

    # final energy consumption per node and industry (TWh/a)
    nodal_df = nodal_production.dot(industry_sector_ratios.T)

    # convert GWh to TWh and ktCO2 to MtCO2
    nodal_df *= 0.001
    

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
        "heat": "low-temperature heat",
    }
    

    nodal_df["current electricity"] = nodal_today["electricity"]

    nodal_df.index.name = "TWh/a (MtCO2/a)"
    nodal_df.rename(columns=rename_sectors, inplace=True)
    new_row_names = {
    'BE1 0': 'BE',
    'DE1 0': 'DE',
    'FR1 0': 'FR',
    'GB0 0': 'GB',
    'NL1 0': 'NL'}
    nodal_df.rename(index=new_row_names, inplace=True)
    
    
    
    for country in countrries:
        #nodal_df.loc[country, 'ammonia'] = clever_Industry.loc[country, 'Total Final Energy Consumption of the ammonia industry']
        nodal_df.loc[country, 'electricity'] = clever_Industry.loc[country, 'Total Final electricity consumption in industry']
        nodal_df.loc[country, 'coal'] = clever_Industry.loc[country, 'Total Final energy consumption from solid fossil fuels (coal ...) in industry']
        nodal_df.loc[country, 'solid biomass'] = clever_Industry.loc[country, 'Total Final energy consumption from solid biomass in industry']
        nodal_df.loc[country, 'methane'] = clever_Industry.loc[country, 'Total Final energy consumption from gas grid / gas consumed locally in industry']
        #nodal_df.loc[country, 'low-temperature heat'] = clever_Industry.loc[country, 'Total Final heat consumption in industry']
        nodal_df.loc[country, 'hydrogen'] = clever_Industry.loc[country, 'Total Final hydrogen consumption in industry'] + clever_Industry.loc[country, 'Non-energy consumption of hydrogen for the feedstock production'].sum()
        nodal_df.loc[country, 'naphtha'] = clever_Industry.loc[country, 'Non-energy consumption of oil for the feedstock production']
    
    new_row_names = {
    'BE': 'BE1 0',
    'DE': 'DE1 0',
    'FR': 'FR1 0',
    'GB': 'GB0 0',
    'NL': 'NL1 0'}
    nodal_df.rename(index=new_row_names, inplace=True)
    fn = snakemake.output.industrial_energy_demand_per_node
    nodal_df.to_csv(fn, float_format="%.2f")
