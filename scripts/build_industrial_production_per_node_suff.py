# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build industrial production per model region.
"""

from itertools import product
import os
import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
paths = os.path.join(os.path.dirname(__file__), '../data/')
def clever_industry_data():
    years = int(snakemake.wildcards.planning_horizons)
    df= pd.read_csv(f'{paths}/clever_Industry_{years}.csv',index_col=0)
    return df

# map JRC/our sectors to hotmaps sector, where mapping exist
sector_mapping = {
    "Electric arc": "Iron and steel",
    "Integrated steelworks": "Iron and steel",
    "DRI + Electric arc": "Iron and steel",
    "Ammonia": "Chemical industry",
    "HVC": "Chemical industry",
    "HVC (mechanical recycling)": "Chemical industry",
    "HVC (chemical recycling)": "Chemical industry",
    "Methanol": "Chemical industry",
    "Chlorine": "Chemical industry",
    "Other chemicals": "Chemical industry",
    "Pharmaceutical products etc.": "Chemical industry",
    "Cement": "Cement",
    "Ceramics & other NMM": "Non-metallic mineral products",
    "Glass production": "Glass",
    "Pulp production": "Paper and printing",
    "Paper production": "Paper and printing",
    "Printing and media reproduction": "Paper and printing",
    "Alumina production": "Non-ferrous metals",
    "Aluminium - primary production": "Non-ferrous metals",
    "Aluminium - secondary production": "Non-ferrous metals",
    "Other non-ferrous metals": "Non-ferrous metals",
}


def build_nodal_industrial_production():
    fn = snakemake.input.industrial_production_per_country_tomorrow
    industrial_production = pd.read_csv(fn, index_col=0)

    fn = snakemake.input.industrial_distribution_key
    keys = pd.read_csv(fn, index_col=0)
    keys["country"] = keys.index.str[:2]

    nodal_production = pd.DataFrame(
        index=keys.index, columns=industrial_production.columns, dtype=float
    )

    countries = keys.country.unique()
    sectors = industrial_production.columns

    for country, sector in product(countries, sectors):
        buses = keys.index[keys.country == country]
        mapping = sector_mapping.get(sector, "population")

        key = keys.loc[buses, mapping]
        nodal_production.loc[buses, sector] = (
            industrial_production.at[country, sector] * key
        )
    countrries = snakemake.params.countries
    clever_Industry = clever_industry_data()

    # material demand per node and industry (kton/a)
    new_row_names = {
    'BE1 0': 'BE',
    'DE1 0': 'DE',
    'FR1 0': 'FR',
    'GB0 0': 'GB',
    'NL1 0': 'NL',
    'AT1 0': 'AT',
    'BG1 0': 'BG',
    'CH1 0': 'CH',
    'CZ1 0': 'CZ',
    'DK1 0': 'DK',
    'EE6 0': 'EE',
    'ES1 0': 'ES',
    'FI2 0': 'FI',
    'GR1 0': 'GR',
    'HR1 0': 'HR',
    'HU1 0': 'HU',
    'IE5 0': 'IE',
    'IT1 0': 'IT',
    'LT6 0': 'LT',
    'LU1 0': 'LU',
    'LV6 0': 'LV',
    'NO2 0': 'NO',
    'PL1 0': 'PL',
    'PT1 0': 'PT',
    'RO1 0': 'RO',
    'SE2 0': 'SE',
    'SI1 0': 'SI',
    'SK1 0': 'SK'}
    nodal_production.rename(index=new_row_names, inplace=True)
    for country in countrries:
        nodal_production.loc[country, 'Electric arc'] = clever_Industry.loc[country, 'Production of primary steel']
        nodal_production.loc[country, 'DRI + Electric arc'] = clever_Industry.loc[country, 'Production of recycled steel']
        nodal_production.loc[country, 'Integrated steelworks'] = 0
        nodal_production.loc[country, 'Cement'] = clever_Industry.loc[country, 'Production of cement']
        nodal_production.loc[country, 'Glass production'] = clever_Industry.loc[country, 'Production of glass']
        nodal_production.loc[country, 'Other chemicals'] = clever_Industry.loc[country, 'Production of high value chemicals ']
        nodal_production.loc[country, 'Paper production'] = clever_Industry.loc[country, 'Production of paper']
    new_row_names = {
    'BE': 'BE1 0',
    'DE': 'DE1 0',
    'FR': 'FR1 0',
    'GB': 'GB0 0',
    'NL': 'NL1 0',
    'AT': 'AT1 0',
    'BG': 'BG1 0',
    'CH': 'CH1 0',
    'CZ': 'CZ1 0',
    'DK': 'DK1 0',
    'EE': 'EE6 0',
    'ES': 'ES1 0',
    'FI': 'FI2 0',
    'GR': 'GR1 0',
    'HR': 'HR1 0',
    'HU': 'HU1 0',
    'IE': 'IE5 0',
    'IT': 'IT1 0',
    'LT': 'LT6 0',
    'LU': 'LU1 0',
    'LV': 'LV6 0',
    'NO': 'NO2 0',
    'PL': 'PL1 0',
    'PT': 'PT1 0',
    'RO': 'RO1 0',
    'SE': 'SE2 0',
    'SI': 'SI1 0',
    'SK': 'SK1 0'}
    nodal_production.rename(index=new_row_names, inplace=True)
    nodal_production.to_csv(snakemake.output.industrial_production_per_node)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industrial_production_per_node",
            simpl="",
            clusters=48,
            planning_horizons=2030,
        )
    params = snakemake.params.planning_horizons
    build_nodal_industrial_production()

