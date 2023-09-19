# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build industrial energy demand per country.
"""

import multiprocessing as mp
from functools import partial

import country_converter as coco
import pandas as pd
from tqdm import tqdm
import os

cc = coco.CountryConverter()
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
paths = os.path.join(os.path.dirname(__file__), '../data/')

ktoe_to_twh = 0.011630

# name in JRC-IDEES Energy Balances
sector_sheets = {
    "Integrated steelworks": "cisb",
    "Electric arc": "cise",
    "Alumina production": "cnfa",
    "Aluminium - primary production": "cnfp",
    "Aluminium - secondary production": "cnfs",
    "Other non-ferrous metals": "cnfo",
    "Basic chemicals": "cbch",
    "Other chemicals": "coch",
    "Pharmaceutical products etc.": "cpha",
    "Basic chemicals feedstock": "cpch",
    "Cement": "ccem",
    "Ceramics & other NMM": "ccer",
    "Glass production": "cgla",
    "Pulp production": "cpul",
    "Paper production": "cpap",
    "Printing and media reproduction": "cprp",
    "Food, beverages and tobacco": "cfbt",
    "Transport Equipment": "ctre",
    "Machinery Equipment": "cmae",
    "Textiles and leather": "ctel",
    "Wood and wood products": "cwwp",
    "Mining and quarrying": "cmiq",
    "Construction": "ccon",
    "Non-specified": "cnsi",
}


fuels = {
    "All Products": "all",
    "Solid Fuels": "solid",
    "Total petroleum products (without biofuels)": "liquid",
    "Gases": "gas",
    "Nuclear heat": "heat",
    "Derived heat": "heat",
    "Biomass and Renewable wastes": "biomass",
    "Wastes (non-renewable)": "waste",
    "Electricity": "electricity",
}

eu28 = cc.EU28as("ISO2").ISO2.tolist()

jrc_names = {"GR": "EL", "GB": "UK"}


def industrial_energy_demand_per_country(country, year, jrc_dir):
    jrc_country = jrc_names.get(country, country)
    fn = f"{jrc_dir}/JRC-IDEES-2015_EnergyBalance_{jrc_country}.xlsx"

    sheets = list(sector_sheets.values())
    df_dict = pd.read_excel(fn, sheet_name=sheets, index_col=0)

    def get_subsector_data(sheet):
        df = df_dict[sheet][year].groupby(fuels).sum()

        df["ammonia"] = 0.0

        df["other"] = df["all"] - df.loc[df.index != "all"].sum()

        return df

    df = pd.concat(
        {sub: get_subsector_data(sheet) for sub, sheet in sector_sheets.items()}, axis=1
    )

    sel = ["Mining and quarrying", "Construction", "Non-specified"]
    df["Other Industrial Sectors"] = df[sel].sum(axis=1)
    df["Basic chemicals"] += df["Basic chemicals feedstock"]

    df.drop(columns=sel + ["Basic chemicals feedstock"], index="all", inplace=True)

    df *= ktoe_to_twh

    return df


def add_ammonia_energy_demand(demand):
    # MtNH3/a
    fn = snakemake.input.ammonia_production
    ammonia = pd.read_csv(fn, index_col=0)[str(year)] / 1e3

    def get_ammonia_by_fuel(x):
        fuels = {
            "gas": params["MWh_CH4_per_tNH3_SMR"],
            "electricity": params["MWh_elec_per_tNH3_SMR"],
        }

        return pd.Series({k: x * v for k, v in fuels.items()})

    ammonia_by_fuel = ammonia.apply(get_ammonia_by_fuel).T
    ammonia_by_fuel = ammonia_by_fuel.unstack().reindex(
        index=demand.index, fill_value=0.0
    )

    ammonia = pd.DataFrame({"ammonia": ammonia * params["MWh_NH3_per_tNH3"]}).T

    demand["Ammonia"] = ammonia.unstack().reindex(index=demand.index, fill_value=0.0)

    demand["Basic chemicals (without ammonia)"] = (
        demand["Basic chemicals"] - ammonia_by_fuel
    )

    demand["Basic chemicals (without ammonia)"].clip(lower=0, inplace=True)

    demand.drop(columns="Basic chemicals", inplace=True)

    return demand


def add_non_eu28_industrial_energy_demand(countries, demand):
    non_eu28 = countries.difference(eu28)
    if non_eu28.empty:
        return demand
    # output in MtMaterial/a
    fn = snakemake.input.industrial_production_per_country
    production = pd.read_csv(fn, index_col=0) / 1e3

    # recombine HVC, Chlorine and Methanol to Basic chemicals (without ammonia)
    chemicals = ["HVC", "Chlorine", "Methanol"]
    production["Basic chemicals (without ammonia)"] = production[chemicals].sum(axis=1)
    production.drop(columns=chemicals, inplace=True)

    eu28_production = production.loc[countries.intersection(eu28)].sum()
    eu28_energy = demand.groupby(level=1).sum()
    eu28_averages = eu28_energy / eu28_production

    demand_non_eu28 = pd.concat(
        {k: v * eu28_averages for k, v in production.loc[non_eu28].iterrows()}
    )

    return pd.concat([demand, demand_non_eu28])


def industrial_energy_demand(countries, year):
    nprocesses = snakemake.threads
    disable_progress = snakemake.config["run"].get("disable_progressbar", False)
    func = partial(
        industrial_energy_demand_per_country, year=year, jrc_dir=snakemake.input.jrc
    )
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countries),
        desc="Build industrial energy demand",
        disable=disable_progress,
    )
    with mp.Pool(processes=nprocesses) as pool:
        demand_l = list(tqdm(pool.imap(func, countries), **tqdm_kwargs))

    demand = pd.concat(demand_l, keys=countries)

    return demand

def clever_industry_data():
    years = int(snakemake.wildcards.planning_horizons)
    df= pd.read_csv(f'{paths}/clever_Industry_{years}.csv',index_col=0)
    return df

pf= pd.read_csv(f'{paths}/Industrial_perc.csv',index_col=0)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_industrial_energy_demand_per_country_today")

    params = snakemake.params.industry
    year = params.get("reference_year", 2015)
    countries = pd.Index(snakemake.params.countries)
    
    clever_Industry = clever_industry_data()
    #countrries = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT', 'SE', 'SI', 'SK']
    countrries = ['BE', 'DE', 'FR', 'GB', 'NL']
    columns_to_reset = [
    'Printing and media reproduction',
    'Pulp production',
    'Machinery Equipment',
    'Pharmaceutical products etc.',
    'Textiles and leather',
    'Transport Equipment',
    'Wood and wood products',
    'Alumina production',
    'Aluminium - primary production',
    'Aluminium - secondary production']

    demand = industrial_energy_demand(countries.intersection(eu28), year)

    demand = add_ammonia_energy_demand(demand)

    demand = add_non_eu28_industrial_energy_demand(countries, demand)

    # for format compatibility
    demand = demand.stack(dropna=False).unstack(level=[0, 2])

    # style and annotation
    demand.index.name = "TWh/a"
    demand.sort_index(axis=1, inplace=True)
    
    for country in countrries:
        for col in columns_to_reset:
            demand[(country, col)].values[:] = 0
        
    for country in countrries:
        #demand.loc[['ammonia'],[(country, 'Ammonia')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the ammonia industry']
        demand.loc[['electricity'],[(country, 'Integrated steelworks')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the steel industry'] * pf.loc['electricity', 'steel perc']
        demand.loc[['gas'],[(country, 'Integrated steelworks')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the steel industry'] * pf.loc['gas', 'steel perc']
        demand.loc[['liquid'],[(country, 'Integrated steelworks')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the steel industry'] * pf.loc['liquid', 'steel perc']
        demand.loc[['solid'],[(country, 'Integrated steelworks')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the steel industry'] * pf.loc['solid', 'steel perc']
        demand.loc[['biomass'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['biomass', 'Cement Perc']
        demand.loc[['electricity'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['electricity', 'Cement Perc']
        demand.loc[['gas'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['gas', 'Cement Perc']
        demand.loc[['liquid'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['liquid', 'Cement Perc']
        demand.loc[['solid'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['solid', 'Cement Perc']
        demand.loc[['waste'],[(country, 'Cement')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the cement industry'] * pf.loc['waste', 'Cement Perc']
        demand.loc[['electricity'],[(country, 'Glass production')]] = clever_Industry.loc[country, 'Total Final Energy Consumptionof the glass industry'] * pf.loc['electricity', 'Glass perc']
        demand.loc[['gas'],[(country, 'Glass production')]] = clever_Industry.loc[country, 'Total Final Energy Consumptionof the glass industry'] * pf.loc['gas', 'Glass perc']
        demand.loc[['liquid'],[(country, 'Glass production')]] = clever_Industry.loc[country, 'Total Final Energy Consumptionof the glass industry'] * pf.loc['liquid', 'Glass perc']
        demand.loc[['biomass'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['biomass', 'Basic Chemicals Percentage']
        demand.loc[['electricity'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['electricity', 'Basic Chemicals Percentage']
        demand.loc[['gas'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['gas', 'Basic Chemicals Percentage']
        demand.loc[['heat'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['heat', 'Basic Chemicals Percentage']
        demand.loc[['liquid'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['liquid', 'Basic Chemicals Percentage']
        demand.loc[['solid'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['solid', 'Basic Chemicals Percentage']
        demand.loc[['waste'],[(country, 'Basic chemicals (without ammonia)')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the chemical industry'] * pf.loc['waste', 'Basic Chemicals Percentage']
        demand.loc[['biomass'],[(country, 'Other chemicals')]] = clever_Industry.loc[country, 'other_chemicals_total'] * pf.loc['biomass', 'other chemicals perc']
        demand.loc[['electricity'],[(country, 'Other chemicals')]] = clever_Industry.loc[country, 'other_chemicals_total'] * pf.loc['electricity', 'other chemicals perc']
        demand.loc[['gas'],[(country, 'Other chemicals')]] = clever_Industry.loc[country, 'other_chemicals_total'] * pf.loc['gas', 'other chemicals perc']
        demand.loc[['liquid'],[(country, 'Other chemicals')]] = clever_Industry.loc[country, 'other_chemicals_total'] * pf.loc['liquid', 'other chemicals perc']
        demand.loc[['electricity'],[(country, 'Other non-ferrous metals')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non-ferrous metals industry'] * pf.loc['electricity', 'other non ferr perc']
        demand.loc[['gas'],[(country, 'Other non-ferrous metals')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non-ferrous metals industry'] * pf.loc['gas', 'other non ferr perc']
        demand.loc[['liquid'],[(country, 'Other non-ferrous metals')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non-ferrous metals industry'] * pf.loc['liquid', 'other non ferr perc']
        demand.loc[['biomass'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['biomass', 'Food perc']
        demand.loc[['electricity'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['electricity', 'Food perc']
        demand.loc[['gas'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['gas', 'Food perc']
        demand.loc[['heat'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['heat', 'Food perc']
        demand.loc[['liquid'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['liquid', 'Food perc']
        demand.loc[['solid'],[(country, 'Food, beverages and tobacco')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the Food, beverage and tobacco industry'] * pf.loc['solid', 'Food perc']
        demand.loc[['biomass'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['biomass', 'paper perc']
        demand.loc[['electricity'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['electricity', 'paper perc']
        demand.loc[['gas'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['gas', 'paper perc']
        demand.loc[['heat'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['heat', 'paper perc']
        demand.loc[['liquid'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['liquid', 'paper perc']
        demand.loc[['solid'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['solid', 'paper perc']
        demand.loc[['waste'],[(country, 'Paper production')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['waste', 'paper perc']
        demand.loc[['biomass'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['biomass', 'other industrial perc']
        demand.loc[['electricity'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['electricity', 'other industrial perc']
        demand.loc[['gas'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['gas', 'other industrial perc']
        demand.loc[['heat'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of the paper, pulp and printing industry'] * pf.loc['heat', 'other industrial perc']
        demand.loc[['liquid'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['liquid', 'other industrial perc']
        demand.loc[['solid'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['solid', 'other industrial perc']
        demand.loc[['waste'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['waste', 'other industrial perc']
        demand.loc[['other'],[(country, 'Other Industrial Sectors')]] = clever_Industry.loc[country, 'Total Final Energy Consumption of other industries (miscellaneous industries)'] * pf.loc['other', 'other industrial perc']
        demand.loc[['electricity'],[(country, 'Ceramics & other NMM')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non metallic minerals industry'] * pf.loc['electricity', 'Ceramics perc']
        demand.loc[['gas'],[(country, 'Ceramics & other NMM')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non metallic minerals industry'] * pf.loc['gas', 'Ceramics perc']
        demand.loc[['liquid'],[(country, 'Ceramics & other NMM')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non metallic minerals industry'] * pf.loc['liquid', 'Ceramics perc']
        demand.loc[['solid'],[(country, 'Ceramics & other NMM')]] = clever_Industry.loc[country, 'Total Final energy consumption of the non metallic minerals industry'] * pf.loc['solid', 'Ceramics perc']
       


    fn = snakemake.output.industrial_energy_demand_per_country_today
    demand.to_csv(fn)
