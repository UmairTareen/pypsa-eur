# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots from summary CSV files.
"""

import logging

logger = logging.getLogger(__name__)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

plt.style.use("ggplot")

from prepare_sector_network import co2_emissions_year


# consolidate and rename
def rename_techs(label):
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        # "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "helmeth",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


def plot_balances(country, study):
    file = f"results/{study}/htmls/ChartData_{country}.xlsx"

    balances_df = pd.read_csv(f"results/{study}/country_csvs/supply_energy.csv")
    balances_df = balances_df[3:]
    year_columns = ['2030', '2040', '2050']
    column_mapping = {
            'cluster': 'energy',
            'Unnamed: 1': 'components',
            'Unnamed: 2': 'techs',
            f'{cluster}': '2020',
            f'{cluster}.1': '2030',
            f'{cluster}.2': '2040',
            f'{cluster}.3': '2050',}

    balances_df.rename(columns=column_mapping, inplace=True)

    elec_imp = pd.read_excel(file, sheet_name="Chart 23", index_col=0)
    elec_imp.columns = elec_imp.iloc[1]
    gas_val = pd.read_excel(file, sheet_name="Chart 25", index_col=0)
    gas_val.columns = gas_val.iloc[1]
    h2_val = pd.read_excel(file, sheet_name="Chart 24", index_col=0)
    h2_val.columns = h2_val.iloc[1]
    lulucf_val = pd.read_excel(file, sheet_name="Chart 2", index_col=0)
    lulucf_val.columns = lulucf_val.iloc[1]
    
    for year in balances_df.columns[3:]:
        condition = (balances_df['energy'] == 'gas') & (balances_df['components'] == 'generators') & (balances_df['techs'] == 'gas')
        balances_df.loc[condition, [year]] = gas_val.loc[str(year), "Natural gas"] * 1e6

    rows_to_append = []

    # Row data for AC Imports
    row_elc_data = {
    'energy': 'AC',
    'components': 'links',
    'techs': 'Imports',}
    for year_col in year_columns:
     row_elc_data[year_col] = 0
    rows_to_append.append(row_elc_data)

    # Row data for H2 Imports
    row_data = {
    'energy': 'H2',
    'components': 'links',
    'techs': 'Imports',}
    for year_col in year_columns:
     row_data[year_col] = 0
    rows_to_append.append(row_data)

    # Row data for LULUCF
    row_data = {
    'energy': 'co2',
    'components': 'stores',
    'techs': 'LULUCF',}
    for year_col in year_columns:
     row_data[year_col] = 0
    rows_to_append.append(row_data)

    df_to_append = pd.DataFrame(rows_to_append)

    # Concatenate original DataFrame with the new DataFrame
    balances_df = pd.concat([balances_df, df_to_append], ignore_index=True)

    for year in year_columns:
     # Update values for AC Imports
     balances_df.loc[(balances_df['energy'] == 'AC') & (balances_df['components'] == 'links') & (balances_df['techs'] == 'Imports'), year] = elec_imp.loc[str(year), "Imports"] * 1e6
    
     # Update values for H2 Imports
     balances_df.loc[(balances_df['energy'] == 'H2') & (balances_df['components'] == 'links') & (balances_df['techs'] == 'Imports'), year] = h2_val.loc[str(year), "Imports"] * 1e6
    
     # Update values for LULUCF
     balances_df.loc[(balances_df['energy'] == 'co2') & (balances_df['components'] == 'stores') & (balances_df['techs'] == 'LULUCF'), year] = lulucf_val.loc[str(year), "Land use and forestry"] * 1e6
        
    balances_df = balances_df.set_index("energy")
    balances_df = balances_df.drop(columns='2020')
    columns_to_convert = ['2030', '2040', '2050']

    # Use pd.to_numeric to convert the specified columns to numeric
    balances_df[columns_to_convert] = balances_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    balances_df = balances_df.fillna(0)
    balances = {i.replace(" ", "_"): [i] for i in balances_df.index}

    balances_to_drop = ['agriculture_machinery_oil', 'biogas', 'coal', 'gas_for_industry', 'kerosene_for_aviation',
                        'land_transport_oil', 'lignite', 'naphtha_for_industry', 'shipping_methanol', 'shipping_oil',
                        'solid_biomass_for_industry', 'uranium', 'urban_central_water_tanks', 'battery', 'home_battery',
                        'residential_rural_water_tanks', 'residential_urban_decentral_water_tanks', 'services_rural_water_tanks',
                        'services_urban_decentral_water_tanks', 'NH3', 'process_emissions']

    balances = {key: value for key, value in balances.items() if key not in balances_to_drop}
    return balances_df, balances
def plot_figures():
 balances_df_ref, balances = plot_balances(country, study_ref)
 balances_df_suff, balances = plot_balances(country, study_suff)

 combined_df = balances_df_ref.merge(balances_df_suff, on=['energy', 'components', 'techs'], how='outer', suffixes=('_ref', '_suff'))
 combined_df = combined_df.fillna(0)
 fig, ax = plt.subplots(figsize=(12, 8))
 for k, v in balances.items():
    df = combined_df.loc[v]
    df = df.groupby(df["techs"]).sum()
    columns_to_convert = ['2030_ref', '2040_ref', '2050_ref','2030_suff', '2040_suff', '2050_suff']
    # convert MWh to TWh
    df[columns_to_convert] = df[columns_to_convert] / 1e6
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    # remove trailing link ports
    df.index = [
        i[:-1]
        if ((i not in ["co2", "NH3"]) and (i[-1:] in ["0", "1", "2", "3"]))
        else i
        for i in df.index
    ]

    df = df.groupby(df.index.map(rename_techs)).sum()

    if df.empty:
        continue

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.columns.sort_values()
    new_columns = new_columns.drop('components', errors='ignore')

    df = df.loc[new_index, new_columns].T


    df.plot(
    kind="bar",
    ax=ax,
    stacked=True,
    color=[snakemake.params.plotting["tech_colors"][i] for i in new_index],
      )

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    if v[0] in co2_carriers:
        ax.set_ylabel("CO2 [MtCO2/a]")
    else:
        ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="x")

    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=[1, 1],
        frameon=False,
    )

    fig.savefig("results/scenario_results/country_graphs/balances-" + k + ".pdf", bbox_inches="tight")

    plt.cla()



if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_summary")

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    cluster = snakemake.params.scenario["clusters"][0]
    studies = ['ref', 'suff']
    country = snakemake.config["country_summary"]

    study_ref = 'ref'
    study_suff = 'suff'

    balances_df_ref, balances = plot_balances(country, study_ref)
    balances_df_suff, balances = plot_balances(country, study_suff)
    plot_figures()