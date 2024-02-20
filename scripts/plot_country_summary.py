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


def plot_balances():
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    file = f"results/{study}/htmls/ChartData_{country}.xlsx"

    balances_df = pd.read_csv(
        snakemake.input.balances)
    balances_df = balances_df[3:]
    column_mapping = {
            'cluster': 'energy',
            'Unnamed: 1': 'components',
            'Unnamed: 2': 'techs',
            '6': '2020',
            '6.1': '2030',
            '6.2': '2040',
            '6.3': '2050',}

    balances_df.rename(columns=column_mapping, inplace=True)

    elec_imp = pd.read_excel(file, sheet_name="Chart 21", index_col=0)
    elec_imp.columns = elec_imp.iloc[1]
    gas_val = pd.read_excel(file, sheet_name="Chart 23", index_col=0)
    gas_val.columns = gas_val.iloc[1]
    h2_val = pd.read_excel(file, sheet_name="Chart 22", index_col=0)
    h2_val.columns = h2_val.iloc[1]
    for year in balances_df.columns[3:]:
        condition = (balances_df['energy'] == 'gas') & (balances_df['components'] == 'generators') & (balances_df['techs'] == 'gas')
        balances_df.loc[condition, [year]] = gas_val.loc[str(year), "Natural gas"] * 1e6
        

    for year in balances_df.columns[3:]:
        row_elc_data = {
            'energy': 'AC',
            'components': 'links',
            'techs': 'Imports',
        }
        row_elc_data[year] = elec_imp.loc[year, 'Imports'] * 1e6

        elc_row_df = pd.DataFrame([row_elc_data])
        balances_df = pd.concat([balances_df, elc_row_df], ignore_index=True)
    for year in balances_df.columns[3:]:
        row_data = {
            'energy': 'H2',
            'components': 'links',
            'techs': 'Imports',
        }
        row_data[year] = h2_val.loc[year, 'Imports'] * 1e6
        h2_row_df = pd.DataFrame([row_data])
        balances_df = pd.concat([balances_df, h2_row_df], ignore_index=True)

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

    fig, ax = plt.subplots(figsize=(12, 8))

    for k, v in balances.items():
            df = balances_df.loc[v]
            df = df.groupby(df["techs"]).sum()

            # convert MWh to TWh
            df[columns_to_convert] = df[columns_to_convert] / 1e6

            # remove trailing link ports
            df.index = [
                i[:-1]
                if ((i not in ["co2", "NH3"]) and (i[-1:] in ["0", "1", "2", "3"]))
                else i
                for i in df.index
            ]

            df = df.groupby(df.index.map(rename_techs)).sum()

            if v[0] in co2_carriers:
                units = "MtCO2/a"
            else:
                units = "TWh/a"

            df = df.drop(columns='components')
          

            if df.empty:
                continue

            new_index = preferred_order.intersection(df.index).append(
                df.index.difference(preferred_order)
            )

            new_columns = df.columns.sort_values()

            df.loc[new_index, new_columns].T.plot(
                kind="bar",
                ax=ax,
                stacked=True,
                color=[snakemake.params.plotting["tech_colors"][i] for i in new_index],)

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

            fig.savefig(f"results/{study}/country_graphs/balances-" + k + ".pdf", bbox_inches="tight")

            plt.cla()



if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_summary")

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    study = snakemake.config["run"]["name"]
    country = snakemake.config["country_summary"]

    plot_balances()
