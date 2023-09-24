#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:23:33 2023

@author: umair
"""
import logging

logger = logging.getLogger(__name__)

import pypsa
import pandas as pd
from pypsa.descriptors import get_switchable_as_dense as as_dense
import plotly.graph_objects as go
import numpy as np
from matplotlib.colors import to_rgba


def prepare_sankey(n):
    '''
    This functions prepare the data for making the sankey diagram. The "gen" specifies generators, "su" storage units 
    "sto" stores and "load" demands for each sector. The function compiles all the data and convert it into a dataframe which is used to plot the 
    sankey plot for a scenario.
    '''
    def combine_rows(df, column_names, list_labels, target_label):
     '''
    Function that combines and sums the rows of the df dataframe for which the values of column_name are in the list_labels.
    For the columns that hold numbers, the values are summed
    For the columns that hold orther objects (such as string), the function checks that they are identical before combining the rows
    :param df: Pandas dataframe to be processes
    :param column_names: list of columns with the labels to be combined
    :param list_labels: List with the strings to be combined
    :param target_label: String
    '''
     from pandas.api.types import is_numeric_dtype
    # find the columns that are not numeric:
     cols_misc = [col for col in df.columns if not is_numeric_dtype(df[col])]
     for column_name in column_names:
        for i in df.index:
            if df.loc[i, column_name] in list_labels:
                df.loc[i, column_name] = target_label
     return df.groupby(cols_misc).sum().reset_index()

    file_industrial_demand = snakemake.input.industrial_energy_demand_per_node
    energy = snakemake.input.energy_name
    countries = snakemake.params.countries
    clever_industry = clever_industry = snakemake.input.clever_industry
    network = pypsa.Network(snakemake.input.network)

    # Flag to include losses or not in the sankey:
    include_losses = True
    
    n=network.copy()
    feedstock_emissions = (
        pd.read_csv(file_industrial_demand, index_col=0)["process emission from feedstock"].sum() * 1e6
      )  # t
    energy_demand = (
            pd.read_csv(energy, index_col=0)).T
    clever_industry = (
            pd.read_csv(clever_industry, index_col=0)).loc[countries].T

    Rail_demand = energy_demand.loc["total rail"].sum()
    H2_nonenergyy = clever_industry.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry = clever_industry.loc["Total Final hydrogen consumption in industry"].sum()
   
    columns = ["label", "source", "target", "value"]

    gen = ((n.snapshot_weightings.generators @ n.generators_t.p)
       .groupby([n.generators.carrier, n.generators.carrier, n.generators.bus.map(n.buses.carrier), ]).sum().div(
       1e6))  # TWh

    gen.index.set_names(columns[:-1], inplace=True)
    gen = gen.reset_index(name="value")
    gen = gen.loc[gen.value > 0.1]
    
    gen = combine_rows(gen, ['label', 'source', 'target'], ['offwind-ac', 'offwind-dc'], 'offshore wind')
    gen = combine_rows(gen, ['label', 'source', 'target'], ['solar', 'solar rooftop'], 'Solar Power')

    gen["source"] = gen["source"].replace({"gas": "fossil gas", "oil": "fossil oil", "onwind": "Onshore Wind"})

    # Storage units:
    su = ((n.snapshot_weightings.generators @ n.storage_units_t.p)
      .groupby(
    [n.storage_units.carrier, n.storage_units.carrier, n.storage_units.bus.map(n.buses.carrier), ]).sum().div(1e6))

    su.index.set_names(columns[:-1], inplace=True)
    su = su.reset_index(name="value")
    su = su.loc[su.value > 0.1]

    # Add hydro dams storage units to the generators:
    hydro_dams = su[(su["source"] == "hydro") & (su["target"] == "AC") & (su["label"] == "hydro")]
    gen.loc[len(gen), :] = hydro_dams.loc[1, :]

    # Combine hydro dams with run of river:
    gen = combine_rows(gen, ['label', 'source', 'target'], ['ror', 'hydro'], 'Hydro')
    
    #Combine the values of hydro in storage units and generatos together
    su_hydro = su.loc[len(su), "value"]
    gen_hydro = gen.loc[gen.label== "Hydro", "value"].sum()
    gen.loc[gen.label== "Hydro", "value"] = su_hydro + gen_hydro
    su = su.drop(su[(su["label"] == "hydro")].index)

    # Storages:
    sto = ((n.snapshot_weightings.generators @ n.stores_t.p)
       .groupby([n.stores.carrier, n.stores.carrier, n.stores.bus.map(n.buses.carrier)]).sum().div(1e6))

    sto.index.set_names(columns[:-1], inplace=True)
    sto = sto.reset_index(name="value")
    sto = sto.loc[sto.value > 0.1]


    load = (
    (n.snapshot_weightings.generators @ as_dense(n, "Load", "p_set"))
    .groupby([n.loads.carrier, n.loads.carrier, n.loads.bus.map(n.buses.carrier)])
    .sum()
    .div(1e6)
    .swaplevel()
       )  # TWh

    load.index.set_names(columns[:-1], inplace=True)
    load = load.reset_index(name="value")

    load = load.loc[~load.label.str.contains("emissions")]

    # Modify the target for residential and tertiary demands
    load.loc[load.label.str.contains("residential"), "target"] = "Residential and Tertiary"
    load.loc[load.label.str.contains("services"), "target"] = "Residential and Tertiary"
    load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "target"] = "Residential and Tertiary"
    load.loc[
    load.label.str.contains("industry electricity") & (load.label == "industry electricity"), "target"] = "Industry"
    load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "target"] = "Industry"
    load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "value"] = H2_industry
    load.loc[
    load.label.str.contains("naphtha for industry") & (load.label == "naphtha for industry"), "target"] = "Non-energy"
    load.loc[load.label.str.contains("land transport fuel cell") & (
            load.label == "land transport fuel cell"), "target"] = "Domestic transport"
    load.loc[load.label.str.contains("land transport oil") & (
            load.label == "land transport oil"), "target"] = "Domestic transport"
    load.loc[load.label.str.contains("shipping oil") & (
            load.label == "shipping oil"), "target"] = "Maritime bunkers"
    load.loc[
    load.label.str.contains("land transport EV") & (load.label == "land transport EV"), "target"] = "Domestic transport"
    load.loc[load.label.str.contains("agriculture electricity") & (
            load.label == "agriculture electricity"), "target"] = "Agriculture"
    load.loc[load.label.str.contains("agriculture heat") & (load.label == "agriculture heat"), "target"] = "Agriculture"
    load.loc[load.label.str.contains("agriculture machinery oil") & (
            load.label == "agriculture machinery oil"), "target"] = "Agriculture"
    load.loc[load.label.str.contains("urban central heat") & (
            load.label == "urban central heat"), "target"] = "Residential and Tertiary"
    load.loc[
    load.label.str.contains("urban central heat") & (load.label == "urban central heat"), "source"] = "District Heating"
    load.loc[load.label.str.contains("low-temperature heat for industry") & (
            load.label == "low-temperature heat for industry"), "target"] = "Industry"
    load.loc[load.label.str.contains("low-temperature heat for industry") & (
            load.label == "low-temperature heat for industry"), "source"] = "District Heating"
    load.loc[load.label.str.contains("NH3") & (load.label == "NH3"), "target"] = "Industry"
    # Subtract the raiway demand from electricity demand
    value=load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"] 
    load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"] = value - Rail_demand
    for i in range(5):
        n.links[f"total_e{i}"] = (n.snapshot_weightings.generators @ n.links_t[f"p{i}"]).div(1e6)  # TWh
        n.links[f"carrier_bus{i}"] = n.links[f"bus{i}"].map(n.buses.carrier)


    def calculate_losses(x, include_losses=include_losses):
        
        if include_losses:
            energy_ports = x.loc[
            x.index.str.contains("carrier_bus") & ~x.str.contains("co2", na=False)
            ].index.str.replace("carrier_bus", "total_e")
            return -x.loc[energy_ports].sum()
        else:
            return 0

    n.links["total_e5"] = n.links.apply(calculate_losses, include_losses=include_losses, axis=1)  # e4 and bus 4 for bAU 2050
    n.links["carrier_bus5"] = "losses"

    df = pd.concat(
    [
        n.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum(numeric_only=True)["total_e" + str(i)]
        for i in range(1, 6)
    ]
    ).reset_index()
    df.columns = columns

    # fix heat pump energy balance
    
    hp = n.links.loc[n.links.carrier.str.contains("heat pump")]

    hp_t_elec = n.links_t.p0.filter(like="heat pump")

    grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
    hp_elec = ((-n.snapshot_weightings.generators @ hp_t_elec)
           .groupby(grouper).sum().div(1e6).reset_index())
    hp_elec.columns = columns

    df = df.loc[~(df.label.str.contains("heat pump") & (df.target == "losses"))]

    df.loc[df.label.str.contains("heat pump"), "value"] -= hp_elec["value"].values

    df.loc[df.label.str.contains("air heat pump"), "source"] = "Ambient Heat"
    df.loc[df.label.str.contains("ground heat pump"), "source"] = "Ambient Heat"

    # Rename the columns
    df.columns = columns
    df = pd.concat([df, hp_elec])
    df = df.set_index(["label", "source", "target"]).squeeze()
    df = pd.concat(
    [
        df.loc[df < 0].mul(-1),
        df.loc[df > 0].swaplevel(1, 2),
    ]
     ).reset_index()
    df.columns = columns

    # ###
    df.loc[df.label.str.contains("urban central gas CHP") & (df.source == "gas"), "source"] = "gas"
    df.loc[
    df.label.str.contains("urban central gas CHP") & (df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central gas CHP CC") & (df.source == "gas"), "source"] = "gas"
    df.loc[df.label.str.contains("urban central gas CHP CC") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central air heat pump") & (df.source == "Ambient Heat"), "source"] = "Ambient Heat"
    df.loc[df.label.str.contains("urban central air heat pump") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central gas boiler") & (df.source == "gas"), "source"] = "gas"
    df.loc[df.label.str.contains("urban central gas boiler") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central resistive heater") & (df.source == "low voltage"), "source"] = "low voltage"
    df.loc[df.label.str.contains("urban central resistive heater") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central water tanks discharger") & (
            df.source == "urban central water tanks"), "source"] = "urban central water tanks"
    df.loc[df.label.str.contains("urban central water tanks discharger") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central water tanks charger") & (
            df.source == "District Heating"), "source"] = "District Heating"
    df.loc[df.label.str.contains("urban central water tanks charger") & (
            df.target == "urban central water tanks"), "source"] = "District Heating"
    df.loc[df.label.str.contains("H2 Electrolysis") & (df.source == "AC"), "source"] = "AC"
    df.loc[df.label.str.contains("H2 Electrolysis") & (df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("H2 Fuel Cell") & (df.source == "H2"), "source"] = "H2"
    df.loc[df.label.str.contains("H2 Fuel Cell") & (df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central solid biomass CHP") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("urban central solid biomass CHP") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("urban central solid biomass CHP CC") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("urban central solid biomass CHP CC") & (
            df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("residential rural biomass boiler") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[
    df.label.str.contains("residential rural biomass boiler") & (df.target == "urban central heat"), "target"] = "heat"
    df.loc[df.label.str.contains("services rural biomass boiler") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("services rural biomass boiler") & (df.target == "urban central heat"), "target"] = "heat"
    df.loc[df.label.str.contains("residential urban decentral biomass boiler") & (
            df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("residential urban decentral biomass boiler") & (
            df.target == "urban central heat"), "target"] = "heat"
    df.loc[df.label.str.contains("services urban decentral biomass boiler") & (
            df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("services urban decentral biomass boiler") & (
            df.target == "urban central heat"), "target"] = "heat"
    df.loc[df.label.str.contains("Fischer-Tropsch") & (df.source == "H2"), "source"] = "H2"
    df.loc[df.label.str.contains("Fischer-Tropsch") & (df.target == "urban central heat"), "target"] = "District Heating"
    df.loc[df.label.str.contains("gas for industry") & (df.source == "gas"), "source"] = "gas"
    df.loc[df.label.str.contains("gas for industry") & (df.target == "gas for industry"), "target"] = "Industry"
    df.loc[df.label.str.contains("gas for industry CC") & (df.source == "gas"), "source"] = "gas"
    df.loc[df.label.str.contains("gas for industry CC") & (df.target == "gas for industry"), "target"] = "Industry"
    df.loc[df.label.str.contains("solid biomass for industry") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("solid biomass for industry") & (
            df.target == "solid biomass for industry"), "target"] = "Industry"
    df.loc[df.label.str.contains("solid biomass for industry CC") & (df.source == "solid biomass"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("solid biomass for industry CC") & (
            df.target == "solid biomass for industry CC"), "target"] = "Industry"
    df.loc[df.label.str.contains("biogas to gas") & (df.source == "biogas"), "source"] = "biogas"
    df.loc[df.label.str.contains("biogas to gas") & (df.target == "gas"), "target"] = "gas"
    df.loc[df.label.str.contains("DAC") & (df.source == "urban central heat"), "source"] = "District Heating"
    df.loc[df.label.str.contains("urban central water tanks charger") & (
            df.source == "urban central heat"), "source"] = "District Heating"
    df.loc[df.label.str.contains("urban central water tanks discharger") & (
            df.target == "urban central water tanks"), "source"] = "TES Central"
    df.loc[df.label.str.contains("Haber-Bosch") & (df.source == "H2"), "target"] = "NH3"
    df.loc[df.label.str.contains("urban central gas boiler") & (df.source == "losses"), "source"] = "gas"
    df.loc[df.label.str.contains("urban central gas boiler") & (df.target == "gas"), "target"] = "losses"
    df.loc[df.label.str.contains("urban central solid biomass CHP") & (df.source == "losses"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("urban central solid biomass CHP") & (df.target == "solid biomass"), "target"] = "losses"
    df.loc[df.label.str.contains("urban central solid biomass CHP CC") & (df.source == "losses"), "source"] = "solid biomass"
    df.loc[df.label.str.contains("urban central solid biomass CHP CC") & (df.target == "solid biomass"), "target"] = "losses"


    # make DAC demand
    df.loc[df.label == "DAC", "target"] = "DAC"

    to_concat = [df, gen, su, sto, load]
    connections = pd.concat(to_concat).sort_index().reset_index(drop=True)

    # aggregation

    src_contains = connections.source.str.contains
    trg_contains = connections.target.str.contains

    connections.loc[src_contains("low voltage"), "source"] = "AC"
    connections.loc[trg_contains("low voltage"), "target"] = "AC"
    connections.loc[src_contains("CCGT"), "source"] = "gas"
    connections.loc[trg_contains("CCGT"), "target"] = "AC"
    connections.loc[src_contains("OCGT"), "source"] = "gas"
    connections.loc[trg_contains("OCGT"), "target"] = "AC"
    connections.loc[src_contains("residential rural water tank"), "source"] = "water tank"
    connections.loc[trg_contains("residential rural water tank"), "target"] = "water tank"
    connections.loc[src_contains("residential urban decentral water tank"), "source"] = "water tank"
    connections.loc[trg_contains("residential urban decentral water tank"), "target"] = "water tank"
    connections.loc[src_contains("urban central water tank"), "source"] = "urban central water tank"
    connections.loc[trg_contains("urban central water tank"), "target"] = "urban central water tank"
    connections.loc[src_contains("solar thermal"), "source"] = "solar thermal"
    connections.loc[src_contains("battery"), "source"] = "battery"
    connections.loc[trg_contains("battery"), "target"] = "battery"
    connections.loc[src_contains("Li ion"), "source"] = "battery"
    connections.loc[trg_contains("Li ion"), "target"] = "battery"
    connections.loc[src_contains("services rural water tank"), "source"] = "water tank"
    connections.loc[trg_contains("services rural water tank"), "target"] = "water tank"
    connections.loc[src_contains("services urban decentral water tank"), "source"] = "water tank"
    connections.loc[trg_contains("services urban decentral water tank"), "target"] = "water tank"
    connections.loc[src_contains("heat") & ~src_contains("demand"), "source"] = "heat"
    connections.loc[trg_contains("heat") & ~trg_contains("demand"), "target"] = "heat"
    new_row1 = {'label': 'Rail Network',
                'source': 'Electricity grid',
                'target': 'Rail Network',
                'value': Rail_demand}
    new_row2 = {'label': 'H2 for non-energy',
                'source': 'H2',
                'target': 'Non-energy',
                'value': H2_nonenergyy}

    connections = connections.append(new_row1, ignore_index=True)

    connections = connections.append(new_row2, ignore_index=True)

    connections = connections.loc[
    ~(connections.source == connections.target)
    & ~connections.source.str.contains("co2")
    & ~connections.target.str.contains("co2")
    & ~connections.source.str.contains("emissions")
    & (connections.value >= 0.1)]

    connections.replace("fossil gas", "Fossil gas", inplace=True)
    connections.replace("gas", "Gas Network", inplace=True)
    connections.replace("AC", "Electricity grid", inplace=True)
    connections.replace("kerosene for aviation", "Aviation bunkers", inplace=True)
    connections.replace("shipping methanol", "Maritime bunkers", inplace=True)
    connections.replace("H2", "H2 network", inplace=True)
    connections.replace("urban central water tank", "TES Central", inplace=True)
    connections.replace("water tank", "TES", inplace=True)
    
    return connections

def plot_sankey(connections):
    '''
    This function plots the sankey diagram. The colors for each energy carrier and technology can be specified 
    by user here or can be updated in the config file.
    '''
    labels = np.unique(connections[["source", "target"]])
    colors = snakemake.params.plotting["tech_colors"]
    colors["electricity grid"] = colors["electricity"]
    colors["ground-sourced ambient"] = colors["ground heat pump"]
    colors["air-sourced ambient"] = colors["air heat pump"]
    colors["co2 atmosphere"] = colors["co2"]
    colors["co2 stored"] = colors["co2"]
    colors["net co2 emissions"] = colors["co2"]
    colors["co2 sequestration"] = colors["co2"]
    colors["fossil oil"] = colors["oil"]
    colors["fossil gas"] = colors["gas"]
    colors["biogas to gas"] = colors["biogas"]
    colors["process emissions from feedstocks"] = colors["process emissions"]

    gas_boilers = [
    "residential rural gas boiler",
    "services rural gas boiler",
    "residential urban decentral gas boiler",
    "services urban decentral gas boiler",
    "urban central gas boiler",
      ]
    for gas_boiler in gas_boilers:
        colors[gas_boiler] = colors["gas boiler"]
        
    oil_boilers = [
        "residential rural oil boiler",
        "services rural oil boiler",
        "residential urban decentral oil boiler",
        "services urban decentral oil boiler",
        "urban central oil boiler",
    ]
    for oil_boiler in oil_boilers:
        colors[oil_boiler] = colors["oil boiler"]

    colors["urban central gas CHP"] = colors["CHP"]
    colors["urban central gas CHP CC"] = colors["CHP"]
    colors["urban central solid biomass CHP"] = colors["CHP"]
    colors["urban central solid biomass CHP CC"] = colors["CHP"]
    colors["Solar Power"] = colors["onwind"]
    colors["Onshore Wind"] = colors["onwind"]
    colors["Offshore Wind"] = colors["onwind"]
    colors["Fossil gas"] = colors["CCGT"]
    colors["Gas Network"] = colors["CCGT"]
    colors["Hydro"] = colors["onwind"]
    colors["TES"] = "red"
    colors["TES Central"] = "red"
    colors["H2 network"] = colors["H2"]
    colors["H2 turbine"] = colors["onwind"]
    colors["District Heating"] = "red"
    colors["DH Network"] = "red"
    colors["Ambient Heat"] = "pink"
    colors["Residential and Tertiary"] = "black"
    colors["Non-energy"] = "black"
    colors["Industry"] = "black"
    colors["Aviation bunkers"] = "black"
    colors["Maritime bunkers"] = "black"
    colors["Agriculture"] = "black"
    colors["Domestic transport"] = "black"
    colors["Electricity grid"] = colors["onwind"]
    colors["battery"] = colors["onwind"]
    colors["V2G"] = colors["onwind"]
    colors["solid biomass biomass to liquid"] = colors["biomass"]
    colors["solid biomass biomass to liquid CC"] = colors["biomass"]
    colors["solid biomass solid biomass to gas"] = colors["biomass"]
    colors["solid biomass solid biomass to gas CC"] = colors["biomass"]
    colors["co2 storage"] = colors["CCS"]
    nodes = pd.Series({v: i for i, v in enumerate(labels)})

    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

    link_colors = [
        "rgba{}".format(to_rgba(node_colors[src], alpha=0.5))
        for src in connections.source
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="freeform",  # [snap, nodepad, perpendicular, fixed]
            valuesuffix="TWh",
            valueformat=".1f",
            node=dict(pad=20, thickness=10, label=nodes.index, color=node_colors),
            link=dict(
                source=connections.source.map(nodes),
                target=connections.target.map(nodes),
                value=connections.value,
                label=connections.label,
                color=link_colors,
              ),
          )
      )
    return fig

def prepare_carbon_sankey(n):
    
    '''
    This function prepare the data for co2 emissions sankey chart. All the emissions from
    the technologies are compiled together
    '''

    collection = []

    # DAC
    value = -(n.snapshot_weightings.generators @ n.links_t.p1.filter(like="DAC")).sum()
    collection.append(
        pd.Series(
            dict(label="DAC", source="co2 atmosphere", target="co2 stored", value=value)
        )
    )

    # process emissions
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="process emissions$")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="process emissions",
                source="process emissions",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    # process emissions CC
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="process emissions CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="process emissions CC",
                source="process emissions",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    value = -(
        n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="process emissions CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="process emissions CC",
                source="process emissions",
                target="co2 stored",
                value=value,
            )
        )
    )

    # OCGT
    value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="OCGT")).sum()
    collection.append(
        pd.Series(dict(label="OCGT", source="gas", target="co2 atmosphere", value=value))
    )

    value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="CCGT")).sum()
    collection.append(
        pd.Series(dict(label="CCGT", source="gas", target="co2 atmosphere", value=value))
    )

    value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="lignite")).sum()
    collection.append(
        pd.Series(dict(label="lignite", source="coal", target="co2 atmosphere", value=value))
    )

    # Sabatier
    value = (n.snapshot_weightings.generators @ n.links_t.p2.filter(like="Sabatier")).sum()
    collection.append(
        pd.Series(dict(label="Sabatier", source="co2 stored", target="gas", value=value))
    )

    # SMR
    value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="SMR$")).sum()
    collection.append(
        pd.Series(dict(label="SMR", source="gas", target="co2 atmosphere", value=value))
    )

    # SMR CC
    value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="SMR CC")).sum()
    collection.append(
        pd.Series(dict(label="SMR CC", source="gas", target="co2 atmosphere", value=value))
    )

    value = -(n.snapshot_weightings.generators @ n.links_t.p3.filter(like="SMR CC")).sum()
    collection.append(
        pd.Series(dict(label="SMR CC", source="gas", target="co2 stored", value=value))
    )

    # gas boiler
    gas_boilers = [
        "residential rural gas boiler",
        "services rural gas boiler",
        "residential urban decentral gas boiler",
        "services urban decentral gas boiler",
        "urban central gas boiler",
    ]
    for gas_boiler in gas_boilers:
        value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like=gas_boiler)
        ).sum()
        collection.append(
            pd.Series(
                dict(label=gas_boiler, source="gas", target="co2 atmosphere", value=value)
            )
        )
    #oil boiler
    oil_boilers = [
        "residential rural oil boiler",
        "services rural oil boiler",
        "residential urban decentral oil boiler",
        "services urban decentral oil boiler",
        "urban central oil boiler",
    ]
    for oil_boiler in oil_boilers:
        value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like=oil_boiler)
        ).sum()
        collection.append(
            pd.Series(
                dict(label=oil_boiler, source="oil", target="co2 atmosphere", value=value)
            )
        )
    #biogas to gas
    value = (
        n.snapshot_weightings.generators @ n.links_t.p2.filter(like="biogas to gas")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="biogas to gas", source="co2 atmosphere", target="biogas", value=value
            )
        )
    )
    collection.append(
        pd.Series(dict(label="biogas to gas", source="biogas", target="gas", value=value))
    )
    
    # solid biomass for industry
    value = (
    n.snapshot_weightings.generators
    @ n.links_t.p0.filter(regex="solid biomass for industry$")
      ).sum()
    collection.append(
    pd.Series(
        dict(
            label="solid biomass for industry",
            source="solid biomass",
            target="co2 atmosphere",
            value=value * 0.37,
        )
         )
          )
    collection.append(
    pd.Series(dict(label="solid biomass for industry", source="co2 atmosphere", target="solid biomass", value=value * 0.37))
      )
    # solid biomass for industry CC
    value = (
        n.snapshot_weightings.generators
        @ n.links_t.p2.filter(like="solid biomass for industry CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass for industry CC",
                source="co2 atmosphere",
                target="BECCS",
                value=value,
            )
        )
    )


    value = -(
        n.snapshot_weightings.generators
        @ n.links_t.p3.filter(like="solid biomass for industry CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass for industry CC",
                source="BECCS",
                target="co2 stored",
                value=value,
            )
        )
    )

    #methanolisation
    value = (
        n.snapshot_weightings.generators @ n.links_t.p3.filter(like="methanolisation")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="methanolisation", source="co2 stored", target="methanol", value=value  # C02 intensity of gas from cost data
            )
        )
    )
    #gas for industry
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="gas for industry$")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="gas for industry", source="gas", target="co2 atmosphere", value=value
            )
        )
    )

    # gas for industry CC
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p2.filter(like="gas for industry CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="gas for industry CC",
                source="gas",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    value = -(
        n.snapshot_weightings.generators @ n.links_t.p3.filter(like="gas for industry CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="gas for industry CC", source="gas", target="co2 stored", value=value
            )
        )
    )
    #solid biomass to gas 
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="solid biomass solid biomass to gas$")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas", source="solid biomass", target="gas", value=value# CO2 stored in bioSNG from cost data
            )
        )
    )

    # solid biomass to gas CC
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p1.filter(like="solid biomass solid biomass to gas CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas CC",
                source="BioSNG",
                target="gas",
                value=value*0.2,
            )
        )
    )

    value = -(
        n.snapshot_weightings.generators @ n.links_t.p2.filter(like="solid biomass solid biomass to gas CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas CC", source="BioSNG", target="co2 stored", value=value
            )
        )
    )
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p3.filter(like="solid biomass solid biomass to gas CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas CC", source="BioSNG", target="co2 atmosphere", value=value
            )
        )
    )
    #biomass boilers
    #residential urban decentral biomass boiler
    tech = "residential urban decentral biomass boiler"

    value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * 0.37)
        )
    )
    collection.append(
        pd.Series(dict(label="residential urban decentral biomass boiler", source="co2 atmosphere", target="solid biomass", value=value * 0.37))
    )
    #services urban decentral biomass boiler
    tech = "services urban decentral biomass boiler"

    value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * 0.37)
        )
    )
    collection.append(
        pd.Series(dict(label="services urban decentral biomass boiler", source="co2 atmosphere", target="solid biomass", value=value * 0.37))
    )
    #residential rural biomass boiler
    tech = "residential rural biomass boiler"

    value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * 0.37)
        )
    )
    collection.append(
        pd.Series(dict(label="residential rural biomass boiler", source="co2 atmosphere", target="solid biomass", value=value * 0.37))
    )
    #services rural biomass boiler
    tech = "services rural biomass boiler"

    value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * 0.37)
        )
    )
    collection.append(
        pd.Series(dict(label="services rural biomass boiler", source="co2 atmosphere", target="solid biomass", value=value * 0.37))
    )
    #Solid biomass to liquid
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p1.filter(like="solid biomass biomass to liquid")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass biomass to liquid", source="solid biomass", target="oil", value=value*0.26  #C02 stored in oil by BTL from costs data 2050
            )
        )
    )
    collection.append(
        pd.Series(dict(label="solid biomass biomass to liquid", source="co2 atmosphere", target="solid biomass", value=value * 0.26))
    )
    # # solid biomass liquid to  CC

    value = -(
        n.snapshot_weightings.generators @ n.links_t.p3.filter(like="solid biomass biomass to liquid CC")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="solid biomass biomass to liquid CC", source="solid biomass", target="co2 stored", value=value
            )
        )
    )
    collection.append(
        pd.Series(dict(label="solid biomass biomass to liquid CC", source="co2 atmosphere", target="solid biomass", value=value))
    )
    #Fischer-Tropsch
    value = (
        n.snapshot_weightings.generators @ n.links_t.p2.filter(like="Fischer-Tropsch")
    ).sum()
    collection.append(
        pd.Series(
            dict(label="Fischer-Tropsch", source="co2 stored", target="oil", value=value)
        )
    )

    #urban central gas CHP
    value = -(
        n.snapshot_weightings.generators @ n.links_t.p3.filter(like="urban central gas CHP")
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="urban central gas CHP",
                source="gas",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    #urban central gas CHP CC
    tech = "urban central gas CHP CC"
    value = -(n.snapshot_weightings.generators @ n.links_t.p3.filter(like=tech)).sum()
    collection.append(
        pd.Series(dict(label=tech, source="gas", target="co2 atmosphere", value=value))
    )

    value = -(n.snapshot_weightings.generators @ n.links_t.p4.filter(like=tech)).sum()
    collection.append(
        pd.Series(dict(label=tech, source="gas", target="co2 stored", value=value))
    )

    #urban solid biomass CHP

    value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(regex="urban central solid biomass CHP$")).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value*0.37)
        )
    )
    collection.append(
        pd.Series(
            dict(label="urban central solid biomass CHP", source="co2 atmosphere", target="solid biomass", value=value*0.37)
        )
    )
    #urban solid biomass CHP CC
    tech = "urban central solid biomass CHP CC"

    value = (n.snapshot_weightings.generators @ n.links_t.p3.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="co2 atmosphere", target="BECCS", value=value)
        )
    )

    value = -(n.snapshot_weightings.generators @ n.links_t.p4.filter(like=tech)).sum()
    collection.append(
        pd.Series(
            dict(label=tech, source="BECCS", target="co2 stored", value=value)
        )
    )

    # oil emissions
    value = -(
        n.snapshot_weightings.generators
        @ as_dense(n, "Load", "p_set").filter(regex="^oil emissions", axis=1)
    ).sum()
    collection.append(
        pd.Series(
            dict(label="oil emissions", source="oil", target="co2 atmosphere", value=value)
        )
    )

    # agriculture machinery oil emissions
    value = -(
        n.snapshot_weightings.generators
        @ as_dense(n, "Load", "p_set").filter(
            like="agriculture machinery oil emissions", axis=1
        )
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="agriculture machinery oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    value = -(
        n.snapshot_weightings.generators
        @ as_dense(n, "Load", "p_set").filter(
            like="land transport oil emissions", axis=1
        )
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="land transport oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    value = -(
        n.snapshot_weightings.generators
        @ as_dense(n, "Load", "p_set").filter(
            like="shipping oil emissions", axis=1
        )
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="shipping oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
        )
    )
    value = -(
        n.snapshot_weightings.generators
        @ as_dense(n, "Load", "p_set").filter(
            like="shipping methanol emissions", axis=1
        )
    ).sum()
    collection.append(
        pd.Series(
            dict(
                label="shipping methanol emissions",
                source="methanol",
                target="co2 atmosphere",
                value=value,
            )
        )
    )

    #LULUCF
    V=n.stores.e_nom_opt.filter(like="LULUCF").sum()
    collection.append(
        pd.Series(
            dict(
                label="LULUCF",
                source="co2 atmosphere",
                target="LULUCF",
                value=V,
            )
        )
    )
    df = pd.concat(collection, axis=1).T
    df.value /= 1e6  # Mt

    # fossil gas
    co2_intensity = 0.2 # t/MWh
    value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="gas")).div(
        1e6
    ).sum() * 0.2
    row = pd.DataFrame(
        [dict(label="fossil gas", source="fossil gas", target="gas", value=value)]
    )

    df = pd.concat([df, row], axis=0)

    # fossil oil
    # co2_intensity = 0.27  # t/MWh
    value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="oil")).div(
        1e6
    ).sum() * 0.26
    row = pd.DataFrame(
        [dict(label="fossil oil", source="fossil oil", target="oil", value=value)]
    )
    df = pd.concat([df, row], axis=0)

    # sequestration
    value = (
        df.loc[df.target == "co2 stored", "value"].sum()
        - df.loc[df.source == "co2 stored", "value"].sum()
    )
    row = pd.DataFrame(
        [
            dict(
                label="co2 sequestration",
                source="co2 stored",
                target="co2 sequestration",
                value=value,
            )
        ]
    )

    df = pd.concat([df, row], axis=0)


    # net co2 emissions
    value = (
        df.loc[df.target == "co2 atmosphere", "value"].sum()
        - df.loc[df.source == "co2 atmosphere", "value"].sum()
    )
    row = pd.DataFrame(
        [
            dict(
                label="net co2 emissions",
                source="co2 atmosphere",
                target="net co2 emissions",
                value=value,
            )
        ]
    )
    df = pd.concat([df, row], axis=0)
    df = df.loc[(df.value >= 0.1)]
    
    return df

def plot_carbon_sankey(collection):
    labels = np.unique(collection[["source", "target"]])

    nodes = pd.Series({v: i for i, v in enumerate(labels)})
    colors = snakemake.params.plotting["tech_colors"]
    colors["LULUCF"] = "greenyellow"
    colors["Methanation"] = colors["Sabatier"]
    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

    link_colors = [
        "rgba{}".format(to_rgba(colors[src], alpha=0.5)) for src in collection.label
    ]

    fig_co2 = go.Figure(
        go.Sankey(
            arrangement="freeform",  # [snap, nodepad, perpendicular, fixed]
            valuesuffix=" MtCO2",
            valueformat=".1f",
            node=dict(pad=20, thickness=20, label=nodes.index, color=node_colors),
            link=dict(
                source=collection.source.map(nodes),
                target=collection.target.map(nodes),
                value=collection.value,
                label=collection.label,
                color=link_colors,
            ),
        )
    )

    return fig_co2

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sankey",
            simpl="",
            opts="",
            clusters="6",
            ll="vopt",
            sector_opts="1H-T-H-B-I-A-dist1",
            planning_horizons="2050",
        )
    logging.basicConfig(level=snakemake.config["logging"]["level"])
    planning_horizons = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)
    connections = prepare_sankey(n)
    collection = prepare_carbon_sankey(n)
    fig = plot_sankey(connections) 
    fig_co2 = plot_carbon_sankey(collection)
    fig.write_html(snakemake.output.sankey)
    fig_co2.write_html(snakemake.output.sankey_carbon)
    connections.to_csv(snakemake.output.sankey_csv)
    collection.to_csv(snakemake.output.sankey_carbon_csv)
