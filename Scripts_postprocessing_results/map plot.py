#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:08:37 2023

"""

import pypsa
import yaml
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import os

SCRIPTS_PATH = "../scripts/"
sys.path.append(os.path.join(SCRIPTS_PATH))
from plot_summary import rename_techs
from plot_network import assign_location
from plot_network import add_legend_circles, add_legend_patches, add_legend_lines
import holoviews as hv
from make_summary import assign_carriers
from plot_summary import preferred_order, rename_techs

hv.extension("bokeh")
hv.output(size=200)
plt.style.use(["bmh", "matplotlibrc"])
xr.set_options(display_style="html")

LL = "vopt"


    
n= pypsa.Network("../simulations/myopic simulations/resultsnocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")
# n= pypsa.Network("../simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")
plt.style.use(["ggplot", "matplotlibrc"])
with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)

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
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif "Li ion" in tech:
        return "battery storage"
    elif "BEV charger" in tech:
        return "V2G"
    elif "load" in tech:
        return "load shedding"
    elif tech == "oil" or tech == "gas":
         return "fossil oil and gas"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    
    else:
        return tech


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def plot_map(
    network,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=1.7e10,
    transmission=True,
    with_legend=True,
):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["fossil oil and gas"] = colors["oil"]
    colors["hydrogen storage"] = colors["H2 Store"]
    colors["load shedding"] = 'black'
    colors["gas-to-power/heat"] = 'darkred'

    n = network.copy()
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)

        if df_c.empty:
            continue

        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)
        

        #logger.debug(f"{comp}, {costs}")

    costs = costs.groupby(costs.columns, axis=1).sum()
    #del costs["CCS"]

    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]


    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    eu_location = config["plotting"].get(
        "eu_node_location", dict(x=-5.5, y=46)
    )
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    n.links.drop(
        n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
        inplace=True,
    )

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        #logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs.drop(to_drop, level=0, inplace=True, axis=0, errors="ignore")

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    threshold = 100e6  # 100 mEUR/a
    carriers = costs.groupby(level=1).sum()
    carriers = carriers.where(carriers > threshold).dropna()
    carriers = list(carriers.index)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 2e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    if LL == "1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        linewidth_factor = 2e3
        line_lower_threshold = 0.0
        title = "added grid"
        

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
            
    else:
        line_widths_a = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths_a = n.links.p_nom_opt - n.links.p_nom_min
        linewidth_factor = 2e3
        line_lower_threshold = 0.0
        title = "added grid"

        if transmission:
            line_widths_t = n.lines.s_nom_opt
            link_widths_t = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "total grid"

    # line_widths_t = line_widths_t.clip(line_lower_threshold, line_upper_threshold)
    # link_widths_t = link_widths_t.clip(line_lower_threshold, line_upper_threshold)

    # line_widths_t = line_widths_t.replace(line_lower_threshold, 0)
    # link_widths_t = link_widths_t.replace(line_lower_threshold, 0)
    
    # line_widths_a = line_widths_a.clip(line_lower_threshold, line_upper_threshold)
    # link_widths_a = link_widths_a.clip(line_lower_threshold, line_upper_threshold)

    # line_widths_a = line_widths_a.replace(line_lower_threshold, 0)
    # link_widths_a = link_widths_a.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Sinusoidal()})
    fig.set_size_inches(10, 10)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=(line_widths_t/linewidth_factor),
        link_widths=(link_widths_t/linewidth_factor),
        ax=ax,
    )
    n.plot(
        ax=ax,
        bus_sizes=0.0,
        link_colors="black",
        line_colors="black",
        link_widths=(link_widths_a/linewidth_factor),
        line_widths=(line_widths_a/linewidth_factor),
        # branch_components=["Link"],
        color_geomap=False,
    )

    #sizes = [20, 10, 5]
    sizes = [30, 20, 10]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.7),
        labelspacing=1,
        frameon=False,
        fontsize=20,
        handletextpad=1,
        title="system cost",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    # sizes = [10, 5]
    # labels = [f"{s} GW" for s in sizes]
    # scale = 1e3 / linewidth_factor
    # sizes = [s * scale for s in sizes]

    # legend_kw = dict(
    #     loc="upper left",
    #     bbox_to_anchor=(0.05, 0.45),
    #     fontsize=20,
    #     frameon=False,
    #     labelspacing=1,
    #     handletextpad=1,
    #     title=title,
    # )

    # add_legend_lines(
    #     ax, sizes, labels, patch_kw=dict(color="black"), legend_kw=legend_kw,
    # )
    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.4),
        fontsize=20,
        frameon=False,
        labelspacing=1,
        handletextpad=1,
        title="added grid",
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="black"), legend_kw=legend_kw,
    )

    # legend_kw = dict(
    #     bbox_to_anchor=(0.5, 1),
    #     frameon=False,
    #     fontsize=20,
    # )

    # if with_legend:
    #     colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
    #     labels = carriers + ["HVAC line", "HVDC link"]

    #     add_legend_patches(
    #         ax,
    #         colors,
    #         labels,
    #         legend_kw=legend_kw,
    #     )
    
plot_map(
        n,
        components=["generators", "links", "stores", "storage_units"],
        bus_size_factor=90e9,
        transmission=True,
    )
plt.title("NO_CDR-2050", fontsize=(25))
plt.rcParams['legend.title_fontsize'] = '20'
plt.tight_layout()

#%%

def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    # group pipe lines connecting the same buses and rename them for plotting
    pipe_capacity = df.groupby(level=0).agg(
        {"p_nom_opt": sum, "bus0": "first", "bus1": "first"}
    )

    return pipe_capacity


def plot_h2_map(network):
    n = network.copy()
    if "H2 pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    h2_storage = n.stores.query("carrier == 'H2'")
    # regions["H2"] = h2_storage.rename(
    #     index=h2_storage.bus.map(n.buses.location)
    # ).e_nom_opt.div(
    #     1e6
    # )  # TWh
    # regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 3e5
    linewidth_factor = 7e3
    # MW below which not drawn
    line_lower_threshold = 750

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )
    
    eu_location = config["plotting"].get(
        "eu_node_location", dict(x=-5.5, y=46)
    )
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]

    if config["foresight"] == "myopic":
        # sum capacitiy for pipelines from different investment periods
        h2_new = group_pipes(h2_new)

        if not h2_retro.empty:
            h2_retro = (
                group_pipes(h2_retro, drop_direction=True)
                .reindex(h2_new.index)
                .fillna(0)
            )

    if not h2_retro.empty:
        positive_order = h2_retro.bus0 < h2_retro.bus1
        h2_retro_p = h2_retro[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
        h2_retro = pd.concat([h2_retro_p, h2_retro_n])

        h2_retro["index_orig"] = h2_retro.index
        h2_retro.index = h2_retro.apply(
            lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
            axis=1,
        )

        retro_w_new_i = h2_retro.index.intersection(h2_new.index)
        h2_retro_w_new = h2_retro.loc[retro_w_new_i]

        retro_wo_new_i = h2_retro.index.difference(h2_new.index)
        h2_retro_wo_new = h2_retro.loc[retro_wo_new_i]
        h2_retro_wo_new.index = h2_retro_wo_new.index_orig

        to_concat = [h2_new, h2_retro_w_new, h2_retro_wo_new]
        h2_total = pd.concat(to_concat).p_nom_opt.groupby(level=0).sum()

    else:
        h2_total = h2_new.p_nom_opt

    link_widths_total = h2_total / linewidth_factor

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).first()
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    proj = ccrs.EqualEarth()
    #regions = regions.to_crs(proj.proj4_init)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"

    bus_colors = {"H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#805394"}

    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
    )

    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        color_geomap=False,
    )

    # regions.plot(
    #     ax=ax,
    #     column="H2",
    #     cmap="Blues",
    #     linewidths=0,
    #     legend=True,
    #     vmax=6,
    #     vmin=0,
    #     legend_kwds={
    #         "label": "Hydrogen Storage [TWh]",
    #         "shrink": 0.7,
    #         "extend": "max",
    #     },
    # )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.5, 0.95),
        labelspacing=1,
        handletextpad=0,
        frameon=False,
        fontsize=15,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.63, 0.95),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        fontsize=15,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = [bus_colors[c] for c in carriers] + [color_h2_pipe, color_retrofit]
    labels = carriers + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.5),
        ncol=2,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")
    
plot_h2_map(n)
plt.title("Suff-2050", fontsize=(25))
plt.rcParams['legend.title_fontsize'] = '20'
plt.tight_layout()

#%%
def plot_ch4_map(network):
    n = network.copy()

    if "gas pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    bus_size_factor = 10e8
    linewidth_factor = 1e4
    # MW below which not drawn
    line_lower_threshold = 1e3

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    fossil_gas_i = n.generators[n.generators.carrier == "gas"].index
    fossil_gas = (
        n.generators_t.p.loc[:, fossil_gas_i]
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
        .groupby(n.generators.loc[fossil_gas_i, "bus"])
        .sum()
        / bus_size_factor
    )
    fossil_gas.rename(index=lambda x: x.replace(" gas", ""), inplace=True)
    fossil_gas = fossil_gas.reindex(n.buses.index).fillna(0)
    # make a fake MultiIndex so that area is correct for legend
    fossil_gas.index = pd.MultiIndex.from_product([fossil_gas.index, ["fossil gas"]])

    methanation_i = n.links[n.links.carrier.isin(["helmeth", "Sabatier"])].index
    methanation = (
        abs(
            n.links_t.p1.loc[:, methanation_i].mul(
                n.snapshot_weightings.generators, axis=0
            )
        )
        .sum()
        .groupby(n.links.loc[methanation_i, "bus1"])
        .sum()
        / bus_size_factor
    )
    methanation = (
        methanation.groupby(methanation.index)
        .sum()
        .rename(index=lambda x: x.replace(" gas", ""))
    )
    # make a fake MultiIndex so that area is correct for legend
    methanation.index = pd.MultiIndex.from_product([methanation.index, ["methanation"]])

    biogas_i = n.stores[n.stores.carrier == "biogas"].index
    biogas = (
        n.stores_t.p.loc[:, biogas_i]
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
        .groupby(n.stores.loc[biogas_i, "bus"])
        .sum()
        / bus_size_factor
    )
    biogas = (
        biogas.groupby(biogas.index)
        .sum()
        .rename(index=lambda x: x.replace(" biogas", ""))
    )
    # make a fake MultiIndex so that area is correct for legend
    biogas.index = pd.MultiIndex.from_product([biogas.index, ["biogas"]])

    bus_sizes = pd.concat([fossil_gas, methanation, biogas])
    bus_sizes.sort_index(inplace=True)
    
    eu_location = config["plotting"].get(
        "eu_node_location", dict(x=-5.5, y=46)
    )
    n.buses.loc["EU gas", "x"] = eu_location["x"]
    n.buses.loc["EU gas", "y"] = eu_location["y"]

    to_remove = n.links.index[~n.links.carrier.str.contains("gas pipeline")]
    n.links.drop(to_remove, inplace=True)

    link_widths_rem = n.links.p_nom_opt / linewidth_factor
    link_widths_rem[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_orig = n.links.p_nom / linewidth_factor
    link_widths_orig[n.links.p_nom < line_lower_threshold] = 0.0

    max_usage = n.links_t.p0.abs().max(axis=0)
    link_widths_used = max_usage / linewidth_factor
    link_widths_used[max_usage < line_lower_threshold] = 0.0

    tech_colors = config["plotting"]["tech_colors"]

    pipe_colors = {
        "gas pipeline": "#f08080",
        "gas pipeline new": "#c46868",
        "gas pipeline (in 2020)": "lightgrey",
        "gas pipeline (available)": "#e8d1d1",
    }

    link_color_used = n.links.carrier.map(pipe_colors)

    n.links.bus0 = n.links.bus0.str.replace(" gas", "")
    n.links.bus1 = n.links.bus1.str.replace(" gas", "")

    bus_colors = {
        "fossil gas": tech_colors["fossil gas"],
        "methanation": tech_colors["methanation"],
        "biogas": "seagreen",
    }

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.EqualEarth()})

    n.plot(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=pipe_colors["gas pipeline (in 2020)"],
        link_widths=link_widths_orig,
        branch_components=["Link"],
        ax=ax,
    )

    n.plot(
        ax=ax,
        bus_sizes=0.0,
        link_colors=pipe_colors["gas pipeline (available)"],
        link_widths=link_widths_rem,
        branch_components=["Link"],
        color_geomap=False,
    )

    n.plot(
        ax=ax,
        bus_sizes=0.0,
        link_colors=link_color_used,
        link_widths=link_widths_used,
        branch_components=["Link"],
        color_geomap=False,
    )

    sizes = [100, 10]
    labels = [f"{s} TWh" for s in sizes]
    sizes = [s / bus_size_factor * 1e6 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.5, 1),
        fontsize=15,
        labelspacing=0.8,
        frameon=False,
        handletextpad=1,
        title="gas sources",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.68, 1),
        frameon=False,
        labelspacing=0.8,
        fontsize=15,
        handletextpad=1,
        title="gas pipeline",
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = list(pipe_colors.values()) + list(bus_colors.values())
    labels = list(pipe_colors.keys()) + list(bus_colors.keys())

    # legend on the side
    # legend_kw = dict(
    #     bbox_to_anchor=(1.47, 1.04),
    #     frameon=False,
    # )

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.6),
        ncol=2,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw=legend_kw,
    )
plot_ch4_map(n)
plt.title("Suff-2050", fontsize=(25))
plt.rcParams['legend.title_fontsize'] = '20'
plt.tight_layout()
