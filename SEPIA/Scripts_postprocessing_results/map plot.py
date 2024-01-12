#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:08:37 2023

@author: umair
"""

import pypsa
import yaml
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_script_dir, "../../scripts/")
sys.path.append(scripts_path)
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
scenario = "ncdr"

    
# n= pypsa.Network("../simulations/myopic simulations/resultsbau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")
# n= pypsa.Network("../simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")
plt.style.use(["ggplot", "matplotlibrc"])
with open("../../config/config.yaml") as file:
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
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        linewidth_factor = 2e3
        line_lower_threshold = 0.0
        title = "added grid"

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "total grid"

    # line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
    # link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)

    # line_widths = line_widths.replace(line_lower_threshold, 0)
    # link_widths = link_widths.replace(line_lower_threshold, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Sinusoidal()})
    fig.set_size_inches(15, 15)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
    )

    #sizes = [20, 10, 5]
    sizes = [30, 20, 10]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / bus_size_factor * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.75),
        labelspacing=2,
        frameon=False,
        fontsize=20,
        handletextpad=1,
        title="investment costs",
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]
    if planning_horizon == 2020:
        value = "current grid"
    else:
        value = "total grid"
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.35),
        fontsize=20,
        frameon=False,
        labelspacing=1,
        handletextpad=1,
        title=value
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="black"), legend_kw=legend_kw,
    )

    legend_kw = dict(
        bbox_to_anchor=(1.3, 1),
        frameon=False,
        fontsize=15,
    )

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )
    return fig

planning_horizons = [2020, 2030, 2040, 2050]
html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
/* Style the tab content */
.tabcontent {
  display: none;
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-top: none;
}

/* Style the tabs */
.tab {
  overflow: hidden;
  border: 1px solid #ccc;
  background-color: #f1f1f1;
}

/* Style the tab buttons */
.tab button {
  background-color: inherit;
  float: left;
  border: none;
  outline: none;
  cursor: pointer;
  padding: 14px 16px;
  transition: 0.3s;
  font-size: 17px;
  margin-top: 15px;  /* Add margin to move the buttons down */
}

/* Change background color of buttons on hover */
.tab button:hover {
  background-color: #ddd;
}

/* Create an active/current tablink class */
.tab button.active {
  background-color: #ccc;
}
</style>
</head>
<body>
"""

html_content = """
<div class="tab">
"""

for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_map(
            n,
            components=["generators", "links", "stores", "storage_units"],
            bus_size_factor=90e9,
            transmission=True,
        )
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_plot_{planning_horizon}.png"
    maps_dir = os.path.join(current_script_dir, "..", "maps")
    if not os.path.exists(maps_dir):
     os.makedirs(maps_dir)
    fig.savefig(output_image_path, bbox_inches="tight")

    # Add tab content for each planning horizon
    html_content += f"""
    <button class="tablinks{' active' if i == 0 else ''}" onclick="openTab(event, '{planning_horizon}')">{planning_horizon}</button>
    """

# Close the tab div
html_content += """
</div>
"""

# Add tab content divs
for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_map(
            n,
            components=["generators", "links", "stores", "storage_units"],
            bus_size_factor=90e9,
            transmission=True,
        )
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_plot_{planning_horizon}.png"
    fig.savefig(output_image_path, bbox_inches="tight")

    html_content += f"""
    <div id="{planning_horizon}" class="tabcontent" style="display: {'block' if i == 0 else 'none'};">
        <h2>Map Plot - {planning_horizon}</h2>
        <img src="{output_image_path}" alt="Map Plot" width="1200" height="800">
    </div>
    """

# Add JavaScript for tab functionality
html_content += """
<script>
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>
</body>
</html>
"""

# Save the entire HTML content to a single file
output_combined_html_path = "../maps/map_plots.html"
with open(output_combined_html_path, "w") as html_file:
    html_file.write(html_content)
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

    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})

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
        bbox_to_anchor=(0.05, 0.75),
        labelspacing=1.2,
        handletextpad=0,
        frameon=False,
        fontsize=15,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.6),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        fontsize=15,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="black"),
        legend_kw=legend_kw,
    )

    colors = [bus_colors[c] for c in carriers] + [color_h2_pipe, color_retrofit]
    labels = carriers + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(1, 0.9),
        ncol=1,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")
    
    return fig
html_content = """
<div class="tab">
"""

planning_horizons = [2030, 2040, 2050]

for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_h2_map(network=n)
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_h2_plot_{planning_horizon}.png"
    fig.savefig(output_image_path, bbox_inches="tight")

    # Add tab content for each planning horizon
    html_content += f"""
    <button class="tablinks{' active' if i == 0 else ''}" onclick="openTab(event, '{planning_horizon}')">{planning_horizon}</button>
    """

# Close the tab div
html_content += """
</div>
"""

# Add tab content divs
for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_h2_map(network=n)
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_h2_plot_{planning_horizon}.png"
    fig.savefig(output_image_path, bbox_inches="tight")

    html_content += f"""
    <div id="{planning_horizon}" class="tabcontent" style="display: {'block' if i == 0 else 'none'};">
        <h2>Map H2 Plot - {planning_horizon}</h2>
        <img src="{output_image_path}" alt="Map H2 Plot" width="1200" height="800">
    </div>
    """

# Add JavaScript for tab functionality
html_content += """
<script>
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>
</body>
</html>
"""

# Save the entire HTML content to a single file
output_combined_html_path = "../maps/map_h2_plots.html"
with open(output_combined_html_path, "w") as html_file:
    html_file.write(html_content)


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

    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": ccrs.EqualEarth()})

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
        bbox_to_anchor=(0, 0.8),
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
        patch_kw=dict(facecolor="black"),
        legend_kw=legend_kw,
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 0.6),
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
        patch_kw=dict(color="black"),
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
        bbox_to_anchor=(1, 0.9),
        ncol=1,
        frameon=False,
        fontsize=15,
    )

    add_legend_patches(
        ax,
        colors,
        labels,
        legend_kw=legend_kw,
    )
    return fig
html_content = """
<div class="tab">
"""

planning_horizons = [2020,2030, 2040, 2050]

for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_ch4_map(network=n)
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_ch4_plot_{planning_horizon}.png"
    fig.savefig(output_image_path, bbox_inches="tight")

    # Add tab content for each planning horizon
    html_content += f"""
    <button class="tablinks{' active' if i == 0 else ''}" onclick="openTab(event, '{planning_horizon}')">{planning_horizon}</button>
    """

# Close the tab div
html_content += """
</div>
"""

# Add tab content divs
for i, planning_horizon in enumerate(planning_horizons):
    # Load network for the current planning horizon
    network_path = f"../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc"

    if not os.path.exists(network_path):
        print(f"Network file not found for {planning_horizon}")
        continue

    n = pypsa.Network(network_path)

    # Plot the map and get the figure
    fig = plot_ch4_map(network=n)
    plt.rcParams['legend.title_fontsize'] = '20'
    # Save the map plot as an image
    output_image_path = f"../maps/map_ch4_plot_{planning_horizon}.png"
    fig.savefig(output_image_path, bbox_inches="tight")

    html_content += f"""
    <div id="{planning_horizon}" class="tabcontent" style="display: {'block' if i == 0 else 'none'};">
        <h2>Map H2 Plot - {planning_horizon}</h2>
        <img src="{output_image_path}" alt="Map H2 Plot" width="1200" height="800">
    </div>
    """

# Add JavaScript for tab functionality
html_content += """
<script>
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>
</body>
</html>
"""

# Save the entire HTML content to a single file
output_combined_html_path = "../maps/map_ch4_plots.html"
with open(output_combined_html_path, "w") as html_file:
    html_file.write(html_content)

#%%
import logging

logger = logging.getLogger(__name__)
def plot_series(carrier="AC", name="test"):
    
    n=pypsa.Network("/home/umair/pypsa-eur/results/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")
    m=pypsa.Network("/home/umair/pypsa-eur/resultsBAU2050/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")
    #o=pypsa.Network("/home/umair/pypsa-eur/results2040/S2040/postnetworks/elec_s_6_lv1.5__Co2L0p15-1H-T-H-B-I-A-dist1_2040.nc")
    p=pypsa.Network("/home/umair/pypsa-eur/resultsSuff2050/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")
    #q=pypsa.Network("/home/umair/pypsa-eur/results2050NOCCS/S2050NOCCS/postnetworks/elec_s_6_lv1.5__Co2L0p15-1H-T-H-B-I-A-dist1_2050.nc")
    r=pypsa.Network("/home/umair/pypsa-eur/resultsS2050LULUCF/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")

    assign_location(n)
    assign_carriers(n)
    assign_location(m)
    assign_carriers(m)
    # assign_location(o)
    # assign_carriers(o)
    assign_location(p)
    assign_carriers(p)
    # assign_location(q)
    # assign_carriers(q)
    assign_location(r)
    assign_carriers(r)

    busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]
    busesm = m.buses.index[m.buses.carrier.str.contains(carrier)]
    # buseso = o.buses.index[o.buses.carrier.str.contains(carrier)]
    busesp = p.buses.index[p.buses.carrier.str.contains(carrier)]
    busesr = r.buses.index[r.buses.carrier.str.contains(carrier)]
    

    supplyn = pd.DataFrame(index=n.snapshots)
    supplym = pd.DataFrame(index=m.snapshots)
    # supplyo = pd.DataFrame(index=o.snapshots)
    supplyp = pd.DataFrame(index=p.snapshots)
    supplyr = pd.DataFrame(index=r.snapshots)
    
    for c in n.iterate_components(n.branch_components):
        n_port = 3 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyn = pd.concat(
                (
                    supplyn,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesn)]
        supplyn = pd.concat(
            (
                supplyn,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in m.iterate_components(m.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplym = pd.concat(
                (
                    supplym,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesm)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in m.iterate_components(m.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesm)]
        supplym = pd.concat(
            (
                supplym,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    # for c in o.iterate_components(o.branch_components):
    #     n_port = 4 if c.name == "Link" else 2     #port3
    #     for i in range(n_port):
    #         supplyo = pd.concat(
    #             (
    #                 supplyo,
    #                 (-1)
    #                 * c.pnl["p" + str(i)]
    #                 .loc[:, c.df.index[c.df["bus" + str(i)].isin(buseso)]]
    #                 .groupby(c.df.carrier, axis=1)
    #                 .sum(),
    #             ),
    #             axis=1,
    #         )

    # for c in o.iterate_components(o.one_port_components):
    #     comps = c.df.index[c.df.bus.isin(buseso)]
    #     supplyo = pd.concat(
    #         (
    #             supplyo,
    #             ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
    #             .groupby(c.df.carrier, axis=1)
    #             .sum(),
    #         ),
    #         axis=1,
    #     )
        
    for c in p.iterate_components(p.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyp = pd.concat(
                (
                    supplyp,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesp)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in p.iterate_components(p.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesp)]
        supplyp = pd.concat(
            (
                supplyp,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in r.iterate_components(r.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyr = pd.concat(
                (
                    supplyr,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesr)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in r.iterate_components(r.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesr)]
        supplyr = pd.concat(
            (
                supplyr,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )

    supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
    supplym = supplym.groupby(rename_techs_tyndp, axis=1).sum()
    # supplyo = supplyo.groupby(rename_techs_tyndp, axis=1).sum()
    supplyp = supplyp.groupby(rename_techs_tyndp, axis=1).sum()
    supplyr = supplyr.groupby(rename_techs_tyndp, axis=1).sum()
    

    bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]
    bothm = supplym.columns[(supplym < 0.0).any() & (supplym > 0.0).any()]
    # botho = supplyo.columns[(supplyo < 0.0).any() & (supplyo > 0.0).any()]
    bothp = supplyp.columns[(supplyp < 0.0).any() & (supplyp > 0.0).any()]
    bothr = supplyr.columns[(supplyr < 0.0).any() & (supplyr > 0.0).any()]

    positive_supplyn = supplyn[bothn]
    negative_supplyn = supplyn[bothn]
    positive_supplym = supplym[bothm]
    negative_supplym = supplym[bothm]
    # positive_supplyo = supplyo[botho]
    # negative_supplyo = supplyo[botho]
    positive_supplyp = supplyp[bothp]
    negative_supplyp = supplyp[bothp]
    positive_supplyr = supplyr[bothr]
    negative_supplyr = supplyr[bothr]

    positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
    negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)
    positive_supplym = positive_supplym.mask(positive_supplym < 0.0, 0.0)
    negative_supplym = negative_supplym.mask(negative_supplym > 0.0, 0.0)
    # positive_supplyo = positive_supplyo.mask(positive_supplyo < 0.0, 0.0)
    # negative_supplyo = negative_supplyo.mask(negative_supplyo > 0.0, 0.0)
    positive_supplyp = positive_supplyp.mask(positive_supplyp < 0.0, 0.0)
    negative_supplyp = negative_supplyp.mask(negative_supplyp > 0.0, 0.0)
    positive_supplyr = positive_supplyr.mask(positive_supplyr < 0.0, 0.0)
    negative_supplyr = negative_supplyr.mask(negative_supplyr > 0.0, 0.0)


    supplyn[bothn] = positive_supplyn
    supplym[bothm] = positive_supplym
    # supplyo[botho] = positive_supplyo
    supplyp[bothp] = positive_supplyp
    supplyr[bothr] = positive_supplyr
    

    suffix = " charging"

    negative_supplyn.columns = negative_supplyn.columns + suffix
    negative_supplym.columns = negative_supplym.columns + suffix
    # negative_supplyo.columns = negative_supplyo.columns + suffix
    negative_supplyp.columns = negative_supplyp.columns + suffix
    negative_supplyr.columns = negative_supplyr.columns + suffix

    supplyn = pd.concat((supplyn, negative_supplyn), axis=1)
    supplym = pd.concat((supplym, negative_supplym), axis=1)
    # supplyo = pd.concat((supplyo, negative_supplyo), axis=1)
    supplyp = pd.concat((supplyp, negative_supplyp), axis=1)
    supplyr = pd.concat((supplyr, negative_supplyr), axis=1)
    
    

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-08-01"
    stop = "2013-08-07"
    

    threshold = 1

    to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]
    to_dropm = supplym.columns[(abs(supplym) < threshold).all()]
    # to_dropo = supplyo.columns[(abs(supplyo) < threshold).all()]
    to_dropp = supplyp.columns[(abs(supplyp) < threshold).all()]
    to_dropr = supplyr.columns[(abs(supplyr) < threshold).all()]

    if len(to_dropn) != 0:
        logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
        supplyn.drop(columns=to_dropn, inplace=True)
    
    if len(to_dropm) != 0:
        logger.info(f"Dropping {to_dropm.tolist()} from supplym")
        supplym.drop(columns=to_dropm, inplace=True)
        
    # if len(to_dropo) != 0:
    #     logger.info(f"Dropping {to_dropo.tolist()} from supplyio")
    #     supplyo.drop(columns=to_dropo, inplace=True)
        
    if len(to_dropp) != 0:
        logger.info(f"Dropping {to_dropp.tolist()} from supplyp")
        supplyp.drop(columns=to_dropp, inplace=True)
        
    if len(to_dropr) != 0:
        logger.info(f"Dropping {to_dropr.tolist()} from supplyr")
        supplyr.drop(columns=to_dropr, inplace=True)

    supplyn.index.name = None
    supplym.index.name = None
    # supplyo.index.name = None
    supplyp.index.name = None
    supplyr.index.name = None
    

    supplyn = supplyn / 1e3
    supplym = supplym / 1e3
    # supplyo = supplyo / 1e3
    supplyp = supplyp / 1e3
    supplyr = supplyr / 1e3
    #del supply["CCS"]

    supplyn.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyn.columns = supplyn.columns.str.replace("residential ", "")
    supplyn.columns = supplyn.columns.str.replace("services ", "")
    supplyn.columns = supplyn.columns.str.replace("urban decentral ", "decentral ")
    
    supplym.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplym.columns = supplym.columns.str.replace("residential ", "")
    supplym.columns = supplym.columns.str.replace("services ", "")
    supplym.columns = supplym.columns.str.replace("urban decentral ", "decentral ")
    
    # supplyo.rename(
    #     columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    # )
    # supplyo.columns = supplyo.columns.str.replace("residential ", "")
    # supplyo.columns = supplyo.columns.str.replace("services ", "")
    # supplyo.columns = supplyo.columns.str.replace("urban decentral ", "decentral ")
    
    supplyp.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyp.columns = supplyp.columns.str.replace("residential ", "")
    supplyp.columns = supplyp.columns.str.replace("services ", "")
    supplyp.columns = supplyp.columns.str.replace("urban decentral ", "decentral ")
    
    supplyr.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyr.columns = supplyr.columns.str.replace("residential ", "")
    supplyr.columns = supplyr.columns.str.replace("services ", "")
    supplyr.columns = supplyr.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(
        [
            "electric demand",
            "transmission lines",
            "hydroelectricity",
            "hydro reservoir",
            "run of river",
            "pumped hydro storage",
            "CHP",
            "onshore wind",
            "offshore wind",
            "solar PV",
            "solar thermal",
            "building retrofitting",
            "ground heat pump",
            "air heat pump",
            "resistive heater",
            "OCGT",
            "gas boiler",
            "gas",
            "natural gas",
            "methanation",
            "hydrogen storage",
            "battery storage",
            "hot water storage",
            "solar curtailment",
            "onshore curtailment",
            "offshore curtailment",
        ]
    )

    new_columnsn = preferred_order.intersection(supplyn.columns).append(
        supplyn.columns.difference(preferred_order)
    )
    new_columnsm = preferred_order.intersection(supplym.columns).append(
        supplym.columns.difference(preferred_order)
    )
    # new_columnso = preferred_order.intersection(supplyo.columns).append(
    #     supplyo.columns.difference(preferred_order)
    # )
    new_columnsp = preferred_order.intersection(supplyp.columns).append(
        supplyp.columns.difference(preferred_order)
    )
    new_columnsr = preferred_order.intersection(supplyr.columns).append(
        supplyr.columns.difference(preferred_order)
    )

    supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()
    supplym = supplym.groupby(supplym.columns, axis=1).sum()
    # supplyo = supplyo.groupby(supplyo.columns, axis=1).sum()
    supplyp = supplyp.groupby(supplyp.columns, axis=1).sum()
    supplyr = supplyr.groupby(supplyr.columns, axis=1).sum()
    
    c_solarn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyn = supplyn.T
    supplyn.loc["solar"] = supplyn.loc["solar"] + c_solarn
    supplyn.loc["offshore wind"] = supplyn.loc["offshore wind"] + c_offwindn
    supplyn.loc["onshore wind"] = supplyn.loc["onshore wind"] + c_onwindn
    supplyn.loc["solar curtailment"] = -abs(c_solarn)
    supplyn.loc["onshore curtailment"] = -abs(c_onwindn)
    supplyn.loc["offshore curtailment"] = -abs(c_offwindn)
    supplyn=supplyn.T
    
    c_solarm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplym = supplym.T
    supplym.loc["solar"] = supplym.loc["solar"] + c_solarm
    supplym.loc["offshore wind"] = supplym.loc["offshore wind"] + c_offwindm
    supplym.loc["onshore wind"] = supplym.loc["onshore wind"] + c_onwindm
    supplym.loc["solar curtailment"] = -abs(c_solarm)
    supplym.loc["onshore curtailment"] = -abs(c_onwindm)
    supplym.loc["offshore curtailment"] = -abs(c_offwindm)
    supplym=supplym.T
    
    # c_solaro=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    # c_onwindo=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    # c_offwindo=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    # supplyo = supplyo.T
    # supplyo.loc["solar"] = supplyo.loc["solar"] + c_solaro
    # supplyo.loc["offshore wind"] = supplyo.loc["offshore wind"] + c_offwindo
    # supplyo.loc["onshore wind"] = supplyo.loc["onshore wind"] + c_onwindo
    # supplyo.loc["solar curtailment"] = -abs(c_solaro)
    # supplyo.loc["onshore curtailment"] = -abs(c_onwindo)
    # supplyo.loc["offshore curtailment"] = -abs(c_offwindo)
    # supplyo=supplyo.T
    
    c_solarp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyp = supplyp.T
    supplyp.loc["solar"] = supplyp.loc["solar"] + c_solarp
    supplyp.loc["offshore wind"] = supplyp.loc["offshore wind"] + c_offwindp
    supplyp.loc["onshore wind"] = supplyp.loc["onshore wind"] + c_onwindp
    supplyp.loc["solar curtailment"] = -abs(c_solarp)
    supplyp.loc["onshore curtailment"] = -abs(c_onwindp)
    supplyp.loc["offshore curtailment"] = -abs(c_offwindp)
    supplyp=supplyp.T
    
    c_solarr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyr = supplyr.T
    supplyr.loc["solar"] = supplyr.loc["solar"] + c_solarr
    supplyr.loc["offshore wind"] = supplyr.loc["offshore wind"] + c_offwindr
    supplyr.loc["onshore wind"] = supplyr.loc["onshore wind"] + c_onwindr
    supplyr.loc["solar curtailment"] = -abs(c_solarr)
    supplyr.loc["onshore curtailment"] = -abs(c_onwindr)
    supplyr.loc["offshore curtailment"] = -abs(c_offwindr)
    supplyr=supplyr.T
    
    
    fig, (ax1,ax2,ax4,ax5) = plt.subplots(4,1)
    fig.set_size_inches((10, 35))

    (
        supplyn.loc[start:stop, supplyn.columns].plot(
            ax=ax1,
            kind="area",
            stacked=True,
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyn.columns
            ],
        )
    )
    
    (
        supplym.loc[start:stop, supplym.columns].plot(
            ax=ax2,
            kind="area",
            stacked=True,
            title = "2030",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplym.columns
            ],
        )
    )
    
    # (
    #     supplyo.loc[start:stop, supplyo.columns].plot(
    #         ax=ax3,
    #         kind="area",
    #         stacked=True,
    #         title = "2040",
    #         legend=False,
    #         linewidth=0.0,
    #         color=[
    #             config["plotting"]["tech_colors"][i.replace(suffix, "")]
    #             for i in supplyo.columns
    #         ],
    #     )
    # )
    
    (
        supplyp.loc[start:stop, supplyp.columns].plot(
            ax=ax4,
            kind="area",
            stacked=True,
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyp.columns
            ],
        )
    )
    
    (
        supplyr.loc[start:stop, supplyr.columns].plot(
            ax=ax5,
            kind="area",
            stacked=True,
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyr.columns
            ],
        )
    )

    handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax2.get_legend_handles_labels()
    # handles, labels = ax3.get_legend_handles_labels()
    # handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax5.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    ax4.legend(new_handles, new_labels, ncol=1, bbox_to_anchor=(1,3.4), loc="upper left", frameon=False, fontsize=20)
    ax1.set_title("Reference Case", fontsize=20)
    ax2.set_title("BAU-2050", fontsize=20)
    # ax3.set_title("2040", fontsize=11)
    ax4.set_title("Suff-2050", fontsize=20)
    ax5.set_title("Suff-2050-NO-CDR", fontsize=20)
    #fig.supxlabel('Electricity Dispatch in winters')

    # ax4.set_xlim([start, stop])
    # ax4.set_ylim([-800, 800])
    # ax4.grid(True)
    ax1.set_ylabel("Power [GW]", fontsize=20)
    ax1.set_ylim(-1250, 1250)
    ax2.set_ylabel("Power [GW]",fontsize=20)
    ax2.set_ylim(-1250, 1250)
    #ax3.set_ylabel("Demand [GW]")
    ax4.set_ylabel("Power [GW]", fontsize=20)
    ax4.set_ylim(-1250, 1250)
    ax5.set_ylabel("Power [GW]", fontsize=20)
    ax5.set_ylim(-1250, 1250)
    ax5.set_xlabel("Electricity Dispatch in Summer",fontsize=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)

    

    
plot_series(carrier="AC")
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
#plot_series(n, carrier="heat")

#%%%
import logging

logger = logging.getLogger(__name__)
def plot_series(carrier="AC", name="test"):
    
    n=pypsa.Network("/home/umair/pypsa-eur/results/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")
    m=pypsa.Network("/home/umair/pypsa-eur/resultsBAU2050/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")
    #o=pypsa.Network("/home/umair/pypsa-eur/results2040/S2040/postnetworks/elec_s_6_lv1.5__Co2L0p15-1H-T-H-B-I-A-dist1_2040.nc")
    p=pypsa.Network("/home/umair/pypsa-eur/resultsSuff2050/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")
    #q=pypsa.Network("/home/umair/pypsa-eur/results2050NOCCS/S2050NOCCS/postnetworks/elec_s_6_lv1.5__Co2L0p15-1H-T-H-B-I-A-dist1_2050.nc")
    r=pypsa.Network("/home/umair/pypsa-eur/resultsS2050LULUCF/postnetworks/elec_s_6_lvopt__Co2L0-1H-T-H-B-I-A-dist1_2050.nc")

    assign_location(n)
    assign_carriers(n)
    assign_location(m)
    assign_carriers(m)
    # assign_location(o)
    # assign_carriers(o)
    assign_location(p)
    assign_carriers(p)
    # assign_location(q)
    # assign_carriers(q)
    assign_location(r)
    assign_carriers(r)

    busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]
    busesm = m.buses.index[m.buses.carrier.str.contains(carrier)]
    # buseso = o.buses.index[o.buses.carrier.str.contains(carrier)]
    busesp = p.buses.index[p.buses.carrier.str.contains(carrier)]
    busesr = r.buses.index[r.buses.carrier.str.contains(carrier)]
    

    supplyn = pd.DataFrame(index=n.snapshots)
    supplym = pd.DataFrame(index=m.snapshots)
    # supplyo = pd.DataFrame(index=o.snapshots)
    supplyp = pd.DataFrame(index=p.snapshots)
    supplyr = pd.DataFrame(index=r.snapshots)
    
    for c in n.iterate_components(n.branch_components):
        n_port = 2 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyn = pd.concat(
                (
                    supplyn,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesn)]
        supplyn = pd.concat(
            (
                supplyn,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in m.iterate_components(m.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplym = pd.concat(
                (
                    supplym,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesm)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in m.iterate_components(m.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesm)]
        supplym = pd.concat(
            (
                supplym,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    # for c in o.iterate_components(o.branch_components):
    #     n_port = 4 if c.name == "Link" else 2     #port3
    #     for i in range(n_port):
    #         supplyo = pd.concat(
    #             (
    #                 supplyo,
    #                 (-1)
    #                 * c.pnl["p" + str(i)]
    #                 .loc[:, c.df.index[c.df["bus" + str(i)].isin(buseso)]]
    #                 .groupby(c.df.carrier, axis=1)
    #                 .sum(),
    #             ),
    #             axis=1,
    #         )

    # for c in o.iterate_components(o.one_port_components):
    #     comps = c.df.index[c.df.bus.isin(buseso)]
    #     supplyo = pd.concat(
    #         (
    #             supplyo,
    #             ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
    #             .groupby(c.df.carrier, axis=1)
    #             .sum(),
    #         ),
    #         axis=1,
    #     )
        
    for c in p.iterate_components(p.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyp = pd.concat(
                (
                    supplyp,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesp)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in p.iterate_components(p.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesp)]
        supplyp = pd.concat(
            (
                supplyp,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in r.iterate_components(r.branch_components):
        n_port = 2 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyr = pd.concat(
                (
                    supplyr,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesr)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in r.iterate_components(r.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesr)]
        supplyr = pd.concat(
            (
                supplyr,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )

    supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
    supplym = supplym.groupby(rename_techs_tyndp, axis=1).sum()
    # supplyo = supplyo.groupby(rename_techs_tyndp, axis=1).sum()
    supplyp = supplyp.groupby(rename_techs_tyndp, axis=1).sum()
    supplyr = supplyr.groupby(rename_techs_tyndp, axis=1).sum()
    

    bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]
    bothm = supplym.columns[(supplym < 0.0).any() & (supplym > 0.0).any()]
    # botho = supplyo.columns[(supplyo < 0.0).any() & (supplyo > 0.0).any()]
    bothp = supplyp.columns[(supplyp < 0.0).any() & (supplyp > 0.0).any()]
    bothr = supplyr.columns[(supplyr < 0.0).any() & (supplyr > 0.0).any()]

    positive_supplyn = supplyn[bothn]
    negative_supplyn = supplyn[bothn]
    positive_supplym = supplym[bothm]
    negative_supplym = supplym[bothm]
    # positive_supplyo = supplyo[botho]
    # negative_supplyo = supplyo[botho]
    positive_supplyp = supplyp[bothp]
    negative_supplyp = supplyp[bothp]
    positive_supplyr = supplyr[bothr]
    negative_supplyr = supplyr[bothr]

    positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
    negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)
    positive_supplym = positive_supplym.mask(positive_supplym < 0.0, 0.0)
    negative_supplym = negative_supplym.mask(negative_supplym > 0.0, 0.0)
    # positive_supplyo = positive_supplyo.mask(positive_supplyo < 0.0, 0.0)
    # negative_supplyo = negative_supplyo.mask(negative_supplyo > 0.0, 0.0)
    positive_supplyp = positive_supplyp.mask(positive_supplyp < 0.0, 0.0)
    negative_supplyp = negative_supplyp.mask(negative_supplyp > 0.0, 0.0)
    positive_supplyr = positive_supplyr.mask(positive_supplyr < 0.0, 0.0)
    negative_supplyr = negative_supplyr.mask(negative_supplyr > 0.0, 0.0)


    supplyn[bothn] = positive_supplyn
    supplym[bothm] = positive_supplym
    # supplyo[botho] = positive_supplyo
    supplyp[bothp] = positive_supplyp
    supplyr[bothr] = positive_supplyr
    

    suffix = " charging"

    negative_supplyn.columns = negative_supplyn.columns + suffix
    negative_supplym.columns = negative_supplym.columns + suffix
    # negative_supplyo.columns = negative_supplyo.columns + suffix
    negative_supplyp.columns = negative_supplyp.columns + suffix
    negative_supplyr.columns = negative_supplyr.columns + suffix

    supplyn = pd.concat((supplyn, negative_supplyn), axis=1)
    supplym = pd.concat((supplym, negative_supplym), axis=1)
    # supplyo = pd.concat((supplyo, negative_supplyo), axis=1)
    supplyp = pd.concat((supplyp, negative_supplyp), axis=1)
    supplyr = pd.concat((supplyr, negative_supplyr), axis=1)
    
    

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-08-01"
    stop = "2013-08-07"
    

    threshold = 1

    to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]
    to_dropm = supplym.columns[(abs(supplym) < threshold).all()]
    # to_dropo = supplyo.columns[(abs(supplyo) < threshold).all()]
    to_dropp = supplyp.columns[(abs(supplyp) < threshold).all()]
    to_dropr = supplyr.columns[(abs(supplyr) < threshold).all()]

    if len(to_dropn) != 0:
        logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
        supplyn.drop(columns=to_dropn, inplace=True)
    
    if len(to_dropm) != 0:
        logger.info(f"Dropping {to_dropm.tolist()} from supplym")
        supplym.drop(columns=to_dropm, inplace=True)
        
    # if len(to_dropo) != 0:
    #     logger.info(f"Dropping {to_dropo.tolist()} from supplyio")
    #     supplyo.drop(columns=to_dropo, inplace=True)
        
    if len(to_dropp) != 0:
        logger.info(f"Dropping {to_dropp.tolist()} from supplyp")
        supplyp.drop(columns=to_dropp, inplace=True)
        
    if len(to_dropr) != 0:
        logger.info(f"Dropping {to_dropr.tolist()} from supplyr")
        supplyr.drop(columns=to_dropr, inplace=True)

    supplyn.index.name = None
    supplym.index.name = None
    # supplyo.index.name = None
    supplyp.index.name = None
    supplyr.index.name = None
    

    supplyn = supplyn / 1e3
    supplym = supplym / 1e3
    # supplyo = supplyo / 1e3
    supplyp = supplyp / 1e3
    supplyr = supplyr / 1e3
    #del supply["CCS"]

    supplyn.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyn.columns = supplyn.columns.str.replace("residential ", "")
    supplyn.columns = supplyn.columns.str.replace("services ", "")
    supplyn.columns = supplyn.columns.str.replace("urban decentral ", "decentral ")
    
    supplym.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplym.columns = supplym.columns.str.replace("residential ", "")
    supplym.columns = supplym.columns.str.replace("services ", "")
    supplym.columns = supplym.columns.str.replace("urban decentral ", "decentral ")
    
    # supplyo.rename(
    #     columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    # )
    # supplyo.columns = supplyo.columns.str.replace("residential ", "")
    # supplyo.columns = supplyo.columns.str.replace("services ", "")
    # supplyo.columns = supplyo.columns.str.replace("urban decentral ", "decentral ")
    
    supplyp.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyp.columns = supplyp.columns.str.replace("residential ", "")
    supplyp.columns = supplyp.columns.str.replace("services ", "")
    supplyp.columns = supplyp.columns.str.replace("urban decentral ", "decentral ")
    
    supplyr.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyr.columns = supplyr.columns.str.replace("residential ", "")
    supplyr.columns = supplyr.columns.str.replace("services ", "")
    supplyr.columns = supplyr.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(
        [
            "electric demand",
            "transmission lines",
            "hydroelectricity",
            "hydro reservoir",
            "run of river",
            "pumped hydro storage",
            "CHP",
            "onshore wind",
            "offshore wind",
            "solar PV",
            "solar thermal",
            "building retrofitting",
            "ground heat pump",
            "air heat pump",
            "resistive heater",
            "OCGT",
            "gas boiler",
            "gas",
            "natural gas",
            "methanation",
            "hydrogen storage",
            "battery storage",
            "hot water storage",
            "solar curtailment",
            "onshore curtailment",
            "offshore curtailment",
        ]
    )

    new_columnsn = preferred_order.intersection(supplyn.columns).append(
        supplyn.columns.difference(preferred_order)
    )
    new_columnsm = preferred_order.intersection(supplym.columns).append(
        supplym.columns.difference(preferred_order)
    )
    # new_columnso = preferred_order.intersection(supplyo.columns).append(
    #     supplyo.columns.difference(preferred_order)
    # )
    new_columnsp = preferred_order.intersection(supplyp.columns).append(
        supplyp.columns.difference(preferred_order)
    )
    new_columnsr = preferred_order.intersection(supplyr.columns).append(
        supplyr.columns.difference(preferred_order)
    )

    supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()
    supplym = supplym.groupby(supplym.columns, axis=1).sum()
    # supplyo = supplyo.groupby(supplyo.columns, axis=1).sum()
    supplyp = supplyp.groupby(supplyp.columns, axis=1).sum()
    supplyr = supplyr.groupby(supplyr.columns, axis=1).sum()
    
    
    
    fig, (ax1,ax2,ax4,ax5) = plt.subplots(4,1)
    fig.set_size_inches((10, 34))

    (
        supplyn.loc[start:stop, supplyn.columns].plot(
            ax=ax1,
            kind="area",
            stacked=True,
            title = "BAU",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyn.columns
            ],
        )
    )
    
    (
        supplym.loc[start:stop, supplym.columns].plot(
            ax=ax2,
            kind="area",
            stacked=True,
            title = "2030",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplym.columns
            ],
        )
    )
    
    # (
    #     supplyo.loc[start:stop, supplyo.columns].plot(
    #         ax=ax3,
    #         kind="area",
    #         stacked=True,
    #         title = "2040",
    #         legend=False,
    #         linewidth=0.0,
    #         color=[
    #             config["plotting"]["tech_colors"][i.replace(suffix, "")]
    #             for i in supplyo.columns
    #         ],
    #     )
    # )
    
    (
        supplyp.loc[start:stop, supplyp.columns].plot(
            ax=ax4,
            kind="area",
            stacked=True,
            title = "2050",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyp.columns
            ],
        )
    )
    
    (
        supplyr.loc[start:stop, supplyr.columns].plot(
            ax=ax5,
            kind="area",
            stacked=True,
            title = "BAU-2050",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyr.columns
            ],
        )
    )

    handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax2.get_legend_handles_labels()
    # handles, labels = ax3.get_legend_handles_labels()
    # handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax5.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    ax4.legend(new_handles, new_labels, ncol=1, bbox_to_anchor=(1,3.4), loc="upper left", frameon=False, fontsize=20)
    #fig.supxlabel('Electricity Dispatch in winters')

    # ax4.set_xlim([start, stop])
    # ax4.set_ylim([-800, 800])
    # ax4.grid(True)
    ax1.set_title("Reference Case", fontsize=20)
    ax2.set_title("BAU-2050", fontsize=20)
    # ax3.set_title("2040", fontsize=11)
    ax4.set_title("Suff-2050", fontsize=20)
    ax5.set_title("Suff-2050-NO-CDR", fontsize=20)
    #fig.supxlabel('Electricity Dispatch in winters')

    # ax4.set_xlim([start, stop])
    # ax4.set_ylim([-800, 800])
    # ax4.grid(True)
    ax1.set_ylabel("Heat [GW]", fontsize=20)
    ax1.set_ylim(-200, 200)
    ax2.set_ylabel("Heat [GW]", fontsize=20)
    ax2.set_ylim(-200, 200)
    # ax3.set_ylabel("Demand [GW]")
    ax4.set_ylabel("Heat [GW]", fontsize=20)
    ax4.set_ylim(-200, 200)
    ax5.set_ylabel("Heat [GW]", fontsize=20)
    ax5.set_ylim(-200, 200)
    ax5.set_xlabel("Heat Dispatch in Summer",fontsize=20)
    
    

    
plot_series(carrier="heat")
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

