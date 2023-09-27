#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:02:14 2023


"""
import pypsa
import matplotlib.pyplot as plt
import pandas as pd

def upload_networks():
    networks = {
        'n': "/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc",
        'm1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'm2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'm3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc",
        'p1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'p2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'p3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc",
        'r1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'r2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'r3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc"
    }
    
    network_objects = {}
    
    for name, path in networks.items():
        try:
            network = pypsa.Network(path)
            network_objects[name] = network
        except Exception as e:
            print(f"Error loading network {name}: {str(e)}")
    
    return network_objects



def process_marginal_price_data(network, carrier):
    marginal_price = network.buses_t.marginal_price.loc[:, network.buses.carrier == carrier].stack()
    sorted_price = marginal_price.sort_values(ascending=False).reset_index(drop=True)
    sorted_price.index = [i / len(marginal_price) * 100 for i in sorted_price.index]
    return sorted_price

def plot_price_duration_curve(networks, carriers):
    for carrier in carriers:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xlabel("Share of Snapshots and Nodes [%]")
        plt.ylabel("Nodal Price [EUR/MWh]")
        plt.axvline(0, linewidth=0.5, linestyle=":", color="grey")
        plt.axvline(100, linewidth=0.5, linestyle=":", color="grey")
        plt.axhline(0, linewidth=0.5, linestyle=":", color="grey")

        if carrier == "H2":
            title = "Hydrogen"
            plt.ylim([-20, 350])
        elif carrier == "AC":
            title = "Electricity"
            plt.ylim([-10, 500])
        elif carrier == "urban central heat":
            title = "District Heating"
            plt.ylim([-10, 500])
        elif carrier == "low voltage":
            title = "Distribution Electricity"
            plt.ylim([-10, 500])
        else:
            title = carrier

        plt.title(title, fontsize=12, color="#343434")

        for i, network in enumerate(networks):
            sorted_price = process_marginal_price_data(network, carrier)
            sorted_price.plot(ax=ax, legend=True, label=f"Network {i + 1}")

        plt.legend()

carriers = ["AC", "H2", "urban central heat", "low voltage"]


networks = upload_networks()


plot_price_duration_curve([networks['n'], networks['m1'], networks['m2'], networks['m3'], 
                          networks['p1'], networks['p2'], networks['p3'], networks['r1'], 
                          networks['r2'], networks['r3']], carriers)


plt.show()

#%%

import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import pandas as pd
import sys
import os
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

def plot_line_loading(network, fn=None, title="Network"):
    line_width_factor = 4e3

    n = network.copy()

    n.mremove("Bus", n.buses.index[n.buses.carrier != "AC"])
    n.mremove("Link", n.links.index[n.links.carrier != "DC"])

    line_loading = (
        n.lines_t.p0.abs().mean() / (n.lines.s_nom_opt * n.lines.s_max_pu) * 100
    )

    link_loading = (
        n.links_t.p0.abs().mean() / (n.links.p_nom_opt * n.links.p_max_pu) * 100
    )

    crs = ccrs.EqualEarth()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": crs})

    cmap = plt.cm.OrRd
    norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    line_colors = pd.Series(
        list(map(mcolors.to_hex, cmap(norm(line_loading)))), index=line_loading.index
    )

    link_colors = pd.Series(
        list(map(mcolors.to_hex, cmap(norm(link_loading)))), index=link_loading.index
    )

    n.plot(
        geomap=True,
        ax=ax,
        bus_sizes=0.08,
        bus_colors="k",
        line_colors=line_colors,
        line_widths=n.lines.s_nom_opt / line_width_factor,
        link_colors=link_colors,
        link_widths=n.links.p_nom_opt / line_width_factor,
    )

    cbar = plt.colorbar(
        sm,
        orientation="vertical",
        shrink=0.7,
        ax=ax,
        label="Average Line Loading",
    )
    sizes = [2, 1]
    labels = [f"{s} GW" for s in sizes]
    scale = line_width_factor / 1e3
    sizes = [s * scale for s in sizes]
    
    add_legend_lines(
        ax,
        sizes,
        labels,
        legend_kw=dict(title="Line Capacity", bbox_to_anchor=(0.6, 1)),
        patch_kw=dict(color="lightgrey"),
    )

    axins = ax.inset_axes([0.05, 0.1, 0.3, 0.2])
    curve = line_loading.sort_values().reset_index(drop=True)
    curve.index = [c / curve.size * 100 for c in curve.index]
    curve.plot(
        ax=axins,
        ylim=(-5, 150),
        yticks=[0, 25, 50, 75, 100, 125, 150],
        c="firebrick",
        linewidth=1.5,
    )
    axins.annotate("Loading [%]", (3, 83), color="darkgrey", fontsize=9)
    axins.annotate("Lines [%]", (55, 7), color="darkgrey", fontsize=9)
    axins.grid(True)
    
    plt.title(title)

# Usage:
plot_line_loading(networks['n'], title="Reff")
plt.show()

#%%
import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import pandas as pd

# Define the function to upload network objects
def upload_networks():
    networks = {
        'n': "/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc",
        'm1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'm2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'm3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/bau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc",
        'p1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'p2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'p3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/suff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc",
        'r1': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2030.nc",
        'r2': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2040.nc",
        'r3': "/home/umair/pypsa-eur_repository/simulations/myopic simulations/nocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc"
    }
    
    network_objects = {}
    
    for name, path in networks.items():
        try:
            network = pypsa.Network(path)
            network_objects[name] = network
        except Exception as e:
            print(f"Error loading network {name}: {str(e)}")
    
    return network_objects

# Load the network objects into a dictionary
networks_dict = upload_networks()

# Create a list of network objects
networks = [networks_dict['n'], networks_dict['m1'], networks_dict['m2'], networks_dict['m3'],
            networks_dict['p1'], networks_dict['p2'], networks_dict['p3'], networks_dict['r1'],
            networks_dict['r2'], networks_dict['r3']]
#%%
def create_line_loading_plot(network_set, titles, filename=None, subplot_width=4, subplot_height=4):
    line_width_factor = 4e3
    cmap = plt.cm.OrRd
    norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    num_networks = len(network_set)
    num_cols = 4  # Number of columns for subplots
    num_rows = 1  # Number of rows (1x4 configuration)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(subplot_width * num_cols, subplot_height * num_rows), subplot_kw={"projection": ccrs.EqualEarth()})
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, (network, title) in enumerate(zip(network_set, titles)):
        n = network.copy()
        n.mremove("Bus", n.buses.index[n.buses.carrier != "AC"])
        n.mremove("Link", n.links.index[n.links.carrier != "DC"])

        line_loading = n.lines_t.p0.abs().mean() / (n.lines.s_nom_opt * n.lines.s_max_pu) * 100
        link_loading = n.links_t.p0.abs().mean() / (n.links.p_nom_opt * n.links.p_max_pu) * 100

        line_colors = pd.Series(list(map(mcolors.to_hex, cmap(norm(line_loading)))), index=line_loading.index)
        link_colors = pd.Series(list(map(mcolors.to_hex, cmap(norm(link_loading)))), index=link_loading.index)

        row_idx = 0  # Always in the first row
        col_idx = i  # Place each plot in a separate column
        ax = axes[col_idx]

        n.plot(
            geomap=True,
            ax=ax,
            bus_sizes=0.08,
            bus_colors="k",
            line_colors=line_colors,
            line_widths=n.lines.s_nom_opt / line_width_factor,
            link_colors=link_colors,
            link_widths=n.links.p_nom_opt / line_width_factor,
        )

        ax.set_title(title, fontsize=15)

        # Add colorbar
        if i == num_networks - 1:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical", label="Average Line Loading")

        
        # sizes = [2, 1]
        # labels = [f"{s} GW" for s in sizes]
        # scale = line_width_factor / 1e3
        # sizes = [s * scale for s in sizes]

        # add_legend_lines(
        #     ax,
        #     sizes,
        #     labels,
        #     legend_kw=dict(title="Line Capacity", bbox_to_anchor=(0.6, 1)),
        #     patch_kw=dict(color="lightgrey"),
        # )

        
        axins = ax.inset_axes([0.05, 0.1, 0.3, 0.2])
        curve = line_loading.sort_values().reset_index(drop=True)
        curve.index = [c / curve.size * 100 for c in curve.index]
        curve.plot(
            ax=axins,
            ylim=(-5, 150),
            yticks=[0,25, 50, 75,100,125, 150],
            c="firebrick",
            linewidth=2,
        )
        axins.annotate("Loading [%]", (3, 83), color="darkgrey", fontsize=9)
        axins.annotate("Lines [%]", (55, 7), color="darkgrey", fontsize=9)
        axins.grid(True)


network_titles = ["Reff", "BAU-2030","BAU-2040","BAU-2050", "Reff","SUff-2030", "SUff-2040", "SUff-2050",
                  "Reff","NO_CDR-2030", "NO_CDR-2040", "NO_CDR-2050"]


n1 = networks_dict["n"]
mf1 = networks_dict["m1"]
mf2 = networks_dict["m2"]
mf3 = networks_dict["m3"]
pf1 = networks_dict["p1"]
pf2 = networks_dict["p2"]
pf3 = networks_dict["p3"]
rf1 = networks_dict["r1"]
rf2 = networks_dict["r2"]
rf3 = networks_dict["r3"]


network_set1 = [n1, mf1, mf2, mf3]
network_set2 = [n1, pf1, pf2, pf3]
network_set3 = [n1, rf1, rf2, rf3]

create_line_loading_plot(network_set1, network_titles[:4], subplot_width=9, subplot_height=6)
create_line_loading_plot(network_set2, network_titles[4:8], subplot_width=8, subplot_height=6)
create_line_loading_plot(network_set3, network_titles[8:], subplot_width=8, subplot_height=6)

