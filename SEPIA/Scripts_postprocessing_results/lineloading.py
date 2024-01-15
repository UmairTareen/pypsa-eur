#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:02:14 2023


"""
import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import os

save_figure = True

scenario = 'bau'

planning_horizons = [2020,2030,2040,2050]
def upload_networks(planning_horizon):
    networks = {
        'n': f'../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_{planning_horizon}.nc'
    }
    
    network_objects = {}
    
    for name, path in networks.items():
        try:
            network = pypsa.Network(path)
            network_objects[name] = network
        except Exception as e:
            print(f"Error loading network {name}: {str(e)}")
    
    return network_objects


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
    if save_figure == True:
     if not os.path.exists(save_folder):
            os.makedirs(save_folder)

     fn_path = os.path.join(save_folder, fn)
     plt.savefig(fn_path, dpi=300, bbox_inches="tight")
    plt.show()
# Usage:
save_folder = f"../../results/{scenario}/plots"
for planning_horizon in planning_horizons:
  networks = upload_networks(planning_horizon)
  fn = f"line_loading_{planning_horizon}.png"
  if planning_horizon == 2020:
     title = f"Reff - {planning_horizon}"
  else:
     title = f"{scenario} - {planning_horizon}"
  plot_line_loading(networks['n'], fn=fn, title=title)
