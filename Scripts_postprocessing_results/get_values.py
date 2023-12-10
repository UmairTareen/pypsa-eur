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



file = '/home/sylvain/svn/pypsa-eur/results/postnetworks/elec_s_6_lvopt__EQ0.7c-1H-T-H-B-I-A-dist1_2020.nc'
#file = '/home/sylvain/temp/resultsbau/postnetworks/elec_s_6_lvopt__EQ0.7c-1H-T-H-B-I-A-dist1_2050.nc'

n= pypsa.Network(file)

# n= pypsa.Network("../simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")

with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)

# DC lines:
idx = n.links[(n.links.carrier=='DC') & (n.links.bus1 =='BE1 0') ].index
dclines = pd.DataFrame(index=idx,columns=['from','to','capacity (MW)'])
dclines['from'] = n.links.bus0[idx]
dclines['to'] = n.links.bus1[idx]
dclines['capacity (MW)'] = n.links.p_nom_opt[idx]
print(dclines)

# AC lines:
idx = n.lines[n.lines.bus0=='BE1 0'].index
aclines = pd.DataFrame(index=idx,columns=['from','to','capacity (MW)'])
aclines['from'] = n.lines.bus0[idx]
aclines['to'] = n.lines.bus1[idx]
aclines['capacity (MW)'] = n.lines.s_nom_opt[idx]
print(aclines)

a = 1