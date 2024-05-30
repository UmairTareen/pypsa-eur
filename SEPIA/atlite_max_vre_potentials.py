#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:49:39 2023

GRID_CODE,CLC_CODE,LABEL1,LABEL2,LABEL3,RGB
1,111,Artificial surfaces,Urban fabric,Continuous urban fabric,230-000-077
2,112,Artificial surfaces,Urban fabric,Discontinuous urban fabric,255-000-000
3,121,Artificial surfaces,"Industrial, commercial and transport units",Industrial or commercial units,204-077-242
4,122,Artificial surfaces,"Industrial, commercial and transport units",Road and rail networks and associated land,204-000-000
5,123,Artificial surfaces,"Industrial, commercial and transport units",Port areas,230-204-204
6,124,Artificial surfaces,"Industrial, commercial and transport units",Airports,230-204-230
7,131,Artificial surfaces,"Mine, dump and construction sites",Mineral extraction sites,166-000-204
8,132,Artificial surfaces,"Mine, dump and construction sites",Dump sites,166-077-000
9,133,Artificial surfaces,"Mine, dump and construction sites",Construction sites,255-077-255
10,141,Artificial surfaces,"Artificial, non-agricultural vegetated areas",Green urban areas,255-166-255
11,142,Artificial surfaces,"Artificial, non-agricultural vegetated areas",Sport and leisure facilities,255-230-255
12,211,Agricultural areas,Arable land,Non-irrigated arable land,255-255-168
13,212,Agricultural areas,Arable land,Permanently irrigated land,255-255-000
14,213,Agricultural areas,Arable land,Rice fields,230-230-000
15,221,Agricultural areas,Permanent crops,Vineyards,230-128-000
16,222,Agricultural areas,Permanent crops,Fruit trees and berry plantations,242-166-077
17,223,Agricultural areas,Permanent crops,Olive groves,230-166-000
18,231,Agricultural areas,Pastures,Pastures,230-230-077
19,241,Agricultural areas,Heterogeneous agricultural areas,Annual crops associated with permanent crops,255-230-166
20,242,Agricultural areas,Heterogeneous agricultural areas,Complex cultivation patterns,255-230-077
21,243,Agricultural areas,Heterogeneous agricultural areas,"Land principally occupied by agriculture, with significant areas of natural vegetation",230-204-077
22,244,Agricultural areas,Heterogeneous agricultural areas,Agro-forestry areas,242-204-166
23,311,Forest and semi natural areas,Forests,Broad-leaved forest,128-255-000
24,312,Forest and semi natural areas,Forests,Coniferous forest,000-166-000
25,313,Forest and semi natural areas,Forests,Mixed forest,077-255-000
26,321,Forest and semi natural areas,Scrub and/or herbaceous vegetation associations,Natural grasslands,204-242-077
27,322,Forest and semi natural areas,Scrub and/or herbaceous vegetation associations,Moors and heathland,166-255-128
28,323,Forest and semi natural areas,Scrub and/or herbaceous vegetation associations,Sclerophyllous vegetation,166-230-077
29,324,Forest and semi natural areas,Scrub and/or herbaceous vegetation associations,Transitional woodland-shrub,166-242-000
30,331,Forest and semi natural areas,Open spaces with little or no vegetation,"Beaches, dunes, sands",230-230-230
31,332,Forest and semi natural areas,Open spaces with little or no vegetation,Bare rocks,204-204-204
32,333,Forest and semi natural areas,Open spaces with little or no vegetation,Sparsely vegetated areas,204-255-204
33,334,Forest and semi natural areas,Open spaces with little or no vegetation,Burnt areas,000-000-000
34,335,Forest and semi natural areas,Open spaces with little or no vegetation,Glaciers and perpetual snow,166-230-204
35,411,Wetlands,Inland wetlands,Inland marshes,166-166-255
36,412,Wetlands,Inland wetlands,Peat bogs,077-077-255
37,421,Wetlands,Maritime wetlands,Salt marshes,204-204-255
38,422,Wetlands,Maritime wetlands,Salines,230-230-255
39,423,Wetlands,Maritime wetlands,Intertidal flats,166-166-230
40,511,Water bodies,Inland waters,Water courses,000-204-242
41,512,Water bodies,Inland waters,Water bodies,128-242-230
42,521,Water bodies,Marine waters,Coastal lagoons,000-255-166
43,522,Water bodies,Marine waters,Estuaries,166-255-230
44,523,Water bodies,Marine waters,Sea and ocean,230-242-255

"""

import geopandas as gpd
import atlite
import matplotlib.pyplot as plt
from atlite.gis import ExclusionContainer, shape_availability
from rasterio.plot import show
import xarray as xr
import os
import base64
import yaml
import functools
import numpy as np
import pandas as pd

def generate_max_onshore_potentials(country, output_html):
    plt.rcParams['figure.figsize'] = [7, 7]
    
    with open("../config/config.yaml") as file:
     config = yaml.safe_load(file)
     
    corine_grid_codes = config['renewable']['onwind']['corine']['grid_codes']
    distance_codes = config['renewable']['onwind']['corine']['distance_grid_codes']
    buffer = config['renewable']['onwind']['corine']['distance']
    capacity = config['renewable']['onwind']['capacity_per_sqkm']
    # Load world shapes
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Filter for specified countries
    shapes = world[world.name.isin(countries)].set_index('name')
    shapes.plot()

    # Define bounds and create cutout
    bounds = shapes.unary_union.buffer(0.5).bounds
    cutout = atlite.Cutout(
        path=f"{country}.nc",
        module="era5",
        bounds=bounds,
        time=slice('2013-01-01', '2013-12-31')
    )

    # Directory to save images
    image_dir = 'max_vre_potential_charts/images_onshore'
    os.makedirs(image_dir, exist_ok=True)

    # Create figures and save them as images
    fig_list = []
    titles = [
        "Country Shape",
        "Eligible Area (Green) - Onshore",
        "Eligible Area with Grid cells",
        "Capacity Factor for eligible area",
        "Maximum Onshore Wind Potential"
    ]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    shapes.plot(ax=ax1,color='green')
    cutout.grid.plot(ax=ax1, edgecolor='grey', color='None')
    fig1_path = os.path.join(image_dir, 'fig1.png')
    fig1.savefig(fig1_path)
    fig_list.append(fig1_path)
    print(f"Saved {fig1_path}")

    CORINE = '../data/bundle/corine/g250_clc06_V18_5.tif'
    excluder = ExclusionContainer()
    excluder.add_raster(CORINE, codes=corine_grid_codes,crs=3035, invert=True)
    excluder.add_raster(CORINE, codes=distance_codes, buffer=buffer,crs=3035)
    exc = shapes.loc[[country]].geometry.to_crs(excluder.crs)
    masked, transform = shape_availability(exc, excluder)
    eligible_share = masked.sum() * excluder.res**2 / exc.geometry.item().area

    fig2, ax2 = plt.subplots()
    show(masked, transform=transform, cmap='Greens', ax=ax2)
    exc.plot(ax=ax2, edgecolor='k', color='None')
    ax2.set_title(f'Eligible area (green) {eligible_share * 100:2.2f}%')
    fig2_path = os.path.join(image_dir, 'fig2.png')
    fig2.savefig(fig2_path)
    fig_list.append(fig2_path)
    print(f"Saved {fig2_path}")

    fig3, ax3 = plt.subplots()
    show(masked, transform=transform, cmap='Greens', ax=ax3)
    exc.plot(ax=ax3, edgecolor='k', color='None')
    cutout.grid.to_crs(excluder.crs).plot(edgecolor='grey', color='None', ax=ax3, ls=':')
    ax3.set_title(f'Eligible area (green) {eligible_share * 100:2.2f}%')
    fig3_path = os.path.join(image_dir, 'fig3.png')
    fig3.savefig(fig3_path)
    fig_list.append(fig3_path)
    print(f"Saved {fig3_path}")

    A = cutout.availabilitymatrix(shapes, excluder)
    A.name = "Capacity Factor"

    fig4, ax4 = plt.subplots()
    A.sel(name=country).plot(cmap='Greens')
    shapes.loc[[country]].plot(ax=ax4, edgecolor='k', color='None')
    cutout.grid.plot(ax=ax4, color='None', edgecolor='grey', ls=':')
    fig4_path = os.path.join(image_dir, 'fig4.png')
    fig4.savefig(fig4_path)
    fig_list.append(fig4_path)
    print(f"Saved {fig4_path}")

    cap_per_sqkm = capacity
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm
    cutout.prepare()
    wind = cutout.wind(matrix=capacity_matrix, turbine=atlite.windturbines.Vestas_V112_3MW, index=shapes.index)

    fig5, ax5 = plt.subplots(figsize=(14, 7))
    wind.to_pandas().div(1e3).plot(ylabel='Onshore Wind [GW]',fontsize=14, ax=ax5, color='green')
    ax5.set_ylabel('Onshore Wind [GW]', fontsize=14)
    fig5_path = os.path.join(image_dir, 'fig5.png')
    fig5.savefig(fig5_path)
    fig_list.append(fig5_path)
    print(f"Saved {fig5_path}")

    # Generate HTML file
    with open(output_html, 'w') as f:
        f.write('<html><head><title>Figures</title></head><body>\n')
        for fig_path, title in zip(fig_list, titles):
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                f.write(f'<h2>Figure:    {title}</h2>\n')
                f.write(f'<img src="data:image/png;base64,{encoded_string}" alt="{os.path.basename(fig_path)}" style="width:60%; height:auto;"><br>\n')
            else:
                print(f"Image {fig_path} not found.")
        f.write('</body></html>')
    print(f"HTML file saved at {output_html}")


def generate_max_offshore_ac_potentials(country, output_html):
    plt.rcParams['figure.figsize'] = [7, 7]
    
    with open("../config/config.yaml") as file:
     config = yaml.safe_load(file)
     
    corine_grid_codes = config['renewable']['offwind-ac']['corine']
    max_shore_distance = config['renewable']['offwind-ac']['max_shore_distance']
    capacity = config['renewable']['offwind-ac']['capacity_per_sqkm']
    ship_threshold = config['renewable']['offwind-ac']['ship_threshold']
    shipping_threshold = (
        ship_threshold * 8760 * 6
    ) 
    func = functools.partial(np.less, shipping_threshold)
    max_depth = config['renewable']['offwind-ac']['max_depth']
    func = functools.partial(np.greater, -max_depth)
    excluder_resolution = config['renewable']['offwind-ac']['excluder_resolution']
    
    country_shape = "../resources/ncdr/country_shapes.geojson"
    eez_path = "../data/bundle/eez/World_EEZ_v8_2014.shp"
    eez = gpd.read_file(eez_path)
    # separator = ", "
    # country= separator.join(country)
    # Filter for specified countries
    shapes = eez[eez['Country'] == country]
    shapes.plot()
    
    # Define bounds and create cutout
    bounds = shapes.unary_union.buffer(0.5).bounds
    cutout = atlite.Cutout(country, module='era5', bounds=bounds, time=slice('2013-01-01', '2013-12-31'))

    # Directory to save images
    image_dir = 'max_vre_potential_charts/images_offshore_ac'
    os.makedirs(image_dir, exist_ok=True)

    # Create figures and save them as images
    fig_list = []
    titles = [
        "Country Shape",
        "Eligible Area (blue) - Offshore-ac",
        "Eligible Area with Grid cells",
        "Capacity Factor for eligible area",
        "Maximum Offshore-ac Wind Potential"
    ]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    shapes.plot(ax=ax1,color='blue')
    cutout.grid.plot(ax=ax1, edgecolor='grey', color='None')
    fig1_path = os.path.join(image_dir, 'fig1.png')
    fig1.savefig(fig1_path)
    fig_list.append(fig1_path)
    print(f"Saved {fig1_path}")

    CORINE = '../data/bundle/corine/g250_clc06_V18_5.tif'
    ship_density = '../resources/ncdr/shipdensity_raster.tif'
    gebco = '../data/bundle/GEBCO_2014_2D.nc'
    excluder = ExclusionContainer(crs=3035, res = excluder_resolution)
    excluder.add_raster(CORINE, codes=corine_grid_codes,crs=3035, invert=True)
    excluder.add_raster(ship_density, codes=func, crs=4326, allow_no_overlap=True)
    excluder.add_raster(gebco, codes=func, crs=4326, nodata=-1000)
    excluder.add_geometry(country_shape, buffer=max_shore_distance, invert=True)
    exc = shapes.geometry.to_crs(excluder.crs)
    masked, transform = shape_availability(exc, excluder)
    eligible_share = masked.sum() * excluder.res**2 / exc.geometry.item().area

    fig2, ax2 = plt.subplots()
    show(masked, transform=transform, cmap='Blues', ax=ax2)
    exc.plot(ax=ax2, edgecolor='k', color='None')
    ax2.set_title(f'Eligible area (blue) {eligible_share * 100:2.2f}%')
    fig2_path = os.path.join(image_dir, 'fig2.png')
    fig2.savefig(fig2_path)
    fig_list.append(fig2_path)
    print(f"Saved {fig2_path}")

    fig3, ax3 = plt.subplots()
    show(masked, transform=transform, cmap='Blues', ax=ax3)
    exc.plot(ax=ax3, edgecolor='k', color='None')
    cutout.grid.to_crs(excluder.crs).plot(edgecolor='grey', color='None', ax=ax3, ls=':')
    ax3.set_title(f'Eligible area (blue) {eligible_share * 100:2.2f}%')
    fig3_path = os.path.join(image_dir, 'fig3.png')
    fig3.savefig(fig3_path)
    fig_list.append(fig3_path)
    print(f"Saved {fig3_path}")

    A = cutout.availabilitymatrix(shapes, excluder)
    A.name = "Capacity Factor"

    fig4, ax4 = plt.subplots()
    A.plot(cmap='Blues')
    shapes.plot(ax=ax4, edgecolor='k', color='None')
    cutout.grid.plot(ax=ax4, color='None', edgecolor='grey', ls=':')
    ax4.set_title(country)
    fig4_path = os.path.join(image_dir, 'fig4.png')
    fig4.savefig(fig4_path)
    fig_list.append(fig4_path)
    print(f"Saved {fig4_path}")

    cap_per_sqkm = capacity
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm
    cutout.prepare()
    wind = cutout.wind(matrix=capacity_matrix, turbine=atlite.windturbines.NREL_ReferenceTurbine_5MW_offshore, index=shapes.index)

    fig5, ax5 = plt.subplots(figsize=(14, 7))
    wind.to_pandas().div(1e3).plot(ylabel='Onshore Wind [GW]',fontsize=14, ax=ax5, color='blue')
    ax5.set_ylabel('Offshore-ac Wind [GW]', fontsize=14)
    fig5_path = os.path.join(image_dir, 'fig5.png')
    fig5.savefig(fig5_path)
    fig_list.append(fig5_path)
    print(f"Saved {fig5_path}")

    # Generate HTML file
    with open(output_html, 'w') as f:
        f.write('<html><head><title>Figures</title></head><body>\n')
        for fig_path, title in zip(fig_list, titles):
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                f.write(f'<h2>Figure:    {title}</h2>\n')
                f.write(f'<img src="data:image/png;base64,{encoded_string}" alt="{os.path.basename(fig_path)}" style="width:60%; height:auto;"><br>\n')
            else:
                print(f"Image {fig_path} not found.")
        f.write('</body></html>')
    print(f"HTML file saved at {output_html}")
    
    
def generate_max_offshore_dc_potentials(country, output_html):
    plt.rcParams['figure.figsize'] = [7, 7]
    
    with open("../config/config.yaml") as file:
     config = yaml.safe_load(file)
     
    corine_grid_codes = config['renewable']['offwind-dc']['corine']
    min_shore_distance = config['renewable']['offwind-dc']['min_shore_distance']
    ship_threshold = config['renewable']['offwind-dc']['ship_threshold']
    capacity = config['renewable']['offwind-dc']['capacity_per_sqkm']
    shipping_threshold = (
        ship_threshold * 8760 * 6
    ) 
    func = functools.partial(np.less, shipping_threshold)
    max_depth = config['renewable']['offwind-dc']['max_depth']
    func = functools.partial(np.greater, -max_depth)
    excluder_resolution = config['renewable']['offwind-dc']['excluder_resolution']
    
    country_shape = "../resources/ncdr/country_shapes.geojson"
    eez_path = "../data/bundle/eez/World_EEZ_v8_2014.shp"
    eez = gpd.read_file(eez_path)
    # separator = ", "
    # country= separator.join(country)
    # Filter for specified countries
    shapes = eez[eez['Country'] == country]
    shapes.plot()
    
    # Define bounds and create cutout
    bounds = shapes.unary_union.buffer(0.5).bounds
    cutout = atlite.Cutout(country, module='era5', bounds=bounds, time=slice('2013-01-01', '2013-12-31'))

    # Directory to save images
    image_dir = 'max_vre_potential_charts/images_offshore_dc'
    os.makedirs(image_dir, exist_ok=True)

    # Create figures and save them as images
    fig_list = []
    titles = [
        "Country Shape",
        "Eligible Area (purple) - Offshore-dc",
        "Eligible Area with Grid cells",
        "Capacity Factor for eligible area",
        "Maximum Offshore-dc Wind Potential"
    ]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    shapes.plot(ax=ax1,color='purple')
    cutout.grid.plot(ax=ax1, edgecolor='grey', color='None')
    fig1_path = os.path.join(image_dir, 'fig1.png')
    fig1.savefig(fig1_path)
    fig_list.append(fig1_path)
    print(f"Saved {fig1_path}")

    CORINE = '../data/bundle/corine/g250_clc06_V18_5.tif'
    ship_density = '../resources/ncdr/shipdensity_raster.tif'
    gebco = '../data/bundle/GEBCO_2014_2D.nc'
    excluder = ExclusionContainer(crs=3035, res = excluder_resolution)
    excluder.add_raster(CORINE, codes=corine_grid_codes,crs=3035, invert=True)
    excluder.add_raster(ship_density, codes=func, crs=4326, allow_no_overlap=True)
    excluder.add_raster(gebco, codes=func, crs=4326, nodata=-1000)
    excluder.add_geometry(country_shape, buffer=min_shore_distance)
    exc = shapes.geometry.to_crs(excluder.crs)
    masked, transform = shape_availability(exc, excluder)
    eligible_share = masked.sum() * excluder.res**2 / exc.geometry.item().area

    fig2, ax2 = plt.subplots()
    show(masked, transform=transform, cmap='Purples', ax=ax2)
    exc.plot(ax=ax2, edgecolor='k', color='None')
    ax2.set_title(f'Eligible area (purple) {eligible_share * 100:2.2f}%')
    fig2_path = os.path.join(image_dir, 'fig2.png')
    fig2.savefig(fig2_path)
    fig_list.append(fig2_path)
    print(f"Saved {fig2_path}")

    fig3, ax3 = plt.subplots()
    show(masked, transform=transform, cmap='Purples', ax=ax3)
    exc.plot(ax=ax3, edgecolor='k', color='None')
    cutout.grid.to_crs(excluder.crs).plot(edgecolor='grey', color='None', ax=ax3, ls=':')
    ax3.set_title(f'Eligible area (purple) {eligible_share * 100:2.2f}%')
    fig3_path = os.path.join(image_dir, 'fig3.png')
    fig3.savefig(fig3_path)
    fig_list.append(fig3_path)
    print(f"Saved {fig3_path}")

    A = cutout.availabilitymatrix(shapes, excluder)
    A.name = "Capacity Factor"

    fig4, ax4 = plt.subplots()
    A.plot(cmap='Purples')
    shapes.plot(ax=ax4, edgecolor='k', color='None')
    cutout.grid.plot(ax=ax4, color='None', edgecolor='grey', ls=':')
    ax4.set_title(country)
    fig4_path = os.path.join(image_dir, 'fig4.png')
    fig4.savefig(fig4_path)
    fig_list.append(fig4_path)
    print(f"Saved {fig4_path}")

    cap_per_sqkm = capacity
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm
    cutout.prepare()
    wind = cutout.wind(matrix=capacity_matrix, turbine=atlite.windturbines.NREL_ReferenceTurbine_5MW_offshore, index=shapes.index)

    fig5, ax5 = plt.subplots(figsize=(14, 7))
    wind.to_pandas().div(1e3).plot(ylabel='Onshore Wind [GW]',fontsize=14, ax=ax5, color='purple')
    ax5.set_ylabel('Offshore-dc Wind [GW]', fontsize=14)
    fig5_path = os.path.join(image_dir, 'fig5.png')
    fig5.savefig(fig5_path)
    fig_list.append(fig5_path)
    print(f"Saved {fig5_path}")

    # Generate HTML file
    with open(output_html, 'w') as f:
        f.write('<html><head><title>Figures</title></head><body>\n')
        for fig_path, title in zip(fig_list, titles):
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                f.write(f'<h2>Figure:    {title}</h2>\n')
                f.write(f'<img src="data:image/png;base64,{encoded_string}" alt="{os.path.basename(fig_path)}" style="width:60%; height:auto;"><br>\n')
            else:
                print(f"Image {fig_path} not found.")
        f.write('</body></html>')
    print(f"HTML file saved at {output_html}")
    
    
def generate_max_solar_potentials(country, output_html):
    plt.rcParams['figure.figsize'] = [7, 7]
    
    with open("../config/config.yaml") as file:
     config = yaml.safe_load(file)
     
    corine_grid_codes = config['renewable']['solar']['corine']
    capacity = config['renewable']['solar']['capacity_per_sqkm']
    # Load world shapes
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Filter for specified countries
    shapes = world[world.name.isin(countries)].set_index('name')
    shapes.plot()
    
    # Define bounds and create cutout
    bounds = shapes.unary_union.buffer(0.5).bounds
    cutout = atlite.Cutout(country, module='era5', bounds=bounds, time=slice('2013-01-01', '2013-12-31'))

    # Directory to save images
    image_dir = 'max_vre_potential_charts/images_solar'
    os.makedirs(image_dir, exist_ok=True)

    # Create figures and save them as images
    fig_list = []
    titles = [
        "Country Shape",
        "Eligible Area (orange) - Solar",
        "Eligible Area with Grid cells",
        "Capacity Factor for eligible area",
        "Maximum solar Potential",
        "Maximum Solar-Rooftop Potential"
    ]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    shapes.plot(ax=ax1,color='orange')
    cutout.grid.plot(ax=ax1, edgecolor='grey', color='None')
    fig1_path = os.path.join(image_dir, 'fig1.png')
    fig1.savefig(fig1_path)
    fig_list.append(fig1_path)
    print(f"Saved {fig1_path}")

    CORINE = '../data/bundle/corine/g250_clc06_V18_5.tif'
    excluder = ExclusionContainer()
    excluder.add_raster(CORINE, codes=corine_grid_codes,crs=3035, invert=True)
    exc = shapes.loc[[country]].geometry.to_crs(excluder.crs)
    masked, transform = shape_availability(exc, excluder)
    eligible_share = masked.sum() * excluder.res**2 / exc.geometry.item().area

    fig2, ax2 = plt.subplots()
    show(masked, transform=transform, cmap='Oranges', ax=ax2)
    exc.plot(ax=ax2, edgecolor='k', color='None')
    ax2.set_title(f'Eligible area (orange) {eligible_share * 100:2.2f}%')
    fig2_path = os.path.join(image_dir, 'fig2.png')
    fig2.savefig(fig2_path)
    fig_list.append(fig2_path)
    print(f"Saved {fig2_path}")

    fig3, ax3 = plt.subplots()
    show(masked, transform=transform, cmap='Oranges', ax=ax3)
    exc.plot(ax=ax3, edgecolor='k', color='None')
    cutout.grid.to_crs(excluder.crs).plot(edgecolor='grey', color='None', ax=ax3, ls=':')
    ax3.set_title(f'Eligible area (orange) {eligible_share * 100:2.2f}%')
    fig3_path = os.path.join(image_dir, 'fig3.png')
    fig3.savefig(fig3_path)
    fig_list.append(fig3_path)
    print(f"Saved {fig3_path}")

    A = cutout.availabilitymatrix(shapes, excluder)
    A.name = "Capacity Factor"

    fig4, ax4 = plt.subplots()
    A.sel(name=country).plot(cmap='Oranges')
    shapes.loc[[country]].plot(ax=ax4, edgecolor='k', color='None')
    cutout.grid.plot(ax=ax4, color='None', edgecolor='grey', ls=':')
    fig4_path = os.path.join(image_dir, 'fig4.png')
    fig4.savefig(fig4_path)
    fig_list.append(fig4_path)
    print(f"Saved {fig4_path}")

    cap_per_sqkm = capacity
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm
    cutout.prepare()
    pv = cutout.pv(matrix=capacity_matrix, panel=atlite.solarpanels.CSi,orientation='latitude_optimal', index=shapes.index)

    fig5, ax5 = plt.subplots(figsize=(14, 7))
    pv.to_pandas().div(1e3).plot(ylabel='Solar [GW]',fontsize=14, ax=ax5, color='orange')
    ax5.set_ylabel('Solar [GW]', fontsize=14)
    fig5_path = os.path.join(image_dir, 'fig5.png')
    fig5.savefig(fig5_path)
    fig_list.append(fig5_path)
    print(f"Saved {fig5_path}")
    
    pop_layout = pd.read_csv(
        "../resources/ncdr/pop_layout_elec_s_6.csv", index_col=0
    )
    pop_solar = pop_layout.total
    potential = 0.1 * 10 * pop_solar
    country_code_map = {
        'Belgium': 'BE',
        'France': 'FR',
        'Germany': 'DE',
        'Netherlands': 'NL',
        'Great Britain': 'GB'}

    # Create a list of codes from the country names
    country_codes = [country_code_map.get(country, 'Unknown')]
    potential=potential[potential.index.str[:2].isin(country_codes)].sum()/1e3
    fig6, ax6 = plt.subplots(figsize=(10, 10))
    ax6.bar(0, potential, color='orange', width=0.3)  # Plotting the float value at position 0
    ax6.set_ylabel('Solar Rooftop Potential [GW]', fontsize=12)  # Labeling the y-axis
    ax6.set_xticks([])  # Removing x-axis ticks as there's only one bar
    ax6.set_xlim(-0.5, 0.5)  # Setting x-axis limits
    ax6.set_title('Solar Rooftop Potential')  # Adding a title
    ax6.tick_params(axis='y', labelsize=12)
    fig6_path = os.path.join(image_dir, 'fig6.png')
    fig6.savefig(fig6_path)
    fig_list.append(fig6_path)
    print(f"Saved {fig6_path}")
    # Generate HTML file
    with open(output_html, 'w') as f:
        f.write('<html><head><title>Figures</title></head><body>\n')
        for fig_path, title in zip(fig_list, titles):
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                f.write(f'<h2>Figure:    {title}</h2>\n')
                f.write(f'<img src="data:image/png;base64,{encoded_string}" alt="{os.path.basename(fig_path)}" style="width:60%; height:auto;"><br>\n')
            else:
                print(f"Image {fig_path} not found.")
        f.write('</body></html>')
    print(f"HTML file saved at {output_html}")
# Usage example
countries = ['Belgium']
output_dir = "max_vre_potential_charts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for country in countries:
 output_html = os.path.join(output_dir, f"{country}_onshore_potentials.html")
 generate_max_onshore_potentials(country, output_html)
 output_html = os.path.join(output_dir, f"{country}_offshore_ac_potentials.html")
 generate_max_offshore_ac_potentials(country, output_html)
 output_html = os.path.join(output_dir, f"{country}_offshore_dc_potentials.html")
 generate_max_offshore_dc_potentials(country, output_html)
 output_html = os.path.join(output_dir, f"{country}_solar_potentials.html")
 generate_max_solar_potentials(country, output_html)

