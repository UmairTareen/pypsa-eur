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

#import geopandas as gpd
#import atlite
import xarray as xr
import plotly.graph_objects as go
import plotly.offline as pyo
#from atlite.gis import shape_availability, ExclusionContainer
from plotly.subplots import make_subplots

offwind_ac = xr.open_dataset('../resources/bau/profile_offwind-ac.nc')

data = xr.open_dataset('../results/bau/prenetworks/elec_s_6_lvopt__EQ0.7c-12H-T-H-B-I-A-dist1_2050.nc')
pmax = data.generators_p_nom_max
pnow = data.generators_p_nom

bus = 'BE1 0'

wind_generators = [gen for gen in pmax.generators_i.values if 'wind' in gen and bus in gen]
solar_generators = [gen for gen in pmax.generators_i.values if bus + ' solar' in gen]

for gen in wind_generators+solar_generators:
    max_power = pmax.sel(generators_i=gen).max().item()
    print(f"Maximum power for {gen}: {max_power}")

for gen in  wind_generators+solar_generators:
    power = pnow.sel(generators_i=gen).max().item()
    print(f"Installed power for {gen}: {power}")





'''
def plot_onshore_wind_potential(shapes, cutout, excluder):
    cap_per_sqkm = 3
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    A = cutout.availabilitymatrix(shapes, excluder)
    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm

    cutout.prepare()
    wind_on = cutout.wind(matrix=capacity_matrix, turbine=atlite.windturbines.Vestas_V112_3MW,
                          index=shapes.index)

    return wind_on

def plot_solar_potential(shapes, cutout, excluder):
    cap_per_sqkm = 2
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))
    A = cutout.availabilitymatrix(shapes, excluder)
    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm

    cutout.prepare()
    pv = cutout.pv(matrix=capacity_matrix, panel=atlite.solarpanels.CSi,
                  orientation='latitude_optimal', index=shapes.index)

    return pv

def plot_offshore_wind_potential(offshore_shapes, cutout, excluder):
    cap_per_sqkm = 3
    area = cutout.grid.set_index(['y', 'x']).to_crs(3035).area / 1e6
    area = xr.DataArray(area, dims=('spatial'))

    A = cutout.availabilitymatrix(offshore_shapes, excluder)
    capacity_matrix = A.stack(spatial=['y', 'x']) * area * cap_per_sqkm

    cutout.prepare()
    wind = cutout.wind(matrix=capacity_matrix, turbine=atlite.windturbines.NREL_ReferenceTurbine_5MW_offshore,
                      index=offshore_shapes.index)

    return wind


def save_combined_plot_to_html(onshore_wind, offshore_wind, solar_pv, country_name, index_column):
    df_onshore = onshore_wind.to_pandas().div(1e3).reset_index()
    df_offshore = offshore_wind.to_pandas().div(1e3).reset_index()
    df_solar = solar_pv.to_pandas().div(1e3).reset_index()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=['Onshore Wind Potential [GW]', 'Offshore Wind Potential [GW]', 'Solar Potential [GW]'])

    for col in df_onshore.columns[1:]:
        fig.add_scatter(x=df_onshore['time'], y=df_onshore[col], mode='lines', name='Onshore Wind', line_color='green', row=1, col=1)

    for col in df_offshore.columns[1:]:
        fig.add_scatter(x=df_offshore['time'], y=df_offshore[col], mode='lines', name='Offshore Wind', line_color='blue', row=2, col=1)

    for col in df_solar.columns[1:]:
        fig.add_scatter(x=df_solar['time'], y=df_solar[col], mode='lines', name='Solar PV', line_color='orange', row=3, col=1)

    fig.update_layout(xaxis_title='Time')
    fig.update_xaxes(tickformat="%b")

    fig.write_html(f"output_charts/{country_name}_combined_potentials.html")

def max_combined_capacities():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    eez_path = "../data/bundle/eez/World_EEZ_v8_2014.shp"
    eez = gpd.read_file(eez_path)
    Countries = ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom']

    for country_name in Countries:
        shapes = world[world.name == country_name].set_index('name')
        offshore_shapes = eez[eez['Country'] == country_name]


        # Load CORINE raster file for onshore wind
        CORINE = '../resources/natura.tiff'
        excluder_onshore = atlite.ExclusionContainer()
        excluder_onshore.add_raster(CORINE, codes=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32], invert=True)

        # Load CORINE raster file for offshore wind
        excluder_offshore = atlite.ExclusionContainer()
        excluder_offshore.add_raster(CORINE, codes=[42, 43, 44, 255], invert=True)

        # Load CORINE raster file for solar PV
        excluder_pv = atlite.ExclusionContainer()
        excluder_pv.add_raster(CORINE, codes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 31, 32], invert=True)

        # Create a cutout for onshore wind, offshore wind, and solar PV
        cutout = atlite.Cutout('../cutouts/europe-2013-era5.nc')

        shapes_crs_onshore = shapes.geometry.to_crs(excluder_onshore.crs)
        shapes_crs_offshore = offshore_shapes.geometry.to_crs(excluder_offshore.crs)
        shapes_crs_pv = shapes.geometry.to_crs(excluder_pv.crs)

        masked_onshore, transform_onshore = shape_availability(shapes_crs_onshore, excluder_onshore)
        masked_offshore, transform_offshore = shape_availability(shapes_crs_offshore, excluder_offshore)
        masked_pv, transform_pv = shape_availability(shapes_crs_pv, excluder_pv)

        onshore_wind = plot_onshore_wind_potential(shapes, cutout, excluder_onshore)
        offshore_wind = plot_offshore_wind_potential(offshore_shapes, cutout, excluder_offshore)
        solar_pv = plot_solar_potential(shapes, cutout, excluder_pv)

        save_combined_plot_to_html(onshore_wind, offshore_wind, solar_pv, country_name, shapes.index[0])

max_combined_capacities()

'''