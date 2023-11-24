#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:49:39 2023

@author: umair
"""

import geopandas as gpd
import atlite
import xarray as xr
import plotly.graph_objects as go
import plotly.offline as pyo
from atlite.gis import shape_availability, ExclusionContainer
from plotly.subplots import make_subplots

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
        excluder_onshore.add_raster(CORINE, codes=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32], invert=False)

        # Load CORINE raster file for offshore wind
        excluder_offshore = atlite.ExclusionContainer()
        excluder_offshore.add_raster(CORINE, codes=[42, 43, 44, 255], invert=False)

        # Load CORINE raster file for solar PV
        excluder_pv = atlite.ExclusionContainer()
        excluder_pv.add_raster(CORINE, codes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 31, 32], invert=False)

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
