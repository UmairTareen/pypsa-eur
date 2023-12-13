#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:44:19 2023

@author: umair
"""

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd

PATH = "/home/umair/pypsa-eur_repository/"
file_path_ac = PATH + f"resources/profile_offwind-ac.nc"
file_dc = PATH + f"resources/profile_offwind-dc.nc"

regions = gpd.read_file(PATH + "resources/regions_offshore.geojson")

# Filter the regions GeoDataFrame to include only Belgium (BE)
belgium_regions = regions[(regions['country'] == 'BE')]

belgium_regions["Area"] = belgium_regions.to_crs(epsg=3035).area.div(1e6)
# Create an empty DataFrame to store the hourly wind power generation
hourly_wind_power_ac = pd.DataFrame()
hourly_wind_power_dc = pd.DataFrame()

# Process file_path_ac
for region in belgium_regions.iterrows():
    region_name = region[1]['name']
    
    ds = xr.open_dataset(file_path_ac)
    
    # Extracting time and profile data
    time_values = ds.time.values
    profile_data = ds.profile.values
    
    # Creating a DataFrame with time as index and profile data as columns
    df = pd.DataFrame(profile_data, index=time_values, columns=[f"{region_name}_{i}" for i in range(profile_data.shape[1])])
    
    # Concatenate the DataFrame to the existing hourly_wind_power_ac DataFrame
    hourly_wind_power_ac = pd.concat([hourly_wind_power_ac, df], axis=1)

# Process file_path_dc
for region in belgium_regions.iterrows():
    region_name = region[1]['name']
    
    ds = xr.open_dataset(file_dc)
    
    # Extracting time and profile data
    time_values = ds.time.values
    profile_data = ds.profile.values
    
    # Creating a DataFrame with time as index and profile data as columns
    df = pd.DataFrame(profile_data, index=time_values, columns=[f"{region_name}_{i}" for i in range(profile_data.shape[1])])
    
    # Concatenate the DataFrame to the existing hourly_wind_power_dc DataFrame
    hourly_wind_power_dc = pd.concat([hourly_wind_power_dc, df], axis=1)

# Summing the values for each hour to get the total wind power generation per hour
hourly_wind_power_sum_ac = hourly_wind_power_ac.sum(axis=1)
hourly_wind_power_sum_dc = hourly_wind_power_dc.sum(axis=1)
hours_in_dataset = len(time_values)
# Combine the potentials from both files
hourly_wind_power_sum_combined = hourly_wind_power_sum_ac + hourly_wind_power_sum_dc

# Convert to GW
hourly_wind_power_sum_gw_combined = hourly_wind_power_sum_combined/1e3 # Convert to GW


# Calculate the total potential using the given turbine capacity
total_potential_combined = hourly_wind_power_sum_gw_combined.sum()

print(f'Total Wind Power Potential for Belgium (Combined): {total_potential_combined:.2f} GW')

# Plotting the time series
plt.figure(figsize=(10, 6))
hourly_wind_power_sum_gw_combined.plot()
plt.title(f'Hourly Wind Power Potential (Belgium - Combined)')
plt.xlabel('Time')
plt.ylabel('Wind Power Potential (GW)')
plt.grid(True)
plt.show()