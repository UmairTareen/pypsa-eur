#!/usr/bin/env python3
# -*- coding: ucf-8 -*-
"""
Created on Wed Apr 19 08:57:23 2023

@author: umair
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import os
from _helpers import mute_print
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging
import re
from pathlib import Path

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
path = os.path.join(os.path.dirname(__file__), '../data/')
PATH = os.path.join(os.path.dirname(__file__), '../')
skip_cols = [0,1, 2, 3]
keep_cols = [i for i in range(34) if i not in skip_cols]
year= 2050

countriess = [
    "FR",
    "DE",
    "GB",
    "IT",
    "ES",
    "PL",
    "SE",
    "NL",
    "BE",
    "FI",
    "CZ",
    "DK",
    "PT",
    "RO",
    "AT",
    "BG",
    "EE",
    "GR",
    "LV",
    "HU",
    "IE",
    "SK",
    "LT",
    "HR",
    "LU",
    "SI",
    "CH",
    "NO",
]

def build_CLEVER_Residential_data_per_country(countriess):
    
    '''
    This function computes CLEVER data of residential sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with residential input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    #cn_countries = config["atlite"]["cn_countries"]
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"


    cf = pd.read_excel(cn_countriess, "Residential", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert 'Total final energy consumption in the residential sector' in cf
    assert 'Final electricity consumption in the residential sector' in cf
    assert 'Total final energy consumption for space heating in the residential sector' in cf
    assert 'Final electricity consumption for space heating in the residential sector' in cf
    assert 'Total final energy consumption for domestic hot water' in cf
    assert 'Final electricity consumption for domestic hot water' in cf
    assert 'Total final energy consumption for domestic cooking' in cf
    assert 'Final electricity consumption for domestic cooking' in cf
    assert 'Final energy consumption from heating networks in the residential sector' in cf
    
    cf["Thermal_uses_residential"] = cf["Total final energy consumption for space heating in the residential sector"] + cf["Total final energy consumption for domestic hot water"] + cf["Total final energy consumption for domestic cooking"] + cf["Final energy consumption from heating networks in the residential sector"].sum()
    
    assert "Thermal_uses_residential" in cf
    
    return cf
    
    
def build_Clever_Residential(countriess, year):
    '''
    This function converts CLEVER residential sector data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input sufficincy residential data
    
    Return multi-index for all countries residential sufficiency data in TWh/a.

    '''

    func = partial(build_CLEVER_Residential_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Residential from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)
    

    return totals.T
clever_residential = build_Clever_Residential(countriess, year) 
clever_residential = clever_residential.loc[:, clever_residential.columns.notna()]
clever_residential.to_csv("/home/umair/pypsa-eur/data/clever_residential_2030.csv")

def build_CLEVER_tertiary_data_per_country(countriess):
    
    '''
    This function computes CLEVER data of tertiary sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with tertiary input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "Tertiary", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert "Total final energy consumption in the tertiary sector" in cf
    assert "Final electricity consumption in the tertiary sector" in cf
    assert "Total final energy consumption for space heating in the tertiary sector (with climatic corrections) " in cf
    assert "Final electricity consumption for space heating in the tertiary sectorr" in cf
    assert "Final electricity consumption for cooling in the tertiary sector" in cf
    assert "Total final energy consumption for hot water in the tertiary sector" in cf
    assert "Final electricity consumption for hot water in the tertiary sector" in cf
    assert "Total Final energy consumption for cooking in the tertiary sector" in cf
    assert "Final electricity consumption for cooking in the tertiary sector" in cf
    assert "Final energy consumption from heating networks in the tertiary sector" in cf
    
    cf["Thermal_uses_tertiary"] = cf["Total final energy consumption for space heating in the tertiary sector (with climatic corrections) "] + cf["Total final energy consumption for hot water in the tertiary sector"] + cf["Total Final energy consumption for cooking in the tertiary sector"] + cf["Final energy consumption from heating networks in the tertiary sector"].sum()
    
    assert "Thermal_uses_tertiary" in cf
    
    
    return cf
def build_Clever_Tertiary(countriess, year):
    ''''
    This function converts CLEVER tertary sector data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input sufficincy tertiary data
    
    Return multi-index for all countries tertiary sufficiency data in TWh/a.

    '''

    func = partial(build_CLEVER_tertiary_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Tertiary from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)
    

    return totals.T
clever_Tertairy = build_Clever_Tertiary(countriess, year) 

'''
Missing data for space heating, hot water and cooking in the tertiary sector.
assuming the percentage of residential sector to fill out the data.
'''
for country_code in ['BE', 'DE', 'FR', 'GB', 'NL']:
    # Perform the calculations for each country
    perc_total_space = (clever_residential.loc[country_code, 'Final electricity consumption for space heating in the residential sector'] / clever_residential.loc[country_code, 'Total final energy consumption for space heating in the residential sector'])
    perc_total_water = (clever_residential.loc[country_code, 'Final electricity consumption for domestic hot water'] / clever_residential.loc[country_code, 'Total final energy consumption for domestic hot water'])
    perc_total_cooking = (clever_residential.loc[country_code, 'Final electricity consumption for domestic cooking'] / clever_residential.loc[country_code, 'Total final energy consumption for domestic cooking'])
    perc_total_cooking_total = (clever_residential.loc[country_code, 'Total final energy consumption for domestic cooking'] / clever_residential.loc[country_code, 'Total final energy consumption in the residential sector'])

    clever_Tertairy.loc[country_code, 'Final electricity consumption for space heating in the tertiary sectorr'] = clever_Tertairy.loc[country_code, 'Total final energy consumption for space heating in the tertiary sector (with climatic corrections) '] * perc_total_space
    clever_Tertairy.loc[country_code, 'Final electricity consumption for hot water in the tertiary sector'] = clever_Tertairy.loc[country_code, 'Total final energy consumption for hot water in the tertiary sector'] * perc_total_water
    clever_Tertairy.loc[country_code, 'Total Final energy consumption for cooking in the tertiary sector'] = clever_Tertairy.loc[country_code, 'Total final energy consumption in the tertiary sector'] * perc_total_cooking_total
    clever_Tertairy.loc[country_code, 'Final electricity consumption for cooking in the tertiary sector'] = clever_Tertairy.loc[country_code, 'Total Final energy consumption for cooking in the tertiary sector'] * perc_total_cooking
    
    clever_Tertairy.loc[country_code, "Thermal_uses_tertiary"] = clever_Tertairy.loc[country_code, "Total final energy consumption for space heating in the tertiary sector (with climatic corrections) "] + clever_Tertairy.loc[country_code, "Final electricity consumption for cooling in the tertiary sector"] + clever_Tertairy.loc[country_code, "Total final energy consumption for hot water in the tertiary sector"] + clever_Tertairy.loc[country_code, "Total Final energy consumption for cooking in the tertiary sector"] + clever_Tertairy.loc[country_code, "Final energy consumption from heating networks in the tertiary sector"].sum()
    

clever_Tertairy = clever_Tertairy.loc[:, clever_Tertairy.columns.notna()]
clever_Tertairy.to_csv("/home/umair/pypsa-eur/data/clever_Tertairy_2050.csv")

def build_CLEVER_transport_data_per_country(countriess):
    
    '''
    This function computes CLEVER data of transport sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with transport input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    #cn_countries = config["atlite"]["cn_countries"]
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "Transport", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert "Total final energy consumption in passenger road mobility" in cf
    assert "Total final energy consumption in road freight transport" in cf
    assert "Final electricity consumption for passenger road mobility" in cf
    assert "Final electricity consumption for road freight transport" in cf
    assert "Total final energy consumption in rail passenger transport" in cf
    assert "Final electricity consumption in rail passenger transport" in cf
    assert "Total final energy consumption in rail freight transport" in cf
    assert "Final electricity consumption in rail freight transport" in cf
    assert "Total final energy consumption for air travel" in cf
    assert "Final liquid fuels consumption on international flights" in cf
    assert "Final liquid fuels consumption on domestic flights" in cf
    assert "Final energy consumption from liquid fuels in national water freight transport" in cf
    assert "Final energy consumption from liquid fuels in international water freight transport" in cf
    
    cf["Total_Road"] = cf["Total final energy consumption in passenger road mobility"] + cf["Total final energy consumption in road freight transport"].sum()
    cf["Electricity_Road"] = cf["Final electricity consumption for passenger road mobility"] + cf["Final electricity consumption for road freight transport"].sum()
    cf["Total_rail"] = cf["Total final energy consumption in rail passenger transport"] + cf["Total final energy consumption in rail freight transport"].sum()
    cf["Electricity_rail"] = cf["Final electricity consumption in rail passenger transport"] + cf["Final electricity consumption in rail freight transport"].sum()
    
    assert "Total_Road" in cf
    assert "Electricity_Road" in cf
    assert "Total_rail" in cf
    assert "Electricity_rail" in cf
    
    return cf
def build_Clever_Transport(countriess, year):
    ''''
    This function converts CLEVER transport sector data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input sufficincy transport data
    
    Return multi-index for all countries transport sufficiency data in TWh/a.

    '''

    func = partial(build_CLEVER_transport_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Transport from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)
    

    return totals.T  
clever_Transport = build_Clever_Transport(countriess, year)
clever_Transport = clever_Transport.loc[:, clever_Transport.columns.notna()]
clever_Transport.to_csv("/home/umair/pypsa-eur/data/clever_Transport_2050.csv") 

def build_CLEVER_agriculture_data_per_country(countriess):
    '''
    This function computes CLEVER data of agriculture sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with agriculture input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "Agriculture", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert "Total Final energy consumption in agriculture" in cf
    assert "Final electricity consumption in agriculture" in cf
    assert "Final oil consumption in agriculture" in cf
    assert "Final energy consumption from solid fossil fuels (coal ...) in agriculture" in cf
    assert "Final energy consumption from solid biomass in agriculture" in cf
    assert "Final energy consumption from gas grid / gas consumed locally in agriculture" in cf
    
    cf["Total_agriculture_heat"] = cf["Final energy consumption from solid fossil fuels (coal ...) in agriculture"] + cf["Final energy consumption from solid biomass in agriculture"] + cf["Final energy consumption from gas grid / gas consumed locally in agriculture"].sum()
    
    assert "Total_agriculture_heat" in cf
    
    return cf
def build_Clever_Agriculture(countriess, year):
    ''''
    This function converts CLEVER agriculture sector data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input sufficincy agriculture data
    
    Return multi-index for all countries agriculture sufficiency data in TWh/a.

    '''

    func = partial(build_CLEVER_agriculture_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Agriculture from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)

    return totals.T 
clever_Agriculture = build_Clever_Agriculture(countriess, year) 
clever_Agriculture = clever_Agriculture.loc[:, clever_Agriculture.columns.notna()]
clever_Agriculture.to_csv("/home/umair/pypsa-eur/data/clever_Agriculture_2050.csv") 

def build_CLEVER_AFOLUB_data_per_country(countriess):
    
    '''
    This function computes CLEVER data of carbon emissions from agriculture sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with emissions input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "AFOLUB (Solagro figures)", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert "Total CO2 emissions from agriculture" in cf
    assert "Total CO2 emissions from the LULUCF sector" in cf
    
    return cf
def build_Clever_AFOLUB(countriess, year):
    ''''
    This function converts CLEVER emissions data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input emissions data
    
    Return multi-index for all countries agriculture emissions data in MtCO2/a.

    '''

    func = partial(build_CLEVER_AFOLUB_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build AFOLUB from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)

    return totals.T
clever_AFOLUB = build_Clever_AFOLUB(countriess, year) 
clever_AFOLUB = clever_AFOLUB.loc[:, clever_AFOLUB.columns.notna()]
clever_AFOLUB.to_csv("/home/umair/pypsa-eur/data/clever_AFOLUB_2050.csv") 

def build_CLEVER_macro_data_per_country(countriess):
    
    ''''
    This function computes CLEVER data of carbon emissions from multiple sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with emissions input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "Macro", index_col=0, skiprows=[1,2], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf= cf.astype(float)
    
    assert "Total net GHG emissions from non-energy sources in agriculture" in cf
    assert "GHG emissions from non-energy sources in industry (process emissions)" in cf
    assert "GHG emissions from non-energy sources in waste management" in cf
    assert "GHG emissions from non-energy sources in other sectors (industrial product use)" in cf
    assert "Total population" in cf
    
    return cf
def build_Clever_Macro(countriess, year):
    ''''
    This function converts CLEVER emissions data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input emissions data
    
    Return multi-index for all countries agriculture emissions data in MtCO2/a.

    '''

    func = partial(build_CLEVER_macro_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Macro from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)

    return totals.T 
clever_Macro = build_Clever_Macro(countriess, year)
clever_Macro = clever_Macro.loc[:, clever_Macro.columns.notna()]
clever_Macro.to_csv("/home/umair/pypsa-eur/data/clever_Macro_2050.csv")  


def build_CLEVER_industry_data_per_country(countriess):
    
    ''''
    This function computes CLEVER data of industrial sector for all eu28 countries

    Parameters
    ----------
    countries : list (country names)
    year : int
    cf : variables with industrial input data read from CLEVER Scenario
    assert: valdation of dataframe
    
    Returns raw input data for individual countries for further processing 
    
    '''
    cn_countriess = f"{path}/Dashboard_{countriess}_v5.38_PNC_Tra_Int.xlsx"
    
    cf = pd.read_excel(cn_countriess, "Industry", index_col=0, skiprows=[1,2,3], usecols=keep_cols)[year]
    cf = cf.fillna(0) 
    cf = cf.apply (pd.to_numeric, errors='coerce')
    cf = cf.drop(cf.index[108:110])
    cf= cf.astype(float)
    
    assert "Total Final Energy Consumption of the ammonia industry" in cf
    assert "Total Final Energy Consumption of the steel industry" in cf
    assert "Total Final Energy Consumption of the cement industry" in cf
    assert "Total Final Energy Consumptionof the glass industry" in cf
    assert "Total Final energy consumption of the non metallic minerals industry" in cf
    assert "Total Final Energy Consumption of the chemical industry" in cf
    assert "Total Final Energy Consumption of the Food, beverage and tobacco industry" in cf
    
    assert "Total Final Energy Consumption of the other chemical industries" in cf
    assert "Total Final energy consumption of the non-ferrous metals industry" in cf
    assert "Total Final Energy Consumption of other industries (miscellaneous industries)" in cf
    assert "Total Final Energy Consumption of the paper, pulp and printing industry" in cf
    assert "Total Final Energy Consumption of the high value chemicals industry" in cf
    
    cf["other_chemicals_total"] = cf["Total Final Energy Consumption of the other chemical industries"] + cf["Total Final Energy Consumption of the high value chemicals industry"].sum()
    return cf
    
    
def build_Clever_Industry(countriess, year):
    ''''
    This function converts CLEVER industrial data into dataframe for all EU-28 countries

    Parameters
    ----------
    countries : dict (dictionary of country names)
    year : int
    totals: variables with input industrial data
    
    Return multi-index for all countries industrial data in TWh/a.

    '''

    func = partial(build_CLEVER_industry_data_per_country)
    tqdm_kwargs = dict(
        ascii=False,
        unit=" country",
        total=len(countriess),
        desc="Build Industry from CLEVER database",
    )
    with mute_print():
        with mp.Pool() as pool:
            totals_list = list(tqdm(pool.imap(func, countriess), **tqdm_kwargs))

    totals = pd.concat(totals_list, axis=1, keys=countriess)

    return totals.T  
clever_Industry = build_Clever_Industry(countriess, year) 
clever_Industry = clever_Industry.loc[:, clever_Industry.columns.notna()]
clever_Industry.to_csv("/home/umair/pypsa-eur/data/clever_Industry_2050.csv")  
    

