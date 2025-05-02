#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:35:29 2025
"""

import pandas as pd
import logging

def Construct_2020_from_JRC_IDEES(country):
    year = 2019
    JRC_year = 2021
    conversion_factor = 11.63 / 1e3 #ktoe to Twh
    #function to create unique entries if there are duplicates
    def rename_duplicates(series_or_df):
        counts = {}
        new_index = []
        for label in series_or_df.index:
            counts[label] = counts.get(label, 0) + 1
            new_index.append(f"{label}_{counts[label]}" if counts[label] > 1 else label)
        series_or_df.index = new_index
        return series_or_df
    
    fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
    fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"
    fn_industry = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Industry_{country}.xlsx"
    fn_transport = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Transport_{country}.xlsx"
    fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
    fn_energy = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_EnergyBalance_{country}.xlsx"
    # Read and convert data
    df_res = pd.read_excel(fn_residential, "RES_hh_fec", index_col=0)[year] * conversion_factor
    df_res_tot = pd.read_excel(fn_residential, "RES_summary", index_col=0)[year] * conversion_factor
    df_ter = pd.read_excel(fn_tertiary, "SER_hh_fec", index_col=0)[year] * conversion_factor
    df_ter_tot = pd.read_excel(fn_tertiary, "SER_summary", index_col=0)[year] * conversion_factor
    df_ind = pd.read_excel(fn_industry, "Ind_Summary_fec", index_col=0)[year] * conversion_factor
    df_ind_ne = pd.read_excel(fn_energy, "index", index_col=0)[year] * conversion_factor
    df_bm = pd.read_excel(fn_energy, "PPRD", index_col=0)[year] * conversion_factor
    df_tra_dom = pd.read_excel(fn_transport, "TrRoad_ene", index_col=0)[year] * conversion_factor
    df_tra_rai = pd.read_excel(fn_transport, "TrRail_ene", index_col=0)[year] * conversion_factor
    df_tra_avi = pd.read_excel(fn_transport, "TrAvia_ene", index_col=0)[year] * conversion_factor
    df_tra_nav = pd.read_excel(fn_transport, "TrNavi_ene", index_col=0)[year] * conversion_factor
    df_agr = pd.read_excel(fn_tertiary, "AGR_fec", index_col=0)[year] * conversion_factor
    df_elc = pd.read_excel(fn_power, "OverviewPG", index_col=0)[year] / 1e3 # GWh to Twh
    df_elc_therm = pd.read_excel(fn_power, "Thermal", index_col=0)[year] / 1e3
    df_elc_chp = pd.read_excel(fn_power, "Thermal_CHP", index_col=0)[year] / 1e3
    df_elc = rename_duplicates(df_elc)
    df_elc_therm = rename_duplicates(df_elc_therm)
    df_elc_chp = rename_duplicates(df_elc_chp)
    
    #extract total heat demand for residential and tertiary
    res_heat = df_res_tot.loc["Energy consumption by fuel - Eurostat structure (ktoe)"].sum()
    res_elc = df_res_tot.loc["Electricity"].sum()
    res_dh = df_res_tot.loc["Distributed heat"].sum()
    res_total_dm = res_heat - res_elc - res_dh
    ter_heat = df_ter_tot.loc["Energy consumption by fuel - Eurostat structure (ktoe)"].sum()
    ter_elc = df_ter_tot.loc["Electricity"].sum()
    ter_dh = df_ter_tot.loc["Distributed heat"].sum()
    ter_total_dm = ter_heat - ter_elc - ter_dh
    total_res_ter = res_total_dm + ter_total_dm
    #Create a dictionary from the JRC data
    def make_row(label, fuels, target, df):
          return {
              "label": label,
              "source": "TWh",
              "target": target,
              "2020": df.loc[fuels].sum() if isinstance(fuels, list) else df.loc[fuels].sum()
          }
      
    # Build list
    ct_totals = [
          make_row("Residential and tertiary gas for heating", "Natural gas", "presgazcfg",df_res),
          make_row("Residential and tertiary oil for heating", ["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"], "prespetcfo",df_res),
          make_row("Residential and tertiary biomass for heating", "Biomass", "presenccfb",df_res),
          make_row("Residential and tertiary DH demand", "Distributed heat", "presvapcfdhs",df_res),
          make_row("Residential and tertiary HP for heating", "Advanced electric heating", "prespaccfaaa",df_res),
          make_row("Residential and tertairy heat from electric heaters", ["Conventional electric heating","Electricity"], "preehplx",df_res),
          make_row("Residential and tertiary gas for heating", ["Conventional gas heaters", "Natural gas"], "presgazcfggg",df_ter),
          make_row("Residential and tertiary oil for heating", ["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"], "prespetcfooo",df_ter),
          make_row("Residential and tertiary biomass for heating", "Biomass", "presenccfbbb",df_ter),
          make_row("Residential and tertiary DH demand", "Distributed heat", "presvapcfdhs",df_ter),
          make_row("Residential and tertiary HP for heating", "Advanced electric heating", "prespaccffff",df_ter),
          make_row("Residential and tertairy heat from electric heaters", ["Conventional electric heating","Electricity"], "preehplyyy",df_ter),
          make_row("electricity demand of residential and tertairy", "Electricity", "preselccfterr",df_ter_tot),
          make_row("electricity demand of residential and tertairy", "Electricity", "preselccfres",df_res_tot),
          make_row("electricity for Industry", ["Lighting","Air compressors","Motor drives","Fans and pumps","Electricity"], "preselccfind",df_ind),
          make_row("coal for industry", ["Solids","Derived gases","Coke"], "cmscfind",df_ind),
          make_row("gas for Industry", ["Natural gas","Refinery gas","LPG"], "presgazcfind",df_ind),
          make_row("Oil for industry", ["Diesel oil","Fuel oil","Other liquids"], "prespetcfind",df_ind),
          make_row("solid biomass for Industry", "Biomass and waste", "presenccfind",df_ind),
          make_row("low-temperature heat for industry", "Distributed steam", "presvapcfind",df_ind),
          make_row("naphtha for non-energy", "Non-energy use industry/transformation/energy", "prespetcfneind",df_ind_ne),
          make_row("oil to transport demand", "by fuel (EUROSTAT DATA)", "preslqfcftra",df_tra_dom),
          make_row("land transport EV", "Electricity", "preselccftra",df_tra_dom),
          make_row("BEV charging", "Electricity", "prebev",df_tra_dom),
          make_row("electricity demand for rail network", "Electricity", "preserail",df_tra_rai),
          make_row("oil demand for rail network", "Diesel oil (blend)", "preserailoil",df_tra_rai),
          make_row("aviation oil demand", "Energy consumption (ktoe)", "preslqfcfavi",df_tra_avi),
          make_row("shipping oil", "Energy consumption (ktoe)", "preslqfcffrewati",df_tra_nav),
          make_row("agriculture electricity", ["Lighting","Ventilation","Motor drives","Electricity"], "preselccfagr",df_agr),
          make_row("agriculture oil", "Diesel oil and liquid biofuels", "prespetcfagr",df_agr),
          make_row("agriculture heat", "Natural gas and biogas", "pregazcfagr",df_agr),
          make_row("Nuclear production", "Nuclear_5", "proelcnuc",df_elc),
          make_row("Total ror production", "Hydro_5", "prohdror",df_elc),
          make_row("Solar photovoltaic Production", "Solar photovoltaics_5", "prospv",df_elc),
          make_row("wind-generated electricity", "Wind_5", "prowind",df_elc),
          make_row("Total hydropower production", "Pumped storage_5", "prohdr",df_elc),
          make_row("Gas-fired power generation", ["Natural gas_5","Biogas_5","Derived gas_5","Refinery gas_5"], "proelcgaz",df_elc_therm),
          make_row("Oil generation", ["Diesel oil_5","Fuel oil_5"], "proelcpet",df_elc_therm),
          make_row("solid biomass power plants", ["Solid biomass_5","Waste_5"], "proelcboi",df_elc_therm),
          make_row("Coal-fired power generation", "Coal_5", "proelccms",df_elc_therm),
          make_row("lignite power generation", "Lignite_5", "proelign",df_elc_therm),
          make_row("Power output from coal-fired CHP plants", ["Coal_5","Lignite_5"], "prbelcchpcms",df_elc_chp),
          make_row("Power output from methane-fired CHP plants", ["Natural gas_5","Biogas_5","Derived gas_5","Refinery gas_5"], "prbelcchpgaz",df_elc_chp),
          make_row("Power output from oil-fired CHP plants", ["Diesel oil_5","Fuel oil_5"], "prbelcchppet",df_elc_chp),
          make_row("Power output from solid biomass CHP plants", ["Solid biomass_5","Waste_5"], "prbelcchpboi",df_elc_chp),
          make_row("Heat output from coal-fired CHP plants", ["Coal_7","Lignite_7"], "prbvapchpcms",df_elc_chp),
          make_row("Heat output from methane-fired CHP plants", ["Natural gas_7","Biogas_7","Derived gas_7","Refinery gas_7"], "prbvapchpgass",df_elc_chp),
          make_row("Heat output from oil-fired CHP plants", ["Diesel oil_7","Fuel oil_7"], "prbvapchppet",df_elc_chp),
          make_row("Heat output from solid biomass CHP plants", ["Solid biomass_7","Waste_7"], "prbvapchpboi",df_elc_chp),
          make_row("Domestic production of solid biomass", "Primary solid biofuels", "prodomboi",df_bm),
             
    ]
    ct_totals.append({
    "label": "Residential and tertiary heat demand",
    "source": "TWh",
    "target": "demandheatc",
    "2020": total_res_ter})
    totals_df = pd.DataFrame(ct_totals)
    
    return totals_df

def Construct_2020_emissions_from_JRC_IDEES(country):
    year = 2019
    JRC_year = 2021
    conversion_factor = 1e3
    #function to create unique entries if there are duplicates
    def rename_duplicates(series_or_df):
        counts = {}
        new_index = []
        for label in series_or_df.index:
            counts[label] = counts.get(label, 0) + 1
            new_index.append(f"{label}_{counts[label]}" if counts[label] > 1 else label)
        series_or_df.index = new_index
        return series_or_df
    
    fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
    fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"
    fn_industry = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Industry_{country}.xlsx"
    fn_transport = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Transport_{country}.xlsx"
    fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
    
    #Using non-energy GHG gases and LULUCF values from Solargo (CLEVER)
    ghg_agri = pd.read_csv(snakemake.input.agri,index_col=0)
    non_energy_ghg_agri_ch4 = ghg_agri.loc[countries, 'Total CH4 emissions from agriculture']
    non_energy_ghg_agri_n2o = ghg_agri.loc[countries, 'Total N2O emissions from agriculture']
    GHG_agri_total = non_energy_ghg_agri_ch4 + non_energy_ghg_agri_n2o 
    GHG_agri_total = GHG_agri_total.T
    non_energy_agr = GHG_agri_total.filter(like=(country)).sum()
    lulucf = ghg_agri.loc[countries, 'Total CO2 emissions from the LULUCF sector']
    lulucf[lulucf > 0] = 0
    lulucf = lulucf.T
    lulucf = lulucf.filter(like=(country)).sum()
    lulucf = -lulucf
    # Read and convert data
    df_res = pd.read_excel(fn_residential, "RES_hh_emi", index_col=0)[year] / conversion_factor
    df_ter = pd.read_excel(fn_tertiary, "SER_hh_emi", index_col=0)[year] / conversion_factor
    df_ind = pd.read_excel(fn_industry, "Ind_Summary_emi", index_col=0)[year] / conversion_factor
    df_ind_ne = pd.read_excel(fn_industry, "CHI_emi", index_col=0)[year] / conversion_factor
    df_tra_dom = pd.read_excel(fn_transport, "TrRoad_emi", index_col=0)[year] / conversion_factor
    df_tra_rai = pd.read_excel(fn_transport, "TrRail_emi", index_col=0)[year] / conversion_factor
    df_tra_avi = pd.read_excel(fn_transport, "TrAvia_emi", index_col=0)[year] / conversion_factor
    df_tra_nav = pd.read_excel(fn_transport, "TrNavi_emi", index_col=0)[year] / conversion_factor
    df_agr = pd.read_excel(fn_tertiary, "AGR_emi", index_col=0)[year] / conversion_factor
    df_elc_therm = pd.read_excel(fn_power, "Thermal_ElecOnly", index_col=0)[year] / conversion_factor
    df_elc_chp = pd.read_excel(fn_power, "Thermal_CHP", index_col=0)[year] / conversion_factor

    df_elc_therm = rename_duplicates(df_elc_therm)
    df_elc_chp = rename_duplicates(df_elc_chp)
      
    #Create a dictionary from the JRC data
    def make_row(label, fuels, target, df):
          return {
              "label": label,
              "source": "MtCO2",
              "target": target,
              "2020": df.loc[fuels].sum() if isinstance(fuels, list) else df.loc[fuels].sum()
          }
      
    # Build list
    ct_totals = [
          make_row("res urb boiler", "Natural gas", "emmresubbo",df_res),
          make_row("res urb boiler", ["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"], "emmresuoil",df_res),
          make_row("residential urban decentral biomass boiler", "Biomass", "emmresbm",df_res),
          make_row("residential urban decentral biomass boiler", "Biomass", "emmresbmatm",df_res),
          make_row("res boiler", "Distributed heat", "emmresbo",df_res),
          make_row("serv urb boiler", ["Conventional gas heaters", "Natural gas"], "emmsesubbo",df_ter),
          make_row("serv urb boiler", ["Solids", "Liquified petroleum gas (LPG)", "Diesel oil"], "emmsesuboil",df_ter),
          make_row("residential urban decentral biomass boiler", "Biomass", "emmserbm",df_ter),
          make_row("residential urban decentral biomass boiler", "Biomass", "emmserbmatm",df_ter),
          make_row("urb boiler", "Distributed heat", "emmurbbbo",df_ter),
          make_row("coal emissions", ["Solids","Derived gases","Coke"], "emmcoal",df_ind),
          make_row("gas for industry", ["Natural gas","Refinery gas","LPG","Low-enthalpy heat"], "emmgas",df_ind),
          make_row("Oil for industry", ["Diesel oil","Fuel oil","Other liquids"], "emmoilind",df_ind),
          make_row("solid biomass for industry", "Biomass and waste", "emmindbm",df_ind),
          make_row("solid biomass for industry", "Biomass and waste", "emmindbmatm",df_ind),
          make_row("process emissions", "Process emissions", "emmprocess",df_ind),
          make_row("naphtha for industry", "Chemicals: Feedstock (energy used as raw material)", "emmoil",df_ind_ne),
          make_row("land transport oil emissions", "by fuel", "emmoiltra",df_tra_dom),
          make_row("Rail oil emissions", "by fuel", "emmrail",df_tra_rai),
          make_row("kerosene for aviation", "CO2 emissions (kt CO2)", "emmavi",df_tra_avi),
          make_row("shipping oil emissions", "CO2 emissions (kt CO2)", "emmoilwati",df_tra_nav),
          make_row("agriculture machinery oil emissions", "Diesel oil and liquid biofuels", "emmoilagr",df_agr),
          make_row("Agriculture gas emissions", "Natural gas and biogas", "emmagrgasz",df_agr),
          make_row("CCGT emissions", ["Natural gas_11","Derived gas_3","Refinery gas_9"], "emmccgt",df_elc_therm),
          make_row("Oil generation", ["Diesel oil_9","Fuel oil_9"], "emmoilgeb",df_elc_therm),
          make_row("biomass powerplants", ["Solid biomass_15","Waste_9"], "emmbnpower",df_elc_therm),
          make_row("biomass powerplants", ["Solid biomass_15","Waste_9"], "emmbmatmp",df_elc_therm),
          make_row("Coal-fired power generation", "Coal_9", "emmcoalgen",df_elc_therm),
          make_row("lignite power generation", "Lignite_9", "emmliggen",df_elc_therm),
          make_row("Coal-fired CHP", ["Coal_9","Lignite_9"], "emmcoalchp",df_elc_chp),
          make_row("urban central gas CHP", ["Natural gas_11","Derived gases_3","Refinery gas_9"], "emmgaschp",df_elc_chp),
          make_row("oil fired CHP", ["Diesel oil_9","Fuel oil_9"], "emmoilchp",df_elc_chp),
          make_row("urban central solid biomass CHP", ["Solid biomass_15","Waste_9"], "emmbmchp",df_elc_chp),
          make_row("urban central solid biomass CHP", ["Solid biomass_15","Waste_9"], "emmbmchpatm",df_elc_chp),
             
    ]
    ct_totals.append({
    "label": "Non-energy GHG Agriculture",
    "source": "MtCO2",
    "target": "emmghgagri",
    "2020": non_energy_agr})
    ct_totals.append({
    "label": "LULUCF",
    "source": "MtCO2",
    "target": "emmluf",
    "2020": lulucf})
    
    emm_totals = pd.DataFrame(ct_totals)
    return emm_totals

def Construct_2015_GB_from_JRC_IDEES(country):
    year = 2015
    JRC_year = 2015
    conversion_factor = 11.63 / 1e3 #ktoe to Twh
    #function to create unique entries if there are duplicates
    def rename_duplicates(series_or_df):
        counts = {}
        new_index = []
        for label in series_or_df.index:
            counts[label] = counts.get(label, 0) + 1
            new_index.append(f"{label}_{counts[label]}" if counts[label] > 1 else label)
        series_or_df.index = new_index
        return series_or_df
    
    fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
    fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"
    fn_industry = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Industry_{country}.xlsx"
    fn_transport = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Transport_{country}.xlsx"
    fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
    fn_energy = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_EnergyBalance_{country}.xlsx"
      
    # Read and convert data
    df_res = pd.read_excel(fn_residential, "RES_hh_fec", index_col=0)[year] * conversion_factor
    df_res_tot = pd.read_excel(fn_residential, "RES_summary", index_col=0)[year] * conversion_factor
    df_ter = pd.read_excel(fn_tertiary, "SER_hh_fec", index_col=0)[year] * conversion_factor
    df_ter_tot = pd.read_excel(fn_tertiary, "SER_summary", index_col=0)[year] * conversion_factor
    df_ind = pd.read_excel(fn_industry, "Ind_Summary_fec", index_col=0)[year] * conversion_factor
    df_ind_ne = pd.read_excel(fn_energy, "index", index_col=0)[year] * conversion_factor
    df_bm = pd.read_excel(fn_energy, "prod", index_col=0)[year] * conversion_factor
    df_tra_dom = pd.read_excel(fn_transport, "TrRoad_ene", index_col=0)[year] * conversion_factor
    df_tra_rai = pd.read_excel(fn_transport, "TrRail_ene", index_col=0)[year] * conversion_factor
    df_tra_avi = pd.read_excel(fn_transport, "TrAvia_ene", index_col=0)[year] * conversion_factor
    df_tra_nav = pd.read_excel(fn_transport, "TrNavi_ene", index_col=0)[year] * conversion_factor
    df_agr = pd.read_excel(fn_tertiary, "AGR_fec", index_col=0)[year] * conversion_factor
    df_elc = pd.read_excel(fn_power, "OverviewPG", index_col=0)[f'{year}'] / 1e3 # GWh to Twh
    df_elc_therm = pd.read_excel(fn_power, "Thermal", index_col=0)[f'{year}'] / 1e3
    df_elc_chp = pd.read_excel(fn_power, "Thermal_CHP", index_col=0)[f'{year}'] / 1e3
    df_elc = rename_duplicates(df_elc)
    df_elc_therm = rename_duplicates(df_elc_therm)
    df_elc_chp = rename_duplicates(df_elc_chp)
      
    #Create a dictionary from the JRC data
    def make_row(label, fuels, target, df):
          return {
              "label": label,
              "source": "TWh",
              "target": target,
              "2020": df.loc[fuels].sum() if isinstance(fuels, list) else df.loc[fuels].sum()
          }
      
    # Build list
    ct_totals = [
          make_row("Residential and tertiary gas for heating", "Gases incl. biogas", "presgazcfg",df_res),
          make_row("Residential and tertiary oil for heating", ["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"], "prespetcfo",df_res),
          make_row("Residential and tertiary biomass for heating", "Biomass and wastes", "presenccfb",df_res),
          make_row("Residential and tertiary DH demand", "Derived heat", "presvapcfdhs",df_res),
          make_row("Residential and tertiary HP for heating", "Advanced electric heating", "prespaccfaaa",df_res),
          make_row("Residential and tertairy heat from electric heaters", ["Conventional electric heating","Electricity"], "preehplx",df_res),
          make_row("Residential and tertiary gas for heating", ["Conventional gas heaters", "Gases incl. biogas"], "presgazcfggg",df_ter),
          make_row("Residential and tertiary oil for heating", ["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"], "prespetcfooo",df_ter),
          make_row("Residential and tertiary biomass for heating", "Biomass and wastes", "presenccfbbb",df_ter),
          make_row("Residential and tertiary DH demand", "Derived heat", "presvapcfdhs",df_ter),
          make_row("Residential and tertiary HP for heating", "Advanced electric heating", "prespaccffff",df_ter),
          make_row("Residential and tertairy heat from electric heaters", ["Conventional electric heating","Electricity"], "preehplyyy",df_ter),
          make_row("electricity demand of residential and tertairy", "Electricity", "preselccfterr",df_ter_tot),
          make_row("electricity demand of residential and tertairy", "Electricity", "preselccfres",df_res_tot),
          make_row("electricity for Industry", ["Lighting","Air compressors","Motor drives","Fans and pumps","Electricity"], "preselccfind",df_ind),
          make_row("coal for industry", ["Solids","Derived gases","Coke"], "cmscfind",df_ind),
          make_row("gas for Industry", ["Natural gas","Refinery gas","LPG"], "presgazcfind",df_ind),
          make_row("Oil for industry", ["Diesel oil","Residual fuel oil","Other liquids"], "prespetcfind",df_ind),
          make_row("solid biomass for Industry", ["Biomass and wastes","Biomass"], "presenccfind",df_ind),
          make_row("low-temperature heat for industry", "Steam distributed", "presvapcfind",df_ind),
          make_row("naphtha for non-energy", "Final Non-Energy Consumption", "prespetcfneind",df_ind_ne),
          make_row("oil to transport demand", "by fuel (EUROSTAT DATA)", "preslqfcftra",df_tra_dom),
          make_row("land transport EV", "Electricity", "preselccftra",df_tra_dom),
          make_row("BEV charging", "Electricity", "prebev",df_tra_dom),
          make_row("electricity demand for rail network", "Electricity", "preserail",df_tra_rai),
          make_row("oil demand for rail network", "Liquids (Petroleum products)", "preserailoil",df_tra_rai),
          make_row("aviation oil demand", "Total energy consumption (ktoe)", "preslqfcfavi",df_tra_avi),
          make_row("shipping oil", "Total energy consumption (ktoe)", "preslqfcffrewati",df_tra_nav),
          make_row("agriculture electricity", ["Lighting","Ventilation","Motor drives","Electricity"], "preselccfagr",df_agr),
          make_row("agriculture oil", "Diesel oil (incl. biofuels)", "prespetcfagr",df_agr),
          make_row("agriculture heat", "Gases (incl. biogas)", "pregazcfagr",df_agr),
          make_row("Nuclear production", "Nuclear_5", "proelcnuc",df_elc),
          make_row("Total ror production", "Hydro_5", "prohdror",df_elc),
          make_row("Solar photovoltaic Production", "Solar photovoltaics_5", "prospv",df_elc),
          make_row("wind-generated electricity", "Wind_5", "prowind",df_elc),
          make_row("Total hydropower production", "Pump storage_5", "prohdr",df_elc),
          make_row("Gas-fired power generation", ["Gas fired_5","Derived gas fired_5","Refinery gas fired_5"], "proelcgaz",df_elc_therm),
          make_row("Oil generation", ["Diesel oil fired_5","Fuel Oil fired_5"], "proelcpet",df_elc_therm),
          make_row("solid biomass power plants", "Solid biomass & waste fired_5", "proelcboi",df_elc_therm),
          make_row("Coal-fired power generation", "Coal fired_5", "proelccms",df_elc_therm),
          make_row("lignite power generation", "Lignite fired_5", "proelign",df_elc_therm),
          make_row("Power output from coal-fired CHP plants", ["Coal fired_5","Lignite fired_5"], "prbelcchpcms",df_elc_chp),
          make_row("Power output from methane-fired CHP plants", ["Gas fired_5","Derived gas fired_5","Refinery gas fired_5"], "prbelcchpgaz",df_elc_chp),
          make_row("Power output from oil-fired CHP plants", ["Diesel oil fired_5","Fuel Oil fired_5"], "prbelcchppet",df_elc_chp),
          make_row("Power output from solid biomass CHP plants", "Solid biomass & waste fired_5", "prbelcchpboi",df_elc_chp),
          make_row("Heat output from coal-fired CHP plants", ["Coal fired_7","Lignite fired_7"], "prbvapchpcms",df_elc_chp),
          make_row("Heat output from methane-fired CHP plants", ["Gas fired_7","Derived gas fired_7","Refinery gas fired_7"], "prbvapchpgaz",df_elc_chp),
          make_row("Heat output from oil-fired CHP plants", ["Diesel oil fired_7","Fuel Oil fired_7"], "prbvapchppet",df_elc_chp),
          make_row("Heat output from solid biomass CHP plants", "Solid biomass & waste fired_7", "prbvapchpboi",df_elc_chp),
          make_row("Domestic production of solid biomass", "Solid biofuels (Wood & Wood waste)", "prodomboi",df_bm),
             
    ]
    totals_df = pd.DataFrame(ct_totals)
    return totals_df


def Construct_2015_GB_emissions_from_JRC_IDEES(country):
    year = 2015
    JRC_year = 2015
    conversion_factor = 1e3
    #function to create unique entries if there are duplicates
    def rename_duplicates(series_or_df):
        counts = {}
        new_index = []
        for label in series_or_df.index:
            counts[label] = counts.get(label, 0) + 1
            new_index.append(f"{label}_{counts[label]}" if counts[label] > 1 else label)
        series_or_df.index = new_index
        return series_or_df
    
    fn_residential = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Residential_{country}.xlsx"
    fn_tertiary = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Tertiary_{country}.xlsx"
    fn_industry = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Industry_{country}.xlsx"
    fn_transport = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_Transport_{country}.xlsx"
    fn_power = f"data/JRC/JRC-IDEES-{JRC_year}_{country}/JRC-IDEES-{JRC_year}_PowerGen_{country}.xlsx"
    
    #Using non-energy GHG gases and LULUCF values from Solargo (CLEVER)
    ghg_agri = pd.read_csv(snakemake.input.agri,index_col=0)
    non_energy_ghg_agri_ch4 = ghg_agri.loc[countriess, 'Total CH4 emissions from agriculture']
    non_energy_ghg_agri_n2o = ghg_agri.loc[countriess, 'Total N2O emissions from agriculture']
    GHG_agri_total = non_energy_ghg_agri_ch4 + non_energy_ghg_agri_n2o 
    GHG_agri_total = GHG_agri_total.T
    non_energy_agr = GHG_agri_total.filter(like=(country)).sum()
    lulucf = ghg_agri.loc[countriess, 'Total CO2 emissions from the LULUCF sector']
    lulucf[lulucf > 0] = 0
    lulucf = lulucf.T
    lulucf = lulucf.filter(like=(country)).sum()
    lulucf = -lulucf
    # Read and convert data
    df_res = pd.read_excel(fn_residential, "RES_hh_emi", index_col=0)[year] / conversion_factor
    df_ter = pd.read_excel(fn_tertiary, "SER_hh_emi", index_col=0)[year] / conversion_factor
    df_ind = pd.read_excel(fn_industry, "Ind_Summary_emi", index_col=0)[year] / conversion_factor
    df_ind_ne = pd.read_excel(fn_industry, "CHI_emi", index_col=0)[year] / conversion_factor
    df_tra_dom = pd.read_excel(fn_transport, "TrRoad_emi", index_col=0)[year] / conversion_factor
    df_tra_rai = pd.read_excel(fn_transport, "TrRail_emi", index_col=0)[year] / conversion_factor
    df_tra_avi = pd.read_excel(fn_transport, "TrAvia_emi", index_col=0)[year] / conversion_factor
    df_tra_nav = pd.read_excel(fn_transport, "TrNavi_emi", index_col=0)[year] / conversion_factor
    df_agr = pd.read_excel(fn_tertiary, "AGR_emi", index_col=0)[year] / conversion_factor
    df_elc_therm = pd.read_excel(fn_power, "Thermal_ElecOnly", index_col=0)[f'{year}'] / conversion_factor
    df_elc_chp = pd.read_excel(fn_power, "Thermal_CHP", index_col=0)[f'{year}'] / conversion_factor

    df_elc_therm = rename_duplicates(df_elc_therm)
    df_elc_chp = rename_duplicates(df_elc_chp)
      
    #Create a dictionary from the JRC data
    def make_row(label, fuels, target, df):
          return {
              "label": label,
              "source": "MtCO2",
              "target": target,
              "2020": df.loc[fuels].sum() if isinstance(fuels, list) else df.loc[fuels].sum()
          }
      
    # Build list
    ct_totals = [
          make_row("res urb boiler", "Gases incl. biogas", "emmresubbo",df_res),
          make_row("res urb boiler", ["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"], "emmresuoil",df_res),
          make_row("residential urban decentral biomass boiler", "Biomass and wastes", "emmresbm",df_res),
          make_row("residential urban decentral biomass boiler", "Biomass and wastes", "emmresbmatm",df_res),
          make_row("res boiler", "Derived heat", "emmresbo",df_res),
          make_row("serv urb boiler", ["Conventional gas heaters", "Gases incl. biogas"], "emmsesubbo",df_ter),
          make_row("serv urb boiler", ["Solids", "Liquified petroleum gas (LPG)", "Gas/Diesel oil incl. biofuels (GDO)"], "emmsesuboil",df_ter),
          make_row("residential urban decentral biomass boiler", "Biomass and wastes", "emmserbm",df_ter),
          make_row("residential urban decentral biomass boiler", "Biomass and wastes", "emmserbmatm",df_ter),
          make_row("urb boiler", "Derived heat", "emmurbbbo",df_ter),
          make_row("coal emissions", ["Solids","Derived gases","Coke"], "emmcoal",df_ind),
          make_row("gas for industry", ["Natural gas","Refinery gas","LPG","Low enthalpy heat"], "emmgas",df_ind),
          make_row("Oil for industry", ["Diesel oil","Residual fuel oil","Other liquids"], "emmoilind",df_ind),
          make_row("solid biomass for industry", ["Biomass and wastes","Biomass"], "emmindbm",df_ind),
          make_row("solid biomass for industry", ["Biomass and wastes","Biomass"], "emmindbmatm",df_ind),
          make_row("process emissions", "Process emissions", "emmprocess",df_ind),
          make_row("naphtha for industry", "Chemicals: Feedstock (energy used as raw material)", "emmoil",df_ind_ne),
          make_row("land transport oil emissions", "by fuel", "emmoiltra",df_tra_dom),
          make_row("Rail oil emissions", "by fuel", "emmrail",df_tra_rai),
          make_row("kerosene for aviation", "by fuel", "emmavi",df_tra_avi),
          make_row("shipping oil emissions", "by fuel", "emmoilwati",df_tra_nav),
          make_row("agriculture machinery oil emissions", "Diesel oil (incl. biofuels)", "emmoilagr",df_agr),
          make_row("Agriculture gas emissions", "Gases (incl. biogas)", "emmagrgasz",df_agr),
          make_row("CCGT emissions", ["Gas_2","Derived gases_2","Refinery gas_2"], "emmccgt",df_elc_therm),
          make_row("Oil generation", ["Diesel oil_2","Fuel Oil_2"], "emmoilgeb",df_elc_therm),
          make_row("biomass powerplants", "Solid biomass & waste_2", "emmbnpower",df_elc_therm),
          make_row("biomass powerplants", "Solid biomass & waste_2", "emmbmatmp",df_elc_therm),
          make_row("Coal-fired power generation", "Coal_2", "emmcoalgen",df_elc_therm),
          make_row("lignite power generation", "Lignite_2", "emmliggen",df_elc_therm),
          make_row("Coal-fired CHP", ["Coal_2","Lignite_2"], "emmcoalchp",df_elc_chp),
          make_row("urban central gas CHP", ["Gas_2","Derived gases_2","Refinery gas_2"], "emmgaschp",df_elc_chp),
          make_row("oil fired CHP", ["Diesel oil_2","Fuel Oil_2"], "emmoilchp",df_elc_chp),
          make_row("urban central solid biomass CHP", "Solid biomass & waste_2", "emmbmchp",df_elc_chp),
          make_row("urban central solid biomass CHP", "Solid biomass & waste_2", "emmbmchpatm",df_elc_chp),
             
    ]
    ct_totals.append({
    "label": "Non-energy GHG Agriculture",
    "source": "MtCO2",
    "target": "emmghgagri",
    "2020": non_energy_agr})
    ct_totals.append({
    "label": "LULUCF",
    "source": "MtCO2",
    "target": "emmluf",
    "2020": lulucf})
    emm_totals = pd.DataFrame(ct_totals)
    return emm_totals

    
    
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "generate_JRC")

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    countries = ['BE', 'DE', 'FR', 'NL']
    countriess = ['GB']
    study = snakemake.params.study
    # Create separate files for each country
    for country in countries:
        totals_df = Construct_2020_from_JRC_IDEES(country)
        emm_df = Construct_2020_emissions_from_JRC_IDEES(country)
        with pd.ExcelWriter(f"results/{study}/sepia/inputs_{country}.xlsx") as writer:
            totals_df.to_excel(writer, sheet_name="Inputs", index=False)
            emm_df.to_excel(writer, sheet_name="Inputs_co2", index=False)

    for country in countriess:
        totals_df = Construct_2015_GB_from_JRC_IDEES(country)
        emm_df = Construct_2015_GB_emissions_from_JRC_IDEES(country)
        with pd.ExcelWriter(f"results/{study}/sepia/inputs_{country}.xlsx") as writer:
            totals_df.to_excel(writer, sheet_name="Inputs", index=False)
            emm_df.to_excel(writer, sheet_name="Inputs_co2", index=False)

    combined_inputs = None
    combined_emissions = None

    for country in countries:
        df_inputs = Construct_2020_from_JRC_IDEES(country).set_index(["label", "source", "target"])
        df_emm = Construct_2020_emissions_from_JRC_IDEES(country).set_index(["label", "source", "target"])
        combined_inputs = df_inputs if combined_inputs is None else combined_inputs.add(df_inputs, fill_value=0)
        combined_emissions = df_emm if combined_emissions is None else combined_emissions.add(df_emm, fill_value=0)

    for country in countriess:
        df_inputs = Construct_2015_GB_from_JRC_IDEES(country).set_index(["label", "source", "target"])
        df_emm = Construct_2015_GB_emissions_from_JRC_IDEES(country).set_index(["label", "source", "target"])
        combined_inputs = combined_inputs.add(df_inputs, fill_value=0)
        combined_emissions = combined_emissions.add(df_emm, fill_value=0)

    # Save to inputsEU.xlsx with two sheets
    with pd.ExcelWriter(f"results/{study}/sepia/inputs_EU.xlsx") as writer:
        combined_inputs.reset_index().to_excel(writer, sheet_name="Inputs", index=False)
        combined_emissions.reset_index().to_excel(writer, sheet_name="Inputs_co2", index=False)
    