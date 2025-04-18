# -*- coding: utf-8 -*-
"""
SEPIA - Simplified Energy Prospective and Interterritorial Analysis tool
"""
__author__ = "Adrien Jacob, négaWatt"
__copyright__ = "Copyright 2022, négaWatt"
__license__ = "GPL"
__version__ = "1.8"
__email__ = "adrien.jacob@negawatt.org"

import SEPIA_additional_functions as saf # Custom functions
import SEPIA_functions as sf # Custom functions
import pandas as pd # Read/analyse data
import datetime # For current time
import logging
import numpy as np
import yaml
import os

def biomass_potentials():
    # Create an empty DataFrame
    ALL_COUNTRIES = snakemake.params.countries
    planning_horizons = snakemake.params.planning_horizons
    cluster = snakemake.params.cluster
    cluster = cluster[0]
    df = pd.DataFrame(index=planning_horizons, columns=ALL_COUNTRIES)

    # Iterate over countries and planning horizons
    for country in ALL_COUNTRIES:
        for planning_horizon in planning_horizons:
            biomass_potentials_file = f"resources/{study}/biomass_potentials_s_{cluster}_{planning_horizon}.csv"
            biomass_p = pd.read_csv(biomass_potentials_file, index_col=0)
            
            # Convert MWh to TWh
            biomass_p = biomass_p / 1E6
            biomass_p.index = biomass_p.index.str[:2]
            biomass_p = biomass_p.groupby(biomass_p.index).sum()
            
            # Assign the summed biomass potential for the country and planning horizon to the DataFrame
            df.loc[planning_horizon, country] = biomass_p.loc[country, 'solid biomass']

    return df

def prepare_sepia(countries):
 '''This function prepares data from excel files for sepia visulaisation'''

 # Import country data
 file = snakemake.input.countries
 COUNTRIES = pd.read_excel(file, index_col=0)
 ALL_COUNTRIES = snakemake.params.countries
 fn = snakemake.input.costs
 options = pd.read_csv(fn ,index_col=[0, 1]).sort_index()

 # Import config data (nodes, processes, general settings etc.)
 file = snakemake.input.sepia_config
 CONFIG = pd.read_excel(file, ["MAIN_PARAMS","NODES","PROCESSES","PROCESSES_2","PROCESSES_3","IMPORT_MIX","INDICATORS","GAS_PRO","OIL_PRO"], index_col=0)

 # Main settings (cf. SEPIA_config for description of all setting constants)
 MAIN_PARAMS = CONFIG["MAIN_PARAMS"].drop('Description',axis=1).to_dict()['Value']

 NODES = CONFIG["NODES"]
 FE_NODES = sf.nodes_by_type(NODES,'FINAL_ENERGIES')
 SE_NODES = sf.nodes_by_type(NODES,'SECONDARY_ENERGIES')
 PE_NODES = sf.nodes_by_type(NODES,'PRIMARY_ENERGIES')
 DS_NODES = sf.nodes_by_type(NODES,'DEMAND_SECTORS')
 II_NODES = sf.nodes_by_type(NODES,'IMPORTS')
 EE_NODES = sf.nodes_by_type(NODES,'EXPORTS')
 GHG_SECTORS = sf.nodes_by_type(NODES,'GHG_SECTORS')

 PROCESSES = CONFIG["PROCESSES"].reset_index()
 PROCESSES['Type'].fillna('', inplace=True)
 PROCESSES_2 = CONFIG["PROCESSES_2"].reset_index()
 PROCESSES_2['Type'].fillna('', inplace=True)
 PROCESSES_3 = CONFIG["PROCESSES_3"].reset_index()
 PROCESSES_3['Type'].fillna('', inplace=True)
 INDICATORS = CONFIG["INDICATORS"]
 LOCAL_GAS = CONFIG["GAS_PRO"]
 LOCAL_GAS = LOCAL_GAS.drop('Unnamed: 5', axis=1)
 LOCAL_OIL = CONFIG["OIL_PRO"]
 LOCAL_OIL = LOCAL_OIL.drop('Unnamed: 5', axis=1)
 # Aggregated results per Country
 tot_results = pd.DataFrame()
 # Dictionnaries storing results of the next section : "Country" => Value
 tot_flows = {} # Energy flow DataFrames
 tot_ghg = {}
 tot_co2 = {}
 total_country = 'EU'
 include_total_country = True
 if include_total_country == True:
      ALL_COUNTRIES.append(total_country)
 else:
      ALL_COUNTRIES = ALL_COUNTRIES

 # Energy system (network graph) creation for all countries
 print("\nEnergy system (network graph) creation\n")
 print("ALL_COUNTRIES:", ALL_COUNTRIES)
 print("snakemake.input.excelfile:", snakemake.input.excelfile)
 
 for country in ALL_COUNTRIES:
    datafile = snakemake.input.excelfile[ALL_COUNTRIES.index(country)]
    datafile = str(datafile)
    
    '''load energy input data for Sepia'''
    data = pd.read_excel(datafile, sheet_name="Inputs", index_col=0, usecols="C:G")
    data.reset_index(drop=True, inplace=False)
    data=data.T
    
    
    '''load co2 input data for Sepia'''
    data_co2 = pd.read_excel(datafile, sheet_name="Inputs_co2", index_col=0, usecols="C:G")
    data_co2.reset_index(drop=True, inplace=False)
    data_co2=data_co2.T
    
    '''Remove any duplicated data'''
    data = data.loc[:,~data.columns.duplicated()] 
    data_co2 = data_co2.loc[:,~data_co2.columns.duplicated()]# Remove duplicate indicators

    '''Consider the coding used in sepia config and put unfound demands from pypsa file to zero'''
    unfound_inputs = []
    unfound_inputs.extend(sf.unfound_indicators(data,PROCESSES,'Value_Code'))
    unfound_inputs.extend(sf.unfound_indicators(data,PROCESSES,'Efficiency_Code'))
    unfound_inputs.extend(sf.unfound_indicators(data,INDICATORS,'Value_Code'))
    if len(unfound_inputs)>0:
        data = data.reindex(columns=[*data.columns.tolist(), *unfound_inputs], fill_value=0)
        print("! Warning: the following indicators have not been found (they have been filled with 0): "+", ".join(unfound_inputs)+" !!!")
        
    unfound_inputs_co2 = []
    unfound_inputs_co2.extend(sf.unfound_indicators(data_co2,PROCESSES_2,'Value_Code'))
    if len(unfound_inputs_co2)>0:
        data_co2 = data_co2.reindex(columns=[*data_co2.columns.tolist(), *unfound_inputs_co2], fill_value=0)
        print("! Warning: the following indicators have not been found (they have been filled with 0): "+", ".join(unfound_inputs_co2)+" !!!")

    # ## Corrections on input data
    # Renaming indicators, based on INDICATORS sheet
    data = data.rename(columns=dict(zip(INDICATORS['Value_Code'],INDICATORS.index)))
    data = data.loc[:,~data.columns.duplicated()] # Remove duplicate indicators
    data_co2 = data_co2.rename(columns=dict(zip(INDICATORS['Value_Code'],INDICATORS.index)))
    data_co2 = data_co2.loc[:,~data_co2.columns.duplicated()]
    data_ghg = data_co2.copy()
    

    ## Creating flows filling values which do not require calculation, directly from input data
    proc_without_calc = PROCESSES[PROCESSES['Value_Code'].isin(data.columns)] # indicator is not empty and found in data
    flows = pd.DataFrame(data[proc_without_calc.Value_Code].values, index=data.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc.Source, proc_without_calc.Target, proc_without_calc.Type)), names=('Source','Target','Type')))
    proc_without_calc_co2 = PROCESSES_2[PROCESSES_2['Value_Code'].isin(data_co2.columns)] # indicator is not empty and found in data
    flows_co2 = pd.DataFrame(data_co2[proc_without_calc_co2.Value_Code].values, index=data_co2.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc_co2.Source, proc_without_calc_co2.Target, proc_without_calc_co2.Type)), names=('Source','Target','Type')))
    proc_without_calc_ghg = PROCESSES_3[PROCESSES_3['Value_Code'].isin(data_ghg.columns)] # indicator is not empty and found in data
    flows_ghg = pd.DataFrame(data_ghg[proc_without_calc_ghg.Value_Code].values, index=data_ghg.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc_ghg.Source, proc_without_calc_ghg.Target, proc_without_calc_ghg.Type)), names=('Source','Target','Type')))


    '''Attaching production from primary and secondary energies to final energy demands'''
    selected_columns = flows.columns.get_level_values('Source').isin(FE_NODES)
    fec_carrier = flows.loc[:, selected_columns]
    grouped_fec = fec_carrier.groupby(level='Source', axis=1).sum()
    fec = grouped_fec
    for en_code in ['vap','elc','gaz','hyd','bev']:
        flows[(en_code+'_se',en_code+'_fe','')] = fec[en_code+'_fe']
        
    selected_columns_pe = flows.columns.get_level_values('Source').isin(FE_NODES)
    fec_carrier_pe = flows.loc[:, selected_columns_pe]
    grouped_fec_pe = fec_carrier_pe.groupby(level='Source', axis=1).sum()
    fec_pe = grouped_fec_pe
    for en_code in ['pac','enc','cms']:
        flows[(en_code+'_pe',en_code+'_fe','')] = fec_pe[en_code+'_fe']
  
    biogas_p = flows['bgl_pe','gaz_se',''].squeeze().rename_axis(None)
    biogas_cc = flows['bgl_pe','gaz_se','cc'].squeeze().rename_axis(None)
    biosng_p = flows['enc_pe','gaz_se',''].squeeze().rename_axis(None)
    meth_p = flows['hyd_se','gaz_se',''].squeeze().rename_axis(None)
   
        
    selected_columns_se = flows.columns.get_level_values('Source').isin(SE_NODES)
    fec_carrier_se = flows.loc[:, selected_columns_se]
    grouped_fec_se = fec_carrier_se.groupby(level='Source', axis=1).sum()
    fec_se = grouped_fec_se
    for en_code in ['gaz']:
        flows[(en_code+'_pe',en_code+'_se','')] = fec_se[en_code+'_se']-biogas_p-biosng_p-meth_p-biogas_cc
    
    ''' Adding missing values for nuclear losses'''
    nuc_eff = options.loc[("nuclear", "efficiency"), "value"]
    nuc_los = (1 - nuc_eff)/nuc_eff
    for en_code in ['ura']:
      value_loss_nuc_2020 = flows.loc['2020', (en_code + '_pe', 'elc_se', 'thm')].sum() * nuc_los
      value_loss_nuc_2030 = flows.loc['2030', (en_code + '_pe', 'elc_se', 'thm')].sum() * nuc_los
      value_loss_nuc_2040 = flows.loc['2040', (en_code + '_pe', 'elc_se', 'thm')].sum() * nuc_los
      value_loss_nuc_2050 = flows.loc['2050', (en_code + '_pe', 'elc_se', 'thm')].sum() * nuc_los
      flows.loc['2020', (en_code + '_pe', 'per', 'thm')] = value_loss_nuc_2020
      flows.loc['2030', (en_code + '_pe', 'per', 'thm')] = value_loss_nuc_2030
      flows.loc['2040', (en_code + '_pe', 'per', 'thm')] = value_loss_nuc_2040
      flows.loc['2050', (en_code + '_pe', 'per', 'thm')] = value_loss_nuc_2050
    
    '''Attaching local production and imports'''
    selected_columns_p = flows.columns.get_level_values('Source').isin(PE_NODES)
    fec_carrier_p = flows.loc[:, selected_columns_p]
    grouped_fec_p = fec_carrier_p.groupby(level='Source', axis=1).sum()
    fec_p = grouped_fec_p
    if country == 'EU':
     for en_code in ['hdr','eon','eof','spv','pac','enc','ura','bgl']:
        flows[('prod',en_code+'_pe','')] = fec_p[en_code+'_pe']
    else: 
     for en_code in ['hdr','eon','eof','spv','pac','ura','bgl']:
        flows[('prod',en_code+'_pe','')] = fec_p[en_code+'_pe'] 
    for en_code in ['cms']:
       flows[('imp',en_code+'_pe','')] = fec_p[en_code+'_pe'] 
    for en_code in ['gaz']:
     if country != 'EU':
      values = fec_p[en_code + '_pe']
      local_val = LOCAL_GAS.loc[country]
      local_val.index = local_val.index.map(str)
      mask = values >= local_val
      flows[('prod', en_code+'_pe', '')] = np.where(mask, local_val, values)
      flows[('imp', en_code+'_pe','')] =  np.where(mask, values - local_val, 0)
     else:
      values = fec_p[en_code + '_pe']
      local_val = LOCAL_GAS.loc[LOCAL_GAS.index.intersection(countries)].sum()
      local_val.index = local_val.index.map(str)
      mask = values >= local_val
      flows[('prod', en_code+'_pe', '')] = np.where(mask, local_val, values)
      flows[('imp', en_code+'_pe','')] =  np.where(mask, values - local_val, 0)
    for en_code in ['pet']:
     if country != 'EU':
      values = fec_p[en_code + '_pe']
      local_val = LOCAL_OIL.loc[country]
      local_val.index = local_val.index.map(str)
      mask = values >= local_val
      flows[('prod', en_code+'_pe', '')] = np.where(mask, local_val, values)
      flows[('imp', en_code+'_pe', '')] = np.where(mask, values - local_val, 0)
     else:
      values = fec_p[en_code + '_pe']
      local_val = LOCAL_OIL.loc[LOCAL_OIL.index.intersection(countries)].sum()
      local_val.index = local_val.index.map(str)
      mask = values >= local_val
      flows[('prod', en_code+'_pe', '')] = np.where(mask, local_val, values)
      flows[('imp', en_code+'_pe', '')] = np.where(mask, values - local_val, 0)
     
    ''' Compute biomass imports and local production from model data'''
    if country != 'EU':
     df[country] = df[country].astype(float)
     bm_potentials = df[[country]]
     bm_potentials = bm_potentials.loc[:, country].values
     for en_code in ['enc']:
      flows[('prod',en_code+'_pe','')] = bm_potentials
      imp_values = fec_p[en_code + '_pe'] - bm_potentials
      flows[('imp', en_code + '_pe', '')] = imp_values
      flows[(en_code + '_pe', 'exp', '')] = bm_potentials - fec_p[en_code + '_pe']
    
    sec_imports = flows.columns.get_level_values('Target').isin(SE_NODES)
    sec_imports = flows.loc[:, sec_imports]
    sec_imports = sec_imports.groupby(level='Target', axis=1).sum()
    if country != 'EU':
     for en_code in ['elc','hyd']:
        values_exp = sec_imports[en_code + '_se'] - fec_se[en_code + '_se']
        values_imp = fec_se[en_code + '_se'] - sec_imports[en_code + '_se']
        values_imp = values_imp.clip(lower=0)
        values_exp = values_exp.clip(lower=0)
        flows[('imp',en_code + '_se', '')] = values_imp
        flows[(en_code+'_se','exp','')] = values_exp
        
    other_imports = flows.columns.get_level_values('Target').isin(FE_NODES)
    other_imports = flows.loc[:, other_imports]
    other_imports = other_imports.groupby(level='Target', axis=1).sum() 
    if country != 'EU':
     for en_code in ['amm','met']:
        values_elec = flows[('elc_se',en_code + '_fe', '')].squeeze().rename_axis(None)
        flows[('met_fe','per', '')] = values_elec
        values_exp = other_imports[en_code + '_fe'] - fec_pe[en_code + '_fe'] - values_elec
        values_imp = fec_pe[en_code + '_fe'] - other_imports[en_code + '_fe'] - values_elec
        values_imp = values_imp.clip(lower=0)
        values_exp = values_exp.clip(lower=0)
        flows[('imp',en_code + '_fe', '')] = values_imp
        flows[(en_code+'_fe','exp','')] = values_exp
    else:
        values_elec = flows[('elc_se','met_fe', '')].squeeze().rename_axis(None)
        flows[('met_fe','per', '')] = values_elec
    '''preparing co2 emissions for carbon sankey'''
    tot_emm_s = flows_co2.columns.get_level_values('Source').isin(GHG_SECTORS)
    tot_emm_s = flows_co2.loc[:, tot_emm_s]
    tot_emm_s = tot_emm_s.groupby(level='Source', axis=1).sum() 
    
    # '''using co2 intensities from pypsa and compuing it from demands as on pypsa they are solved on EU level'''
    co2_intensity_gas = options.loc[("gas", "CO2 intensity"), "value"]
    co2_intensity_met = options.loc[("methanolisation", "carbondioxide-input"), "value"]
    # demand_side_emm = flows.columns.get_level_values('Target').isin(DS_NODES)
    # demand_side_emm = flows.loc[:, demand_side_emm]
    # demand_side_emm = demand_side_emm.groupby(level='Target', axis=1).sum() 
    # for en_code in ['fol']:
    #     values_oil_emm = fec_p['pet_pe']
    #     flows_co2[(en_code + '_ghg', 'oil_ghg', '')] = values_oil_emm * co2_intensity_oil
    if country != 'EU':     
     for en_code in ['fgs']:
        # values_agr_emm = flows[('gaz_fe','agr', '')].squeeze().rename_axis(None) * co2_intensity_gas
        values_gas_emm = fec_p['gaz_pe'] 
        flows_co2[(en_code + '_ghg', 'gas_ghg', '')] = values_gas_emm * co2_intensity_gas
    
    # for en_code in ['oil']:
    #     value_so = flows[('pet_fe', 'wati', '')].squeeze().rename_axis(None) * co2_intensity_oil
    #     value_naph = flows[('pet_fe', 'neind', '')].squeeze().rename_axis(None)
    #     value_ker = flows[('pet_fe', 'avi', '')].squeeze().rename_axis(None)
    #     value_tra = flows[('pet_fe', 'tra', '')].squeeze().rename_axis(None) * co2_intensity_oil
    #     value_tot =  value_naph * co2_intensity_oil
    #     flows_co2[(en_code + '_ghg', 'atm', 'so')] = value_so
    #     flows_co2[(en_code + '_ghg', 'atm', 'oil')] = value_tot
    #     flows_co2[(en_code + '_ghg', 'atm', 'tra')] = value_tra
  
    
    
    tot_emm = flows_co2.columns.get_level_values('Target').isin(GHG_SECTORS)
    tot_emm = flows_co2.loc[:, tot_emm]
    tot_emm = tot_emm.groupby(level='Target', axis=1).sum() 
    for en_code in ['net']:
        # bm_cap = flows_co2[('atm', 'bec' + '_ghg', '')].squeeze().rename_axis(None)
        blg_cap = flows_co2[('atm', 'blg' + '_ghg', '')].squeeze().rename_axis(None)
        blg_cap_cc = flows_co2[('atm', 'blg' + '_ghg', 'cc')].squeeze().rename_axis(None)
        dac_cap = flows_co2[('atm', 'stm', '')].squeeze().rename_axis(None)
        # bm_cap = bm_cap.sum(axis=1)
        values_atm = tot_emm['atm'] - tot_emm['bm_ghg'] - tot_emm['luf_ghg'] - dac_cap - blg_cap - blg_cap_cc
        flows_co2[('atm',en_code + '_ghg',  'net')] = values_atm
        
    if country != 'EU':
     for en_code in ['met']:
      met_dem = fec_pe[en_code + '_fe']
      met_pro = flows[('hyd_se',en_code + '_fe', '')].squeeze().rename_axis(None)
      imp_met = (met_dem - met_pro) * co2_intensity_met 
      flows_co2[('hth_ghg',en_code + '_ghg', '')] = imp_met
      exp_met = (met_pro - met_dem) * co2_intensity_met
      flows_co2[(en_code + '_ghg',  'eth_ghg', '')] = exp_met
    # for en_code in ['pet']:
    #     flows_ghg[('ind_ghg',  en_code + '_pe', 'oil')] = value_tot
    #     flows_ghg[('tra_ghg',  en_code + '_pe', '')] = value_tra
       
    # for en_code in ['wati']:
    #     flows_ghg[(en_code + '_ghg', 'pet_pe',  '')] =value_so
    
    # for en_code in ['pet']:
    #     if flows[('hyd_se',en_code + '_fe',  '')].squeeze().rename_axis(None).sum()>0:
    #         pet_pro = flows[('hyd_se', en_code + '_fe', '')].squeeze().rename_axis(None)
    #         pet_bm = flows[('enc_pe', en_code + '_fe', '')].squeeze().rename_axis(None)
    #         value_agr = flows[('pet_fe', 'agr', '')].squeeze().rename_axis(None) 
    #         tot_pet = pet_pro + pet_bm 
    #         exp_pet = tot_pet - value_ker - value_naph -  value_agr
    #         flows[(en_code + '_fe', 'exp', '')] = exp_pet
    if country != 'EU':
     for en_code in ['gaz']:
        if (
    (flows[('hyd_se', en_code + '_se',  '')].squeeze().rename_axis(None).sum() > 0) or
    (flows[('bgl_pe', en_code + '_se',  'cc')].squeeze().rename_axis(None).sum() > 0)):
            tot_pro = biogas_p + biosng_p + meth_p + biogas_cc
            gas_dem = flows[(en_code + '_se', en_code + '_fe','')].squeeze().rename_axis(None)
            gas_los = flows[(en_code + '_se', 'per','')].squeeze().rename_axis(None)
            if isinstance(gas_los, pd.DataFrame):
              gas_los = gas_los.sum(axis=1)
            gas_elc = flows[(en_code + '_se', 'elc_se', '')].squeeze().rename_axis(None)
            if isinstance(gas_elc, pd.DataFrame):
              gas_elc = gas_elc.sum(axis=1)
            gas_vap = flows[(en_code + '_se', 'vap_se', '')].squeeze().rename_axis(None)
            if isinstance(gas_vap, pd.DataFrame):
              gas_vap = gas_vap.sum(axis=1)
            gas_smr = flows[(en_code + '_se', 'hyd_se', 'smr')].squeeze().rename_axis(None)
            if isinstance(gas_smr, pd.DataFrame):
              gas_smr = gas_smr.sum(axis=1)
            tot_dem = gas_dem + gas_los + gas_elc + gas_vap + gas_smr
            exp_gas = tot_pro - tot_dem
            flows[(en_code + '_se', 'exp', '')] = exp_gas
    for en_code in ['exg']:
        exp_emm = flows[('gaz' + '_se', 'exp', '')].squeeze().rename_axis(None) * co2_intensity_gas
        flows_co2[('gas_ghg',en_code + '_ghg', '')] = exp_emm
    # for en_code in ['ext']:
    #     exp_emm_p = flows[('pet' + '_fe', 'exp', '')].squeeze().rename_axis(None) * co2_intensity_oil
    #     flows_co2[('oil_ghg',en_code + '_ghg', '')] = exp_emm_p
    
    
    for en_code in ['gaz']: 
      value_smr = flows.loc['2020', (en_code + '_se', 'hyd_se', 'smr')].sum() 
      flows.loc['2020', (en_code + '_se', 'hyd_fe', 'ddd')] = value_smr 
      flows.loc['2020', (en_code + '_se', 'hyd_se', 'smr')] = 0
      flows.loc['2020', ('hyd_se', 'hyd_fe', '')] = 0
      
    filtered_flows = [
    ('imp', 'gaz_pe', ''),
    ('imp', 'pet_pe', ''),
    ('imp', 'elc_se', ''),
    ('imp', 'hyd_se', ''),
    ('imp', 'enc_pe', ''),
    ('imp', 'amm_fe', ''),
    ('imp', 'met_fe', ''),
    ('imp', 'cms_pe', '')]
    selected_imports = pd.DataFrame()
    for flow in filtered_flows:
     if flow in flows.columns:
        selected_imports["_".join([flow[0], flow[1]]).replace(" ", "_")] = flows[flow]
    local_production = [
    ('prod', 'gaz_pe', ''),
    ('prod', 'pet_pe', ''),
    ('prod', 'hdr_pe', ''),
    ('prod', 'eon_pe', ''),
    ('prod', 'eof_pe', ''),
    ('prod', 'enc_pe', ''),
    ('prod', 'spv_pe', ''),
    ('prod', 'pac_pe', ''),
    ('prod', 'cms_pe', ''),
    ('prod', 'ura_pe', ''),
    ('prod', 'bgl_pe', ''),]
    local_prod = pd.DataFrame()
    for flow in local_production:
     if flow in flows.columns:
        local_prod["_".join([flow[0], flow[1]]).replace(" ", "_")] = flows[flow]
    output_dir = f"results/{study}/country_csvs"
    os.makedirs(output_dir, exist_ok=True)
    selected_imports.to_csv(f"{output_dir}/total_imports_{country}.csv", index=True)
    local_prod.to_csv(f"{output_dir}/local_product_{country}.csv", index=True) 
    ## Storing energy flows, non-energy GHG values and other relevant values for each country
    tot_flows[country] = flows
    tot_ghg[country] = flows_ghg
    tot_co2[country] = flows_co2
    
    country_results = pd.DataFrame()
    tot_results = pd.concat([tot_results, country_results], axis=1)


 def generate_results(flows, tot_results, country):
    xls_file_name = snakemake.output.excelfile[countries.index(country)]
    xls_file_name = str(xls_file_name)
 
    file_handle = open(xls_file_name, 'wb')
    results_xls_writer = pd.ExcelWriter(file_handle, engine="openpyxl")

    if country in ALL_COUNTRIES:
        show_total = False
        country_list = COUNTRIES

    country_results = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Indicator','Sub_indicator']))
    global interval_time

    flows_bk =  flows.copy()
    
  
    selected_columns = flows_bk.columns.get_level_values('Source').isin(FE_NODES)
    # Sum the selected columns to calculate the FEC carrier
    fec_carrier = flows_bk.loc[:, selected_columns]
    grouped_fec = fec_carrier.groupby(level='Source', axis=1).sum()
    fec_carrier = grouped_fec
    selected_columns = flows_bk.columns.get_level_values('Target').isin(DS_NODES)

    # Create a new DataFrame containing only the selected columns
    fec_sector = flows_bk.loc[:, selected_columns]

    # Group the columns by 'Target' (nodes) and sum each group separately
    grouped_fec_sec = fec_sector.groupby(level='Target', axis=1).sum()
    fec_sector = grouped_fec_sec

      
    '''preparing data for local production area charts'''

    selected_columns_E = flows_bk.columns.get_level_values('Target').isin(EE_NODES)
    export_carrier = flows_bk.loc[:, selected_columns_E]
    grouped_export = export_carrier.groupby(level='Source', axis=1).sum()
    cov_exports = grouped_export
    selected_columns_I = flows_bk.columns.get_level_values('Source').isin(II_NODES)
    import_carrier = flows_bk.loc[:, selected_columns_I]
    grouped_import = import_carrier.groupby(level='Target', axis=1).sum()
    cov_imports = grouped_import
    
    impexp_carriers = list(set(cov_imports.columns.to_list() + cov_exports.columns.to_list())) # Carriers with imports and/or exports only
    # merged_carriers = pd.concat([grouped_export, grouped_import], axis=1).fillna(0)
    target_flows_list = ['elc_se', 'cms_pe', 'met_fe', 'hyd_se', 'gaz_pe', 'amm_fe', 'enc_pe', 'vap_se', 'pet_pe']
    ps_cons = pd.DataFrame()
    for target_flow in target_flows_list:
     flows_sum = flows.xs(target_flow, level='Target', axis=1, drop_level=True).sum(axis=1)
     ps_cons[target_flow] = flows_sum
    cov_ratios = 100 * ps_cons.subtract(cov_imports, fill_value=0).filter(impexp_carriers) / ps_cons.subtract(cov_exports, fill_value=0).filter(impexp_carriers)
    value_gaz = flows_bk[('gaz_pe', 'gaz_se', '')].squeeze().rename_axis(None)
    value_biogas = flows_bk[('bgl_pe', 'gaz_se', '')].squeeze().rename_axis(None)
    value_biogas_cc = flows_bk[('bgl_pe', 'gaz_se', 'cc')].squeeze().rename_axis(None)
    value_bl = flows_bk[('enc_pe', 'gaz_se', '')].squeeze().rename_axis(None)
    value_hy = flows_bk[('hyd_se', 'gaz_se', '')].squeeze().rename_axis(None)
    value_total = ((value_biogas + value_biogas_cc + value_bl + value_hy)/(value_gaz + value_biogas + value_biogas_cc + value_bl + value_hy))*100
    cov_ratios['gaz_se'] = value_total
    value_petr = flows_bk[('pet_pe', 'pet_fe', '')].squeeze().rename_axis(None)
    value_biml = flows_bk[('enc_pe', 'pet_fe', '')].squeeze().rename_axis(None)
    value_fish = flows_bk[('hyd_se', 'pet_fe', '')].squeeze().rename_axis(None)
    value_toltal = ((value_biml + value_fish)/(value_petr + value_biml + value_fish))*100
    cov_ratios['pet_fe'] = value_toltal
    cov_ratios = cov_ratios.clip(upper=100)
    
    '''preparing data for renewable share in each energy vector'''
    ren_cov_ratios=pd.DataFrame()
    gfec_breakdown=pd.DataFrame()
    flows_from_node_cum=pd.DataFrame()
    selected_columns_cr = flows_bk.columns.get_level_values('Source').isin(PE_NODES)
    flows_from_node = flows_bk.loc[:, selected_columns_cr]
    flows_from_node_cum = pd.concat([flows_from_node_cum,flows_from_node], axis=1).groupby(axis=1,level=[0,1,2]).sum()
    flows_from_node = flows_from_node.groupby(level='Source', axis=1).sum()
    selected_columns_de = flows_bk.columns.get_level_values('Source').isin(FE_NODES)
    flows_to_node = flows_bk.loc[:, selected_columns_de]
    flows_to_node = flows_to_node.groupby(level='Source', axis=1).sum()
    ren_columns = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe', 'pac_pe','bgl_pe']
    fos_columns = ['cms_pe', 'gaz_pe', 'pet_pe']
    nuk_columns = ['ura_pe']

    ren_sum = flows_from_node[ren_columns].sum(axis=1)
    fos_sum = flows_from_node[fos_columns].sum(axis=1)
    nuk_sum = flows_from_node[nuk_columns].sum(axis=1)

    flows_from_node['ren'] = ren_sum.clip(lower=0)
    flows_from_node['fos'] = fos_sum.clip(lower=0)
    flows_from_node['nuk'] = nuk_sum.clip(lower=0)

    flows_from_node_t = flows_from_node.drop(columns=ren_columns + fos_columns + nuk_columns)
    gfec_breakdown = flows_from_node_t.loc[:, (flows_from_node_t != 0).any()]
    
    tot_columns = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe', 'pac_pe','cms_pe', 'gaz_pe', 'pet_pe','ura_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in tot_columns and col[1] == 'elc_se']
    result_elc = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_elc_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_elc = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe']
    ren_elc = result_elc[ren_elc].sum(axis=1)
    ren_cov_ratios = pd.DataFrame()
    ren_cov_ratios['elc_fe'] = (ren_elc/result_elc_t)*100
    ren_cov_ratios['elc_fe'] = ren_cov_ratios['elc_fe']
    
    gas_columns = ['bgl_pe','enc_pe', 'gaz_pe','hyd_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in gas_columns and col[1] == 'gaz_se']
    result_gas = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_gas_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_gas = ['bgl_pe','enc_pe', 'hyd_se']
    ren_gas = result_gas[ren_gas].sum(axis=1)
    ren_cov_ratios['gaz_fe'] = (ren_gas/result_gas_t)*100
    ren_cov_ratios['gaz_fe'] = ren_cov_ratios['gaz_fe']
    
    pet_columns = ['pet_pe','enc_pe','hyd_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in pet_columns and col[1] in ['pet_fe', 'lqf_se']]
    result_pet = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_pet_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_pet = ['enc_pe','hyd_se']
    ren_pet = result_pet[ren_pet].sum(axis=1)
    ren_cov_ratios['pet_fe'] = (ren_pet/result_pet_t)*100
    ren_cov_ratios['pet_fe'] = ren_cov_ratios['pet_fe']
    
    hyd_columns = ['elc_se','gaz_se','imp']
    filtered_columns = [col for col in flows_bk.columns if col[0] in hyd_columns and col[1] == 'hyd_se']
    result_hyd = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_hyd_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_hyd = ['elc_se','imp']
    ren_hyd = result_hyd[ren_hyd].sum(axis=1)
    ren_cov_ratios['hyd_fe'] = (ren_hyd/result_hyd_t)*100
    ren_cov_ratios['hyd_fe'] = ren_cov_ratios['hyd_fe']
    
    bm_columns = ['enc_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in bm_columns and col[1] in ['gaz_se', 'lqf_se', 'elc_se','vap_se','enc_fe']]
    result_bm = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_bm_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_bm = ['enc_pe']
    ren_bm = result_bm[ren_bm].sum(axis=1)
    ren_cov_ratios['enc_fe'] = (ren_bm/result_bm_t)*100
    ren_cov_ratios['enc_fe'] = ren_cov_ratios['enc_fe']
    
    dh_columns = ['enc_pe','cms_pe','pet_pe','gaz_se','hyd_se','bgl_pe','elc_se','pac_pe','tes_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in dh_columns and col[1] in ['vap_se']]
    result_dh = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_dh_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_dh = ['enc_pe','bgl_pe','elc_se','pac_pe', 'hyd_se', 'tes_se']
    ren_dh = result_dh[ren_dh].sum(axis=1)
    ren_cov_ratios['vap_fe'] = (ren_dh/result_dh_t)*100
    ren_cov_ratios['vap_fe'] = ren_cov_ratios['vap_fe']
    
    am_columns = ['pac_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in am_columns and col[1] in ['vap_se','pac_fe']]
    result_am = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_am_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_am = ['pac_pe']
    ren_am = result_am[ren_am].sum(axis=1)
    ren_cov_ratios['pac_fe'] = (ren_am/result_am_t)*100
    ren_cov_ratios['pac_fe'] = ren_cov_ratios['pac_fe']
    
    nh_columns = ['elc_se','hyd_se','imp']
    filtered_columns = [col for col in flows_bk.columns if col[0] in nh_columns and col[1] in ['amm_fe']]
    result_nh = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_nh_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_nh = ['elc_se','hyd_se','imp']
    ren_nh = result_nh[ren_nh].sum(axis=1)
    ren_cov_ratios['amm_fe'] = (ren_nh/result_nh_t)*100
    ren_cov_ratios['amm_fe'] = ren_cov_ratios['amm_fe']
    
    me_columns = ['elc_se','hyd_se','imp']
    filtered_columns = [col for col in flows_bk.columns if col[0] in me_columns and col[1] in ['met_fe']]
    result_me = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_me_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_me = ['elc_se','hyd_se','imp']
    ren_me = result_me[ren_me].sum(axis=1)
    ren_cov_ratios['met_fe'] = (ren_me/result_me_t)*100
    ren_cov_ratios['met_fe'] = ren_cov_ratios['met_fe']
    
    gfec_breakdown_pct = sf.share_percent(gfec_breakdown,100)
    ren_cov_ratios['total'] = gfec_breakdown_pct['ren']
    ren_cov_ratios = ren_cov_ratios.clip(upper=100)
    
    # interval_time = sf.calc_time('Graph analysis (consumption breakdown)', interval_time)

    '''Preparing data foe emission charts'''
    flows_co2 = tot_co2[country]
    ghg_sector = tot_ghg[country]
    ghg_sector = ghg_sector.groupby(level='Source', axis=1).sum()
    ghg_sector['lufnes_ghg'] = -ghg_sector['lufnes_ghg']
    ghg_sector['blg_ghg'] = -ghg_sector['blg_ghg']
    ghg_sector['dac_ghg'] = -ghg_sector['dac_ghg']
    ghg_sector['blq_ghg'] = -ghg_sector['blq_ghg']
    ghg_sector['bmc_ghg'] = -ghg_sector['bmc_ghg']
    ghg_source = tot_ghg[country].groupby(level='Target', axis=1).sum()
    ghg_source['lufnes_ghg'] = -ghg_source['lufnes_ghg']
    ghg_source['blg_ghg'] = -ghg_source['blg_ghg']
    ghg_source['dac_ghg'] = -ghg_source['dac_ghg']
    ghg_source['blq_ghg'] = -ghg_source['blq_ghg']
    ghg_source['bmc_ghg'] = -ghg_source['bmc_ghg']
    
    ghg_sector_cum = ghg_sector.copy()
    ghg_sector_cum.loc['2030'] *= 10
    ghg_sector_cum.loc['2040'] *= 10
    ghg_sector_cum.loc['2050'] *= 10
    #multiplying by 10 for cumulative emissions
    ghg_source_cum = ghg_source.copy()
    ghg_source_cum.loc['2030'] *= 10
    ghg_source_cum.loc['2040'] *= 10
    ghg_source_cum.loc['2050'] *= 10
    
    ghg_sector_cum.to_csv(f'results/{study}/country_csvs/ghg_sector_cum_{country}.csv', index=True)
    
    ## Start HTML output
    html_items_emissions = {}
    html_items_sankeys = {}
    html_items_FEC = {}
    
    #load html flag file
    with open(snakemake.input.plots_html, 'r') as file:
     plots = yaml.safe_load(file)
    sepia_plots = plots.get("Sepia_plots", {})
    id_section = -1
    sections = [('ghg','CO2 emissions by sector'),('ghg','CO2 emissions by source'),('ghg','Cumulative CO2 emissions by sector'),('ghg','Cumulative CO2 emissions by source'),('sankey','Sankey diagram'),('carbon sankey','Carbon Sankey diagram'),('res','Renewable energy share'),('res','Final energy consumption by origin'),('carrier','Share of domestic production'),('cons','Final energy consumption by each sector'),('cons','Mix of secondary energies'),('cons','Final energy consumption by carrier for each sector')]
    sections = [(category, label) for category, label in sections if sepia_plots.get(label, False)]
    
    if MAIN_PARAMS['HTML_TEMPLATE'] == "raw": sections += [('input','Input data')]
    html_items_emissions['TABLE_OF_CONTENTS'] = '<ol>'
    html_items_sankeys['TABLE_OF_CONTENTS'] = '<ol>'
    html_items_FEC['TABLE_OF_CONTENTS'] = '<ol>'
    
    sections_emissions = [(anchor, title) for anchor, title in sections if "CO2" in title or "emissions" in title]
    sections_sankeys = [(anchor, title) for anchor, title in sections if "Sankey" in title]
    sections_FEC = [(anchor, title) for anchor, title in sections if "Final energy consumption" in title or "carrier" in title]
    for (anchor, title) in sections_emissions:
     html_items_emissions['TABLE_OF_CONTENTS'] += "<li><a href='#" + anchor + "'>" + title + "</a></li>"
    html_items_emissions['TABLE_OF_CONTENTS'] += "</ol>"
    html_items_emissions['MAIN'] = ''

    for (anchor, title) in sections_sankeys:
     html_items_sankeys['TABLE_OF_CONTENTS'] += "<li><a href='#" + anchor + "'>" + title + "</a></li>"
    html_items_sankeys['TABLE_OF_CONTENTS'] += "</ol>"
    html_items_sankeys['MAIN'] = ''

    for (anchor, title) in sections_FEC:
     html_items_FEC['TABLE_OF_CONTENTS'] += "<li><a href='#" + anchor + "'>" + title + "</a></li>"
    html_items_FEC['TABLE_OF_CONTENTS'] += "</ol>"
    html_items_FEC['MAIN'] = ''
    
    #load HTML Texts
    def load_html_texts(file_path):
     html_texts = {}
     with open(file_path, 'r') as file:
        for line in file:
            # Split by ": " to get the key and the HTML content
            if ": " in line:
                key, html_content = line.split(": ", 1)
                html_texts[key.strip()] = html_content.strip()
     return html_texts
 
    def get_country_specific_desc(country, html_texts):
    # Extract descriptions specific to the selected country
     country_desc = {}
     if country == 'EU':
        country_desc = {key: content for key, content in html_texts.items() if key.startswith("EU_")}
     elif country == 'BE':
        country_desc = {key: content for key, content in html_texts.items() if key.startswith("BE_")}
     else:
        # For all other countries, load the generic descriptors
        country_desc = {key: content for key, content in html_texts.items() if not key.startswith(("EU_", "BE_"))}

     return country_desc

    # Load the HTML texts from file
    html_texts = load_html_texts(file_path)
    country_desc = get_country_specific_desc(country, html_texts)
    # GHG
    if sepia_plots["CO2 emissions by sector"] == True:
     id_section += 1
     html_items_emissions['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_emissions['MAIN'] += country_desc.get(f"{country}_CO2_emissions_sector", '')
     html_items_emissions['MAIN'] += saf.combine_charts([('',ghg_sector)], MAIN_PARAMS, NODES,'', 'ghgchart',  results_xls_writer, 'MtCO<sub>2</sub>eq') #('by sect. - power & heat dispatched',ghg_sector_2),
    else:
     saf.combine_charts([('',ghg_sector)], MAIN_PARAMS, NODES,'', 'ghgchart',  results_xls_writer, 'MtCO<sub>2</sub>eq')
    if sepia_plots["CO2 emissions by source"] == True:
     id_section += 1
     html_items_emissions['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_emissions['MAIN'] += country_desc.get(f"{country}_CO2_emissions_source", '')
     html_items_emissions['MAIN'] += saf.combine_charts([('',ghg_source)], MAIN_PARAMS, NODES,'', 'ghgchart', results_xls_writer, 'MtCO<sub>2</sub>eq')
    else:
     saf.combine_charts([('',ghg_source)], MAIN_PARAMS, NODES,'', 'ghgchart', results_xls_writer, 'MtCO<sub>2</sub>eq')
    if sepia_plots["Cumulative CO2 emissions by sector"] == True:
     id_section += 1
     html_items_emissions['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_emissions['MAIN'] += country_desc.get(f"{country}_Cumulative_CO2_emissions_sector", '')
     html_items_emissions['MAIN'] += saf.combine_charts([('',saf.cumul(ghg_sector_cum, 2020))], MAIN_PARAMS, NODES,'', 'ghgchart',  results_xls_writer, 'MtCO<sub>2</sub>eq')
    else:
     saf.combine_charts([('',saf.cumul(ghg_sector_cum, 2020))], MAIN_PARAMS, NODES,'', 'ghgchart',  results_xls_writer, 'MtCO<sub>2</sub>eq')
    if sepia_plots["Cumulative CO2 emissions by source"] == True:
     id_section += 1
     html_items_emissions['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_emissions['MAIN'] += country_desc.get(f"{country}_Cumulative_CO2_emissions_source", '')
     html_items_emissions['MAIN'] += saf.combine_charts([('',saf.cumul(ghg_source_cum,2020))], MAIN_PARAMS, NODES, '','ghgchart', results_xls_writer, 'MtCO<sub>2</sub>')
    else:
     saf.combine_charts([('',saf.cumul(ghg_source_cum,2020))], MAIN_PARAMS, NODES, '','ghgchart', results_xls_writer, 'MtCO<sub>2</sub>')
    # Sankeys
    if sepia_plots["Sankey diagram"] == True:
     id_section += 1
     html_items_sankeys['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_sankeys['MAIN'] += country_desc.get(f"{country}_sankey_diagram", '')
     html_items_sankeys['MAIN'] += saf.combine_charts([('Sankey diagram',flows)], MAIN_PARAMS, NODES, '', 'sankey', sk_proc=PROCESSES) #('upstream flows from final energies',flows_from_node_cum),('Sankey diagram without import mix',flows)
    if sepia_plots["Carbon Sankey diagram"] == True:
     id_section += 1
     html_items_sankeys['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_sankeys['MAIN'] += country_desc.get(f"{country}_carbon_sankey_diagram", '')
     html_items_sankeys['MAIN'] += saf.combine_charts([('Carbon Sankey diagram',flows_co2)], MAIN_PARAMS, NODES, '', 'carbon sankey', sk_proc=PROCESSES_2) #('upstream flows from final energies',flows_from_node_cum),('Sankey diagram without import mix',flows)
    # RES share
    if sepia_plots["Renewable energy share"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_RES_share", '')
     html_items_FEC['MAIN'] += sf.chart_to_output(sf.create_node_chart(ren_cov_ratios, NODES, MAIN_PARAMS, 'linechart', '', results_xls_writer, '%'))
    else:
     sf.chart_to_output(sf.create_node_chart(ren_cov_ratios, NODES, MAIN_PARAMS, 'linechart', '', results_xls_writer, '%'))
    # html_items['MAIN'] += '<p>Renewable shares per final energies are calculated by analysing all energy flows going through different transformation processes (electricity and heat production processes, power-to-gas etc.) as described by the Sankey diagram. An algorithm goes upstream through this complex energy system, from a given final energy to all relevant primary energies, and determines their respective shares. For example, a renewable share of 50% for final electricity means that 50% of the electricity consumed has been produced by renewable means, either directly from renewable power technologies such as wind of PV, or indirectly - for example if gas cogeneration has been used with a share of renewables in the gas mix.'
    # if MAIN_PARAMS['USE_IMPORT_MIX'] and not show_total: html_items['MAIN'] += ' NB: the mix of imported secondary energy carriers (power, gas...) is calculated from exports of other EU countries, and may thus contain some level of renewables, included in this calculation as well.'
    if sepia_plots["Final energy consumption by origin"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_FEC_origin", '')
     if show_total: html_items_FEC['MAIN'] += sf.chart_to_output(sf.create_map(tot_results[('ren_cov_ratio','total')], country_list, 'RES share in final consumption', MAIN_PARAMS, unit='%', min_scale=0,max_scale=100))
     html_items_FEC['MAIN'] += sf.chart_to_output(sf.create_node_chart(gfec_breakdown, NODES, MAIN_PARAMS, 'area', '', results_xls_writer))
    else:
     sf.chart_to_output(sf.create_node_chart(gfec_breakdown, NODES, MAIN_PARAMS, 'area', '', results_xls_writer))
    # Energy carrier share balance
    if sepia_plots["Share of domestic production"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_Domestic_production_share", '')
     html_items_FEC['MAIN'] += sf.chart_to_output(sf.create_node_chart(cov_ratios, NODES, MAIN_PARAMS, 'linechart', 'Local production coverage ratios', results_xls_writer, unit='%'))
    # html_items['MAIN'] += '<p>Local production coverage ratios are simply defined as the ratio between the local production of a given energy carrier, and its local consumption (including final and non final uses). A ratio above 100% thus means that the country is more than self-sufficient (net exporter), while a ratio below 100% means that the country is a net importer.</p>'
     if show_total:
        node_list = tot_results['cov_ratio'].columns.unique(level='Sub_indicator').to_list()
        sf.put_item_in_front(node_list, 'total') # We put total first
        combinations = [(NODES.loc[node,'Label'],tot_results[('cov_ratio',node)]) for node in node_list]
        html_items_FEC['MAIN'] += saf.combine_charts(combinations, MAIN_PARAMS, country_list, 'Local prod coverage ratios -', 'map', results_xls_writer, '%', min_scale=0, mid_scale=50)
    
    combinations = [('All energies',fec_sector)]
    grouped_flows = flows.T.groupby(['Source', 'Target', 'Type']).sum().T
    for energy in FE_NODES:
        combinations += [(NODES.loc[energy,'Label'], sf.node_consumption(grouped_flows, energy, direction='forward', splitby='target'))]
    if sepia_plots["Final energy consumption by each sector"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_FEC_sector", '')
     html_items_FEC['MAIN'] += saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by sector -', 'areachart', results_xls_writer)
    else:
     saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by sector -', 'areachart', results_xls_writer)
    combinations = []
    for energy in SE_NODES:
        df = sf.node_consumption(grouped_flows, energy, direction='backward', splitby='source')
        combinations += [(NODES.loc[energy,'Label'], df)]
        country_results = sf.add_indicator_to_results(country_results, df, 'sec_mix.'+energy, False)
    if sepia_plots["Mix of secondary energies"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_Grid_carrier_contribution", '')
     html_items_FEC['MAIN'] += saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Mix of secondary energies -', 'areachart', results_xls_writer)
    else:
     saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Mix of secondary energies -', 'areachart', results_xls_writer)
    # html_items['MAIN'] += '<p>The above chart describes the contribution of misc. technologies (and possibly imports) to the production of a given secondary energy carrier.</p>'
    combinations = []
    combinations = [('All sectors',fec_carrier)]
    for sector in DS_NODES:
       df = sf.node_consumption(grouped_flows, sector, direction='backwards', splitby='target')
       combinations += [(NODES.loc[sector,'Label'], df)]
       country_results = sf.add_indicator_to_results(country_results, df, 'fec.'+sector)
     # Energy consumption
    if sepia_plots["Final energy consumption by carrier for each sector"] == True:
     id_section += 1
     html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
     html_items_FEC['MAIN'] += country_desc.get(f"{country}_FEC_carrier", '')
     html_items_FEC['MAIN'] += saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by carrier -', 'areachart', results_xls_writer)
    else:
     saf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by carrier -', 'areachart', results_xls_writer)

    # Indicator calculation for inter-territorial analysis
    # Multidimensionnal indicators not already defined above (some indicators have several nomenclatures: by energy / sector / origin etc.)
    for (indicator,df) in [('fec',fec_carrier),('fec',fec_sector),('cov_ratio',cov_ratios),('ren_cov_ratio',ren_cov_ratios),('ghg_sector',ghg_sector),('ghg_source',ghg_source)]:
        country_results = sf.add_indicator_to_results(country_results, df, indicator)
    
    for indicator in ['fec','ghg_sector']:
        country_results[(indicator,'reduc')] = sf.reduction_rate(country_results[(indicator,'total')],100)
    
    ## List of input files
    if MAIN_PARAMS['HTML_TEMPLATE'] == "raw":
        id_section += 1
        html_items_emissions['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
        html_items_sankeys['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
        html_items_FEC['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
        
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %Hh%M")
    html_items_emissions['MAIN'] += f'<p style="text-align:right;font-size:small;">SEPIA v{__version__} @ {current_time}</p>'
    html_items_sankeys['MAIN'] += f'<p style="text-align:right;font-size:small;">SEPIA v{__version__} @ {current_time}</p>'
    html_items_FEC['MAIN'] += f'<p style="text-align:right;font-size:small;">SEPIA v{__version__} @ {current_time}</p>'
    
    ## Writing HTML file
    # for country in ALL_COUNTRIES:
    template = snakemake.input.template
    with open(template) as f:
        template_content = f.read()
    html_output_emissions = template_content  # Create a copy
    for label in html_items_emissions:
     html_output_emissions = html_output_emissions.replace('{{'+label+'}}', html_items_emissions[label])
    emissions_filename = snakemake.output.htmlfile_emissions[ALL_COUNTRIES.index(country)]
    emissions_filename = str(emissions_filename)
    with open(emissions_filename, 'w') as f:
     f.write(html_output_emissions)
    html_output_sankeys = template_content  # Create a fresh copy
    for label in html_items_sankeys:
     html_output_sankeys = html_output_sankeys.replace('{{'+label+'}}', html_items_sankeys[label])
    sankeys_filename = snakemake.output.htmlfile_sankeys[ALL_COUNTRIES.index(country)]
    sankeys_filename = str(sankeys_filename)
    with open(sankeys_filename, 'w') as f:
     f.write(html_output_sankeys)
    html_output_FEC = template_content  # Create a fresh copy
    for label in html_items_FEC:
     html_output_FEC = html_output_FEC.replace('{{'+label+'}}', html_items_FEC[label])
    FEC_filename = snakemake.output.htmlfile_fec[ALL_COUNTRIES.index(country)]
    FEC_filename = str(FEC_filename)
    with open(FEC_filename, 'w') as f:
     f.write(html_output_FEC)
    
    
    # Table of content for ChartData
    toc = results_xls_writer.book.create_sheet('TOC')
    for (i,sheet) in enumerate(results_xls_writer.sheets):
        if sheet != "TOC":
            toc['A'+str(i+1)].value = sheet
            toc['B'+str(i+1)].hyperlink = "#'"+sheet+"'!A1"
            toc['B'+str(i+1)].value = results_xls_writer.sheets[sheet]['A1'].value
            toc['B'+str(i+1)].style = 'Hyperlink'
            back_to_toc = results_xls_writer.sheets[sheet]['A2']
            back_to_toc.hyperlink = "#TOC!A1"
            back_to_toc.value = 'Back to table of contents'
            back_to_toc.style = 'Hyperlink'
            disclaimer = 'Results shown below are not to be disseminated to third parties' if MAIN_PARAMS['DRAFT'] else 'This work is licensed under a Creative Commons Attribution 4.0 International License.'
            disclaimer += f' SEPIA v{__version__} @ {current_time}'
            results_xls_writer.sheets[sheet]['D2'] = disclaimer
    results_xls_writer.book.active = toc
    results_xls_writer.close()
    return tot_results

 
 for country in ALL_COUNTRIES:
    tot_results = generate_results(tot_flows[country], tot_results, country)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "generate_sepia")

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    countries = snakemake.params.countries
    study = snakemake.params.study
    file_path = snakemake.input.file_path
    df = biomass_potentials()
    prepare_sepia(countries)

