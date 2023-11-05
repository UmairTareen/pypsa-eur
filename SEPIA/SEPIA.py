# -*- coding: utf-8 -*-
"""
SEPIA - Simplified Energy Prospective and Interterritorial Analysis tool
"""
__author__ = "Adrien Jacob, négaWatt"
__copyright__ = "Copyright 2022, négaWatt"
__license__ = "GPL"
__version__ = "1.8"
__email__ = "adrien.jacob@negawatt.org"

import SEPIA_functions as sf # Custom functions

import pandas as pd # Read/analyse data
import numpy as np # for np.inf
import re # Regex
import os # File system
import time # For performance measurement
import datetime # For current time
import warnings # to manage user warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl') # Disable warning from openpyxl

start_time = interval_time = time.time()

DIRNAME = os.path.dirname(__file__)

## Import country data
file = pd.ExcelFile(os.path.join(DIRNAME, r'COUNTRIES.xlsx'), engine="openpyxl")
COUNTRIES = pd.read_excel(file, "COUNTRIES", index_col=0)
COUNTRIES = COUNTRIES[COUNTRIES['Input_File'].notna()]
# EU_COUNTRIES = COUNTRIES[COUNTRIES['EU_member']].index.to_list()
ALL_COUNTRIES = COUNTRIES.index.to_list()
country_groups = {'ALL':ALL_COUNTRIES}
# if len(EU_COUNTRIES)>0: country_groups['EU'] = EU_COUNTRIES

## Import config data (nodes, processes, general settings etc.)
file = pd.ExcelFile(os.path.join(DIRNAME, r'SEPIA_config.xlsx'), engine="openpyxl")
CONFIG = pd.read_excel(file, ["MAIN_PARAMS","NODES","PROCESSES","PROCESSES_2","PROCESSES_3","IMPORT_MIX","INDICATORS"], index_col=0)

# Simulation data:
datafile = os.path.join(DIRNAME, "../results/sepia/inputs.xlsx")

# Main settings (cf. SEPIA_config for description of all setting constants)
MAIN_PARAMS = CONFIG["MAIN_PARAMS"].drop('Description',axis=1).to_dict()['Value']

NODES = CONFIG["NODES"]
FE_NODES = sf.nodes_by_type(NODES,'FINAL_ENERGIES')
SE_NODES = sf.nodes_by_type(NODES,'SECONDARY_ENERGIES')
PE_NODES = sf.nodes_by_type(NODES,'PRIMARY_ENERGIES')
DS_NODES = sf.nodes_by_type(NODES,'DEMAND_SECTORS')
SI_NODES = sf.nodes_by_type(NODES,'SECONDARY_IMPORTS')
II_NODES = sf.nodes_by_type(NODES,'IMPORTS')
EE_NODES = sf.nodes_by_type(NODES,'EXPORTS')
GHG_ENERGIES = NODES[NODES['Emission_Factor'] > 0]
GHG_SECTORS = sf.nodes_by_type(NODES,'GHG_SECTORS')

PROCESSES = CONFIG["PROCESSES"].reset_index()
PROCESSES['Type'].fillna('', inplace=True)
PROCESSES_2 = CONFIG["PROCESSES_2"].reset_index()
PROCESSES_2['Type'].fillna('', inplace=True)
PROCESSES_3 = CONFIG["PROCESSES_3"].reset_index()
PROCESSES_3['Type'].fillna('', inplace=True)

IMPORT_MIX = CONFIG["IMPORT_MIX"].set_index('Category', append=True).T.interpolate(limit_direction='backward')

INDICATORS = CONFIG["INDICATORS"]

CATEGORIES = ['ren','fos','nuk']

# Aggregated results per Country
tot_results = pd.DataFrame()
tot_debug = pd.DataFrame()
# Aggregated imports and exports per network / category (same structure as IMPORT_MIX)
tot_se_export_mix = pd.DataFrame(index=IMPORT_MIX.index, columns=pd.MultiIndex.from_product([SE_NODES,CATEGORIES], names=['Network','Category']))
tot_se_import_mix = tot_se_export_mix.copy()
# Dictionnaries storing results of the next section : "Country" => Value
tot_flows = {} # Energy flow DataFrames

# Energy system (network graph) creation for all countries
print("\nEnergy system (network graph) creation\n")

for country in ALL_COUNTRIES:
    country_debug = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['Indicator','Sub_indicator','Country']))
    print("||| "+COUNTRIES.loc[country,'Label']+" |||")
    ##Import country data
    country_input_file = COUNTRIES.loc[country,'Input_File']+'.xlsx'
    #file = pd.ExcelFile(os.path.join(DIRNAME,'Inputs',country_input_file), engine="openpyxl")
    data = pd.read_excel(datafile, sheet_name="Inputs", index_col=0, usecols="C:F")
    data.reset_index(drop=True, inplace=False)
    data=data.T
    
    data_co2 = pd.read_excel(datafile, sheet_name="Inputs_co2", index_col=0, usecols="C:F")
    data_co2.reset_index(drop=True, inplace=False)
    data_co2=data_co2.T
    # data = data.rename_axis('Year')
    # country_params = pd.read_excel(file, "Parameters", index_col=0, usecols="G,H")

    interval_time = sf.calc_time('Excel file reading', interval_time)

#     ## Data cleanup
#     for sheet in data:
#         data[sheet] = sf.db_cleanup(data[sheet])
    # data = pd.concat(data.values(), axis=1) # input dataframe
    data = data.loc[:,~data.columns.duplicated()] # Remove duplicate indicators
#     country_params = sf.db_cleanup(country_params, False)['Value'].to_dict()
#     r = re.compile(r'mrtrch\d+')
    # if MAIN_PARAMS['HEAT_MERIT_ORDER']:
    #     # heat_prod_processes = ([country_params[key] for key in list(filter(r.match, country_params.keys()))])
    #     for process in PROCESSES['Input_Label'][~PROCESSES['Input_Label'] * ~PROCESSES['Input_Label'].isnull()]:
    #         print("! Warning: {} not found in heat processes.".format(process))
    # else:
    #     heat_prod_processes = PROCESSES['Input_Label'][~PROCESSES['Input_Label'].isnull()]
    # pcs_pci = country_params['pcspci']
    
    # Consistency check : are all indicators found in input data?
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
    # unfound_inputs_ghg = []
    # unfound_inputs_ghg.extend(sf.unfound_indicators(data_ghg,PROCESSES_3,'Value_Code'))
    # if len(unfound_inputs_ghg)>0:
    #     data_ghg = data_ghg.reindex(columns=[*data_ghg.columns.tolist(), *unfound_inputs_ghg], fill_value=0)
    #     print("! Warning: the following indicators have not been found (they have been filled with 0): "+", ".join(unfound_inputs)+" !!!")
    # If electricity production is available we use it, otherwise we try to calculate from capacity & load factor
    # for en_code in ['enm','eon','eof','spv','ght']:
    #     data_filter = data['pro'+en_code].eq(0) & data['cai'+en_code].notnull() & data['fch'+en_code].notnull()
    #     data.loc[data_filter,'pro'+en_code] = data['cai'+en_code] * data['fch'+en_code] * 8.760
    # We use total elec losses if defined, otherwise we use disagragated values (transmission + distribution + storage if available)
    # data.loc[data['perreselc'] == 0, 'perreselc'] = data['pertraelc'] + data['perdiselc'] + data.get('perstorelc',0)
    # # We group fatal heat from wastewater & industry
    # data.loc[data['prbrchfat'] == 0, 'prbrchfat'] = data['prbrchepu'] + data['prbrchind']
    # # Converting max volumetric share of H2 into max energy share
    # data['pchydirg'] = data['pchydirg'] *119930*0.08988/(50020*0.6512)
    # # Default renewable share in waste is 50%
    # data.loc[data['pcenrwst'] == 0, 'pcenrwst'] = 0.5

    ## Creating flows and efficiencies DataFrames and filling values which do not require calculation, directly from input data
    proc_without_calc = PROCESSES[PROCESSES['Value_Code'].isin(data.columns)] # indicator is not empty and found in data
    flows = pd.DataFrame(data[proc_without_calc.Value_Code].values, index=data.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc.Source, proc_without_calc.Target, proc_without_calc.Type)), names=('Source','Target','Type')))
    proc_without_calc_co2 = PROCESSES_2[PROCESSES_2['Value_Code'].isin(data_co2.columns)] # indicator is not empty and found in data
    flows_co2 = pd.DataFrame(data_co2[proc_without_calc_co2.Value_Code].values, index=data_co2.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc_co2.Source, proc_without_calc_co2.Target, proc_without_calc_co2.Type)), names=('Source','Target','Type')))
    proc_without_calc_ghg = PROCESSES_3[PROCESSES_3['Value_Code'].isin(data_ghg.columns)] # indicator is not empty and found in data
    flows_ghg = pd.DataFrame(data_ghg[proc_without_calc_ghg.Value_Code].values, index=data_ghg.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_without_calc_ghg.Source, proc_without_calc_ghg.Target, proc_without_calc_ghg.Type)), names=('Source','Target','Type')))
#     proc_with_eff = PROCESSES[PROCESSES['Efficiency_Code'].isin(data.columns)]
#     efficiencies = pd.DataFrame(data[proc_with_eff.Efficiency_Code].values, index=data.index, columns=pd.MultiIndex.from_tuples(list(zip(proc_with_eff.Source, proc_with_eff.Target, proc_with_eff.Type)), names=('Source','Target','Type')))

    # Net final energy consumption
    selected_columns = flows.columns.get_level_values('Source').isin(FE_NODES)
    fec_carrier = flows.loc[:, selected_columns]
    grouped_fec = fec_carrier.groupby(level='Source', axis=1).sum()
    fec = grouped_fec
    for en_code in ['vap','elc','gaz','hyd']:
        flows[(en_code+'_se',en_code+'_fe','')] = fec[en_code+'_fe']
        
    selected_columns_pe = flows.columns.get_level_values('Source').isin(FE_NODES)
    fec_carrier_pe = flows.loc[:, selected_columns_pe]
    grouped_fec_pe = fec_carrier_pe.groupby(level='Source', axis=1).sum()
    fec_pe = grouped_fec_pe
    for en_code in ['pac','enc']:
        flows[(en_code+'_pe',en_code+'_fe','')] = fec_pe[en_code+'_fe']
    fischer_tropsch_p = flows['hyd_se','pet_fe']
    biomass_liquid_p = flows['blq_pe','pet_fe']
    value = fischer_tropsch_p + biomass_liquid_p.sum()
    for en_code in ['pet']:
        flows[(en_code+'_pe',en_code+'_fe','')] = fec_pe[en_code+'_fe']-value
        
    selected_columns_se = flows.columns.get_level_values('Source').isin(SE_NODES)
    fec_carrier_se = flows.loc[:, selected_columns_se]
    grouped_fec_se = fec_carrier_se.groupby(level='Source', axis=1).sum()
    fec_se = grouped_fec_se
    for en_code in ['gaz']:
        flows[(en_code+'_pe',en_code+'_se','')] = fec_se[en_code+'_se']
        
    # ## Direct transfer of primary energies to their corresponding networks
    # for (en_code,network) in [('gaz','gaz'),('blq','lqf')]:
    #     flows[(en_code+'_pe',network+'_se','')] = flows[('prod',en_code+'_pe','')]

    # ## Heat : applying merit order, and downsizing heat output if needed
    # heat_dem = fec['vap_fe'] + flows[('vap_se','per')]
    # for heat_proc_label in heat_prod_processes:
    #     if heat_proc_label not in PROCESSES['Input_Label'].values:
    #         print("! Warning: "+heat_proc_label+" not found in config file")
    #         continue
    #     heat_proc = PROCESSES.query("Input_Label == '" + heat_proc_label +"'").iloc[0]  # using iloc to get the first found item in PROCESSES corresponding to "heat_proc_label"
    #     heat_source = heat_proc.Source
    #     heat_proc_type = heat_proc.Type
    #     heat_proc = (heat_source,'vap_se',heat_proc_type)
    #     productible = flows[heat_proc].copy()
        # if MAIN_PARAMS['HEAT_MERIT_ORDER']: flows[heat_proc] = pd.concat([productible,heat_dem], axis=1).min(axis=1) # Minimum of productible and heat demand
        # heat_dem -= flows[heat_proc]
#         usage_ratio = (flows[heat_proc] / productible).fillna(1)
#         if usage_ratio[usage_ratio<1].count() > 0: country_debug[('usage_ratio',heat_proc_label,country)] = usage_ratio * 100
#         if heat_proc in efficiencies.columns: # Energy involving transformation losses
#             loss_proc = (heat_source,'per',heat_proc_type)
#             if heat_proc_type == 'chp':
#                 power_proc = (heat_source,'elc_se',heat_proc_type)
#                 if MAIN_PARAMS['HEAT_MERIT_ORDER']: flows[power_proc] *= usage_ratio # Correcting power output from real heat produced
#                 primary_cons = flows[power_proc] / efficiencies[power_proc]
#                 flows[loss_proc] = primary_cons - flows[heat_proc] - flows[power_proc]
#             else:
#                 primary_cons = flows[heat_proc] / efficiencies[heat_proc]
#                 if heat_source != 'pac_pe':
#                     flows[loss_proc] = primary_cons - flows[heat_proc]
#                 else:
#                     # For centralised solar thermal, we replace part of the pumped heat by electricity (according to pump efficiency)
#                     flows[('elc_se','vap_se',heat_proc_type)] = primary_cons
#                     flows[heat_proc] -= primary_cons
#             if loss_proc in flows.columns and (flows[loss_proc] < 0).any():
#                 print("! Warning: total efficiency above 100% for process '{}'. Assuming 100% efficiency instead.".format(heat_proc_label))
#             if (primary_cons == np.inf).any():
#                 print("! Warning: null efficiencies with non null prod values for process '{}'.".format(heat_proc_label))

#     sf.balance_node(flows,'vap_se','def','exc')
#     if flows[('def','vap_se','')].sum() > 0.1:
#         # print("! Warning: insufficient heat production on district heating networks !")
#         country_debug[('missing_heat','',country)] = flows[('def','vap_se','')]
#     if flows[('vap_se','exc','')].sum() > 0.1:
#         # print("! Warning: excess heat production on district heating networks !")
#         country_debug[('excess_heat','',country)] = flows[('vap_se','exc','')]

#     ## Liquid motor fuels
#     if MAIN_PARAMS['BIOFUEL_SHARE']:
#         flows[('blq_pe','lqf_se','')] = fec['lqf_fe'] * data['pcenrcfcltra']
#         sf.balance_node(flows, 'blq_pe', exp='lqf_se') # If local liquid fuel production > demand from motor fuels, excess is still mixed and will go to exports
#     else:
#         flows[('blq_pe','lqf_se','')] = flows[('prod','blq_pe','')]
#     sf.balance_node(flows, 'lqf_se')

#     ## Transformation losses, other than cogeneration and boilers (handled previously)
#     eff_columns = efficiencies.columns
#     for (source, target, proc_type) in eff_columns[~eff_columns.get_level_values('Type').isin(['chp','rch'])]: # filtering processes types, which are NOT 'chp' or 'rch'
#         if (efficiencies[(source,target,proc_type)] > 1).any():
#             print("! Warning: efficiency above 100% for process '{}'. Assuming 100% efficiency instead.".format(proc_type))
#         else:
#             flows[(source,'per',proc_type)] = flows[(source,target,proc_type)] * (1/efficiencies[(source,target,proc_type)] - 1)
#         if (flows[(source,'per',proc_type)] == np.inf).any():
#             print("! Warning: null efficiencies with non null prod values for process '{0}' going grom '{1}' to '{2}'.".format(proc_type, source, target))
    
#     # Setting negative flows to 0 (happens when efficiencies exceed 100%)
#     flows[flows < 0] = 0
    
#     ## Default reffinery consumption (if not specified in final demand of energy industry), calculated from primary oil consumption without non-energy
#     if 0 in data['gazcfens'].unique() or 0 in data['petcfens'].unique():
#         print("Info: gaz and/or oil consumption of energy industry have 0 values, calculating them from primary oil consumption.")
#         pec_pet = sf.node_consumption(flows, 'pet_pe').squeeze() + flows.get(('imp','lqf_se',''),0) - flows.get(('pet_fe','neind',''),0) # we consider imported liquid motor fuels as fossil
#         flows.loc[data['gazcfens'] == 0, ('gaz_fe','ens','')] = pec_pet * data['defgazcfens']
#         flows.loc[data['petcfens'] == 0, ('pet_fe','ens','')] = pec_pet * data['defpetcfens']
#         # Readjusting primary/secondary consumptions
#         flows[('gaz_se','gaz_fe','ntw')] += sf.node_consumption(flows, 'gaz_fe').squeeze() - fec['gaz_fe']
#         flows[('pet_pe','pet_fe','')] += sf.node_consumption(flows, 'pet_fe').squeeze() - fec['pet_fe']

#     ## Direct hydrogen injection in gas grid : all excess H2, below a given volumetric share of gas network
#     gas_on_grid = pd.concat([sf.node_consumption(flows, 'gaz_se', 'forward'), sf.node_consumption(flows, 'gaz_se', 'backwards')], axis=1).max(axis=1) # Max of in and out flows on gas network
#     sf.balance_node(flows, 'hyd_se', exp='gaz_se', procexp='irg')
#     flows[('hyd_se','gaz_se','irg')] = pd.concat([gas_on_grid * (1/(1-data['pchydirg']) - 1), flows[('hyd_se','gaz_se','irg')]], axis=1).min(axis=1)

    ## Local (renewable) production
    # gas_threshold = 2000  # TWh
    # oil_threshold = 1500  #TWh
    
    selected_columns_p = flows.columns.get_level_values('Source').isin(PE_NODES)
    fec_carrier_p = flows.loc[:, selected_columns_p]
    grouped_fec_p = fec_carrier_p.groupby(level='Source', axis=1).sum()
    fec_p = grouped_fec_p
    for en_code in ['hdr','eon','eof','spv','cms','pac','ura','enc','bgl','blq']:
        flows[('prod',en_code+'_pe','')] = fec_p[en_code+'_pe']
    for en_code in ['gaz']:
     values = fec_p[en_code + '_pe']
     # prod_values = values.clip(upper=gas_threshold)
     # imp_values = values - prod_values
     imp_values = values
    
     # flows[('prod', en_code + '_pe', '')] = prod_values
     flows[('imp', en_code + '_pe', '')] = imp_values
    for en_code in ['pet']:
     values = fec_p[en_code + '_pe']
     # prod_values = values.clip(upper=oil_threshold)
     # imp_values = values - prod_values
     imp_values = values
    
     # flows[('prod', en_code + '_pe', '')] = prod_values
     flows[('imp', en_code + '_pe', '')] = imp_values

   
    # ## (Re)balancing all primary and secondary energies with imports/exports
    # for node in PE_NODES + SE_NODES:
    #     sf.balance_node(flows, node)
    
    # ## Splitting renewable and non-renewable parts of waste
    # for level in [0,1]:
    #     flows_wst = flows.columns[(flows.columns.get_level_values(level) == 'wst_pe')]
    #     flows_ren_wst = flows_wst.set_levels(flows_wst.levels[level].str.replace('wst_pe', 'wst_ren_pe'), level=level)
    #     flows_fos_wst = flows_wst.set_levels(flows_wst.levels[level].str.replace('wst_pe', 'wst_fos_pe'), level=level)
    #     for i in range(len(flows_wst)):
    #         flows[flows_ren_wst[i]] = flows[flows_wst[i]] * data['pcenrwst']
    #         flows[flows_fos_wst[i]] = flows[flows_wst[i]] -  flows[flows_ren_wst[i]]
    #     flows = flows.drop(flows_wst, axis=1)

    ## Flows cleanup : deleting flows/columns, which are too small
    # flows = flows.drop(flows.columns[flows.max()<1E-4], axis=1)

    interval_time = sf.calc_time('Energy system (network graph) creation', interval_time)
    
    ## Storing energy flows, non-energy GHG values and other relevant DB values
    tot_flows[country] = flows
    # country_results = pd.DataFrame()
    # data_pop=data.copy()
    # data_pop['pop'] = 525000
    # country_results[('pop')] = data_pop['pop']
    # ghg_nes = data.loc[:,list('ghg'+sector+'nes' for sector in ['agr','ind','wst','oth'])]
    # ghg_nes.columns = ['agr','ind','wst','oth']
    # # Splitting AFOLUB into agr and LULUCF
    # ghg_nes['luf'] = sum(data['ghg'+ghg+'luf'] for ghg in ['co2','ch4','n2o','oth'])
    # ghg_nes['agr'] -= ghg_nes['luf']
    # country_results = sf.add_indicator_to_results(country_results, ghg_nes, 'ghg_nes')
    # country_results[('ghgco2_nes','total')] = data['ghgco2luf'] + data['ghgco2agr'] + data['ghgindnes'] # Non-energy related CO2 emissions (waste and "other" are neglected, industrial process emissions are considered 100% CO2)
    # country_results[('ghgch4_nes','total')] = data['ghgch4luf'] + data['ghgch4agr'] + data['ghgwstnes'] # Non-energy related CH4 emissions (industrial process and "other" are neglected, waste emissions are considered 100% CH4)
   
    # # Adding country results to total results DataFrame
    # country_results.columns = pd.MultiIndex.from_tuples(map(lambda x: (x[0], x[1], country), country_results.columns), names=('Indicator','Sub_indicator','Country'))
    # tot_results = pd.concat([tot_results, country_results], axis=1)
    
    # ## Adding country debug results to total debug DataFrame
    # tot_debug = pd.concat([tot_debug, country_debug], axis=1)
    
    # ## Calculating secondary energy export mix
    # for network in SE_NODES:
    #     if (network,'exp','') in flows.columns:
    #         # Evaluating share of primary sources in secondary energy "network"
    #         primary_exports = sf.node_consumption(sf.shares_from_node(flows,network,PE_NODES),PE_NODES)
    #         # Multiplying by the ratio of exports / total energy going through "network"
    #         exports_ratio = flows[(network,'exp','')].squeeze() / sf.node_consumption(flows, network).sum(axis=1)
    #         # Breakdown of primary energies by categories
    #         primary_exports_breakdown = sf.share_primary_category(primary_exports.multiply(exports_ratio,axis=0), NODES)
    #         tot_se_export_mix[network] = tot_se_export_mix[network].add(primary_exports_breakdown, fill_value=0)

# ## Interterritorial analysis
# print("\nInterterritorial flows analysis\n")

# Concatenation of country flow DataFrames and other indicators
# for perimeter in country_groups:
#     country_list = country_groups[perimeter]
#     for country in country_list:
#         tot_flows[perimeter] = tot_flows[country] if perimeter not in tot_flows else tot_flows[perimeter].add(tot_flows[country], fill_value=0)
#     for (indicator,subindicator) in tot_results.columns.droplevel("Country"):
#         tot_results[(indicator,subindicator,perimeter)] = tot_results[[(indicator, subindicator, country) for country in country_list]].sum(axis=1)
    
# ## Secondary energy import mix calculation
# tot_imports = sf.node_consumption(tot_flows['ALL'], 'imp', splitby='target')
# tot_se_imports = tot_imports.filter(SE_NODES)
# tot_se_excess = tot_se_export_mix.groupby(level='Network',axis=1).sum().sub(tot_se_imports,fill_value=0)
# tot_se_deficit = - tot_se_excess
# tot_se_excess[tot_se_excess < 0] = 0
# tot_se_deficit[tot_se_deficit < 0] = 0
# # Intra-EU imports : total imports - deficit
# intra_eu_imports = tot_se_imports.sub(tot_se_deficit,fill_value=0)
# # Extra-eu imports : total_imports - intra-eu imports
# extra_eu_imports = tot_se_imports.sub(intra_eu_imports,fill_value=0)
# # Intra-EU export mix : conversion from TWh to %
# for network in tot_se_export_mix.columns.get_level_values(0):
#     tot_se_export_mix[network] = sf.share_percent(tot_se_export_mix[network])
# # Import mix in TWh : intra-eu imports (TWh) x intra-eu export mix (%) + extra-eu imports (TWh) x IMPORT_MIX (%)
# for network in tot_se_imports.columns:
#     # intra-eu imports x intra-eu export mix
#     tot_se_import_mix[network] = tot_se_export_mix[network].mul(intra_eu_imports[network], axis=0)
#     # extra-eu imports x IMPORT_MIX
#     tot_se_import_mix[network] = tot_se_import_mix[network].add(IMPORT_MIX[network].mul(extra_eu_imports[network], axis=0), fill_value=0)
# sf.consistency_check(tot_se_import_mix,tot_se_imports,'Debug: max difference between secondary energy imports and breakdown by category (should be 0)')

# # Weighted average import mix in %
# for network in tot_se_imports.columns:
#     tot_se_import_mix[network] = sf.share_percent(tot_se_import_mix[network])

# # Net import/export balances
# def imp_exp_balance(df,perimeter,imp_flow,exp_flow):
#     if imp_flow in df[perimeter].columns and exp_flow in df[perimeter].columns:
#             excess = df[perimeter][exp_flow] - df[perimeter][imp_flow]
#             deficit = - excess
#             excess[excess < 0] = 0
#             deficit[deficit < 0] = 0
#             df[perimeter][exp_flow] = excess.squeeze()
#             df[perimeter][imp_flow] = deficit.squeeze()   
# for perimeter in country_groups:
#     for node in PE_NODES + SE_NODES:
#         imp_flow = ('imp',node,'')
#         exp_flow = (node,'exp','')
#         imp_exp_balance(tot_flows,perimeter,imp_flow,exp_flow)
#     imp_exp_balance(tot_flows,perimeter,('def','vap_se',''),('vap_se','exc',''))

# print("\nIndicator calculation\n")

### Function to generate results
# Writes output result files and fills the tot_results dataFrame
# proc_labels = PROCESSES.set_index(['Source','Target','Type'])#[['Label']]
# links = pd.concat([flows.T, proc_labels], axis=1)
def generate_results(flows, country):
    xls_file_name = 'ChartData_'+country+'.xlsx'
    results_xls_writer = pd.ExcelWriter(open(os.path.join(DIRNAME,'Results',xls_file_name), 'wb'), engine="openpyxl")

    if country in ALL_COUNTRIES:
        country_label = COUNTRIES.loc[country,'Label']
        country_label_w_flag = country_label + ' <img style="height:22px" src="https://flagicons.lipis.dev/flags/4x3/' + country.replace('UK','GB').replace('EL','GR').lower() + '.svg" />'
        show_total = False
        country_list = COUNTRIES
        input_file_label = COUNTRIES.loc[country,'Input_File']
    # else:
    #     country_label = 'All countries (EU27 + UK, CH & NO)' if country=='ALL' else 'Europe (EU27)'
    #     country_label_w_flag = country_label
    #     if country=='EU': country_label_w_flag += '<img style="height:22px" src="https://flagicons.lipis.dev/flags/4x3/eu.svg" />'
    #     show_total = True
    #     country_list = COUNTRIES.loc[country_groups[country]]
    #     input_file_label = '<ul><li>' + '</li><li>'.join(country_list['Input_File']) + '</li></ul>'
    #     # tot_results = tot_results.sort_index(axis=1) # To improve performance
    # print("||| "+country+" |||")

    country_results = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Indicator','Sub_indicator']))
    global interval_time

    ## Changes in bunker perimeter as required (if EU : we override settings)
    sectors_to_remove = []
    # aviation_bunker = MAIN_PARAMS['INT_AVIATION_EU27'] if country=='EU' else MAIN_PARAMS['INT_AVIATION']
    # maritime_bunker = MAIN_PARAMS['INT_MARITIME_EU27'] if country=='EU' else MAIN_PARAMS['INT_MARITIME']
    # if not aviation_bunker: sectors_to_remove += ['avi']
    # if not maritime_bunker: sectors_to_remove += ['wati']
    # eu_bunker_change = True if (not aviation_bunker or maritime_bunker) else False
    flows_bk =  flows.copy()
    # for sector in sectors_to_remove:
    #     flows_bk = flows_bk.sub(sf.shares_from_node(flows_bk,sector,include_losses=True),fill_value=0)
    #     # We cleanup remaining flows connected to removed sector, and negative values
    #     flows_bk = flows_bk.drop(flows_bk.columns[flows_bk.columns.get_level_values('Target') == sector], axis=1)
    #     flows_bk[flows_bk < 0] = 0
    # # Force include aviation bunkers and remove marine bunkers for EU indicators, regardless of settings (if not already the case)
    # if eu_bunker_change:
    #     flows_bk_eu = flows.sub(sf.shares_from_node(flows,'wati',include_losses=True),fill_value=0)
    #     # We cleanup remaining flows connected to removed sector, and negative values
    #     flows_bk_eu = flows_bk_eu.drop(flows_bk_eu.columns[flows_bk_eu.columns.get_level_values('Target') == 'wati'], axis=1)
    #     flows_bk_eu[flows_bk_eu < 0] = 0
    # else:
    #     flows_bk_eu = flows_bk
    
    ## Aggregated consumptions and productions
    # Net final energy consumption by energy carrier and sector
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

        
    # # Gross final energy consumption (net + network losses)
    gross_fec_carrier = fec_carrier.copy()
    # for en_code in ['elc','gaz','vap']:
    #     gross_fec_carrier[en_code+'_fe'] = gross_fec_carrier.get(en_code+'_fe',0)
    gross_net_ratio = gross_fec_carrier / fec_carrier
    
    # # Primary energy production
    # pep = sf.node_consumption(flows_bk, 'prod', splitby='target')
    # pep_breakdown = sf.share_primary_category(pep, NODES)

    # Local coverage ratios per carrier : (cons - imports) / (cons - exports)
    # Adding heat deficit & excess to imports & exports
    # cov_exports = sf.node_consumption(flows, ['exp','exc'], direction='backwards', splitby='target')
    selected_columns_E = flows_bk.columns.get_level_values('Source').isin(EE_NODES)
    export_carrier = flows_bk.loc[:, selected_columns_E]
    grouped_export = export_carrier.groupby(level='Target', axis=1).sum()
    cov_exports = grouped_export
    selected_columns_I = flows_bk.columns.get_level_values('Source').isin(II_NODES)
    import_carrier = flows_bk.loc[:, selected_columns_I]
    grouped_import = import_carrier.groupby(level='Target', axis=1).sum()
    cov_imports = grouped_import
    
    impexp_carriers = list(set(cov_imports.columns.to_list() + cov_exports.columns.to_list())) # Carriers with imports and/or exports only
    # merged_carriers = pd.concat([grouped_export, grouped_import], axis=1).fillna(0)

    ps_cons = sf.node_consumption(flows_bk, impexp_carriers)
    cov_ratios = 100 * ps_cons.subtract(cov_imports, fill_value=0).filter(impexp_carriers) / ps_cons.subtract(cov_exports, fill_value=0).filter(impexp_carriers)
    
  # interval_time = sf.calc_time('Aggregated consumptions', interval_time)
    
    # # Alternative flows dataframe, using import mixes
    # flows_imp = flows_with_imports(flows_bk, se_import_mix)
    # flows_imp_eu = flows_with_imports(flows_bk_eu, se_import_mix) if eu_bunker_change else flows_imp
    
    # Share of renewables in gross final energy consumption using network graph exploration
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
    ren_columns = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe', 'pac_pe','bgl_pe','blq_pe']
    fos_columns = ['cms_pe', 'gaz_pe', 'pet_pe']
    nuk_columns = ['ura_pe']

    ren_sum = flows_from_node[ren_columns].sum(axis=1)
    fos_sum = flows_from_node[fos_columns].sum(axis=1)
    nuk_sum = flows_from_node[nuk_columns].sum(axis=1)

    flows_from_node['ren'] = ren_sum
    flows_from_node['fos'] = fos_sum
    flows_from_node['nuk'] = nuk_sum

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
    
    gas_columns = ['bgl_pe','enc_pe', 'gaz_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in gas_columns and col[1] == 'gaz_se']
    result_gas = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_gas_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_gas = ['bgl_pe','enc_pe']
    ren_gas = result_gas[ren_gas].sum(axis=1)
    ren_cov_ratios['gaz_fe'] = (ren_gas/result_gas_t)*100
    
    pet_columns = ['pet_pe','enc_pe','hyd_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in pet_columns and col[1] in ['pet_fe', 'lqf_se']]
    result_pet = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_pet_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_pet = ['enc_pe','hyd_se']
    ren_pet = result_pet[ren_pet].sum(axis=1)
    ren_cov_ratios['pet_fe'] = (ren_pet/result_pet_t)*100
    
    hyd_columns = ['elc_se','gaz_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in hyd_columns and col[1] == 'hyd_se']
    result_hyd = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_hyd_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_hyd = ['elc_se']
    ren_hyd = result_hyd[ren_hyd].sum(axis=1)
    ren_cov_ratios['hyd_fe'] = (ren_hyd/result_hyd_t)*100
    
    bm_columns = ['enc_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in bm_columns and col[1] in ['gaz_se', 'lqf_se', 'elc_se','vap_se']]
    result_bm = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_bm_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_bm = ['enc_pe']
    ren_bm = result_bm[ren_bm].sum(axis=1)
    ren_cov_ratios['enc_fe'] = (ren_bm/result_bm_t)*100
    
    dh_columns = ['enc_pe','cms_pe','pet_pe','gaz_se','bgl_pe','elc_se','pac_pe','tes_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in dh_columns and col[1] in ['vap_se']]
    result_dh = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_dh_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_dh = ['enc_pe','bgl_pe','elc_se','pac_pe']
    ren_dh = result_dh[ren_dh].sum(axis=1)
    ren_cov_ratios['vap_fe'] = (ren_dh/result_dh_t)*100
    
    am_columns = ['pac_pe']
    filtered_columns = [col for col in flows_bk.columns if col[0] in am_columns and col[1] in ['vap_se','pac_fe']]
    result_am = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_am_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_am = ['pac_pe']
    ren_am = result_am[ren_am].sum(axis=1)
    ren_cov_ratios['pac_fe'] = (ren_am/result_am_t)*100
    
    nh_columns = ['elc_se','hyd_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in nh_columns and col[1] in ['amm_fe']]
    result_nh = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_nh_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_nh = ['elc_se','hyd_se']
    ren_nh = result_nh[ren_nh].sum(axis=1)
    ren_cov_ratios['amm_fe'] = (ren_nh/result_nh_t)*100
    
    me_columns = ['elc_se','hyd_se']
    filtered_columns = [col for col in flows_bk.columns if col[0] in me_columns and col[1] in ['met_fe']]
    result_me = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    result_me_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    ren_me = ['elc_se','hyd_se']
    ren_me = result_me[ren_me].sum(axis=1)
    ren_cov_ratios['met_fe'] = (ren_me/result_me_t)*100
    
    gfec_breakdown_pct = sf.share_percent(gfec_breakdown,100)
    ren_cov_ratios['total'] = gfec_breakdown_pct['ren']
    # # Share of renewable following EU methodology
    res_share_eu=pd.DataFrame()
    # # No change for overal RES share
    # res_share_eu['total'] =  ren_cov_ratios_eu['total']
    # # RES-E is calculated as gross ren. production / gross final consumption (including internal uses), renewable imports are not taken into account
    # cons_at_primary = sf.share_primary_category(sf.node_consumption(sf.shares_from_node(flows,'elc_se',PE_NODES),PE_NODES+['imp']), NODES)
    # res_share_eu['elc_fe'] = 100 * cons_at_primary.get('ren',0) / (cons_at_primary.sum(axis=1))
    # # RES-T
    # ## TD : improve multiplication factors, remove non-compliant biofuels, remove international freight, remove kerosene, use RES-E from 2 years ago (capped to 100%) 
    # denominator_tra = sf.node_consumption(flows,['tra','avi'],'backwards','target')
    # denominator_tra['lqf_fe'] += flows.get(('lqf_fe','agr',''),0) # Adding liquid fuels used in agriculture
    # numerator_tra = denominator_tra.multiply(ren_cov_ratios_eu,fill_value=0)
    # for (energy,multiplicator) in [('elc',3),('lqf',1.5)]:
    #     denominator_tra[energy+'_fe'] = denominator_tra.get(energy+'_fe',0) + numerator_tra[energy+'_fe'] * (multiplicator-1) / 100
    #     numerator_tra[energy+'_fe'] *= multiplicator
    # res_share_eu['tra'] = numerator_tra.sum(axis=1) / denominator_tra.sum(axis=1)
    # # RES-H&C is calculated as renewable share of final energies except electricity & transportation uses
    # # TD : remove non-compliant biofuels
    # denominator_hc = fec_carrier.subtract(denominator_tra, fill_value=0).drop(columns=['elc_fe','lqf_fe'], errors="ignore")
    # numerator_hc = denominator_hc.multiply(ren_cov_ratios_eu, fill_value=0)
    # res_share_eu['chfcli'] = numerator_hc.sum(axis=1) / denominator_hc.sum(axis=1)

    # Detailled breakdown of net final energy consumption per primary energy (and secondary imports)
    # fec_breakdown = sf.node_consumption(flows_from_node_cum,PE_NODES+SI_NODES)
    
    interval_time = sf.calc_time('Graph analysis (consumption breakdown)', interval_time)

    ## Primary energy consumption
    # Primary production + imports 
    pec = calculate_pec(flows)
    pec_breakdown = sf.share_primary_category(pec, NODES)
    # pec_eu = calculate_pec(flows_imp_eu) if eu_bunker_change else pec
    # pec_eu = sf.subtract_cons_from_node(pec_eu, flows_imp_eu, 'neind', end_nodes=PE_NODES).drop(columns=['sth_pe','pac_pe'], errors="ignore") # Removing ambient heat & non-energy consumption
    # heatpower_columns = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe', 'pac_pe','cms_pe', 'gaz_pe', 'pet_pe','ura_pe']
    # filtered_columns = [col for col in flows_bk.columns if col[0] in tot_columns and col[1] == 'elc_se']
    # result_elc = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum()
    # result_elc_t = flows_bk[filtered_columns].groupby(level='Source', axis=1).sum().sum(axis=1)
    # ren_elc = ['spv_pe', 'eon_pe', 'eof_pe', 'hdr_pe', 'enc_pe']
    # ren_elc = result_elc[ren_elc].sum(axis=1)
    # ren_cov_ratios = pd.DataFrame()
    # ren_cov_ratios['elc_fe'] = (ren_elc/result_elc_t)*100
    # ## GHG emissions
    # ghg_sector=pd.DataFrame() # GHG emissions by sector (CRF nomenclature)
    # ghg_sector_2=pd.DataFrame() # GHG emissions by sector with power & heat allocated to other sectors
    # ghg_sector_eu=ghg_sector # GHG emissions by sector on the EU perimeter, by default equivalent to ghg_sector
    # ghg_source=pd.DataFrame() # GHG emissions by source (primary energies / non-energy)
    # ghg_source_CO2=pd.DataFrame() # Same for CO2-only emissions
    # ghg_source_CH4=pd.DataFrame() # Same for CH4-only emissions
    # # Share of power & heat use (used to allocate power & heat emissions to final demand)
    # elc_vap_uses = {}
    # for ntw in ['elc_se','vap_se']:
    #     elc_vap_uses[ntw] = sf.node_consumption(sf.shares_from_node(flows_imp,ntw,DS_NODES,'forward',include_losses=True,normalised=True),DS_NODES+['exp','exc'],'backwards')
    # # Fuel combustion emission: (primary energy consumption - non energy uses) x emission factors
    # for primary_node in GHG_ENERGIES.index:
    #     flow_sector = sf.shares_from_node(flows_imp,primary_node,['elc_se','vap_se'],'forward',include_losses=True)
    #     pec_ghg_sector = sf.node_consumption(flow_sector, DS_NODES+['elc_se','vap_se','exp'], 'backwards')
    #     if pec_ghg_sector.empty: continue
    #     country_results = sf.add_indicator_to_results(country_results, pec_ghg_sector, 'pec_uses.'+primary_node, False)

    #     # Removing feedstocks and exports...
    #     pec_ghg_sector = pec_ghg_sector.drop(['neind','exp'], axis=1, errors='ignore')
    #     # ...except when originating from fossil methane-sourced hydrogen (attributed to energy industry)
    #     if primary_node == 'gaz_pe': pec_ghg_sector['ens'] = pec_ghg_sector.get('ens',0) + flow_sector.get(('hyd_fe','neind',''),0) + flow_sector.get(('hyd_se','exp',''),0)

    #     # Power and heat emissions allocation to final demand sectors
    #     pec_ghg_sector_2 = pec_ghg_sector.drop(['elc_se','vap_se'], axis=1, errors='ignore')
    #     for ntw in ['elc_se','vap_se']:
    #         # Readjusting to actual primary consumption of power & heat production
    #         adj_elc_vap_uses = elc_vap_uses[ntw].mul(pec_ghg_sector.get(ntw,0),axis=0)
    #         pec_ghg_sector_2 = pd.concat([pec_ghg_sector_2,adj_elc_vap_uses], axis=1).groupby(axis=1, level=0).sum()
       
    #     # Merging power and heat
    #     pec_ghg_sector['elc_vap'] = pec_ghg_sector.get('elc_se',0) + pec_ghg_sector.get('vap_se',0)
    #     pec_ghg_sector = pec_ghg_sector.drop(['elc_se','vap_se'], axis=1, errors='ignore')
        
    #     # Applying emission factors and adding to total GHG dataframes
    #     ghg_en_sector = pec_ghg_sector * NODES.loc[primary_node,'Emission_Factor']
    #     ghg_en_sector_2 = pec_ghg_sector_2 * NODES.loc[primary_node,'Emission_Factor']
    #     ghg_source[primary_node] = ghg_en_sector.sum(axis=1)
    #     ghg_source_CO2[primary_node] = (pec_ghg_sector * NODES.loc[primary_node,'Emission_Factor_CO2']).sum(axis=1)
    #     ghg_source_CH4[primary_node] = (pec_ghg_sector * NODES.loc[primary_node,'Emission_Factor_CH4']).sum(axis=1)
    #     country_results = sf.add_indicator_to_results(country_results, ghg_en_sector, 'ghg_en.'+primary_node, False)
    #     ghg_sector = pd.concat([ghg_sector,ghg_en_sector], axis=1).groupby(axis=1, level=0).sum()
    #     ghg_sector_2 = pd.concat([ghg_sector_2,ghg_en_sector_2], axis=1).groupby(axis=1, level=0).sum()
    # if eu_bunker_change:
    #     if aviation_bunker:
    #         ghg_sector_eu = ghg_sector.drop('wati', axis=1, errors='ignore')
    #     else:
    #         for primary_node in GHG_ENERGIES.index:
    #             flow_sector = sf.shares_from_node(flows_imp_eu,primary_node,['elc_se','vap_se'],'forward',include_losses=True)
    #             pec_ghg_sector_eu = sf.node_consumption(flow_sector, DS_NODES+['elc_se','vap_se','exp'], 'backwards')
    #             if pec_ghg_sector_eu.empty: continue
    #             pec_ghg_sector_eu = pec_ghg_sector_eu.drop(['neind','exp'], axis=1, errors='ignore')
    #             if primary_node == 'gaz_pe': pec_ghg_sector_eu['ens'] = pec_ghg_sector_eu.get('ens',0) + flow_sector.get(('hyd_fe','neind',''),0) + flow_sector.get(('hyd_se','exp',''),0)
    #             pec_ghg_sector_eu['elc_vap'] = pec_ghg_sector_eu.get('elc_se',0) + pec_ghg_sector_eu.get('vap_se',0)
    #             pec_ghg_sector_eu = pec_ghg_sector_eu.drop(['elc_se','vap_se'], axis=1, errors='ignore')
    #             ghg_en_sector_eu = pec_ghg_sector_eu * NODES.loc[primary_node,'Emission_Factor']
    #             ghg_sector_eu = pd.concat([ghg_sector_eu,ghg_en_sector_eu], axis=1).groupby(axis=1, level=0).sum()
    # country_results[('ghg_en','total')] = ghg_sector.sum(axis=1)
    # # Adding non-energy emissions
    # ghg_nes = tot_results[[('ghg_nes',sector,country) for sector in ['agr','ind','wst','oth','luf']]].rename(columns=lambda x:x+'nes')
    # ghg_nes.columns = ghg_nes.columns.droplevel(['Indicator','Country'])
    # ghg_sector = ghg_cleanup(ghg_sector,ghg_nes)
    # ghg_sector_2 = ghg_cleanup(ghg_sector_2,ghg_nes)
    # ghg_sector_eu = ghg_cleanup(ghg_sector_eu,ghg_nes) if eu_bunker_change else ghg_sector
    # ghg_source['nes'] = ghg_nes.sum(axis=1)
    # ghg_source_CO2['nes'] = tot_results[('ghgco2_nes','total',country)]
    # ghg_source_CH4['nes'] = tot_results[('ghgch4_nes','total',country)]
    ghg_sector = flows_ghg.copy()
    ghg_sector = ghg_sector.groupby(level='Source', axis=1).sum()
    ghg_sector['lufnes_ghg'] = -ghg_sector['lufnes_ghg']
    ghg_sector['blg_ghg'] = -ghg_sector['blg_ghg']
    ghg_source = flows_ghg.groupby(level='Target', axis=1).sum()
    ghg_source['lufnes_ghg'] = -ghg_source['lufnes_ghg']
    ghg_source['bgl_pe'] = -ghg_source['bgl_pe']
    ## Start HTML output
    html_items = {}
    # html_items['COUNTRY'] = country_label_w_flag

    # html_items['INTRO'] = f'This page provides detailed data visualisation of the trajectory for {country_label}, as suggested by the <a href="https://clever-energy-scenario.eu/">CLEVER project</a>. To access pathways for another country (or EU perimeter), you may use the dropdown menu in the top right menu. You may also <a href="{xls_file_name}" target="_blank">click here</a> to download the raw data behind all graphs shown below.'

    # html_items['DISCLAIMER'] = '<b>DISCLAIMER:</b> this document is work in progress, including the result of the "V0" (the first EU aggregation of some bottom-up national trajectories and top-down normalised trajectories for remaining countries). Those trajectories will continue to be harmonised and reinforced.<br /><span style="font-weight:bold;color:red">Results shown below are not to be disseminated to third parties</span>.' if MAIN_PARAMS['DRAFT'] else 'This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License <img alt="Creative Commons License" style="display: inline; vertical-align: middle;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>'
    # License => https://creativecommons.org/choose/results-one?license_code=by&amp;jurisdiction=&amp;version=4.0&amp;lang=en

    # html_items['METHODO'] = '<p>Methodological notes: unless specified, calculations below have been made <b><span style="color:red">' + ('without' if 'avi' in sectors_to_remove else 'with') + '</span> international aviation and <span style="color:red">' + ('without' if 'wati' in sectors_to_remove else 'with') + '</span> international maritime transport</b>.'
    # if MAIN_PARAMS['DRAFT']: html_items['METHODO'] += 'The mix of imported energy carriers is '+('calculated from other european trajectories' if MAIN_PARAMS['USE_IMPORT_MIX'] else 'considered fossil-sourced by default') +'. The share of biofuels in liquid fuels is '+('defined by a dashboard indicator' if MAIN_PARAMS['BIOFUEL_SHARE'] else 'adjusted, according to available biofuel')+'.</p>'
    
    id_section = -1
    sections = [('ghg','GHG'),('sankey','Sankey diagram'),('carbon sankey','Carbon Sankey diagram'),('res','Renewable energy share'),('carrier','Energy carrier balance'),('cons','Energy consumption'),('eu','EU indicators & objectives')]
    if MAIN_PARAMS['HTML_TEMPLATE'] == "raw": sections += [('input','Input data')]
    html_items['MENU'] = '<ol>'
    for (anchor,title) in sections:
        html_items['MENU'] += '<li><a href="#'+anchor+'">'+title+'</a></li>'
    html_items['MENU'] += '</ol>'

    html_items['MAIN'] = ''

    # GHG
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    html_items['MAIN'] += sf.combine_charts([('by sector',ghg_sector),('by source',ghg_source),('cumulated since 2030 by sector',sf.cumul(ghg_sector,2030)),('cumulated since 2030 by source',sf.cumul(ghg_source,2030))], MAIN_PARAMS, NODES, 'All GHG emissions', 'areachart', results_xls_writer, 'MtCO<sub>2</sub>eq') #('by sect. - power & heat dispatched',ghg_sector_2),
    if show_total:
        html_items['MAIN'] += sf.combine_charts([('total',tot_results[('ghg_source','percap')]),('energy only',tot_results[('ghg_en','percap')]),('non-energy only',tot_results[('ghg_nes','percap')])], MAIN_PARAMS, country_list, 'GHG emissions per capita -', 'map', results_xls_writer, 'tCO<sub>2</sub>eq/cap/year', reverse=True)
    html_items['MAIN'] += sf.combine_charts([('cumulated since 2030',sf.cumul(ghg_source,2030)),('yearly emissions',ghg_source)], MAIN_PARAMS, NODES, 'CO2 only emissions', 'areachart', results_xls_writer, 'MtCO<sub>2</sub>')

    # Sankeys
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    html_items['MAIN'] += sf.combine_charts([('Sankey diagram',flows)], MAIN_PARAMS, NODES, '', 'sankey', sk_proc=PROCESSES) #('upstream flows from final energies',flows_from_node_cum),('Sankey diagram without import mix',flows)
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    html_items['MAIN'] += sf.combine_charts([('Carbon Sankey diagram',flows_co2)], MAIN_PARAMS, NODES, '', 'carbon sankey', sk_proc=PROCESSES_2) #('upstream flows from final energies',flows_from_node_cum),('Sankey diagram without import mix',flows)
    # RES share
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    html_items['MAIN'] += sf.chart_to_output(sf.create_node_chart(ren_cov_ratios, NODES, MAIN_PARAMS, 'linechart', 'Renewable share per final energies', results_xls_writer, '%'))
    # html_items['MAIN'] += '<p>Renewable shares per final energies are calculated by analysing all energy flows going through different transformation processes (electricity and heat production processes, power-to-gas etc.) as described by the Sankey diagram. An algorithm goes upstream through this complex energy system, from a given final energy to all relevant primary energies, and determines their respective shares. For example, a renewable share of 50% for final electricity means that 50% of the electricity consumed has been produced by renewable means, either directly from renewable power technologies such as wind of PV, or indirectly - for example if gas cogeneration has been used with a share of renewables in the gas mix.'
    # if MAIN_PARAMS['USE_IMPORT_MIX'] and not show_total: html_items['MAIN'] += ' NB: the mix of imported secondary energy carriers (power, gas...) is calculated from exports of other EU countries, and may thus contain some level of renewables, included in this calculation as well.'
    html_items['MAIN'] += '<p>'
    if show_total: html_items['MAIN'] += sf.chart_to_output(sf.create_map(tot_results[('ren_cov_ratio','total')], country_list, 'RES share in final consumption', MAIN_PARAMS, unit='%', min_scale=0,max_scale=100))
    html_items['MAIN'] += sf.chart_to_output(sf.create_node_chart(gfec_breakdown, NODES, MAIN_PARAMS, 'area', 'Final consumption by origin', results_xls_writer))
    
    # Energy carrier share balance
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    html_items['MAIN'] += sf.chart_to_output(sf.create_node_chart(cov_ratios, NODES, MAIN_PARAMS, 'linechart', 'Local production coverage ratios', results_xls_writer, unit='%'))
    # html_items['MAIN'] += '<p>Local production coverage ratios are simply defined as the ratio between the local production of a given energy carrier, and its local consumption (including final and non final uses). A ratio above 100% thus means that the country is more than self-sufficient (net exporter), while a ratio below 100% means that the country is a net importer.</p>'
    if show_total:
        node_list = tot_results['cov_ratio'].columns.unique(level='Sub_indicator').to_list()
        sf.put_item_in_front(node_list, 'total') # We put total first
        combinations = [(NODES.loc[node,'Label'],tot_results[('cov_ratio',node)]) for node in node_list]
        html_items['MAIN'] += sf.combine_charts(combinations, MAIN_PARAMS, country_list, 'Local prod coverage ratios -', 'map', results_xls_writer, '%', min_scale=0, mid_scale=50)
    combinations = [('All energies',fec_sector)]
    for energy in FE_NODES:
        combinations += [(NODES.loc[energy,'Label'], sf.node_consumption(flows_bk, energy, direction='forward', splitby='target'))]
    html_items['MAIN'] += sf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by sector -', 'areachart', results_xls_writer)
    combinations = []
    for energy in SE_NODES:
        df = sf.node_consumption(flows, energy, direction='backwards', splitby='target')
        combinations += [(NODES.loc[energy,'Label'], df)]
        country_results = sf.add_indicator_to_results(country_results, df, 'sec_mix.'+energy, False)
    html_items['MAIN'] += sf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Mix of secondary energies -', 'areachart', results_xls_writer)
    # html_items['MAIN'] += '<p>The above chart describes the contribution of misc. technologies (and possibly imports) to the production of a given secondary energy carrier.</p>'
    combinations = []
    # for energy in SE_NODES + ['enc_pe']:
    #     df = sf.node_consumption(sf.shares_from_node(flows_bk,energy,SE_NODES,'forward',include_losses=True),DS_NODES+SE_NODES+['per','exp','exc'],'backwards')
    #     combinations += [(NODES.loc[energy,'Label']+ " - included losses", df)]
    #     country_results = sf.add_indicator_to_results(country_results, df, 'sec_uses.'+energy)
    #     df = sf.node_consumption(sf.shares_from_node(flows_bk,energy,SE_NODES,'forward',include_losses=False),DS_NODES+SE_NODES+['per','exp','exc'],'backwards')
    #     combinations += [(NODES.loc[energy,'Label']+ " - separated losses", df)]
    # html_items['MAIN'] += sf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Internal and final uses -', 'areachart', results_xls_writer)
    # html_items['MAIN'] += '<p>The above chart illustrates final and internal (non final) uses of a given secondary energy carrier. Losses are either included in each internal use, or separated (in both case, the total is the same):</p><ul><li><b>"Included losses"</b>: transformation & network losses are attributed to each internal use & final demand sectors</li><li><b>"Separated losses"</b>: losses are grouped as a separate category</li></ul>'

    # Energy consumption
    id_section += 1
    html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    combinations = [('All sectors',fec_carrier)]
    for sector in DS_NODES:
        df = sf.node_consumption(flows_bk, sector, direction='backwards', splitby='target')
        combinations += [(NODES.loc[sector,'Label'], df)]
        country_results = sf.add_indicator_to_results(country_results, df, 'fec.'+sector)
    html_items['MAIN'] += sf.combine_charts(combinations, MAIN_PARAMS, NODES, 'Final consumption by carrier -', 'areachart', results_xls_writer)
    # if show_total:
    #     html_items['MAIN'] += sf.combine_charts([('Final energy consumption',tot_results[('fec','reduc')])], MAIN_PARAMS, country_list, 'Reduction vs. 2015 -', 'map', results_xls_writer, '%')
    # html_items['MAIN'] += sf.combine_charts([('consumption',pec),('consumption by type',pec_breakdown)], MAIN_PARAMS, NODES, 'Primary energy', 'areachart', results_xls_writer)

    ## EU indicators & objectives
    # targets = {}
    # targets_res = {'total':{},'elc_fe':{},'tra':{},'chfcli':{}}
    # id_section += 1
    # html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
    # html_items['MAIN'] += '<p>In this section, indicators are calculated according to Eurostat methodology, and compared with official EU objectives (when available).</p>'
    # if eu_bunker_change: html_items['MAIN'] += '<p><b>Caution</b>: the perimeter for bunkers is different in this section than above graphs - international aviation is included, international maritime transport is excluded.</p>'
    # # GHG
    # if country == 'EU': targets = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030],'y':[3622,3497,2027]}
    # html_items['MAIN'] += sf.combine_charts([('yearly emissions', ghg_sector_eu, targets)], MAIN_PARAMS, NODES, 'All GHG emissions', 'areachart', results_xls_writer, 'MtCO<sub>2</sub>eq')

    # # RES
    # if country == 'EUR':
    #     targets_res['total'] = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030],'y':[17.8,19.9,42.5]}
    #     targets_res['elc_fe'] = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030,2030],'y':[29.66,34.09,65,69]}
    #     targets_res['tra'] = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030,2030],'y':[6.75,8.8,28,32]}
    #     targets_res['chfcli'] = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030,2030],'y':[20.31,22.43,38,46]}
    # html_items['MAIN'] += sf.combine_charts([(label,res_share_eu[[indicator]],targets_res[indicator]) for (label,indicator) in [('all energies','total'),('electricity','elc_fe'),('transportation','tra'),('heating & cooling','chfcli')]], MAIN_PARAMS, NODES, 'Renewable energy share -', 'linechart', results_xls_writer, '%')
    # html_items['MAIN'] += '<ul><li><b>Overall RES</b> has not been adapted.</li><li>The <b>RES-E</b> (share of renewables in power sector) has been adapted to follow Eurostat methodology and is calculated as gross renewable production / gross final consumption (including internal uses such as electrolysis and e-fuels). It may thus be above 100% (in case of high share of renewables, and exports). Renewable power imports are not included in the numerator.</li><li><b>RES-T</b> (share of renewable in transportation) only <u>partially follows</u> Eurostat methodology. Main differences: multipliers are approximated (3x for electricity use as a whole, 1.5x for biofuels), non compliant biofuels not excluded, international freight and kerosene not excluded, RES-E calculated from current year mix (not 2 years ago).</li><li><b>RES-H&C</b> (share of renewable in heating & cooling) follows Eurostat methodology: share of renewables in final energies excluding electricity and transportation uses.</li></ul>'

    # # Final energy consumption
    # if country == 'EUR': targets = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030],'y':[10912,11253,8882]}
    # html_items['MAIN'] += sf.combine_charts([('All energies',fec_sector_eu,targets)], MAIN_PARAMS, NODES, 'Final consumption by sector -', 'areachart', results_xls_writer)
    # html_items['MAIN'] += '<p>This indicator is equivalent to Eurostat\'s "Final energy consumption (Europe 2020-2030)", it is equal to the final energy consumption calculated previously, without ambient heat, non-energy consumption, international maritime consumption and the energy sector (except blast furnaces).</p>'

    # Primary energy consumption 
    # if country == 'EUR': targets = {'title':'ESTAT / EU objective','mode':'markers','x':[2015,2019,2030],'y':[15731,15745,11544]}
    # html_items['MAIN'] += sf.combine_charts([('consumption',pec,targets)], MAIN_PARAMS, NODES, 'Primary energy', 'areachart',  results_xls_writer)
    # # html_items['MAIN'] += '<p>This indicator is equivalent to Eurostat\'s "Primary energy consumption (Europe 2020-2030)", it is equal to the primary energy consumption calculated previously, without ambient heat and non-energy consumption.</p>'
    
    interval_time = sf.calc_time('Plotting & file writting', interval_time)

    # Indicator calculation for inter-territorial analysis
    # Multidimensionnal indicators not already defined above (some indicators have several nomenclatures: by energy / sector / origin etc.)
    for (indicator,df) in [('fec',fec_carrier),('fec',fec_sector),('cov_ratio',cov_ratios),('ren_cov_ratio',ren_cov_ratios),('ghg_sector',ghg_sector),('ghg_source',ghg_source)]:
        country_results = sf.add_indicator_to_results(country_results, df, indicator)
    # Per capita indicators
    # for indicator in ['fec','ghg_sector','ghg_source']:
    #     country_results[(indicator,'percap')] = country_results[(indicator,'total')] * 1000 / tot_results[('pop','total',country)]
    # country_results['ghg_nes','percap'] = ghg_source['nes'] * 1000 / tot_results[('pop','total',country)]
    # Relative reduction
    for indicator in ['fec','ghg_sector']:
        country_results[(indicator,'reduc')] = sf.reduction_rate(country_results[(indicator,'total')],100)
    
    ## List of input files
    if MAIN_PARAMS['HTML_TEMPLATE'] == "raw":
        id_section += 1
        html_items['MAIN'] += sf.title_to_output(sections[id_section][1], sections[id_section][0], MAIN_PARAMS['HTML_TEMPLATE'])
        html_items['MAIN'] += '<p>Input data taken from: <b>'+input_file_label+'</b></p>'
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %Hh%M")
    html_items['MAIN'] += f'<p style="text-align:right;font-size:small;">SEPIA v{__version__} @ {current_time}</p>'
    
    ## Writing HTML file
    with open(os.path.join(DIRNAME,'Template',MAIN_PARAMS['HTML_TEMPLATE']+'.html')) as f:
        html_output = f.read()
    for label in html_items:
        html_output = html_output.replace('{{'+label+'}}', html_items[label])
    with open(os.path.join(DIRNAME,'Results','Results_'+country+'.html'), 'w') as f:
        f.write(html_output)
    
    # Adding country results to total results DataFrame
    country_results.columns = pd.MultiIndex.from_tuples(map(lambda x: (x[0], x[1], country), country_results.columns))
    # tot_results = pd.concat([tot_results, country_results], axis=1)
    
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

# def flows_with_imports(flows, se_import_mix):
#     flows_imp = flows.copy()
#     for network in SE_NODES:
#         if ('imp',network,'') in flows_imp.columns:
#             flows_imp = flows_imp.drop([('imp',network,'')], axis=1)
#             for cat in CATEGORIES:
#                 if network == 'gaz_se' and cat == 'fos': # If fossil gas import, we add it to the existing primary gas node
#                     flows_imp[('gaz_pe','gaz_se','')] = flows_imp.get(('gaz_pe','gaz_se',''),0) + flows[('imp','gaz_se','')] * se_import_mix[('gaz_se',cat)]
#                     sf.balance_node(flows_imp, 'gaz_pe')
#                 elif network == 'lqf_se' and cat == 'fos': # If fossil liquid fuel import, we add it to the existing primary petroleum node
#                     flows_imp[('pet_pe','lqf_se','')] = flows_imp.get(('pet_pe','lqf_se',''),0) + flows[('imp','lqf_se','')] * se_import_mix[('lqf_se',cat)]
#                     sf.balance_node(flows_imp, 'pet_pe')
#                 elif (network,cat) in se_import_mix.columns:
#                     flows_imp[('imp_'+cat,network,'')] = flows[('imp',network,'')] * se_import_mix[(network,cat)]
#     for cat in CATEGORIES:
#         sf.balance_node(flows_imp, 'imp_'+cat)
#     return flows_imp
# def calculate_res(flows_bk, gross_net_ratio,gfec_breakdown):
#     ren_cov_ratios=pd.DataFrame()
#     gfec_breakdown_carrier=pd.DataFrame()
#     for final_node in FE_NODES:
#         # Share paths from final energy to primary energy nodes
#         selected_columns_cr = flows_bk.columns.get_level_values('Source').isin(PE_NODES)
#         flows_from_node = flows_bk.loc[:, selected_columns_cr]
#         flows_from_node = flows_from_node.groupby(level='Source', axis=1).sum()
#         cons_at_primary = sf.node_consumption(flows_from_node,PE_NODES+SI_NODES)
#         # Adding network losses for gross final consumption
#         if final_node in gross_net_ratio.columns: cons_at_primary = cons_at_primary.multiply(gross_net_ratio[final_node], axis=0)
#         ren_cov_ratios[final_node] = sf.share_percent(gfec_breakdown,100).get('ren',0)
#         gfec_breakdown = pd.concat([gfec_breakdown,gfec_breakdown_carrier], axis=1).groupby(axis=1, level=0).sum()
#     gfec_breakdown_pct = sf.share_percent(gfec_breakdown,100)
#     ren_cov_ratios['total'] = gfec_breakdown_pct['ren']
#     return ren_cov_ratios

def calculate_pec(flows):
    pec = sf.node_consumption(flows, PE_NODES)
    # Adding imports of renewable energy for gas and liquid fuels (fossil imports are already connected to PE_NODES)
    for energy in ['gaz','lqf']:
        if ('imp_ren',energy+'_se','') in flows.columns: pec['imp_ren'] = pec.get('imp_ren',0) + flows[('imp_ren',energy+'_se','')]
    # Subtracting exports of primary and secondary energies (excluding electricity, hydrogen and district heating)
    pec = sf.subtract_cons_from_node(pec, flows, 'exp', ['elc_se','hyd_se','vap_se'], PE_NODES)
    return pec

#   # Adding energy and non-energy GHGs + renaming, sorting and removing empty sectors
# def ghg_cleanup(ghg_sector,ghg_nes):
#     ghg_sector = pd.concat([ghg_sector,ghg_nes], axis=1)
#     ghg_sector = ghg_sector.rename(columns=lambda x:x+'_ghg').reindex(columns=GHG_SECTORS)
#     ghg_sector = ghg_sector.drop(ghg_sector.columns[ghg_sector.fillna(0).max()==0], axis=1)
#     return ghg_sector

## Iterating through all countries + Europe
# for country in ALL_COUNTRIES + list(country_groups):
#     se_import_mix = tot_se_import_mix if MAIN_PARAMS['USE_IMPORT_MIX'] and country in ALL_COUNTRIES else IMPORT_MIX
tot_results = generate_results(flows, country)
    # tot_results = generate_results(tot_flows[country], tot_results, country, se_import_mix)

# Raw csv results
for (filename, df) in [('Results_bigtable',tot_results),('Debug',tot_debug)]:
    if not df.empty:
        df = df.stack("Country").round(4)
        df.columns = df.columns.map('.'.join).str.strip('.')
        df.to_csv(os.path.join(DIRNAME,'Results',filename+'.csv'),encoding='utf-8',sep=';',decimal=',')
interval_time = sf.calc_time('File writing', interval_time)

sf.calc_time('Total run',start_time)