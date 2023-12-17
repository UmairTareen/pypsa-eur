

import pandas as pd # Read/analyse data
import pypsa
import os
import shutil
from pypsa.descriptors import get_switchable_as_dense as as_dense
import logging

scenario = "ncdr"
def prepare_files(simpl, cluster, opt, sector_opt, ll):
    """This function copies and renames the .nc file for the year 2020 to have similar wildcards for the excel generator"""

    file_name = 'elec_s_6_lv1.0__Co2L0.7-12H-T-H-B-I-A-dist1_2020.nc'
    new_file_name = f'elec_s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_2020.nc'
    source_directory = '../results/reff/postnetworks/'
    destination_directory = f'../results/{scenario}/postnetworks/'
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, new_file_name)
    shutil.copy(source_path, destination_path)
    
    # CSV files
    csv_source_directory = '../resources/reff/'
    csv_destination_directory = f'../resources/{scenario}/'
    csv_files = [f"energy_totals_s{simpl}_{cluster}_2020.csv", f"industrial_energy_demand_elec_s{simpl}_{cluster}_2020.csv"]

    for csv_file in csv_files:
        source_csv_path = os.path.join(csv_source_directory, csv_file)
        destination_csv_path = os.path.join(csv_destination_directory, csv_file)
        shutil.copy(source_csv_path, destination_csv_path)

prepare_files(simpl="", cluster="6", opt="", sector_opt="EQ0.7c-12H-T-H-B-I-A-dist1", ll="vopt")
    
    
def build_filename(simpl,cluster,opt,sector_opt,ll ,planning_horizon,prefix=f"../results/{scenario}/postnetworks/elec_"):
    return prefix+"s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc".format(
        simpl=simpl,
        cluster=cluster,
        opt=opt,
        sector_opt=sector_opt,
        ll=ll,
        planning_horizon=planning_horizon
    )

def process_network(simpl,cluster,opt,sector_opt,ll ,planning_horizon):
    results_dict = {}
    for country in countries:
     filename = build_filename(simpl,cluster,opt,sector_opt,ll ,planning_horizon)
     n = pypsa.Network(filename)
     config = snakemake.config
     
     energy_demand =pd.read_csv(f"../resources/{scenario}/energy_totals_s_"+str(cluster)+"_"+str(planning_horizon)+".csv", index_col=0).T
     
     if planning_horizon == 2020:
         H2_nonenergyy =0
     else:
      clever_industry = (
         pd.read_csv("../data/clever_Industry_"+str(planning_horizon)+".csv", index_col=0)).T

      H2_nonenergyy = clever_industry.loc["Non-energy consumption of hydrogen for the feedstock production"].filter(like=country).sum()
      H2_industry = clever_industry.loc["Total Final hydrogen consumption in industry"].filter(like=country).sum()
     industry_demand =pd.read_csv(f"../resources/{scenario}/industrial_energy_demand_elec_s_"+str(cluster)+"_"+str(planning_horizon)+".csv",index_col=0).T
     Rail_demand = energy_demand.loc["total rail"].filter(like=country).sum()
     agriculture_machinery_oil =energy_demand.loc["total agriculture machinery"]

     aviation_p = energy_demand.loc["total international aviation"] 

     navig_d = energy_demand.loc["total domestic navigation"]

     navig_i =energy_demand.loc["total international navigation"]
     naphta_t = industry_demand.loc["naphtha"]
     if planning_horizon == 2020:
         ammonia = 0
     else:   
         ammonia_t = industry_demand.loc["ammonia"]
         ammonia = ammonia_t.filter(like=country).sum()
     
     collection = []
     agri_oil = agriculture_machinery_oil.filter(like=country).sum()
     avaition = aviation_p.filter(like=country).sum()
     navigation = navig_d.filter(like=country).sum() + navig_i.filter(like=country).sum()
     if planning_horizon == 2020:
        navigation_oil = navigation * config["sector"]["shipping_oil_share"][2020]
        navigation_methanol = navigation * config["sector"]["shipping_methanol_share"][2020]
     if planning_horizon == 2030:
        navigation_oil = navigation * config["sector"]["shipping_oil_share"][2030]
        navigation_methanol = navigation * config["sector"]["shipping_methanol_share"][2030]
     elif planning_horizon == 2040:
        navigation_oil = navigation * config["sector"]["shipping_oil_share"][2040]
        navigation_methanol = navigation *config["sector"]["shipping_methanol_share"][2040]
     elif planning_horizon == 2050:
        navigation_oil = navigation * config["sector"]["shipping_oil_share"][2050]
        navigation_methanol = navigation * config["sector"]["shipping_methanol_share"][2050]
     naphta = naphta_t.filter(like=country).sum()
     collection.append(
        pd.Series(
            dict(label="agriculture machinery oil", source="oil", target="Agriculture", value=agri_oil)
        )
     )
     collection.append(
        pd.Series(
            dict(label="kerosene for aviation", source="oil", target="kerosene for aviation", value=avaition)
        )
     )
     collection.append(
        pd.Series(
            dict(label="naphtha for industry", source="oil", target="Non-energy", value=naphta)
        )
     )
     collection.append(
        pd.Series(
            dict(label="NH3", source="H2", target="ammonia for industry", value=ammonia)
        )
     )
     collection.append(
         pd.Series(
               dict(label="shipping oil", source="oil", target="shipping oil", value=navigation_oil)
         )
      )
     collection.append(
         pd.Series(
               dict(label="shipping methanol", source="methanol", target="shipping methanol", value=navigation_methanol)
         )
      )
     collection = pd.concat(collection, axis=1).T
    
     columns = ["label", "source", "target", "value"]

     gen = (
        (n.snapshot_weightings.generators @ n.generators_t.p).filter(like=country)
        .groupby(
            [
                n.generators.carrier,
                n.generators.carrier,
                n.generators.bus.map(n.buses.carrier),
            ]
        )
        .sum()
        .div(1e6)
     )  # TWh

     gen.index.set_names(columns[:-1], inplace=True)
     gen = gen.reset_index(name="value")
     gen = gen.loc[gen.value > 0.1]

     gen["source"] = gen["source"].replace({"gas": "fossil gas", "oil": "fossil oil"})
     gen["label"] = gen["label"].replace({"gas": "fossil gas", "oil": "fossil oil"})

     sto = (
        (n.snapshot_weightings.generators @ n.stores_t.p).filter(like=country)
        .groupby(
            [n.stores.carrier, n.stores.carrier, n.stores.bus.map(n.buses.carrier)]
        )
        .sum()
        .div(1e6)
     )
     sto.index.set_names(columns[:-1], inplace=True)
     sto = sto.reset_index(name="value")
     sto = sto.loc[sto.value > 0.1]

     su = (
        (n.snapshot_weightings.generators @ n.storage_units_t.p).filter(like=country)
        .groupby(
            [
                n.storage_units.carrier,
                n.storage_units.carrier,
                n.storage_units.bus.map(n.buses.carrier),
            ]
        )
        .sum()
        .div(1e6)
     )
     su.index.set_names(columns[:-1], inplace=True)
     su = su.reset_index(name="value")
     su = su.loc[su.value > 0.1]

     load = (
        (n.snapshot_weightings.generators @ as_dense(n, "Load", "p_set")).filter(like=country)
        .groupby([n.loads.carrier, n.loads.carrier, n.loads.bus.map(n.buses.carrier)])
        .sum()
        .div(1e6)
        .swaplevel()
     )  # TWh
     load.index.set_names(columns[:-1], inplace=True)
     load = load.reset_index(name="value")

     load = load.loc[~load.label.str.contains("emissions")]
     load.target += " demand"
     if planning_horizon != 2020:
      load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "value"] = H2_industry
     value=load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"]
     load.loc[load.label.str.contains("AC") & (load.label == "electricity"), "value"] = value - Rail_demand
     
     p_values = [int(key[1:]) for key in n.links_t.keys() if key.startswith("p") and key[1:].isdigit()]
     max_i = max(p_values, default=-1) + 1
     for i in range(max_i):
        n.links[f"total_e{i}"] = (
            n.snapshot_weightings.generators @ n.links_t[f"p{i}"]
        ).filter(like=country).div(
            1e6
        )  # TWh
        n.links[f"carrier_bus{i}"] = n.links[f"bus{i}"].map(n.buses.carrier).filter(like=country)

     def calculate_losses(x):
        energy_ports = x.loc[
            x.index.str.contains("carrier_bus") & ~x.str.contains("co2", na=False)
        ].index.str.replace("carrier_bus", "total_e")
        return -x.loc[energy_ports].sum()

     n.links[f"total_e{max_i}"] = n.links.apply(calculate_losses, axis=1).filter(like=country)    #e4 and bus 4 for bAU 2050
     n.links[f"carrier_bus{max_i}"] = "losses"

     df = pd.concat(
        [
            n.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum()[
                "total_e" + str(i)
            ]
            for i in range(1, max_i + 1)
        ]
     ).reset_index()
     df.columns = columns

    # fix heat pump energy balance


     hp = n.links.loc[n.links.carrier.str.contains("heat pump")]

     hp_t_elec = n.links_t.p0.filter(like="heat pump").filter(like=country)

     grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
     hp_elec = (
        (-n.snapshot_weightings.generators @ hp_t_elec).filter(like=country)
        .groupby(grouper)
        .sum()
        .div(1e6)
        .reset_index()
     )
     hp_elec.columns = columns

     df = df.loc[~(df.label.str.contains("heat pump") & (df.target == "losses"))]

     df.loc[df.label.str.contains("heat pump"), "value"] -= hp_elec["value"].values

     df.loc[df.label.str.contains("air heat pump"), "source"] = "air-sourced ambient"
     df.loc[
        df.label.str.contains("ground heat pump"), "source"
     ] = "ground-sourced ambient"

     df = pd.concat([df, hp_elec])
     df = df.set_index(["label", "source", "target"]).squeeze()
     df = pd.concat(
        [
            df.loc[df < 0].mul(-1),
            df.loc[df > 0].swaplevel(1, 2),
        ]
     ).reset_index()
     df.columns = columns

    # make DAC demand
     df.loc[df.label == "DAC", "target"] = "DAC"

     to_concat = [df, gen, su, sto, load,collection]
     connections = pd.concat(to_concat).sort_index().reset_index(drop=True)

    # aggregation

     src_contains = connections.source.str.contains
     trg_contains = connections.target.str.contains

     connections.loc[src_contains("low voltage"), "source"] = "AC"
     connections.loc[trg_contains("low voltage"), "target"] = "AC"
     connections.loc[src_contains("CCGT"), "source"] = "gas"
     connections.loc[trg_contains("CCGT"), "target"] = "AC"
     connections.loc[src_contains("OCGT"), "source"] = "gas"
     connections.loc[trg_contains("OCGT"), "target"] = "AC"
     connections.loc[src_contains("water tank"), "source"] = "water tank"
     connections.loc[trg_contains("water tank"), "target"] = "water tank"
     connections.loc[src_contains("solar thermal"), "source"] = "solar thermal"
     connections.loc[src_contains("battery"), "source"] = "battery"
     connections.loc[trg_contains("battery"), "target"] = "battery"
     connections.loc[src_contains("Li ion"), "source"] = "battery"
     connections.loc[trg_contains("Li ion"), "target"] = "battery"

     connections.loc[src_contains("heat") & ~src_contains("demand"), "source"] = "heat"
     connections.loc[trg_contains("heat") & ~trg_contains("demand"), "target"] = "heat"
     new_row1 = {'label': 'Rail Network',
            'source': 'Electricity grid',
            'target': 'Rail Network',
            'value': Rail_demand}
     new_row2 = {'label': 'H2 for non-energy',
            'source': 'hyd',
            'target': 'Non-energy',
            'value': H2_nonenergyy}

     connections.loc[len(connections)] = pd.Series(new_row1)
     connections.loc[len(connections)] = pd.Series(new_row2)

     connections = connections.loc[
        ~(connections.source == connections.target)
        & ~connections.source.str.contains("co2")
        & ~connections.target.str.contains("co2")
        & ~connections.source.str.contains("emissions")
        & ~connections.source.isin(["gas for industry", "solid biomass for industry"])
        & (connections.value >= 0.1)
     ]

     where = connections.label == "urban central gas boiler"
     connections.loc[where] = connections.loc[where].replace("losses", "fossil gas")

     connections.replace("AC", "electricity grid", inplace=True)
    
     suffix_counter = {}

     def generate_new_label(label):
      if label in suffix_counter:
        suffix_counter[label] += 1
      else:
        suffix_counter[label] = 1

      if suffix_counter[label] > 1: 
        return f"{label}_{suffix_counter[label]}"
      return label
     
     connections['label'] = connections['label'].apply(generate_new_label)
     if country == 'BE' and planning_horizon !=2020:
      new_value = n.links_t.p0['BE1 0 BEV charger'].sum() / 1e6
      connections.loc[connections.label.str.contains("BEV charger"), "value"] = new_value
      connections.loc[connections.label.str.contains("BEV charger_2"), "value"] = new_value * (1-config["sector"]["bev_charge_efficiency"])
     connections.rename(columns={'value': str(planning_horizon)}, inplace=True)
     results_dict[country] = connections

    return results_dict
#%%
entries_to_select = ['solar', 'solar rooftop', 'onwind','offwind',
                     'offwind-ac', 'offwind-dc', 'hydro', 'ror','nuclear',
                     'coal', 'lignite', 'CCGT', 'urban central solid biomass CHP', 'urban central solid biomass CHP_2',
                     'H2 Electrolysis', 'methanolisation', 'Haber-Bosch', 'SMR', 'SMR CC', 'nuclear_2', 'coal_2', 'lignite_2',
                     'CCGT_2', 'urban central solid biomass CHP_3', 'H2 Electrolysis_2', 'methanolisation_2', 'Haber-Bosch_2',
                     'residential rural biomass boiler_2', 'residential urban decentral biomass boiler_2', 'services rural biomass boiler_2',
                     'services urban decentral biomass boiler_2','residential rural gas boiler_2', 'residential urban decentral gas boiler_2',
                     'services rural gas boiler_2','services urban decentral gas boiler_2','residential rural oil boiler_2',
                     'residential urban decentral oil boiler_2','services rural oil boiler_2','services urban decentral oil boiler_2', 
                     'residential rural resistive heater_3', 'residential urban decentral resistive heater_3', 'services rural resistive heater_3', 'services urban decentral resistive heater_3',
                     'residential rural resistive heater_4', 'residential urban decentral resistive heater_4', 'services rural resistive heater_4', 'services urban decentral resistive heater_4',
                     'electricity', 'Rail Network', 'residential rural biomass boiler','residential urban decentral biomass boiler',
                     'services rural biomass boiler','services urban decentral biomass boiler','residential rural gas boiler','residential urban decentral gas boiler',
                     'services rural gas boiler','services urban decentral gas boiler','residential rural oil boiler','residential urban decentral oil boiler',
                     'services rural oil boiler','services urban decentral oil boiler','residential rural ground heat pump',
                     'residential rural ground heat pump_2','residential urban decentral air heat pump','residential urban decentral air heat pump_2',
                     'services rural ground heat pump','services rural ground heat pump_2','services urban decentral air heat pump','services urban decentral air heat pump_2',
                     'residential rural ground heat pump_3','residential rural ground heat pump_4','residential urban decentral air heat pump_3','residential urban decentral air heat pump_4',
                     'services rural ground heat pump_3','services rural ground heat pump_4','services urban decentral air heat pump_3','services urban decentral air heat pump_4',
                     'residential rural resistive heater','residential rural resistive heater_2','residential urban decentral resistive heater','residential urban decentral resistive heater_2',
                     'services rural resistive heater','services rural resistive heater_2','services urban decentral resistive heater','services urban decentral resistive heater_2',
                     'land transport oil','land transport fuel cell','land transport EV','kerosene for aviation','shipping oil','shipping methanol',
                     'solid biomass for industry','solid biomass for industry CC','gas for industry','gas for industry CC','industry electricity',
                     'low-temperature heat for industry','H2 for industry','naphtha for industry','H2 for non-energy','agriculture machinery oil','agriculture electricity',
                     'agriculture heat','BEV charger','BEV charger_2','V2G','V2G_2','Haber-Bosch_3','NH3','residential rural water tanks charger','residential rural water tanks discharger',
                     'residential urban decentral water tanks charger','residential urban decentral water tanks discharger','services rural water tanks charger',
                     'services rural water tanks discharger','services urban decentral water tanks charger','services urban decentral water tanks discharger',
                     'urban central air heat pump','urban central air heat pump_2','urban central gas boiler','urban central gas boiler_2',
                     'urban central oil boiler','urban central resistive heater','urban central resistive heater_2','urban central water tanks charger',
                     'urban central water tanks discharger','residential urban decentral water tanks charger_2','residential urban decentral water tanks discharger_2',
                     'services rural water tanks charger_2','services rural water tanks discharger_2','services urban decentral water tanks charger_2',
                     'services urban decentral water tanks discharger_2','urban central water tanks charger_2','urban central water tanks discharger_2',
                     'urban central air heat pump_2_2','urban central air heat pump_3','urban central air heat pump_4','residential rural resistive heater_2_2',
                     'residential urban decentral resistive heater_2_2','services rural resistive heater_2_2','services urban decentral resistive heater_2_2',
                     'urban central resistive heater_2_2','residential rural ground heat pump_2_2','residential urban decentral air heat pump_2_2',
                     'services rural ground heat pump_2_2','services urban decentral air heat pump_2_2','solid biomass for industry CC_2',
                     'gas for industry CC_2','SMR_2','SMR CC_2','methanolisation_3','urban central heat','oil','oil_2','biomass to liquid','biomass to liquid_2',
                     'residential rural water tanks charger_2','residential rural water tanks discharger_2','battery charger','battery charger_2',
                     'battery discharger','battery discharger_2','H2 turbine','H2 turbine_2','Fischer-Tropsch','Fischer-Tropsch_2','Fischer-Tropsch_3',
                     'urban central gas CHP','urban central gas CHP_2','urban central gas CHP_3','OCGT','OCGT_2','biogas to gas','BioSNG','Sabatier',
                     'urban central solid biomass CHP CC','urban central solid biomass CHP CC_2','urban central solid biomass CHP CC_3',
                     'DAC','DAC_2','DAC_3','H2 for shipping'] # Add moe entries if needed

entry_label_mapping = {
    'solar': {'label': 'Solar photovoltaic Production', 'source': 'TWh', 'target': 'prospv'},
    'solar rooftop': {'label': 'Solar photovoltaic Production Rooftop', 'source': 'TWh', 'target': 'prospvr'},
    'onwind': {'label': 'Onshore wind-generated electricity', 'source': 'TWh', 'target': 'proeon'},
    'offwind': {'label': 'Offshore wind-generated electricity', 'source': 'TWh', 'target': 'proeof'},
    'offwind-ac': {'label': 'Offshore wind-generated electricity', 'source': 'TWh', 'target': 'proeofac'},
    'offwind-dc': {'label': 'Offshore wind-generated electricity', 'source': 'TWh', 'target': 'proeofdc'},
    'hydro': {'label': 'Total hydropower production', 'source': 'TWh', 'target': 'prohdr'},
    'ror': {'label': 'Total ror production', 'source': 'TWh', 'target': 'prohdror'},
    'nuclear': {'label': 'Nuclear production', 'source': 'TWh', 'target': 'proelcnuc'},
    'coal': {'label': 'Coal-fired power generation', 'source': 'TWh', 'target': 'proelccms'},
    'lignite': {'label': 'lignite power generation', 'source': 'TWh', 'target': 'proelign'},
    'CCGT': {'label': 'Gas-fired power generation', 'source': 'TWh', 'target': 'proelcgaz'},
    'urban central solid biomass CHP': {'label': 'Power output from solid biomass CHP plants', 'source': 'TWh', 'target': 'prbelcchpboi'},
    'urban central solid biomass CHP_2': {'label': 'Heat output from solid biomass CHP plants', 'source': 'TWh', 'target': 'prbvapchpboi'},
    'H2 Electrolysis': {'label': 'Production of H2 via electrolysis', 'source': 'TWh', 'target': 'prohyd'},
    'methanolisation': {'label': 'Production of methanol via methanolisation', 'source': 'TWh', 'target': 'promethanol'},
    'Haber-Bosch': {'label': 'Production of liquid ammonia from electricity', 'source': 'TWh', 'target': 'prohydcl'},
    'SMR': {'label': 'Production of H2 via steam methane reforming', 'source': 'TWh', 'target': 'prohydgaz'},
    'SMR CC': {'label': 'Production of H2 via steam methane reforming', 'source': 'TWh', 'target': 'prohydgazcc'},
    'nuclear_2': {'label': 'Nuclear production', 'source': 'TWh', 'target': 'proelcura'},
    'coal_2': {'label': 'Transformation losses (coal-fired powerplants)', 'source': 'TWh', 'target': 'losscoal'},
    'lignite_2': {'label': 'Transformation losses (coal-fired powerplants)', 'source': 'TWh', 'target': 'losslig'},
    'CCGT_2': {'label': 'Transformation losses (gas-fired powerplants)', 'source': 'TWh', 'target': 'lossgas'},
    'urban central solid biomass CHP_3': {'label': 'Transformation losses (solid biomass CHP plants)', 'source': 'TWh', 'target': 'lossbchp'},
    'H2 Electrolysis_2': {'label': 'Transformation losses (electrolysis)', 'source': 'TWh', 'target': 'losshely'},
    'methanolisation_2': {'label': 'Transformation losses (methanolisation)', 'source': 'TWh', 'target': 'lossmet'},
    'Haber-Bosch_2': {'label': 'Transformation losses (Production of ammonia from electricity)', 'source': 'TWh', 'target': 'lossef'},
    'residential rural biomass boiler_2': {'label': 'Transformation losses (solid biomass boilers)', 'source': 'TWh', 'target': 'lossbb'},
    'residential urban decentral biomass boiler_2': {'label': 'Transformation losses (solid biomass boilers)', 'source': 'TWh', 'target': 'lossbbb'},
    'services rural biomass boiler_2': {'label': 'Transformation losses (solid biomass boilers)', 'source': 'TWh', 'target': 'losssb'},
    'services urban decentral biomass boiler_2': {'label': 'Transformation losses (solid biomass boilers)', 'source': 'TWh', 'target': 'losssbbb'},
    'residential rural gas boiler_2': {'label': 'Transformation losses (gas-fired boilers)', 'source': 'TWh', 'target': 'lossgb'},
    'residential urban decentral gas boiler_2': {'label': 'Transformation losses (gas-fired boilers)', 'source': 'TWh', 'target': 'lossgbb'},
    'services rural gas boiler_2': {'label': 'Transformation losses (gas-fired boilers)', 'source': 'TWh', 'target': 'lossgbs'},
    'services urban decentral gas boiler_2': {'label': 'Transformation losses (gas-fired boilers)', 'source': 'TWh', 'target': 'lossgbss'},
    'residential rural oil boiler_2': {'label': 'Transformation losses (oil-fired boilers)', 'source': 'TWh', 'target': 'lossob'},
    'residential urban decentral oil boiler_2': {'label': 'Transformation losses (oil-fired boilers)', 'source': 'TWh', 'target': 'lossobb'},
    'services rural oil boiler_2': {'label': 'Transformation losses (oil-fired boilers)', 'source': 'TWh', 'target': 'lossobs'},
    'services urban decentral oil boiler_2': {'label': 'Transformation losses (oil-fired boilers)', 'source': 'TWh', 'target': 'lossobss'},
    'residential rural resistive heater_3': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrh'},
    'residential urban decentral resistive heater_3': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhh'},
    'services rural resistive heater_3': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhs'},
    'services urban decentral resistive heater_3': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhss'},
    'residential rural resistive heater_4': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhsh'},
    'residential urban decentral resistive heater_4': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhshs'},
    'services rural resistive heater_4': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhshsh'},
    'services urban decentral resistive heater_4': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhshss'},
    'electricity': {'label': 'electricity demand of residential and tertairy', 'source': 'TWh', 'target': 'preselccfres'},
    'Rail Network': {'label': 'electricity demand for rail network', 'source': 'TWh', 'target': 'preserail'},
    'urban central heat': {'label': 'Residential and tertiary DH demand', 'source': 'TWh', 'target': 'presvapcfdhs'},
    'residential rural biomass boiler': {'label': 'Residential and tertiary biomass for heating', 'source': 'TWh', 'target': 'presenccfres'},
    'residential urban decentral biomass boiler': {'label': 'Residential and tertiary biomass for heating', 'source': 'TWh', 'target': 'presenccfb'},
    'services rural biomass boiler': {'label': 'Residential and tertiary biomass for heating', 'source': 'TWh', 'target': 'presenccfbb'},
    'services urban decentral biomass boiler': {'label': 'Residential and tertiary biomass for heating', 'source': 'TWh', 'target': 'presenccfbbb'},
    'residential rural gas boiler': {'label': 'Residential and tertiary gas for heating', 'source': 'TWh', 'target': 'presgazcfres'},
    'residential urban decentral gas boiler': {'label': 'Residential and tertiary gas for heating', 'source': 'TWh', 'target': 'presgazcfg'},
    'services rural gas boiler': {'label': 'Residential and tertiary gas for heating', 'source': 'TWh', 'target': 'presgazcfgg'},
    'services urban decentral gas boiler': {'label': 'Residential and tertiary gas for heating', 'source': 'TWh', 'target': 'presgazcfggg'},
    'residential rural oil boiler': {'label': 'Residential and tertiary oil for heating', 'source': 'TWh', 'target': 'prespetcfres'},
    'residential urban decentral oil boiler': {'label': 'Residential and tertiary oil for heating', 'source': 'TWh', 'target': 'prespetcfo'},
    'services rural oil boiler': {'label': 'Residential and tertiary oil for heating', 'source': 'TWh', 'target': 'prespetcfoo'},
    'services urban decentral oil boiler': {'label': 'Residential and tertiary oil for heating', 'source': 'TWh', 'target': 'prespetcfooo'},
    'residential rural ground heat pump': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfres'},
    'residential rural ground heat pump_2': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfra'},
    'residential urban decentral air heat pump': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfraa'},
    'residential urban decentral air heat pump_2': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfaaa'},
    'services rural ground heat pump': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfta'},
    'services rural ground heat pump_2': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccftaa'},
    'services urban decentral air heat pump': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccfftt'},
    'services urban decentral air heat pump_2': {'label': 'Residential and tertiary ambient sources for heating', 'source': 'TWh', 'target': 'prespaccffff'},
    'residential rural ground heat pump_3': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehhp'},
    'residential rural ground heat pump_4': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehhpp'},
    'residential urban decentral air heat pump_3': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehhh'},
    'residential urban decentral air heat pump_4': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehhhh'},
    'services rural ground heat pump_3': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehpp'},
    'services rural ground heat pump_4': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehppp'},
    'services urban decentral air heat pump_3': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplm'},
    'services urban decentral air heat pump_4': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplmm'},
    'residential rural resistive heater': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehpln'},
    'residential rural resistive heater_2': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplnn'},
    'residential urban decentral resistive heater': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplx'},
    'residential urban decentral resistive heater_2': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplxx'},
    'services rural resistive heater': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehply'},
    'services rural resistive heater_2': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplyy'},
    'services urban decentral resistive heater': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplyyy'},
    'services urban decentral resistive heater_2': {'label': 'Residential and tertairy heat from electric heaters and pumps', 'source': 'TWh', 'target': 'preehplz'},
    'land transport oil': {'label': 'oil to transport demand', 'source': 'TWh', 'target': 'preslqfcftra'},
    'land transport fuel cell': {'label': 'land transport hydrogen demand', 'source': 'TWh', 'target': 'preshydcftra'},
    'land transport EV': {'label': 'land transport EV', 'source': 'TWh', 'target': 'preselccftra'},
    'kerosene for aviation': {'label': 'aviation oil demand', 'source': 'TWh', 'target': 'preslqfcfavi'},
    'shipping oil': {'label': 'shipping oil', 'source': 'TWh', 'target': 'preslqfcffrewati'},
    'shipping methanol': {'label': 'shipping methanol', 'source': 'TWh', 'target': 'presngvcffrewati'},
    'H2 for shipping': {'label': 'shipping methanol', 'source': 'TWh', 'target': 'preshydwati'},
    'solid biomass for industry': {'label': 'solid biomass for Industry', 'source': 'TWh', 'target': 'presenccfind'},
    'solid biomass for industry CC': {'label': 'solid biomass for Industry CC', 'source': 'TWh', 'target': 'presenccfindd'},
    'gas for industry': {'label': 'gas for Industry', 'source': 'TWh', 'target': 'presgazcfind'},
    'gas for industry CC': {'label': 'gas for Industry CC', 'source': 'TWh', 'target': 'presgazcfindd'},
    'industry electricity': {'label': 'electricity for Industry', 'source': 'TWh', 'target': 'preselccfind'},
    'low-temperature heat for industry': {'label': 'low-temperature heat for industry', 'source': 'TWh', 'target': 'presvapcfind'},
    'H2 for industry': {'label': 'hydrogen for industry', 'source': 'TWh', 'target': 'preshydcfind'},
    'naphtha for industry': {'label': 'naphtha for non-energy', 'source': 'TWh', 'target': 'prespetcfneind'},
    'H2 for non-energy': {'label': 'H2 for non-energy', 'source': 'TWh', 'target': 'preshydcfneind'},
    'agriculture machinery oil': {'label': 'agriculture oil', 'source': 'TWh', 'target': 'prespetcfagr'},
    'agriculture electricity': {'label': 'agriculture electricity', 'source': 'TWh', 'target': 'preselccfagr'},
    'agriculture heat': {'label': 'agriculture heat', 'source': 'TWh', 'target': 'presvapcfagr'},
    'BEV charger': {'label': 'BEV charging', 'source': 'TWh', 'target': 'prebev'},
    'BEV charger_2': {'label': 'BEV charging losses', 'source': 'TWh', 'target': 'prebevloss'},
    'V2G': {'label': 'vehicle to grid', 'source': 'TWh', 'target': 'prevtg'},
    'V2G_2': {'label': 'vehicle to grid losses', 'source': 'TWh', 'target': 'prevtgloss'},
    'Haber-Bosch_3': {'label': 'Production of ammonia from hydrogen', 'source': 'TWh', 'target': 'prohydclam'},
    'NH3': {'label': 'ammonia for industry', 'source': 'TWh', 'target': 'preammind'},
    'residential rural water tanks charger': {'label': 'TES charging', 'source': 'TWh', 'target': 'prechates'},
    'residential rural water tanks discharger': {'label': 'TES discharging', 'source': 'TWh', 'target': 'pretesdis'},
    'residential urban decentral water tanks charger': {'label': 'TES charging', 'source': 'TWh', 'target': 'prechatess'},
    'residential urban decentral water tanks discharger': {'label': 'TES discharging', 'source': 'TWh', 'target': 'pretesdiss'},
    'services rural water tanks charger': {'label': 'TES charging', 'source': 'TWh', 'target': 'prechatesss'},
    'services rural water tanks discharger': {'label': 'TES discharging', 'source': 'TWh', 'target': 'pretesdisss'},
    'services urban decentral water tanks charger': {'label': 'TES charging', 'source': 'TWh', 'target': 'prechatesx'},
    'services urban decentral water tanks discharger': {'label': 'TES discharging', 'source': 'TWh', 'target': 'pretesdissx'},
    'urban central air heat pump': {'label': 'Heat energy output from centralised heat pumps', 'source': 'TWh', 'target': 'prbrchpac'},
    'urban central air heat pump_2': {'label': 'Heat energy output from centralised heat pumps', 'source': 'TWh', 'target': 'prbrchpacc'},
    'urban central gas boiler': {'label': 'Heat energy output from gas-fired boilers', 'source': 'TWh', 'target': 'prbrchgaz'},
    'urban central gas boiler_2': {'label': 'Heat energy output from gas-fired boilers', 'source': 'TWh', 'target': 'prbrchgazz'},
    'urban central oil boiler': {'label': 'Heat energy output from oil-fired boilers', 'source': 'TWh', 'target': 'prbrchpet'},
    'urban central resistive heater': {'label': 'Heat energy output from centralised resistive heaters', 'source': 'TWh', 'target': 'prbchresh'},
    'urban central resistive heater_2': {'label': 'Heat energy output from centralised resistive heaters', 'source': 'TWh', 'target': 'prbchreshh'},
    'urban central water tanks charger': {'label': 'TES charging', 'source': 'TWh', 'target': 'pretesdhs'},
    'urban central water tanks discharger': {'label': 'TES discharging', 'source': 'TWh', 'target': 'pretesdhd'},
    'residential urban decentral water tanks charger_2': {'label': 'TES charging losses', 'source': 'TWh', 'target': 'pretesslos'},
    'residential urban decentral water tanks discharger_2': {'label': 'TES discharging losses', 'source': 'TWh', 'target': 'pretesdlos'},
    'services rural water tanks charger_2': {'label': 'TES charging losses', 'source': 'TWh', 'target': 'pretessloss'},
    'services rural water tanks discharger_2': {'label': 'TES discharging losses', 'source': 'TWh', 'target': 'pretesdloss'},
    'services urban decentral water tanks charger_2': {'label': 'TES charging losses', 'source': 'TWh', 'target': 'pretesslosss'},
    'services urban decentral water tanks discharger_2': {'label': 'TES discharging losses', 'source': 'TWh', 'target': 'pretesdlosss'},
    'urban central water tanks charger_2': {'label': 'TES charging losses', 'source': 'TWh', 'target': 'predhsclos'},
    'urban central water tanks discharger_2': {'label': 'TES discharging losses', 'source': 'TWh', 'target': 'predhsdlos'},
    'urban central air heat pump_2_2': {'label': 'Heat energy output from centralised electric heat pumps', 'source': 'TWh', 'target': 'prbrchpee'},
    'urban central air heat pump_3': {'label': 'Heat energy output from centralised electric heat pumps', 'source': 'TWh', 'target': 'prbrchpeee'},
    'urban central air heat pump_4': {'label': 'Heat energy output from centralised electric heat pumps', 'source': 'TWh', 'target': 'prbrchpeeee'},
    'residential rural resistive heater_2_2': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhx'},
    'residential urban decentral resistive heater_2_2': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhxx'},
    'services rural resistive heater_2_2': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhy'},
    'services urban decentral resistive heater_2_2': {'label': 'Transformation losses (resistive heaters)', 'source': 'TWh', 'target': 'lossrhyy'},
    'urban central resistive heater_2_2': {'label': 'Losses from centralised resistive heaters', 'source': 'TWh', 'target': 'losselchh'},
    'residential rural ground heat pump_2_2': {'label': 'Residential and tertiary electric HP for heating', 'source': 'TWh', 'target': 'preehpx'},
    'residential urban decentral air heat pump_2_2': {'label': 'Residential and tertiary electric HP for heating', 'source': 'TWh', 'target': 'preehpxx'},
    'services rural ground heat pump_2_2': {'label': 'Residential and tertiary electric HP for heating', 'source': 'TWh', 'target': 'preehpy'},
    'services urban decentral air heat pump_2_2': {'label': 'Residential and tertiary electric HP sources for heating', 'source': 'TWh', 'target': 'preehpyy'},
    'solid biomass for industry CC_2': {'label': 'Transformation losses biomass for industry CC', 'source': 'TWh', 'target': 'lossbmind'},
    'gas for industry CC_2': {'label': 'Transformation losses gas for industry CC', 'source': 'TWh', 'target': 'lossgasind'},
    'SMR_2': {'label': 'Transformation losses (steam methane reforming)', 'source': 'TWh', 'target': 'lossmr'},
    'SMR CC_2': {'label': 'Transformation losses (steam methane reforming)', 'source': 'TWh', 'target': 'lossmrr'},
    'methanolisation_3': {'label': 'electricity to metaholisation', 'source': 'TWh', 'target': 'pretareen'},
    'residential rural water tanks charger_2': {'label': 'TES charging', 'source': 'TWh', 'target': 'preclochar'},
    'residential rural water tanks discharger_2': {'label': 'TES discharging', 'source': 'TWh', 'target': 'preclocharr'},
    'oil': {'label': 'Oil generation losses', 'source': 'TWh', 'target': 'pof'},
    'oil_2': {'label': 'Oil generation', 'source': 'TWh', 'target': 'proelcpet'},
    'biomass to liquid': {'label': 'biomass to liquid', 'source': 'TWh', 'target': 'probmliqu'},
    'biomass to liquid_2': {'label': 'biomass to liquid losses', 'source': 'TWh', 'target': 'probmliqlos'},
    'battery charger': {'label': 'battery charger', 'source': 'TWh', 'target': 'probattchg'},
    'battery charger_2': {'label': 'battery charger losses', 'source': 'TWh', 'target': 'probattchlos'},
    'battery discharger': {'label': 'battery discharger', 'source': 'TWh', 'target': 'probattdhg'},
    'battery discharger_2': {'label': 'battery discharger losses', 'source': 'TWh', 'target': 'probattdhlos'},
    'H2 turbine': {'label': 'hydrogen turbine', 'source': 'TWh', 'target': 'proelchyd'},
    'H2 turbine_2': {'label': 'hydrogen turbine losses', 'source': 'TWh', 'target': 'fftfy'},
    'Fischer-Tropsch': {'label': 'Fischer-Tropsch', 'source': 'TWh', 'target': 'profischer'},
    'Fischer-Tropsch_2': {'label': 'Fischer-Tropsch heat', 'source': 'TWh', 'target': 'profischerh'},
    'Fischer-Tropsch_3': {'label': 'Fischer-Tropsch losses', 'source': 'TWh', 'target': 'profischerlo'},
    'urban central gas CHP': {'label': 'Power output from gas CHP plants', 'source': 'TWh', 'target': 'prbelcchpgaz'},
    'urban central gas CHP_2': {'label': 'Heat output from gas CHP plants', 'source': 'TWh', 'target': 'prbvapchpgaz'},
    'urban central gas CHP_3': {'label': 'Losses from gas CHP plants', 'source': 'TWh', 'target': 'dqzf'},
    'OCGT': {'label': 'Gas-fired power generation', 'source': 'TWh', 'target': 'proelcgazoc'},
    'OCGT_2': {'label': 'Gas-fired power generation losses', 'source': 'TWh', 'target': 'proelcgazoclo'},
    'biogas to gas': {'label': 'biogas to gas', 'source': 'TWh', 'target': 'prodombgr'},
    'BioSNG': {'label': 'BioSNG', 'source': 'TWh', 'target': 'probiosng'},
    'Sabatier': {'label': 'Sabatier', 'source': 'TWh', 'target': 'presaba'},
    'urban central solid biomass CHP CC': {'label': 'Power output from solid biomass CHP plants', 'source': 'TWh', 'target': 'prbelccbmcc'},
    'urban central solid biomass CHP CC_2': {'label': 'Heat output from solid biomass CHP plants', 'source': 'TWh', 'target': 'prbelccbmcch'},
    'urban central solid biomass CHP CC_3': {'label': 'losses solid biomass CHP plants', 'source': 'TWh', 'target': 'prbelccbmccl'},
    'DAC': {'label': 'DAC', 'source': 'TWh', 'target': 'predacelc'},
    'DAC_2': {'label': 'DAC', 'source': 'TWh', 'target': 'predache'},
    'DAC_3': {'label': 'DAC', 'source': 'TWh', 'target': 'predachee'},
}


# %%
def prepare_emissions(simpl,cluster,opt,sector_opt,ll,planning_horizon):
    '''
    This function prepare the data for co2 emissions sankey chart. All the emissions from
    the technologies are compiled together
    '''
    results_dict_co2 = {}
    for country in countries:
     filename = build_filename(simpl,cluster,opt,sector_opt,ll ,planning_horizon)
     n = pypsa.Network(filename)
     fn = snakemake.input.costs
     options = pd.read_csv(fn ,index_col=[0, 1]).sort_index()

     collection = []

     # DAC
     value = -(n.snapshot_weightings.generators @ n.links_t.p1.filter(like="DAC").filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label="DAC", source="co2 atmosphere", target="co2 stored", value=value)
        )
         )

     # process emissions
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="process emissions$")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="process emissions",
                source="process emissions",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

     # process emissions CC
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="process emissions CC")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="process emissions CC",
                source="process emissions",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

     value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="process emissions CC")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="process emissions CC",
                source="process emissions",
                target="co2 stored",
                value=value,
            )
             )
              )

     # OCGT
     value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="OCGT").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="OCGT", source="gas", target="co2 atmosphere", value=value))
     )
     
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="OCGT").filter(like=country)
     ).sum() * options.loc[("gas", "CO2 intensity"), "value"]
     collection.append(
         pd.Series(dict(label="OCGT", source="gas", target="co2 atmosphere", value=value))
     )

     value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="CCGT").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="CCGT", source="gas", target="co2 atmosphere", value=value))
     )
     
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="CCGT").filter(like=country)
     ).sum() * options.loc[("gas", "CO2 intensity"), "value"]
     collection.append(
         pd.Series(dict(label="CCGT", source="gas", target="co2 atmosphere", value=value))
     )
     

     value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="lignite").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="lignite", source="coal", target="co2 atmosphere", value=value))
     )
     
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="lignite").filter(like=country)
     ).sum() * options.loc[("lignite", "CO2 intensity"), "value"]
     collection.append(
         pd.Series(dict(label="lignite", source="coal", target="co2 atmosphere", value=value))
     )

     value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(like="coal").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="coal", source="coal", target="co2 atmosphere", value=value))
     )
     
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="coal").filter(like=country)
     ).sum() * options.loc[("coal", "CO2 intensity"), "value"]
     collection.append(
         pd.Series(dict(label="coal", source="coal", target="co2 atmosphere", value=value))
     )

     # Sabatier
     value = (n.snapshot_weightings.generators @ n.links_t.p2.filter(like="Sabatier").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="Sabatier", source="co2 stored", target="gas", value=value))
     )

     # SMR
     value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="SMR$").filter(like=country)).sum()
     collection.append(
        pd.Series(dict(label="SMR", source="gas", target="co2 atmosphere", value=value))
     )

     # SMR CC
     if "p3" in n.links_t:
      value = -(n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="SMR CC").filter(like=country)).sum()
      collection.append(
        pd.Series(dict(label="SMR CC", source="gas", target="co2 atmosphere", value=value))
      )

      value = -(n.snapshot_weightings.generators @ n.links_t.p3.filter(like="SMR CC").filter(like=country)).sum()
      collection.append(
        pd.Series(dict(label="SMR CC", source="gas", target="co2 stored", value=value))
      )

    # gas boiler
     gas_boilers = [
        "residential rural gas boiler",
        "services rural gas boiler",
        "residential urban decentral gas boiler",
        "services urban decentral gas boiler",
        "urban central gas boiler",
      ]
     for gas_boiler in gas_boilers:
        value = -(
                n.snapshot_weightings.generators @ n.links_t.p2.filter(like=gas_boiler)
        .filter(like=country)).sum()
        collection.append(
            pd.Series(
                dict(label=gas_boiler, source="gas", target="co2 atmosphere", value=value)
            )
        )
    # oil boiler
     oil_boilers = [
        "residential rural oil boiler",
        "services rural oil boiler",
        "residential urban decentral oil boiler",
        "services urban decentral oil boiler",
        "urban central oil boiler",
      ]
     for oil_boiler in oil_boilers:
        value = -(
                n.snapshot_weightings.generators @ n.links_t.p2.filter(like=oil_boiler)
        .filter(like=country)).sum()
        collection.append(
            pd.Series(
                dict(label=oil_boiler, source="oil", target="co2 atmosphere", value=value)
            )
         )
     # biogas to gas
     value = (
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like="biogas to gas")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="biogas to gas", source="co2 atmosphere", target="biogas", value=value
            )
             )
              )
     collection.append(
        pd.Series(dict(label="biogas to gas", source="biogas", target="gas", value=value))
     )

     # solid biomass for industry
     value = (
            n.snapshot_weightings.generators
            @ n.links_t.p0.filter(regex="solid biomass for industry$")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="solid biomass for industry",
                source="solid biomass",
                target="co2 atmosphere",
                value=value * options.loc[("solid biomass", "CO2 intensity"), "value"],
            )
             )
              )
     collection.append(
        pd.Series(dict(label="solid biomass for industry", source="co2 atmosphere", target="solid biomass",
                       value=value * options.loc[("solid biomass", "CO2 intensity"), "value"]))
     )
     # solid biomass for industry CC
     if "p3" in n.links_t:
      value = (
            n.snapshot_weightings.generators
            @ n.links_t.p2.filter(like="solid biomass for industry CC")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="solid biomass for industry CC",
                source="co2 atmosphere",
                target="BECCS",
                value=value,
            )
             )
              )

      value = -(
            n.snapshot_weightings.generators
            @ n.links_t.p3.filter(like="solid biomass for industry CC")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="solid biomass for industry CC",
                source="BECCS",
                target="co2 stored",
                value=value,
            )
             )
              )

     # methanolisation
     if "p3" in n.links_t:
      value = (
            n.snapshot_weightings.generators @ n.links_t.p3.filter(like="methanolisation")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="methanolisation", source="co2 stored", target="methanol", value=value
                # C02 intensity of gas from cost data
            )
             )
              )
     # gas for industry
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(regex="gas for industry$")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="gas for industry", source="gas", target="co2 atmosphere", value=value
            )
             )
              )

     # gas for industry CC
     if "p3" in n.links_t:
      value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like="gas for industry CC")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="gas for industry CC",
                source="gas",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

      value = -(
            n.snapshot_weightings.generators @ n.links_t.p3.filter(like="gas for industry CC")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="gas for industry CC", source="gas", target="co2 stored", value=value
            )
             )
              )
     # solid biomass to gas
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p1.filter(regex="solid biomass solid biomass to gas$")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas", source="solid biomass", target="gas", value=value * options.loc[("BioSNG", "CO2 stored"), "value"]
                # CO2 stored in bioSNG from cost data
            )
             )
              )

     # solid biomass to gas CC
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p1.filter(like="solid biomass solid biomass to gas CC")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas CC",
                source="solid biomass",
                target="gas",
                value=value * options.loc[("BioSNG", "CO2 stored"), "value"],
            )
             )
              )
     collection.append(
        pd.Series(dict(label="solid biomass solid biomass to gas CC", source="co2 atmosphere", target="solid biomass",
                       value=value * options.loc[("BioSNG", "CO2 stored"), "value"]))
     )
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like="solid biomass solid biomass to gas CC")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="solid biomass solid biomass to gas CC", source="solid biomass", target="co2 stored", value=value
            )
             )
              )
     collection.append(
        pd.Series(dict(label="solid biomass solid biomass to gas CC", source="co2 atmosphere", target="solid biomass",
                       value=value))
     )
     # biomass boilers
     # residential urban decentral biomass boiler
     tech = "residential urban decentral biomass boiler"

     value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech).filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
         )
          )
     collection.append(
        pd.Series(
            dict(label="residential urban decentral biomass boiler", source="co2 atmosphere", target="solid biomass",
                 value=value * options.loc[("solid biomass", "CO2 intensity"), "value"]))
     )
     # services urban decentral biomass boiler
     tech = "services urban decentral biomass boiler"

     value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech).filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
        )
         )
     collection.append(
        pd.Series(dict(label="services urban decentral biomass boiler", source="co2 atmosphere", target="solid biomass",
                       value=value * options.loc[("solid biomass", "CO2 intensity"), "value"]))
     )
     # residential rural biomass boiler
     tech = "residential rural biomass boiler"

     value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech).filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
        )
         )
     collection.append(
        pd.Series(dict(label="residential rural biomass boiler", source="co2 atmosphere", target="solid biomass",
                       value=value * options.loc[("solid biomass", "CO2 intensity"), "value"]))
     )
    # services rural biomass boiler
     tech = "services rural biomass boiler"

     value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(like=tech).filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label=tech, source="solid biomass", target="co2 atmosphere", value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
        )
         )
     collection.append(
        pd.Series(dict(label="services rural biomass boiler", source="co2 atmosphere", target="solid biomass",
                       value=value * options.loc[("solid biomass", "CO2 intensity"), "value"]))
     )
     # Solid biomass to liquid
     value = -(
            n.snapshot_weightings.generators @ n.links_t.p1.filter(like="solid biomass biomass to liquid")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="solid biomass biomass to liquid", source="solid biomass", target="oil", value=value * options.loc[("BtL", "CO2 stored"), "value"]
                # C02 stored in oil by BTL from costs data 2050
            )
             )
              )
     collection.append(
        pd.Series(dict(label="solid biomass biomass to liquid", source="co2 atmosphere", target="solid biomass",
                       value=value *options.loc[("BtL", "CO2 stored"), "value"]))
      )
     # # solid biomass liquid to  CC
     if "p3" in n.links_t:
      value = -(
            n.snapshot_weightings.generators @ n.links_t.p3.filter(like="solid biomass biomass to liquid CC")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="solid biomass biomass to liquid CC", source="solid biomass", target="co2 stored", value=value
            )
             )
              )
      collection.append(
        pd.Series(dict(label="solid biomass biomass to liquid CC", source="co2 atmosphere", target="solid biomass",
                       value=value))
      )
     # Fischer-Tropsch
     value = (
            n.snapshot_weightings.generators @ n.links_t.p2.filter(like="Fischer-Tropsch")
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label="Fischer-Tropsch", source="co2 stored", target="oil", value=value)
        )
         )

     # urban central gas CHP
     if "p3" in n.links_t:
      value = -(
            n.snapshot_weightings.generators @ n.links_t.p3.filter(like="urban central gas CHP")
      .filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(
                label="urban central gas CHP",
                source="gas",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

    # urban central gas CHP CC
     if "p4" in n.links_t:
      tech = "urban central gas CHP CC"
      value = -(n.snapshot_weightings.generators @ n.links_t.p3.filter(like=tech).filter(like=country)).sum()
      collection.append(
        pd.Series(dict(label=tech, source="gas", target="co2 atmosphere", value=value))
      )

      value = -(n.snapshot_weightings.generators @ n.links_t.p4.filter(like=tech).filter(like=country)).sum()
      collection.append(
        pd.Series(dict(label=tech, source="gas", target="co2 stored", value=value))
      )

    # urban solid biomass CHP

     value = (n.snapshot_weightings.generators @ n.links_t.p0.filter(regex="urban central solid biomass CHP").filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label="urban central solid biomass CHP", source="solid biomass", target="co2 atmosphere",
                 value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
        )
          )
     collection.append(
        pd.Series(
            dict(label="urban central solid biomass CHP", source="co2 atmosphere", target="solid biomass",
                 value=value * options.loc[("solid biomass", "CO2 intensity"), "value"])
        )
         )
     
     if "p4" in n.links_t:
      # urban solid biomass CHP CC
      tech = "urban central solid biomass CHP CC"

      value = (n.snapshot_weightings.generators @ n.links_t.p3.filter(like=tech).filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(label=tech, source="co2 atmosphere", target="BECCS", value=value)
        )
       )

      value = -(n.snapshot_weightings.generators @ n.links_t.p4.filter(like=tech).filter(like=country)).sum()
      collection.append(
        pd.Series(
            dict(label=tech, source="BECCS", target="co2 stored", value=value)
        )
         )

    # oil emissions
     value = -(
            n.snapshot_weightings.generators
            @ as_dense(n, "Load", "p_set").filter(regex="^oil emissions", axis=1)
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(label="oil emissions", source="oil", target="co2 atmosphere", value=value)
        )
         )

     # agriculture machinery oil emissions
     value = -(
            n.snapshot_weightings.generators
            @ as_dense(n, "Load", "p_set").filter(
        like="agriculture machinery oil emissions", axis=1
     )
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="agriculture machinery oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

     value = -(
            n.snapshot_weightings.generators
            @ as_dense(n, "Load", "p_set").filter(
        like="land transport oil emissions", axis=1
     )
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="land transport oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

     value = -(
            n.snapshot_weightings.generators
            @ as_dense(n, "Load", "p_set").filter(
        like="shipping oil emissions", axis=1
     )
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="shipping oil emissions",
                source="oil",
                target="co2 atmosphere",
                value=value,
            )
             )
              )
     value = -(
            n.snapshot_weightings.generators
            @ as_dense(n, "Load", "p_set").filter(
        like="shipping methanol emissions", axis=1
     )
     .filter(like=country)).sum()
     collection.append(
        pd.Series(
            dict(
                label="shipping methanol emissions",
                source="methanol",
                target="co2 atmosphere",
                value=value,
            )
             )
              )

     
     cf = pd.concat(collection, axis=1).T
     cf.value /= 1e6  # Mt

     # fossil gas
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="gas").filter(like=country)).div(
        1e6
     ).sum() * options.loc[("gas", "CO2 intensity"), "value"]
     row = pd.DataFrame(
        [dict(label="fossil gas", source="fossil gas", target="gas", value=value)]
     )

     cf = pd.concat([cf, row], axis=0)

     # fossil oil
     # co2_intensity = 0.27  # t/MWh
     value = (n.snapshot_weightings.generators @ n.generators_t.p.filter(like="oil").filter(like=country)).div(
        1e6
     ).sum() * options.loc[("oil", "CO2 intensity"), "value"]
     row = pd.DataFrame(
        [dict(label="fossil oil", source="fossil oil", target="oil", value=value)]
     )
     cf = pd.concat([cf, row], axis=0)
     
     
     # sequestration
     value = (
            cf.loc[cf.target == "co2 stored", "value"].sum()
            - cf.loc[cf.source == "co2 stored", "value"].sum()
     )
     row = pd.DataFrame(
        [
            dict(
                label="co2 sequestration",
                source="co2 stored",
                target="co2 sequestration",
                value=value,
            )
             ]
              )

     cf = pd.concat([cf, row], axis=0)

     # net co2 emissions
     value = (
            cf.loc[cf.target == "co2 atmosphere", "value"].sum()
            - cf.loc[cf.source == "co2 atmosphere", "value"].sum()
     )
     row = pd.DataFrame(
        [
            dict(
                label="net co2 emissions",
                source="co2 atmosphere",
                target="net co2 emissions",
                value=value,
            )
             ]
              )
     cf = pd.concat([cf, row], axis=0)
     cf = cf.loc[(cf.value >= 0.1)]
     # LULUCF
     if planning_horizon != 2020:
      LULUCF = (
         pd.read_csv(f"../resources/{scenario}/co2_totals_s_"+str(cluster)+"_"+str(planning_horizon)+".csv", index_col=0)).T
      V = LULUCF.loc['LULUCF'].filter(like=country).sum()
      V=-V
      row = pd.DataFrame(
      [
        dict(
            label="LULUCF",
            source="co2 atmosphere",
            target="LULUCF",
            value=V,
        )
       ]
       )
      cf = pd.concat([cf, row], axis=0)
     suffix_counter = {}

     def generate_new_label(label):
        if label in suffix_counter:
            suffix_counter[label] += 1
        else:
            suffix_counter[label] = 1

        if suffix_counter[label] > 1:
            return f"{label}_{suffix_counter[label]}"
        return label

     cf['label'] = cf['label'].apply(generate_new_label)
     cf.rename(columns={'value': str(planning_horizon)}, inplace=True)
     results_dict_co2[country] = cf

    return results_dict_co2

#%%
entries_to_select_c = ['process emissions','process emissions CC','process emissions CC_2','CCGT','lignite','SMR CC','SMR CC_2',
                      'residential rural gas boiler','services rural gas boiler','residential urban decentral gas boiler',
                      'services urban decentral gas boiler','urban central gas boiler','residential rural oil boiler',
                      'services rural oil boiler','residential urban decentral oil boiler','services urban decentral oil boiler',
                      'solid biomass for industry','solid biomass for industry_2','methanolisation','gas for industry',
                      'residential urban decentral biomass boiler','residential urban decentral biomass boiler_2',
                      'services urban decentral biomass boiler','services urban decentral biomass boiler_2','residential rural biomass boiler',
                      'residential rural biomass boiler_2','services rural biomass boiler','services rural biomass boiler_2',
                      'oil emissions','agriculture machinery oil emissions','land transport oil emissions','shipping oil emissions',
                      'shipping methanol emissions','LULUCF','fossil gas','fossil oil','net co2 emissions','gas for industry CC',
                      'gas for industry CC_3','gas for industry CC_2','solid biomass for industry CC','solid biomass for industry CC_2',
                      'urban central solid biomass CHP','urban central solid biomass CHP_2','coal','solid biomass biomass to liquid',
                       'solid biomass biomass to liquid_2','biogas to gas','biogas to gas_2','urban central gas CHP','Fischer-Tropsch',
                       'OCGT','SMR','urban central solid biomass CHP CC','urban central solid biomass CHP CC_2','DAC','co2 sequestration',
                       'solid biomass solid biomass to gas','solid biomass solid biomass to gas CC','solid biomass solid biomass to gas CC_2',
                       'solid biomass solid biomass to gas CC_3','solid biomass solid biomass to gas CC_4','Sabatier'
                       ] # Add moe entries if needed

entry_label_mapping_c = {
    'process emissions': {'label': 'process emissions', 'source': 'MtCO2', 'target': 'emmprocess'},
    'process emissions CC': {'label': 'process emissions CC', 'source': 'MtCO2', 'target': 'emmprocesscc'},
    'process emissions CC_2': {'label': 'process emissions CC', 'source': 'MtCO2', 'target': 'emmprocessccst'},
    'CCGT': {'label': 'CCGT emissions', 'source': 'MtCO2', 'target': 'emmccgt'},
    'OCGT': {'label': 'OCGT emissions', 'source': 'MtCO2', 'target': 'emmocgt'},
    'lignite': {'label': 'lignite emissions', 'source': 'MtCO2', 'target': 'emmlig'},
    'SMR CC': {'label': 'SMR emissions', 'source': 'MtCO2', 'target': 'emmsmr'},
    'SMR CC_2': {'label': 'SMR emissions captored', 'source': 'MtCO2', 'target': 'emmsmrcc'},
    'residential rural gas boiler': {'label': 'res boiler', 'source': 'MtCO2', 'target': 'emmresbo'},
    'services rural gas boiler': {'label': 'serv boiler', 'source': 'MtCO2', 'target': 'emmserbo'},
    'residential urban decentral gas boiler': {'label': 'res urb boiler', 'source': 'MtCO2', 'target': 'emmresubbo'},
    'services urban decentral gas boiler': {'label': 'serv urb boiler', 'source': 'MtCO2', 'target': 'emmsesubbo'},
    'urban central gas boiler': {'label': 'urb boiler', 'source': 'MtCO2', 'target': 'emmurbbbo'},
    'residential rural oil boiler': {'label': 'res boiler', 'source': 'MtCO2', 'target': 'emmresoil'},
    'services rural oil boiler': {'label': 'serv boiler', 'source': 'MtCO2', 'target': 'emmserroil'},
    'residential urban decentral oil boiler': {'label': 'res urb boiler', 'source': 'MtCO2', 'target': 'emmresuoil'},
    'services urban decentral oil boiler': {'label': 'serv urb boiler', 'source': 'MtCO2', 'target': 'emmsesuboil'},
    'solid biomass for industry': {'label': 'solid biomass for industry', 'source': 'MtCO2', 'target': 'emmindbm'},
    'solid biomass for industry_2': {'label': 'solid biomass for industry', 'source': 'MtCO2', 'target': 'emmindbmatm'},
    'methanolisation': {'label': 'methanolisation', 'source': 'MtCO2', 'target': 'emmmet'},
    'gas for industry': {'label': 'gas for industry', 'source': 'MtCO2', 'target': 'emmgas'},
    'residential urban decentral biomass boiler': {'label': 'residential urban decentral biomass boiler', 'source': 'MtCO2', 'target': 'emmresbm'},
    'residential urban decentral biomass boiler_2': {'label': 'residential urban decentral biomass boiler', 'source': 'MtCO2', 'target': 'emmresbmatm'},
    'services urban decentral biomass boiler': {'label': 'residential urban decentral biomass boiler', 'source': 'MtCO2', 'target': 'emmserbm'},
    'services urban decentral biomass boiler_2': {'label': 'residential urban decentral biomass boiler', 'source': 'MtCO2', 'target': 'emmserbmatm'},
    'residential rural biomass boiler': {'label': 'res biomass boiler', 'source': 'MtCO2', 'target': 'emmresbmm'},
    'residential rural biomass boiler_2': {'label': 'res biomass boiler', 'source': 'MtCO2', 'target': 'emmresbmmatm'},
    'services rural biomass boiler': {'label': 'serv biomass boiler', 'source': 'MtCO2', 'target': 'emmserbmm'},
    'services rural biomass boiler_2': {'label': 'serv biomass boiler', 'source': 'MtCO2', 'target': 'emmserbmmatm'},
    'oil emissions': {'label': 'oil emissions', 'source': 'MtCO2', 'target': 'emmoil'},
    'agriculture machinery oil emissions': {'label': 'agriculture machinery oil emissions', 'source': 'MtCO2', 'target': 'emmoilagr'},
    'land transport oil emissions': {'label': 'land transport oil emissions', 'source': 'MtCO2', 'target': 'emmoiltra'},
    'shipping oil emissions': {'label': 'shipping oil emissions', 'source': 'MtCO2', 'target': 'emmoilwati'},
    'shipping methanol emissions': {'label': 'shipping methanol emissions', 'source': 'MtCO2', 'target': 'emmmetwati'},
    'LULUCF': {'label': 'LULUCF', 'source': 'MtCO2', 'target': 'emmluf'},
    'fossil gas': {'label': 'fossil gas', 'source': 'MtCO2', 'target': 'emmfossgas'},
    'fossil oil': {'label': 'fossil oil', 'source': 'MtCO2', 'target': 'emmfossoil'},
    'net co2 emissions': {'label': 'net co2 emissions', 'source': 'MtCO2', 'target': 'emmnet'},
    'gas for industry CC': {'label': 'gas for industry CC', 'source': 'MtCO2', 'target': 'emmgascc'},
    'gas for industry CC_2': {'label': 'gas for industry CC', 'source': 'MtCO2', 'target': 'emmgasccx'},
    'gas for industry CC_3': {'label': 'gas for industry CC', 'source': 'MtCO2', 'target': 'emmgasccxx'},
    'solid biomass for industry CC': {'label': 'solid biomass for industry CC', 'source': 'MtCO2', 'target': 'emmindatmbm'},
    'solid biomass for industry CC_2': {'label': 'solid biomass for industry CC', 'source': 'MtCO2', 'target': 'emmindbeccs'},
    'urban central solid biomass CHP': {'label': 'urban central solid biomass CHP', 'source': 'MtCO2', 'target': 'emmbmchp'},
    'urban central solid biomass CHP_2': {'label': 'urban central solid biomass CHP', 'source': 'MtCO2', 'target': 'emmbmchpatm'},
    'coal': {'label': 'coal emissions', 'source': 'MtCO2', 'target': 'emmcoal'},
    'solid biomass biomass to liquid': {'label': 'solid biomass to liquid', 'source': 'MtCO2', 'target': 'emmbmliq'},
    'solid biomass biomass to liquid_2': {'label': 'solid biomass to liquid', 'source': 'MtCO2', 'target': 'emmbmliqat'},
    'biogas to gas': {'label': 'biogas to gas', 'source': 'MtCO2', 'target': 'emmbiogasat'},
    'biogas to gas_2': {'label': 'biogas to gas', 'source': 'MtCO2', 'target': 'emmbiogas'},
    'urban central gas CHP': {'label': 'urban central gas CHP', 'source': 'MtCO2', 'target': 'emmgaschp'},
    'Fischer-Tropsch': {'label': 'Fischer-Tropsch', 'source': 'MtCO2', 'target': 'emmfischer'},
    'SMR': {'label': 'SMR emissions', 'source': 'MtCO2', 'target': 'emmsmrsm'},
    'urban central solid biomass CHP CC': {'label': 'urban central solid biomass CHP', 'source': 'MtCO2', 'target': 'emmbmchpcc'},
    'urban central solid biomass CHP CC_2': {'label': 'urban central solid biomass CHP', 'source': 'MtCO2', 'target': 'emmbmchpatmcc'},
    'DAC': {'label': 'DAC', 'source': 'MtCO2', 'target': 'emmdac'},
    'co2 sequestration': {'label': 'co2 sequestration', 'source': 'MtCO2', 'target': 'emmseq'},
    'solid biomass solid biomass to gas': {'label': 'solid biomass solid biomass to gas', 'source': 'MtCO2', 'target': 'emmbmsng'},
    'solid biomass solid biomass to gas CC': {'label': 'solid biomass solid biomass to gas CC', 'source': 'MtCO2', 'target': 'emmbmsngcc'},
    'solid biomass solid biomass to gas CC_2': {'label': 'solid biomass solid biomass to gas CC', 'source': 'MtCO2', 'target': 'emmbmsngccc'},
    'solid biomass solid biomass to gas CC_3': {'label': 'solid biomass solid biomass to gas CC', 'source': 'MtCO2', 'target': 'emmbmsngccca'},
    'solid biomass solid biomass to gas CC_4': {'label': 'solid biomass solid biomass to gas CC', 'source': 'MtCO2', 'target': 'emmbmsngcccb'},
    'Sabatier': {'label': 'Sabatier"', 'source': 'MtCO2', 'target': 'emmsaba'},
}
def write_to_excel(simpl, cluster, opt, sector_opt, ll, planning_horizons,countries,filename='../SEPIA/inputs_country.xlsx'):
    '''
    Function that writes the simulation results to the SEPIA excel input file
    :param filename_template: Template for the excel file name with a placeholder for the country
    '''

    for country in countries:
        # Generate the filename for the current country

        merged_df = process_network(simpl, cluster, opt, sector_opt, ll, planning_horizons[0])
        merged_df = merged_df[country]

        for planning_horizon in planning_horizons[1:]:
            temp = process_network(simpl, cluster, opt, sector_opt, ll, planning_horizon)
            temp = temp[country]
            merged_df = pd.merge(merged_df, temp, on=['label', 'source', 'target'], how='outer')

        # Fill missing values with 0
        merged_df = merged_df.fillna(0)
        connections = merged_df
        suffix_counter = {}

        def generate_new_label(label):
            if label in suffix_counter:
                suffix_counter[label] += 1
            else:
                suffix_counter[label] = 1

            if suffix_counter[label] > 1:
                return f"{label}_{suffix_counter[label]}"
            return label

        connections['label'] = connections['label'].apply(generate_new_label)

        df = connections
        selected_entries_df = pd.DataFrame()

        country_filename = ''.join(filename)[:-5] + country + ".xlsx"

        with pd.ExcelWriter(country_filename, engine='openpyxl') as writer:
            for entry in entries_to_select:
                selected_df = df[df['label'] == entry].copy()  # Create a copy of the DataFrame

                # Get the label mapping for the current entry
                label_mapping = entry_label_mapping.get(entry, {})

                # Replace the values in the selected DataFrame based on the mapping
                selected_df.loc[:, 'label'] = label_mapping.get('label', '')
                selected_df.loc[:, 'source'] = label_mapping.get('source', '')
                selected_df.loc[:, 'target'] = label_mapping.get('target', '')

                # Concatenate the selected entry to the main DataFrame
                selected_entries_df = pd.concat([selected_entries_df, selected_df])

            # Write the concatenated DataFrame to a new sheet
            selected_entries_df.to_excel(writer, sheet_name='Inputs', index=False)

        print(f'Excel file "{filename}" created with the selected entries on the "SelectedEntries" sheet.')

        list_as_set = set(entries_to_select)
        df_as_set = set(map(str, connections.label))
        # Find the different elements
        different_elements = list_as_set.symmetric_difference(df_as_set)
        # Convert the result back to a list
        different_elements_list = list(different_elements)
        print("Different elements between the list and DataFrame:", different_elements_list)


        #Deal with the emissions:
    for country in countries:
        merged_emissions = prepare_emissions(simpl, cluster, opt, sector_opt, ll, planning_horizons[0])
        merged_emissions = merged_emissions[country]
        for planning_horizon in planning_horizons[1:]:
            temp = prepare_emissions(simpl, cluster, opt, sector_opt, ll, planning_horizon)
            temp = temp[country]
            merged_emissions = pd.merge(merged_emissions, temp, on=['label', 'source', 'target'], how='outer')

    # Fill missing values with 0
        merged_emissions.fillna(0,inplace=True)
        country_filename = ''.join(filename)[:-5] + country + ".xlsx"

        selected_entries_cf = pd.DataFrame()
        with pd.ExcelWriter(country_filename, engine='openpyxl', mode='a',if_sheet_exists='overlay') as writer:  # Use 'a' to append to the existing file
            for entry in entries_to_select_c:
                selected_cf = merged_emissions[merged_emissions['label'] == entry].copy()  # Create a copy of the DataFrame

            # Get the label mapping for the current entry
                label_mapping_c = entry_label_mapping_c.get(entry, {})

            # Replace the values in the selected DataFrame based on the mapping
                selected_cf.loc[:, 'label'] = label_mapping_c.get('label', '')
                selected_cf.loc[:, 'source'] = label_mapping_c.get('source', '')
                selected_cf.loc[:, 'target'] = label_mapping_c.get('target', '')

            # Concatenate the selected entry to the main DataFrame
                selected_entries_cf = pd.concat([selected_entries_cf, selected_cf])

        # Write the concatenated DataFrame to a new sheet
            selected_entries_cf.to_excel(writer, sheet_name='Inputs_co2', index=False)

        print(f'Excel file "{filename}" updated with the emissions data.')

        list_as_set = set(entries_to_select_c)
        cf_as_set = set(map(str, merged_emissions.label))
        # Find the different elements
        different_elements_c = list_as_set.symmetric_difference(cf_as_set)
        # Convert the result back to a list
        different_elements_list_c = list(different_elements_c)
        print("Different elements between the list and the Emissions dataFrame:", different_elements_list_c)
        
    
        

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_sepia")
        

        # Updating the configuration from the standard config file to run in standalone:
        snakemake.params.scenario["simpl"] = [""]
        snakemake.params.scenario["clusters"] = [6]
        snakemake.params.scenario["opts"] = [""]
        snakemake.params.scenario["sector_opts"] = ["EQ0.7c-12H-T-H-B-I-A-dist1"]
        snakemake.params.scenario["ll"] = ["vopt"]
        snakemake.params.scenario["planning_horizons"] = [2020, 2030, 2040, 2050]

    # List the input files for  this script:
    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): "results/"
        + snakemake.params.RDIR
        + f"/postnetworks/elec_s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for simpl in snakemake.params.scenario["simpl"]
        for cluster in snakemake.params.scenario["clusters"]
        for opt in snakemake.params.scenario["opts"]
        for sector_opt in snakemake.params.scenario["sector_opts"]
        for ll in snakemake.params.scenario["ll"]
        for planning_horizon in snakemake.params.scenario["planning_horizons"]
    }

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    
    # TODO: embed this function call in a loop for the case where there is more than one scenario
    # for country in countries:
    countries = snakemake.params.countries
    
    write_to_excel(
        snakemake.params.scenario["simpl"][0],
        snakemake.params.scenario["clusters"][0],
        snakemake.params.scenario["opts"][0],
        snakemake.params.scenario["sector_opts"][0],
        snakemake.params.scenario["ll"][0],
        snakemake.params.scenario["planning_horizons"],
        countries,  # Pass the current country as a list
        filename=snakemake.output.excelfile,
     )


