

import pandas as pd # Read/analyse data
import pypsa
from pypsa.descriptors import get_switchable_as_dense as as_dense



n=pypsa.Network("/home/umair/1H-32c simulations/suff/results/postnetworks/elec_s_32_lvopt__1H-T-H-B-I-A-dist1_2030.nc")
m=pypsa.Network("/home/umair/1H-32c simulations/suff/results/postnetworks/elec_s_32_lvopt__1H-T-H-B-I-A-dist1_2040.nc")
o=pypsa.Network("/home/umair/1H-32c simulations/suff/results/postnetworks/elec_s_32_lvopt__1H-T-H-B-I-A-dist1_2050.nc")
countries=['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT', 'SE', 'SI', 'SK', 'RO']

def process_network_1(n):
    energy_demand =pd.read_csv("/home/umair/1H-32c simulations/suff/resources/energy_totals_s_32_2030.csv", index_col=0).T
    clever_industry = (
         pd.read_csv("/home/umair/pypsa-eur_repository/data/clever_Industry_2030.csv", index_col=0)).loc[countries].T

    Rail_demand = energy_demand.loc["total rail"].sum()
    H2_nonenergyy = clever_industry.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry = clever_industry.loc["Total Final hydrogen consumption in industry"].sum()
    columns = ["label", "source", "target", "value"]

    gen = (
        (n.snapshot_weightings.generators @ n.generators_t.p)
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
        (n.snapshot_weightings.generators @ n.stores_t.p)
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
        (n.snapshot_weightings.generators @ n.storage_units_t.p)
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
        (n.snapshot_weightings.generators @ as_dense(n, "Load", "p_set"))
        .groupby([n.loads.carrier, n.loads.carrier, n.loads.bus.map(n.buses.carrier)])
        .sum()
        .div(1e6)
        .swaplevel()
    )  # TWh
    load.index.set_names(columns[:-1], inplace=True)
    load = load.reset_index(name="value")

    load = load.loc[~load.label.str.contains("emissions")]
    load.target += " demand"
    load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "value"] = H2_industry
    value=load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"] 
    load.loc[load.label.str.contains("AC") & (load.label == "electricity"), "value"] = value - Rail_demand
    for i in range(5):
        n.links[f"total_e{i}"] = (
            n.snapshot_weightings.generators @ n.links_t[f"p{i}"]
        ).div(
            1e6
        )  # TWh
        n.links[f"carrier_bus{i}"] = n.links[f"bus{i}"].map(n.buses.carrier)

    def calculate_losses(x):
        energy_ports = x.loc[
            x.index.str.contains("carrier_bus") & ~x.str.contains("co2", na=False)
        ].index.str.replace("carrier_bus", "total_e")
        return -x.loc[energy_ports].sum()

    n.links["total_e5"] = n.links.apply(calculate_losses, axis=1)    #e4 and bus 4 for bAU 2050
    n.links["carrier_bus5"] = "losses"

    df = pd.concat(
        [
            n.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum()[
                "total_e" + str(i)
            ]
            for i in range(1, 6)
        ]
    ).reset_index()
    df.columns = columns

    # fix heat pump energy balance
    

    hp = n.links.loc[n.links.carrier.str.contains("heat pump")]

    hp_t_elec = n.links_t.p0.filter(like="heat pump")

    grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
    hp_elec = (
        (-n.snapshot_weightings.generators @ hp_t_elec)
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

    to_concat = [df, gen, su, sto, load]
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

    connections = connections.append(new_row1, ignore_index=True)

    connections = connections.append(new_row2, ignore_index=True)    
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
    connections.rename(columns={'value': '2030'}, inplace=True)

    return connections

def process_network_2(m):
    energy_demand =pd.read_csv("/home/umair/1H-32c simulations/suff/resources/energy_totals_s_32_2040.csv", index_col=0).T
    clever_industry = (
         pd.read_csv("/home/umair/pypsa-eur_repository/data/clever_Industry_2040.csv", index_col=0)).loc[countries].T

    Rail_demand = energy_demand.loc["total rail"].sum()
    H2_nonenergyy = clever_industry.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry = clever_industry.loc["Total Final hydrogen consumption in industry"].sum()
    columns = ["label", "source", "target", "value"]

    gen = (
        (m.snapshot_weightings.generators @ m.generators_t.p)
        .groupby(
            [
                m.generators.carrier,
                m.generators.carrier,
                m.generators.bus.map(m.buses.carrier),
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
        (m.snapshot_weightings.generators @ m.stores_t.p)
        .groupby(
            [m.stores.carrier, m.stores.carrier, m.stores.bus.map(m.buses.carrier)]
        )
        .sum()
        .div(1e6)
    )
    sto.index.set_names(columns[:-1], inplace=True)
    sto = sto.reset_index(name="value")
    sto = sto.loc[sto.value > 0.1]

    su = (
        (m.snapshot_weightings.generators @ m.storage_units_t.p)
        .groupby(
            [
                m.storage_units.carrier,
                m.storage_units.carrier,
                m.storage_units.bus.map(m.buses.carrier),
            ]
        )
        .sum()
        .div(1e6)
    )
    su.index.set_names(columns[:-1], inplace=True)
    su = su.reset_index(name="value")
    su = su.loc[su.value > 0.1]
    
    load = (
        (m.snapshot_weightings.generators @ as_dense(m, "Load", "p_set"))
        .groupby([m.loads.carrier, m.loads.carrier, m.loads.bus.map(m.buses.carrier)])
        .sum()
        .div(1e6)
        .swaplevel()
    )  # TWh
    load.index.set_names(columns[:-1], inplace=True)
    load = load.reset_index(name="value")

    load = load.loc[~load.label.str.contains("emissions")]
    load.target += " demand"
    load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "value"] = H2_industry
    value=load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"] 
    load.loc[load.label.str.contains("AC") & (load.label == "electricity"), "value"] = value - Rail_demand
    for i in range(5):
        m.links[f"total_e{i}"] = (
            m.snapshot_weightings.generators @ m.links_t[f"p{i}"]
        ).div(
            1e6
        )  # TWh
        m.links[f"carrier_bus{i}"] = m.links[f"bus{i}"].map(m.buses.carrier)

    def calculate_losses(x):
        energy_ports = x.loc[
            x.index.str.contains("carrier_bus") & ~x.str.contains("co2", na=False)
        ].index.str.replace("carrier_bus", "total_e")
        return -x.loc[energy_ports].sum()

    m.links["total_e5"] = m.links.apply(calculate_losses, axis=1)    #e4 and bus 4 for bAU 2050
    m.links["carrier_bus5"] = "losses"

    df = pd.concat(
        [
            m.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum()[
                "total_e" + str(i)
            ]
            for i in range(1, 6)
        ]
    ).reset_index()
    df.columns = columns

    # fix heat pump energy balance
    

    hp = m.links.loc[m.links.carrier.str.contains("heat pump")]

    hp_t_elec = m.links_t.p0.filter(like="heat pump")

    grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
    hp_elec = (
        (-m.snapshot_weightings.generators @ hp_t_elec)
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

    to_concat = [df, gen, su, sto, load]
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

    connections = connections.append(new_row1, ignore_index=True)

    connections = connections.append(new_row2, ignore_index=True)    
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
    connections.rename(columns={'value': '2040'}, inplace=True)

    return connections

def process_network_3(o):
    energy_demand =pd.read_csv("/home/umair/1H-32c simulations/suff/resources/energy_totals_s_32_2050.csv", index_col=0).T
    clever_industry = (
         pd.read_csv("/home/umair/pypsa-eur_repository/data/clever_Industry_2050.csv", index_col=0)).loc[countries].T

    Rail_demand = energy_demand.loc["total rail"].sum()
    H2_nonenergyy = clever_industry.loc["Non-energy consumption of hydrogen for the feedstock production"].sum()
    H2_industry = clever_industry.loc["Total Final hydrogen consumption in industry"].sum()
    columns = ["label", "source", "target", "value"]

    gen = (
        (o.snapshot_weightings.generators @ o.generators_t.p)
        .groupby(
            [
                o.generators.carrier,
                o.generators.carrier,
                o.generators.bus.map(o.buses.carrier),
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
        (o.snapshot_weightings.generators @ o.stores_t.p)
        .groupby(
            [o.stores.carrier, o.stores.carrier, o.stores.bus.map(o.buses.carrier)]
        )
        .sum()
        .div(1e6)
    )
    sto.index.set_names(columns[:-1], inplace=True)
    sto = sto.reset_index(name="value")
    sto = sto.loc[sto.value > 0.1]

    su = (
        (o.snapshot_weightings.generators @ o.storage_units_t.p)
        .groupby(
            [
                o.storage_units.carrier,
                o.storage_units.carrier,
                o.storage_units.bus.map(o.buses.carrier),
            ]
        )
        .sum()
        .div(1e6)
    )
    su.index.set_names(columns[:-1], inplace=True)
    su = su.reset_index(name="value")
    su = su.loc[su.value > 0.1]

    load = (
        (o.snapshot_weightings.generators @ as_dense(o, "Load", "p_set"))
        .groupby([o.loads.carrier, o.loads.carrier, o.loads.bus.map(o.buses.carrier)])
        .sum()
        .div(1e6)
        .swaplevel()
    )  # TWh
    load.index.set_names(columns[:-1], inplace=True)
    load = load.reset_index(name="value")

    load = load.loc[~load.label.str.contains("emissions")]
    load.target += " demand"
    load.loc[load.label.str.contains("H2 for industry") & (load.label == "H2 for industry"), "value"] = H2_industry
    value=load.loc[load.label.str.contains("electricity") & (load.label == "electricity"), "value"] 
    load.loc[load.label.str.contains("AC") & (load.label == "electricity"), "value"] = value - Rail_demand
    for i in range(5):
        o.links[f"total_e{i}"] = (
            o.snapshot_weightings.generators @ o.links_t[f"p{i}"]
        ).div(
            1e6
        )  # TWh
        o.links[f"carrier_bus{i}"] = o.links[f"bus{i}"].map(o.buses.carrier)

    def calculate_losses(x):
        energy_ports = x.loc[
            x.index.str.contains("carrier_bus") & ~x.str.contains("co2", na=False)
        ].index.str.replace("carrier_bus", "total_e")
        return -x.loc[energy_ports].sum()

    o.links["total_e5"] = o.links.apply(calculate_losses, axis=1)    #e4 and bus 4 for bAU 2050
    o.links["carrier_bus5"] = "losses"

    df = pd.concat(
        [
            o.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum()[
                "total_e" + str(i)
            ]
            for i in range(1, 6)
        ]
    ).reset_index()
    df.columns = columns

    # fix heat pump energy balance
    

    hp = o.links.loc[o.links.carrier.str.contains("heat pump")]

    hp_t_elec = o.links_t.p0.filter(like="heat pump")

    grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
    hp_elec = (
        (-o.snapshot_weightings.generators @ hp_t_elec)
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

    to_concat = [df, gen, su, sto, load]
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

    connections = connections.append(new_row1, ignore_index=True)

    connections = connections.append(new_row2, ignore_index=True)    
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
    connections.rename(columns={'value': '2050'}, inplace=True)
   
    return connections
#%%

connections_n = process_network_1(n)
connections_m = process_network_2(m)
connections_o = process_network_3(o)

# First, rename the columns to avoid conflicts
connections_n = connections_n.rename(columns={'2030': 'value_2030'})
connections_m = connections_m.rename(columns={'2040': 'value_2040'})
connections_o = connections_o.rename(columns={'2050': 'value_2050'})

# Merge the dataframes based on 'label,' 'source,' and 'target' columns
merged_df = pd.merge(connections_n, connections_m, on=['label', 'source', 'target'], how='outer')
merged_df = pd.merge(merged_df, connections_o, on=['label', 'source', 'target'], how='outer')

# Fill missing values with 0
merged_df = merged_df.fillna(0)
connections=merged_df
connections = connections.rename(columns={'value_2030': '2030', 'value_2040': '2040', 'value_2050': '2050'})
connections = connections[['label', 'source', 'target', '2030', '2040', '2050']]
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
                     'gas for industry CC_2','SMR_2','SMR CC_2','methanolisation_3',
                     'residential rural water tanks charger_2','residential rural water tanks discharger_2'] # Add moe entries if needed

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
    'nuclear_2': {'label': 'Transformation losses (nuclear powerplants)', 'source': 'TWh', 'target': 'lossnuc'},
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
    
}


excel_file = 'input.xlsx'
df=connections
selected_entries_df = pd.DataFrame()

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
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
    

print(f'Excel file "{excel_file}" created with the selected entries on the "SelectedEntries" sheet.')

list_as_set = set(entries_to_select)
df_as_set = set(map(str, connections.label))

# Find the different elements
different_elements = list_as_set.symmetric_difference(df_as_set)

# Convert the result back to a list
different_elements_list = list(different_elements)

print("Different elements between the list and DataFrame:", different_elements_list)

