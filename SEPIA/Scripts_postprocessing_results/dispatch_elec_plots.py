#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:17:16 2023

"""
import logging

logger = logging.getLogger(__name__)
import pypsa
import pandas as pd
import os
import sys
import yaml
import matplotlib.pyplot as plt

SCRIPTS_PATH = "../scripts/"
sys.path.append(os.path.join(SCRIPTS_PATH))
from plot_summary import rename_techs
from plot_network import assign_location
from make_summary import assign_carriers
from plot_summary import preferred_order, rename_techs

with open("../config/config.yaml") as file:
    config = yaml.safe_load(file)

def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "hydrogen storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif "Li ion" in tech:
        return "battery storage"
    elif "BEV charger" in tech:
        return "V2G"
    elif "load" in tech:
        return "load shedding"
    elif tech == "oil" or tech == "gas":
         return "fossil oil and gas"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    
    else:
        return tech
def plot_series(carrier="AC", name="test"):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["fossil oil and gas"] = colors["oil"]
    colors["hydrogen storage"] = colors["H2 Store"]
    colors["load shedding"] = 'black'
    colors["gas-to-power/heat"] = 'darkred'
    n=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/Overnight simulations/resultsreff/postnetworks/elec_s_6_lv1.0__Co2L0.8-1H-T-H-B-I-A-dist1_2020.nc")
    m=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsbau/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")
    p=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultssuff/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")
    r=pypsa.Network("/home/umair/pypsa-eur_repository/simulations/myopic simulations/resultsnocdr/postnetworks/elec_s_6_lvopt__1H-T-H-B-I-A-dist1_2050.nc")

    assign_location(n)
    assign_carriers(n)
    assign_location(m)
    assign_carriers(m)
    # assign_location(o)
    # assign_carriers(o)
    assign_location(p)
    assign_carriers(p)
    # assign_location(q)
    # assign_carriers(q)
    assign_location(r)
    assign_carriers(r)

    busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]
    busesm = m.buses.index[m.buses.carrier.str.contains(carrier)]
    # buseso = o.buses.index[o.buses.carrier.str.contains(carrier)]
    busesp = p.buses.index[p.buses.carrier.str.contains(carrier)]
    busesr = r.buses.index[r.buses.carrier.str.contains(carrier)]
    

    supplyn = pd.DataFrame(index=n.snapshots)
    supplym = pd.DataFrame(index=m.snapshots)
    # supplyo = pd.DataFrame(index=o.snapshots)
    supplyp = pd.DataFrame(index=p.snapshots)
    supplyr = pd.DataFrame(index=r.snapshots)
    
    for c in n.iterate_components(n.branch_components):
        n_port = 3 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyn = pd.concat(
                (
                    supplyn,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesn)]
        supplyn = pd.concat(
            (
                supplyn,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in m.iterate_components(m.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplym = pd.concat(
                (
                    supplym,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesm)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in m.iterate_components(m.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesm)]
        supplym = pd.concat(
            (
                supplym,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    # for c in o.iterate_components(o.branch_components):
    #     n_port = 4 if c.name == "Link" else 2     #port3
    #     for i in range(n_port):
    #         supplyo = pd.concat(
    #             (
    #                 supplyo,
    #                 (-1)
    #                 * c.pnl["p" + str(i)]
    #                 .loc[:, c.df.index[c.df["bus" + str(i)].isin(buseso)]]
    #                 .groupby(c.df.carrier, axis=1)
    #                 .sum(),
    #             ),
    #             axis=1,
    #         )

    # for c in o.iterate_components(o.one_port_components):
    #     comps = c.df.index[c.df.bus.isin(buseso)]
    #     supplyo = pd.concat(
    #         (
    #             supplyo,
    #             ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
    #             .groupby(c.df.carrier, axis=1)
    #             .sum(),
    #         ),
    #         axis=1,
    #     )
        
    for c in p.iterate_components(p.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyp = pd.concat(
                (
                    supplyp,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesp)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in p.iterate_components(p.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesp)]
        supplyp = pd.concat(
            (
                supplyp,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )
        
    for c in r.iterate_components(r.branch_components):
        n_port = 4 if c.name == "Link" else 2     #port3
        for i in range(n_port):
            supplyr = pd.concat(
                (
                    supplyr,
                    (-1)
                    * c.pnl["p" + str(i)]
                    .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesr)]]
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

    for c in r.iterate_components(r.one_port_components):
        comps = c.df.index[c.df.bus.isin(busesr)]
        supplyr = pd.concat(
            (
                supplyr,
                ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"]))
                .groupby(c.df.carrier, axis=1)
                .sum(),
            ),
            axis=1,
        )

    supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
    supplym = supplym.groupby(rename_techs_tyndp, axis=1).sum()
    # supplyo = supplyo.groupby(rename_techs_tyndp, axis=1).sum()
    supplyp = supplyp.groupby(rename_techs_tyndp, axis=1).sum()
    supplyr = supplyr.groupby(rename_techs_tyndp, axis=1).sum()
    

    bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]
    bothm = supplym.columns[(supplym < 0.0).any() & (supplym > 0.0).any()]
    # botho = supplyo.columns[(supplyo < 0.0).any() & (supplyo > 0.0).any()]
    bothp = supplyp.columns[(supplyp < 0.0).any() & (supplyp > 0.0).any()]
    bothr = supplyr.columns[(supplyr < 0.0).any() & (supplyr > 0.0).any()]

    positive_supplyn = supplyn[bothn]
    negative_supplyn = supplyn[bothn]
    positive_supplym = supplym[bothm]
    negative_supplym = supplym[bothm]
    # positive_supplyo = supplyo[botho]
    # negative_supplyo = supplyo[botho]
    positive_supplyp = supplyp[bothp]
    negative_supplyp = supplyp[bothp]
    positive_supplyr = supplyr[bothr]
    negative_supplyr = supplyr[bothr]

    positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
    negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)
    positive_supplym = positive_supplym.mask(positive_supplym < 0.0, 0.0)
    negative_supplym = negative_supplym.mask(negative_supplym > 0.0, 0.0)
    # positive_supplyo = positive_supplyo.mask(positive_supplyo < 0.0, 0.0)
    # negative_supplyo = negative_supplyo.mask(negative_supplyo > 0.0, 0.0)
    positive_supplyp = positive_supplyp.mask(positive_supplyp < 0.0, 0.0)
    negative_supplyp = negative_supplyp.mask(negative_supplyp > 0.0, 0.0)
    positive_supplyr = positive_supplyr.mask(positive_supplyr < 0.0, 0.0)
    negative_supplyr = negative_supplyr.mask(negative_supplyr > 0.0, 0.0)


    supplyn[bothn] = positive_supplyn
    supplym[bothm] = positive_supplym
    # supplyo[botho] = positive_supplyo
    supplyp[bothp] = positive_supplyp
    supplyr[bothr] = positive_supplyr
    

    suffix = " charging"

    negative_supplyn.columns = negative_supplyn.columns + suffix
    negative_supplym.columns = negative_supplym.columns + suffix
    # negative_supplyo.columns = negative_supplyo.columns + suffix
    negative_supplyp.columns = negative_supplyp.columns + suffix
    negative_supplyr.columns = negative_supplyr.columns + suffix

    supplyn = pd.concat((supplyn, negative_supplyn), axis=1)
    supplym = pd.concat((supplym, negative_supplym), axis=1)
    # supplyo = pd.concat((supplyo, negative_supplyo), axis=1)
    supplyp = pd.concat((supplyp, negative_supplyp), axis=1)
    supplyr = pd.concat((supplyr, negative_supplyr), axis=1)
    
    

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-02-01"
    stop = "2013-02-07"
    

    threshold = 1

    to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]
    to_dropm = supplym.columns[(abs(supplym) < threshold).all()]
    # to_dropo = supplyo.columns[(abs(supplyo) < threshold).all()]
    to_dropp = supplyp.columns[(abs(supplyp) < threshold).all()]
    to_dropr = supplyr.columns[(abs(supplyr) < threshold).all()]

    if len(to_dropn) != 0:
        logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
        supplyn.drop(columns=to_dropn, inplace=True)
    
    if len(to_dropm) != 0:
        logger.info(f"Dropping {to_dropm.tolist()} from supplym")
        supplym.drop(columns=to_dropm, inplace=True)
        
    # if len(to_dropo) != 0:
    #     logger.info(f"Dropping {to_dropo.tolist()} from supplyio")
    #     supplyo.drop(columns=to_dropo, inplace=True)
        
    if len(to_dropp) != 0:
        logger.info(f"Dropping {to_dropp.tolist()} from supplyp")
        supplyp.drop(columns=to_dropp, inplace=True)
        
    if len(to_dropr) != 0:
        logger.info(f"Dropping {to_dropr.tolist()} from supplyr")
        supplyr.drop(columns=to_dropr, inplace=True)

    supplyn.index.name = None
    supplym.index.name = None
    # supplyo.index.name = None
    supplyp.index.name = None
    supplyr.index.name = None
    

    supplyn = supplyn / 1e3
    supplym = supplym / 1e3
    # supplyo = supplyo / 1e3
    supplyp = supplyp / 1e3
    supplyr = supplyr / 1e3
    #del supply["CCS"]

    supplyn.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyn.columns = supplyn.columns.str.replace("residential ", "")
    supplyn.columns = supplyn.columns.str.replace("services ", "")
    supplyn.columns = supplyn.columns.str.replace("urban decentral ", "decentral ")
    
    supplym.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplym.columns = supplym.columns.str.replace("residential ", "")
    supplym.columns = supplym.columns.str.replace("services ", "")
    supplym.columns = supplym.columns.str.replace("urban decentral ", "decentral ")
    
    # supplyo.rename(
    #     columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    # )
    # supplyo.columns = supplyo.columns.str.replace("residential ", "")
    # supplyo.columns = supplyo.columns.str.replace("services ", "")
    # supplyo.columns = supplyo.columns.str.replace("urban decentral ", "decentral ")
    
    supplyp.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyp.columns = supplyp.columns.str.replace("residential ", "")
    supplyp.columns = supplyp.columns.str.replace("services ", "")
    supplyp.columns = supplyp.columns.str.replace("urban decentral ", "decentral ")
    
    supplyr.rename(
        columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
    )
    supplyr.columns = supplyr.columns.str.replace("residential ", "")
    supplyr.columns = supplyr.columns.str.replace("services ", "")
    supplyr.columns = supplyr.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(
        [
            "electric demand",
            "transmission lines",
            "hydroelectricity",
            "hydro reservoir",
            "run of river",
            "pumped hydro storage",
            "CHP",
            "onshore wind",
            "offshore wind",
            "solar PV",
            "solar thermal",
            "building retrofitting",
            "ground heat pump",
            "air heat pump",
            "resistive heater",
            "OCGT",
            "gas boiler",
            "gas",
            "natural gas",
            "methanation",
            "hydrogen storage",
            "battery storage",
            "hot water storage",
            "solar curtailment",
            "onshore curtailment",
            "offshore curtailment",
        ]
    )

    new_columnsn = preferred_order.intersection(supplyn.columns).append(
        supplyn.columns.difference(preferred_order)
    )
    new_columnsm = preferred_order.intersection(supplym.columns).append(
        supplym.columns.difference(preferred_order)
    )
    # new_columnso = preferred_order.intersection(supplyo.columns).append(
    #     supplyo.columns.difference(preferred_order)
    # )
    new_columnsp = preferred_order.intersection(supplyp.columns).append(
        supplyp.columns.difference(preferred_order)
    )
    new_columnsr = preferred_order.intersection(supplyr.columns).append(
        supplyr.columns.difference(preferred_order)
    )

    supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()
    supplym = supplym.groupby(supplym.columns, axis=1).sum()
    # supplyo = supplyo.groupby(supplyo.columns, axis=1).sum()
    supplyp = supplyp.groupby(supplyp.columns, axis=1).sum()
    supplyr = supplyr.groupby(supplyr.columns, axis=1).sum()
    
    c_solarn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindn=((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyn = supplyn.T
    supplyn.loc["solar"] = supplyn.loc["solar"] + c_solarn
    supplyn.loc["offshore wind"] = supplyn.loc["offshore wind"] + c_offwindn
    supplyn.loc["onshore wind"] = supplyn.loc["onshore wind"] + c_onwindn
    supplyn.loc["solar curtailment"] = -abs(c_solarn)
    supplyn.loc["onshore curtailment"] = -abs(c_onwindn)
    supplyn.loc["offshore curtailment"] = -abs(c_offwindn)
    supplyn=supplyn.T
    
    c_solarm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindm=((m.generators_t.p_max_pu * m.generators.p_nom_opt) - m.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplym = supplym.T
    supplym.loc["solar"] = supplym.loc["solar"] + c_solarm
    supplym.loc["offshore wind"] = supplym.loc["offshore wind"] + c_offwindm
    supplym.loc["onshore wind"] = supplym.loc["onshore wind"] + c_onwindm
    supplym.loc["solar curtailment"] = -abs(c_solarm)
    supplym.loc["onshore curtailment"] = -abs(c_onwindm)
    supplym.loc["offshore curtailment"] = -abs(c_offwindm)
    supplym=supplym.T
    
    # c_solaro=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    # c_onwindo=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    # c_offwindo=((o.generators_t.p_max_pu * o.generators.p_nom_opt) - o.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    # supplyo = supplyo.T
    # supplyo.loc["solar"] = supplyo.loc["solar"] + c_solaro
    # supplyo.loc["offshore wind"] = supplyo.loc["offshore wind"] + c_offwindo
    # supplyo.loc["onshore wind"] = supplyo.loc["onshore wind"] + c_onwindo
    # supplyo.loc["solar curtailment"] = -abs(c_solaro)
    # supplyo.loc["onshore curtailment"] = -abs(c_onwindo)
    # supplyo.loc["offshore curtailment"] = -abs(c_offwindo)
    # supplyo=supplyo.T
    
    c_solarp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindp=((p.generators_t.p_max_pu * p.generators.p_nom_opt) - p.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyp = supplyp.T
    supplyp.loc["solar"] = supplyp.loc["solar"] + c_solarp
    supplyp.loc["offshore wind"] = supplyp.loc["offshore wind"] + c_offwindp
    supplyp.loc["onshore wind"] = supplyp.loc["onshore wind"] + c_onwindp
    supplyp.loc["solar curtailment"] = -abs(c_solarp)
    supplyp.loc["onshore curtailment"] = -abs(c_onwindp)
    supplyp.loc["offshore curtailment"] = -abs(c_offwindp)
    supplyp=supplyp.T
    
    c_solarr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='solar', axis=1).sum(axis=1)/1e3
    c_onwindr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='onwind', axis=1).sum(axis=1)/1e3
    c_offwindr=((r.generators_t.p_max_pu * r.generators.p_nom_opt) - r.generators_t.p).filter(like='offwind', axis=1).sum(axis=1)/1e3
    supplyr = supplyr.T
    supplyr.loc["solar"] = supplyr.loc["solar"] + c_solarr
    supplyr.loc["offshore wind"] = supplyr.loc["offshore wind"] + c_offwindr
    supplyr.loc["onshore wind"] = supplyr.loc["onshore wind"] + c_onwindr
    supplyr.loc["solar curtailment"] = -abs(c_solarr)
    supplyr.loc["onshore curtailment"] = -abs(c_onwindr)
    supplyr.loc["offshore curtailment"] = -abs(c_offwindr)
    supplyr=supplyr.T
    
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.set_size_inches((10, 15))
    # fig, ax = plt.subplots()
    # fig.set_size_inches((10, 6))

    # (
    #     supplyn.loc[start:stop, supplyn.columns].plot(
    #         ax=ax,
    #         kind="area",
    #         stacked=True,
    #         legend=False,
    #         linewidth=0.0,
    #         color=[
    #             config["plotting"]["tech_colors"][i.replace(suffix, "")]
    #             for i in supplyn.columns
    #         ],
    #     )
    # )
    
    (
        supplym.loc[start:stop, supplym.columns].plot(
            ax=ax1,
            kind="area",
            stacked=True,
            title = "2030",
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplym.columns
            ],
        )
    )
    
    # (
    #     supplyo.loc[start:stop, supplyo.columns].plot(
    #         ax=ax3,
    #         kind="area",
    #         stacked=True,
    #         title = "2040",
    #         legend=False,
    #         linewidth=0.0,
    #         color=[
    #             config["plotting"]["tech_colors"][i.replace(suffix, "")]
    #             for i in supplyo.columns
    #         ],
    #     )
    # )
    
    (
        supplyp.loc[start:stop, supplyp.columns].plot(
            ax=ax2,
            kind="area",
            stacked=True,
            legend=False,
            linewidth=0.0,
            color=[
                config["plotting"]["tech_colors"][i.replace(suffix, "")]
                for i in supplyp.columns
            ],
        )
    )
    
    # (
    #     supplyr.loc[start:stop, supplyr.columns].plot(
    #         ax=ax5,
    #         kind="area",
    #         stacked=True,
    #         legend=False,
    #         linewidth=0.0,
    #         color=[
    #             config["plotting"]["tech_colors"][i.replace(suffix, "")]
    #             for i in supplyr.columns
    #         ],
    #     )
    # )

    # handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax2.get_legend_handles_labels()
    # handles, labels = ax3.get_legend_handles_labels()
    # handles, labels = ax4.get_legend_handles_labels()
    # handles, labels = ax5.get_legend_handles_labels()

    # handles.reverse()
    # labels.reverse()

    new_handles = []
    new_labels = []

    # for i, item in enumerate(labels):
    #     if "charging" not in item:
    #         new_handles.append(handles[i])
    #         new_labels.append(labels[i])

    # ax4.legend(new_handles, new_labels, ncol=1, bbox_to_anchor=(1,3.4), loc="upper left", frameon=False, fontsize=20)
    # ax.set_title("Reff", fontsize=20)
    ax1.set_title("BAU-2050", fontsize=20)
    # ax3.set_title("2040", fontsize=11)
    ax2.set_title("Suff-2050", fontsize=20)
    # ax5.set_title("NO_CDR_2050", fontsize=20)
    #fig.supxlabel('Electricity Dispatch in winters')

    # ax4.set_xlim([start, stop])
    # ax4.set_ylim([-800, 800])
    # ax4.grid(True)
    ax1.set_ylabel("Power [GW]", fontsize=20)
    ax1.set_ylim(-1200, 1200)
    ax2.set_ylabel("Power [GW]",fontsize=20)
    ax2.set_ylim(-1200, 1200)
    #ax3.set_ylabel("Demand [GW]")
    # ax4.set_ylabel("Power [GW]", fontsize=20)
    # ax4.set_ylim(-1200, 1200)
    # ax5.set_ylabel("Power [GW]", fontsize=20)
    # ax5.set_ylim(-1200, 1200)
    ax2.set_xlabel("Electricity Dispatch in Winter",fontsize=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)

    

    
plot_series(carrier="AC")

