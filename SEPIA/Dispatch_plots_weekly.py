#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:11:33 2024
"""
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import pypsa
import logging
import os
import sys
import panel as pn
import plotly.graph_objects as go
current_script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_script_dir, "../scripts/")
sys.path.append(scripts_path)
from plot_summary import rename_techs
from plot_power_network import assign_location
from make_summary import assign_carriers


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    # if "heat pump" in tech or "resistive heater" in tech:
    #     return "power-to-heat"
    if tech in ["H2 Electrolysis", "methanation", 'methanolisation',"helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif "H2 pipeline" in tech:
        return "H2 pipeline"
    elif tech in ["H2 Store", "H2 storage"]:
        return "hydrogen storage"
    elif tech in [ "CHP", "H2 Fuel Cell"]:
        return "CHP"
    # elif "solar rooftop" in tech:
    #     return "solar rooftop"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech in ["CO2 sequestration", "co2", "SMR CC", "process emissions CC","process emissions", "solid biomass for industry CC", "gas for industry CC"]:
         return "CCS"
    elif tech in ["biomass", "biomass boiler", "solid biomass", "solid biomass for industry"]:
         return "biomass"
    elif "Li ion" in tech:
        return "battery storage"
    # elif "EV charger" in tech:
    #     return "V2G"
    elif "load" in tech:
        return "load shedding"
    elif tech == "coal" or tech == "lignite":
          return "coal"
    else:
        return tech



def build_filename(simpl,cluster,opt,sector_opt,ll ,planning_horizon):
    prefix=f"results/{study}/postnetworks/elec_"
    return prefix+"s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc".format(
        simpl=simpl,
        cluster=cluster,
        opt=opt,
        sector_opt=sector_opt,
        ll=ll,
        planning_horizon=planning_horizon
    )
def load_file(filename):
    # Use pypsa.Network to load the network from the filename
    return pypsa.Network(filename)

def load_files(study, planning_horizons, simpl, cluster, opt, sector_opt, ll):
    files = {}
    for planning_horizon in planning_horizons:
        filename = build_filename(simpl, cluster, opt, sector_opt, ll, planning_horizon)
        files[planning_horizon] = load_file(filename)
    return files

def plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons,title):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["fossil oil and gas"] = colors["oil"]
    colors["hydrogen storage"] = colors["H2 Store"]
    colors["load shedding"] = 'black'
    colors["CHP"] = 'darkred'
    colors["load"] = 'black'
    colors["Imports_Exports"] = "dimgray"
    colors["EV charger"] = colors["V2G"]
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        n = loaded_files[planning_horizon]

        assign_location(n)
        assign_carriers(n)
        carrier = 'AC'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]].filter(like=country)
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
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"])).filter(like=country)
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )

        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()
        filtered_ac_lines = n.lines.bus0.str[:2] == country
        ac_lines = n.lines_t.p0.filter(items=filtered_ac_lines[filtered_ac_lines == True].index).sum(axis=1)
        filtered_ac_lines_r = n.lines.bus1.str[:2] == country
        ac_lines_r = n.lines_t.p1.filter(items=filtered_ac_lines_r[filtered_ac_lines_r == True].index).sum(axis=1)
        filtered_dc_lines = (n.links.carrier == 'DC') & (n.links.bus0.str[:2] == country)
        dc_lines = n.links_t.p0.filter(items=filtered_dc_lines[filtered_dc_lines == True].index).sum(axis=1)
        filtered_dc_lines_r = (n.links.carrier == 'DC') & (n.links.bus1.str[:2] == country)
        dc_lines_r = n.links_t.p1.filter(items=filtered_dc_lines_r[filtered_dc_lines_r == True].index).sum(axis=1)
        merged_series = pd.concat([ac_lines,ac_lines_r, dc_lines, dc_lines_r], axis=1)
        imp_exp = merged_series.sum(axis=1)
        imp_exp = imp_exp.rename('Imports_Exports')
        imp_exp=-imp_exp
        supplyn['Imports_Exports'] = imp_exp
        

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)



        threshold = 0.1

        to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]

        if len(to_dropn) != 0:
            logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
            supplyn.drop(columns=to_dropn, inplace=True)

        supplyn.index.name = None

        supplyn = supplyn / 1e3


        supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()

        c_solarn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="solar", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        c_onwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="onwind", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        c_offwindn = ((n.generators_t.p_max_pu * n.generators.p_nom_opt) - n.generators_t.p).filter(
            like="offwind", axis=1
        ).filter(like=country).sum(axis=1) / 1e3
        supplyn = supplyn.T
        if "solar" in supplyn.index:
         supplyn.loc["solar"] = supplyn.loc["solar"] + c_solarn
         supplyn.loc["solar curtailment"] = -abs(c_solarn)
        if "onshore wind" in supplyn.index:
         supplyn.loc["onshore wind"] = supplyn.loc["onshore wind"] + c_onwindn
         supplyn.loc["onshore curtailment"] = -abs(c_onwindn)
        if "offshore wind" in supplyn.index:
         supplyn.loc["offshore wind"] = supplyn.loc["offshore wind"] + c_offwindn
         supplyn.loc["offshore curtailment"] = -abs(c_offwindn)
        if "H2 pipeline" in supplyn.index:
           supplyn = supplyn.drop('H2 pipeline')
        supplyn = supplyn.T
        if "V2G" in n.carriers.index:
             v2g = n.links_t.p1.filter(like=country).filter(like="V2G").sum(axis=1)
             v2g = v2g.to_frame()
             v2g = v2g.rename(columns={v2g.columns[0]: 'V2G'})
             v2g = v2g/1e3
             supplyn['electricity distribution grid'] = supplyn['electricity distribution grid'] + v2g['V2G']
             supplyn['V2G'] = v2g['V2G'].abs()
         
        positive_supplyn = supplyn[supplyn >= 0].fillna(0)
        negative_supplyn = supplyn[supplyn < 0].fillna(0)
        positive_supplyn = positive_supplyn.applymap(lambda x: x if x >= 0.1 else 0)
        negative_supplyn = negative_supplyn.applymap(lambda x: x if x <= -0.1 else 0)
        positive_supplyn = positive_supplyn.loc[:, (positive_supplyn > 0).any()]
        negative_supplyn = negative_supplyn.loc[:, (negative_supplyn < 0).any()]
        weeks = positive_supplyn.index.isocalendar().week.unique()
        fig = go.Figure()
        slider_steps = []
        for i, week in enumerate(weeks[1:]): 
         visible_weeks = [False] * len(weeks)
         visible_weeks[i+1] = True

         slider_steps.append({'args': [
        {'visible': visible_weeks},
        {'title': f'{title} - {country} - Week {week}'}],
         'label': str(week),
         'method': 'update'})

        for col in positive_supplyn.columns:
         for i, week in enumerate(weeks):
           fig.add_trace(go.Scatter(
            x=positive_supplyn.index[positive_supplyn.index.isocalendar().week == week],
            y=positive_supplyn[col][positive_supplyn.index.isocalendar().week == week],
            mode='lines',
            line=dict(color=colors.get(col, 'black')),
            stackgroup='positive',
            showlegend=False,
            hovertemplate='%{y:.2f}',
            name=f'{col} - Week {week}'))

        for col in negative_supplyn.columns:
         for i, week in enumerate(weeks):
          fig.add_trace(go.Scatter(
            x=negative_supplyn.index[negative_supplyn.index.isocalendar().week == week],
            y=negative_supplyn[col][negative_supplyn.index.isocalendar().week == week],
            mode='lines',
            line=dict(color=colors.get(col, 'black')),
            stackgroup='negative',
            showlegend=False,
            hovertemplate='%{y:.2f}',
            name=f'{col} - Week {week}'))
          
        for col in positive_supplyn.columns.union(negative_supplyn.columns):
            for i, week in enumerate(weeks):
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color=colors.get(col, 'black'), width=4),  # Set the line width here
                    legendgroup='supply',
                    showlegend=True,
                    name=f'{col}'
                ))
        fig.update_layout(
         xaxis=dict(title='Time', tickformat="%m-%d"),
         yaxis=dict(title='Power [GW]',),
         title=f'{title} - {country} - {planning_horizon} - Week {weeks[0]}',
         width=1200,
         height=600,
         hovermode='x',
         sliders=[{'active': 0,
              'steps': slider_steps}])
    

            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", fig))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))
        
        html_filename = f'Power_Dispatch-{country}_{planning_horizon}.html'
        output_folder = f'results/{study}/htmls/raw_html'
        os.makedirs(output_folder, exist_ok=True)
        html_filepath = os.path.join(output_folder, html_filename)
        tabs.save(html_filepath)
        
    
def plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons,title):
    tech_colors = config["plotting"]["tech_colors"]
    colors = tech_colors 
    colors["agriculture heat"] = "grey"
    colors["CHP"] = "orange"
    colors["centralised electric boiler"] = "#6488ea"
    tabs = pn.Tabs()

    for country in countries:
     tabs = pn.Tabs()

     for planning_horizon in planning_horizons:
        tab = pn.Tabs()
        n = loaded_files[planning_horizon]

        assign_location(n)
        assign_carriers(n)
        carrier = 'heat'
        busesn = n.buses.index[n.buses.carrier.str.contains(carrier)]

        supplyn = pd.DataFrame(index=n.snapshots)

        for c in n.iterate_components(n.branch_components):
            n_port = 4 if c.name == "Link" else 2  # port3
            for i in range(n_port):
                supplyn = pd.concat(
                    (
                        supplyn,
                        (-1)
                        * c.pnl["p" + str(i)]
                        .loc[:, c.df.index[c.df["bus" + str(i)].isin(busesn)]].filter(like=country)
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
                    ((c.pnl["p"].loc[:, comps]).multiply(c.df.loc[comps, "sign"])).filter(like=country)
                    .groupby(c.df.carrier, axis=1)
                    .sum(),
                ),
                axis=1,
            )
        supplyn = supplyn.rename(columns={"urban central resistive heater": "centralised electric boiler"})
        supplyn = supplyn.groupby(rename_techs_tyndp, axis=1).sum()

        bothn = supplyn.columns[(supplyn < 0.0).any() & (supplyn > 0.0).any()]

        positive_supplyn = supplyn[bothn]
        negative_supplyn = supplyn[bothn]

        positive_supplyn = positive_supplyn.mask(positive_supplyn < 0.0, 0.0)
        negative_supplyn = negative_supplyn.mask(negative_supplyn > 0.0, 0.0)

        supplyn[bothn] = positive_supplyn

        supplyn = pd.concat((supplyn, negative_supplyn), axis=1)


        threshold = 0.1

        to_dropn = supplyn.columns[(abs(supplyn) < threshold).all()]

        if len(to_dropn) != 0:
            logger.info(f"Dropping {to_dropn.tolist()} from supplyn")
            supplyn.drop(columns=to_dropn, inplace=True)

        supplyn.index.name = None

        supplyn = supplyn / 1e3
        supplyn.rename(
            columns={"electricity": "electric demand", "heat": "heat demand"}, inplace=True
        )
        supplyn.columns = supplyn.columns.str.replace("residential ", "")
        supplyn.columns = supplyn.columns.str.replace("services ", "")
        supplyn.columns = supplyn.columns.str.replace("urban decentral ", "decentral ")


        supplyn = supplyn.groupby(supplyn.columns, axis=1).sum()
        positive_supplyn = supplyn[supplyn >= 0].fillna(0)
        negative_supplyn = supplyn[supplyn < 0].fillna(0)
        positive_supplyn = positive_supplyn.applymap(lambda x: x if x >= 0.1 else 0)
        negative_supplyn = negative_supplyn.applymap(lambda x: x if x <= -0.1 else 0)
        positive_supplyn = positive_supplyn.loc[:, (positive_supplyn > 0).any()]
        negative_supplyn = negative_supplyn.loc[:, (negative_supplyn < 0).any()]
        weeks = positive_supplyn.index.isocalendar().week.unique()
        fig = go.Figure()
        slider_steps = []
        for i, week in enumerate(weeks[1:]): 
         visible_weeks = [False] * len(weeks)
         visible_weeks[i+1] = True

         slider_steps.append({'args': [
        {'visible': visible_weeks},
        {'title': f'{title} - {country} - Week {week}'}],
         'label': str(week),
         'method': 'update'})

        for col in positive_supplyn.columns:
         for i, week in enumerate(weeks):
           fig.add_trace(go.Scatter(
            x=positive_supplyn.index[positive_supplyn.index.isocalendar().week == week],
            y=positive_supplyn[col][positive_supplyn.index.isocalendar().week == week],
            mode='lines',
            line=dict(color=colors.get(col, 'black')),
            stackgroup='positive',
            showlegend=False,
            hovertemplate='%{y:.2f}',
            name=f'{col} - Week {week}'))

        for col in negative_supplyn.columns:
         for i, week in enumerate(weeks):
          fig.add_trace(go.Scatter(
            x=negative_supplyn.index[negative_supplyn.index.isocalendar().week == week],
            y=negative_supplyn[col][negative_supplyn.index.isocalendar().week == week],
            mode='lines',
            line=dict(color=colors.get(col, 'black')),
            stackgroup='negative',
            showlegend=False,
            hovertemplate='%{y:.2f}',
            name=f'{col} - Week {week}'))
        
        for col in positive_supplyn.columns.union(negative_supplyn.columns):
            for i, week in enumerate(weeks):
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color=colors.get(col, 'black'), width=4),  # Set the line width here
                    legendgroup='supply',
                    showlegend=True,
                    name=f'{col}'
                ))
        fig.update_layout(
         xaxis=dict(title='Time', tickformat="%m-%d"),
         yaxis=dict(title='Heat [GW]',),
         title=f'{title} - {country} - {planning_horizon} - Week {weeks[0]}',
         width=1200,
         height=600,
         hovermode='x',
         sliders=[{'active': 0,
              'steps': slider_steps}])
    

            # Add the plot to the tabs
        tab.append((f"{planning_horizon}", fig))

            # Add the tab for the planning horizon to the main Tabs
        tabs.append((f"{planning_horizon}", tab))
        
        html_filename = f'Heat_Dispatch-{country}_{planning_horizon}.html'
        output_folder = f'results/{study}/htmls/raw_html'
        os.makedirs(output_folder, exist_ok=True)
        html_filepath = os.path.join(output_folder, html_filename)
        tabs.save(html_filepath)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_dispatch_plots")
        

        # Updating the configuration from the standard config file to run in standalone:
    simpl = snakemake.params.scenario["simpl"][0]
    cluster = snakemake.params.scenario["clusters"][0]
    opt = snakemake.params.scenario["opts"][0]
    sector_opt = snakemake.params.scenario["sector_opts"][0]
    ll = snakemake.params.scenario["ll"][0]
    planning_horizons = [2020, 2030, 2040, 2050]


    countries = snakemake.params.countries 
    logging.basicConfig(level=snakemake.config["logging"]["level"])
    config = snakemake.config
    study = snakemake.params.study
    loaded_files = load_files(study, planning_horizons, simpl, cluster, opt, sector_opt, ll)
      
    plot_series_power(simpl, cluster, opt, sector_opt, ll, planning_horizons,title="Power Dispatch")
    plot_series_heat(simpl, cluster, opt, sector_opt, ll, planning_horizons,title="Heat Dispatch")