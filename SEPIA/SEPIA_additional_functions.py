# -*- coding: utf-8 -*-
"""
Misc. addtional functions used by SEPIA
"""
__author__ = "Umair Tareen, Sylvain Quoilin"


import pandas as pd # Read/analyse data
import numpy as np # for np.nan
import re # Regex
import plotly.graph_objects as go # Plotly figure object
import plotly.express as px
import time # For performance measurement

## General functions
# Show calculation time

from SEPIA_functions import calc_time
from SEPIA_functions import db_cleanup
from SEPIA_functions import unfound_indicators
from SEPIA_functions import put_item_in_front
from SEPIA_functions import node_consumption
from SEPIA_functions import create_line_chart
from SEPIA_functions import format_chart
from SEPIA_functions import chart_to_output
from SEPIA_functions import add_sankey_label
from SEPIA_functions import add_data_to_output
from SEPIA_functions import create_map
from SEPIA_functions import hex_to_rgba
from SEPIA_functions import create_node_chart

# Cumulative results
def cumul(input_df, start_year=0):
    # Convert the index to integers if it's stored as strings
    input_df.index = input_df.index.astype(int)
    # Perform the cumsum after the index type conversion
    return input_df.loc[input_df.index >= start_year].cumsum()
def create_ghg_chart(results, NODES, main_params, type="area", title='', xls_writer=None, unit='TWh/year', targets={}):
    df = results.round(main_params['DECIMALS']).T
    df = pd.concat([df, NODES.loc[results.columns,'Label']], axis=1).set_index('Label', drop=True)
    add_data_to_output(xls_writer, title, unit, df.T)
    df = pd.melt(df, ignore_index=False, var_name='Year').reset_index()
    color_map = NODES.loc[results.columns].set_index('Label', drop=True)['Color'].to_dict()

    df_positive = df[df['value'] >= 0]
    df_positive = df_positive[df_positive['value'] != 0]
    df_negative = df[df['value'] < 0]

    fig = go.Figure()

    for label, data in df_positive.groupby('Label'):
        fig.add_trace(go.Scatter(
            x=data['Year'],
            y=data['value'],
            mode='lines',
            line=dict(color=color_map[label]),
            stackgroup='positive',
            name=label
        ))

    for label, data in df_negative.groupby('Label'):
        fig.add_trace(go.Scatter(
            x=data['Year'],
            y=data['value'],
            mode='lines',
            line=dict(color=color_map[label]),
            stackgroup='negative',
            name=label
        ))

    updatemenus = [dict(
        buttons=[
            dict(
                args=[{"groupnorm": ''}, {"yaxis": {"ticksuffix": '', "title": unit}}],
                label="Absolute",
                method="update"),
        ],
        type="buttons",
        direction="down"
    )]

    line_width = 1
    fig.update_layout(hovermode='x', legend_title_text='', yaxis_title=unit, title=title)
    if len(targets) > 0:
        fig.add_scatter(y=targets['y'], x=targets['x'], mode=targets['mode'], name=targets['title'], marker_size=15,
                        marker_color='black')
    else:
        fig.update_layout(updatemenus=updatemenus)
    if type == "area":
        fig.add_scatter(y=results.sum(axis=1).to_list(), x=results.index.to_list(), mode='lines', name='Total',
                        line_color="black")
    fig.update_traces(hovertemplate='%{y:.1f}', line_width=line_width)
    format_chart(fig, type, main_params)
    return fig
# Sankey
# Create Sankeys with slider (every 'interval_year' years)
def create_sankey(flows, nodes, processes, main_params, interval_year=5, title="Sankey diagram"):
    flows = flows.round(main_params['DECIMALS'])
    # Hide very small flows
    flows[flows<1E-4] = 0
    # To have transparent links
    nodes['ColorOpacity'] = nodes['Color'].apply(hex_to_rgba, opacity=0.4) 
    # Adding Labels to flows (when defined in processes) and transposing flows for Sankey rendering
    proc_labels = processes.set_index(['Source','Target','Type'])#[['Label']]
    links = pd.concat([flows.T, proc_labels], axis=1).reset_index()

    fig = go.Figure()
    steps = []
    years = range(int(flows.index.min()), int(flows.index.max()) + 1, interval_year)

    nodes_with_links = pd.concat([links.groupby(by="Source").sum(numeric_only=True),links.groupby(by="Target").sum(numeric_only=True)])
    # Removing nodes without links, or with 0 flows on all years, to avoid bugs with positionning
    never_empty_nodes = nodes_with_links.loc[nodes_with_links.prod(axis=1) > 0].index.drop_duplicates().to_list()
    never_empty_nodes_sorted = nodes.loc[never_empty_nodes].sort_values(by='PositionX') # Sorting nodes by increasing X position, cf. https://stackoverflow.com/a/70855245/7496027
    for i, year in enumerate(years):
        if str(year) in nodes_with_links.columns:
         new_not_empty_nodes = nodes_with_links.loc[nodes_with_links[str(year)] > 0].index.drop_duplicates()
        # Rest of your code for processing this year
        else:
         print(f"Column '{year}' not found in nodes_with_links DataFrame")
        new_not_empty_nodes = new_not_empty_nodes[~new_not_empty_nodes.isin(never_empty_nodes)].to_list()
        year_nodes = pd.concat([never_empty_nodes_sorted, nodes.loc[new_not_empty_nodes]])
        year_nodes['RowIndex'] = range(len(year_nodes))
        # Removing links with 0 flow
        year_links = links.loc[links[str(year)] > 0]
        sk = go.Sankey(
            valueformat = ".0f",
            valuesuffix = " TWh",
            visible = False,
            arrangement = "snap",
            node={
                "label": year_nodes['Label'],
                "color": year_nodes['Color'],
                "x": year_nodes['PositionX'],
                "y": year_nodes['PositionY'],
                "pad": 5,
            },
            link={
                "source": year_nodes.loc[year_links['Source'],'RowIndex'],
                "target": year_nodes.loc[year_links['Target'],'RowIndex'],
                "color": year_nodes.loc[year_links['Source'],'ColorOpacity'],
                "label": year_links['Label'].fillna(''),
                "value": year_links[str(year)],
                # "hovertemplate": '<b>%{label}</b><br /><b>From:</b> %{source.label}<br /><b>To:</b> %{target.label}',
                # "customdata": year_links['Type'],
                # "hovertemplate": '<b>From:</b> %{source.label}<br /><b>To:</b> %{target.label}<br /><b>Process:</b> %{label} %{customdata}',
            },
        )
        fig.add_trace(sk)
        step = dict(
            label = str(year),
            method = "update",
            args = [{"visible": [False] * len(years)},
                    {"title": {"text": title+" in " + str(year)}}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make last trace visible
    visible_step = len(years)-1
    fig.data[visible_step].visible = True
    fig.update_layout(
        title_text = title+" in " + str(years[visible_step]),
        hovermode = 'x',
        margin_t=100,
        sliders=[dict(active = visible_step, currentvalue = {"visible": False}, steps=steps)])
    # Adding labels for types of nodes
    add_sankey_label(fig, 0, 'Source')
    add_sankey_label(fig, nodes.loc['cms_pe','PositionX'], 'Primary energy')
    add_sankey_label(fig, (nodes.loc['cms_pe','PositionX']+nodes.loc['elc_fe','PositionX'])/2, 'Secondary energy & networks')
    add_sankey_label(fig, nodes.loc['elc_fe','PositionX'], 'Final energy')
    add_sankey_label(fig, 1, 'Demand sector')
    format_chart(fig, "sankey", main_params)
    return fig

def create_carbon_sankey(flows_co2, nodes, processes, main_params, interval_year=5, title="Carbon Sankey diagram"):
    flows_co2 = flows_co2.round(main_params['DECIMALS'])
    # Hide very small flows
    # flows_co2[flows_co2<1E-4] = 0
    # To have transparent links
    nodes['ColorOpacity'] = nodes['Color'].apply(hex_to_rgba, opacity=0.4) 
    # Adding Labels to flows (when defined in processes) and transposing flows for Sankey rendering
    proc_labels = processes.set_index(['Source','Target','Type'])#[['Label']]
    links = pd.concat([flows_co2.T, proc_labels], axis=1).reset_index()

    fig = go.Figure()
    steps = []
    years = range(int(flows_co2.index.min()), int(flows_co2.index.max()) + 1, interval_year)

    nodes_with_links = pd.concat([links.groupby(by="Source").sum(numeric_only=True),links.groupby(by="Target").sum(numeric_only=True)])
    # Removing nodes without links, or with 0 flows on all years, to avoid bugs with positionning
    never_empty_nodes = nodes_with_links.loc[nodes_with_links.prod(axis=1) > 0].index.drop_duplicates().to_list()
    never_empty_nodes_sorted = nodes.loc[never_empty_nodes].sort_values(by='PositionX') # Sorting nodes by increasing X position, cf. https://stackoverflow.com/a/70855245/7496027
    for i, year in enumerate(years):
        if str(year) in nodes_with_links.columns:
         new_not_empty_nodes = nodes_with_links.loc[nodes_with_links[str(year)] > 0].index.drop_duplicates()
        # Rest of your code for processing this year
        else:
         print(f"Column '{year}' not found in nodes_with_links DataFrame")
        new_not_empty_nodes = new_not_empty_nodes[~new_not_empty_nodes.isin(never_empty_nodes)].to_list()
        year_nodes = pd.concat([never_empty_nodes_sorted, nodes.loc[new_not_empty_nodes]])
        year_nodes['RowIndex'] = range(len(year_nodes))
        # Removing links with 0 flow
        year_links = links.loc[links[str(year)] > 0]
        sk = go.Sankey(
            valueformat = ".0f",
            valuesuffix = " MtCO2",
            visible = False,
            arrangement = "snap",
            node={
                "label": year_nodes['Label'],
                "color": year_nodes['Color'],
                "x": year_nodes['PositionX'],
                "y": year_nodes['PositionY'],
                "pad": 5,
            },
            link={
                "source": year_nodes.loc[year_links['Source'],'RowIndex'],
                "target": year_nodes.loc[year_links['Target'],'RowIndex'],
                "color": year_nodes.loc[year_links['Source'],'ColorOpacity'],
                "label": year_links['Label'].fillna(''),
                "value": year_links[str(year)],
                # "hovertemplate": '<b>%{label}</b><br /><b>From:</b> %{source.label}<br /><b>To:</b> %{target.label}',
                # "customdata": year_links['Type'],
                # "hovertemplate": '<b>From:</b> %{source.label}<br /><b>To:</b> %{target.label}<br /><b>Process:</b> %{label} %{customdata}',
            },
        )
        fig.add_trace(sk)
        step = dict(
            label = str(year),
            method = "update",
            args = [{"visible": [False] * len(years)},
                    {"title": {"text": title+" in " + str(year)}}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make last trace visible
    visible_step = len(years)-1
    fig.data[visible_step].visible = True
    fig.update_layout(
        title_text = title+" in " + str(years[visible_step]),
        hovermode = 'x',
        margin_t=100,
        sliders=[dict(active = visible_step, currentvalue = {"visible": False}, steps=steps)])
    # Adding labels for types of nodes
    add_sankey_label(fig, 0, 'Source')
    # add_sankey_label(fig, nodes.loc['fgs_ghg','PositionX'], 'Primary emmissions')
    # add_sankey_label(fig, (nodes.loc['fgs_ghg','PositionX']+nodes.loc['net_ghg','PositionX'])/2, 'Final Emissions')
    # add_sankey_label(fig, nodes.loc['met_ghg','PositionX'], 'Secondary Emissions')
    add_sankey_label(fig, 1, 'Final Emissions')
    format_chart(fig, "carbon sankey", main_params)
    return fig

# Combine several charts into one (with buttons)
# combinations is a list of tuples (button_label,dataframe) or (button_label,dataframe,targets)
# chart_type can be 'map', 'sankey', 'areachart', 'linechart'
def combine_charts(combinations, main_params, description=pd.DataFrame(), title='', chart_type='map', xls_writer=None, interval_year=10, unit='TWh/year', sk_proc=pd.DataFrame(), min_scale=np.NaN, mid_scale=np.NaN, max_scale=np.NaN, reverse=False):
    buttons = []
    for i,combination in enumerate(combinations):
        button_label = combination[0]
        df = combination[1]
        targets = combination[2] if len(combination)>2 else {}
        fig_title = title + ' ' + button_label
        button_args = [{}, {"title": {"text": fig_title}}]
        if chart_type == 'map':
            temp_fig = create_map(df, description, fig_title, main_params, xls_writer=xls_writer, interval_year=interval_year, unit=unit, min_scale=min_scale, mid_scale=mid_scale, max_scale=max_scale, reverse=reverse)
        elif chart_type == 'sankey':
            temp_fig = create_sankey(df, description, sk_proc, main_params, title=fig_title, interval_year=interval_year)
        elif chart_type == 'carbon sankey':
            temp_fig = create_carbon_sankey(df, description, sk_proc, main_params, title=fig_title, interval_year=interval_year)
        elif chart_type == 'ghgchart':
            temp_fig = create_ghg_chart(df, description, main_params, "area", fig_title, xls_writer=xls_writer, unit=unit, targets=targets)
        elif chart_type == 'areachart':
            temp_fig = create_node_chart(df, description, main_params, "area", fig_title, xls_writer=xls_writer, unit=unit, targets=targets)
        elif chart_type == 'linechart':
            temp_fig = create_node_chart(df, description, main_params, "line", fig_title, xls_writer=xls_writer, unit=unit, targets=targets)
        elif chart_type == 'singleline':
            temp_fig = create_line_chart(df, main_params, fig_title, xls_writer=xls_writer, unit=unit)
        else:
            print("!Warning, please define chart_type")
            return False
        
        if i==0:
            fig=temp_fig
        else:
            for trace in temp_fig.data:
                trace.visible = False
                fig.add_trace(trace)
        
        if len(temp_fig.layout.sliders)>0: # Charts with sliders (sankey, map)
            slider = temp_fig.layout.sliders[0]
            for j in range(len(slider.steps)):
                visibility_toggles = [False] * len(slider.steps) * len(combinations)
                visibility_toggles[i*len(slider.steps)+j] = True
                slider.steps[j].args[0]['visible'] = visibility_toggles
            if i==0: fig.layout.sliders[0].steps = slider.steps
            button_args[1]["sliders"] = [slider]
        else:
            visibility_toggles = []
            for j,combination2 in enumerate(combinations):
                number_traces = (len(combination2[1].columns) if isinstance(combination2[1], pd.DataFrame) else 1)
                if len(combination2)>2 and len(combination2[2])>0:
                    number_traces += 1 # In case a "targets" trace has been defined
                # if chart_type == 'areachart': number_traces += 1 # For the "total" line
                # visibility_toggles += [j==i] * number_traces
        button_args[0] = {"visible": visibility_toggles}
        buttons.append(dict(args = button_args,
            label=button_label[0].upper() + button_label[1:],
            method="update"))
    update_menus = [fig.layout.updatemenus[0]] if len(fig.layout.updatemenus) > 0 else [] # We keep existing update menus (in the case of areacharts for ex)
    if len(buttons)>1: update_menus += [dict(buttons=buttons, x=1,y=1.15)]
    fig.update_layout(updatemenus=update_menus)
    return chart_to_output(fig)