# -*- coding: utf-8 -*-
"""
Misc. functions used by SEPIA
"""
__author__ = "Adrien Jacob, négaWatt"
__copyright__ = "Copyright 2022, négaWatt"
__license__ = "GPL"
__version__ = "1.8"
__email__ = "adrien.jacob@negawatt.org"

import pandas as pd # Read/analyse data
import numpy as np # for np.nan
import re # Regex
import plotly.graph_objects as go # Plotly figure object
import plotly.express as px
import time # For performance measurement

## General functions
# Show calculation time
def calc_time(title, start_time):
    print("--- "+title+": %s seconds ---" % round(time.time() - start_time, 2))
    return time.time()
    
## Data processing
# Cleanup data taken from Dashboards
def db_cleanup(df, timeseries=True):
    df = df[df.index.notnull()] # Remove lines without "indicator code"
    if timeseries:
        df = df.T.astype('float64') # Transpose & cast as float
        df.index = df.index.rename('Year')
        # Adding missing years with interpolation
        df = (df.reindex(range(df.index.min(), df.index.max()+1))
            .interpolate())
    return df.fillna(0)

# Check if df has all indicators listed in column "column" of description dataFrame, returns list of unfound indicators and adds missing indicators to df
def unfound_indicators(df,description,column):
    unfound = description.loc[~description[column].isin(df.columns) * description[column].notnull(),column].values
    return unfound

# Add results to an indicator DataFrame (and calculate total)
def add_indicator_to_results(result_df,input_df,indicator_label,add_total=True):
    df = input_df.copy()
    if (indicator_label,'total') not in result_df.columns and 'total' not in input_df.columns and add_total: df['total'] = input_df.sum(axis=1)
    df.columns = pd.MultiIndex.from_product([[indicator_label],df.columns])
    result_df = pd.concat([result_df, df], axis=1)
    return result_df

# Cumulative results
def cumul(input_df, start_year=0):
    # Convert the index to integers if it's stored as strings
    input_df.index = input_df.index.astype(int)
    
    # Perform the cumsum after the index type conversion
    return input_df.loc[input_df.index >= start_year].cumsum()
# Place an item at the head of a list (to place 'total' in front for example)
def put_item_in_front(list, item):
    if item in list: list.insert(0, list.pop(list.index(item)))

## Graph network functions
# Calculate sum of all inbound (or outbound) energy flows on one or several nodes
# Direction may be backwards or forward, and flows may be grouped by selected nodes or their targets
def node_consumption(flows, nodes, direction='forward', splitby='nodes'):
    fields = {True:'Source',False:'Target'}
    filter_field_bool = split_field_bool = (direction == 'forward')
    if splitby != 'nodes': split_field_bool = not(split_field_bool)
    filter_field = fields[filter_field_bool]
    split_field = fields[split_field_bool]
    nodes_list = [nodes] if isinstance(nodes, str) else nodes
    selected_columns = flows.columns[flows.columns.get_level_values(filter_field).isin(nodes_list)]
    result = flows[selected_columns].groupby(level=split_field, axis=1).sum()
    if splitby == 'nodes': result = result[[node for node in nodes_list if node in result.columns]] # Sorting columns, to respect initial order
    return result

# Balance nodes, by default deficit & excess are sent to imports and exports
def balance_node(flows, node, imp='imp', exp='exp', procimp='', procexp=''):
    imp_flow = (imp,node,procimp)
    exp_flow = (node,exp,procexp)
    in_flow_cons = node_consumption(flows, node, 'backwards').sub(flows.get(imp_flow,0), axis=0)
    out_flow_cons = node_consumption(flows, node, 'forward').sub(flows.get(exp_flow,0), axis=0)
    if not in_flow_cons.empty or not out_flow_cons.empty:
        excess = in_flow_cons.sub(out_flow_cons, fill_value=0).squeeze()
        deficit = - excess
        excess[excess < 0] = 0
        deficit[deficit < 0] = 0
        flows[exp_flow] = excess
        flows[imp_flow] = deficit

# List of node codes of a given type (primary energy, secondary energy etc.)
def nodes_by_type(NODES, type):
    return list(NODES.loc[NODES['Type']==type].index)

# Subtract flows going backwards from a given node
def subtract_cons_from_node(consumptions, flows, node, exclude_nodes=[], end_nodes=[]):
    return consumptions.subtract(node_consumption(shares_from_node(flows,node,end_nodes + exclude_nodes,include_losses=True),end_nodes), fill_value=0)

# Breakdown of sources or targets starting from a given node, exploring recursively all possible paths (including possible loops)
# Returns a flow dataFrame with all flows connecting the source to target nodes
## Parameters:
# flows(dataFrame): flows dataFrame
# node(string): starting point
# end_nodes(array - optionnal): endpoints of the algorithm (default none)
# direction(string - optionnal): direction of the algorithm (default backwards)
# share(Series - optionnal): share of the starting point (used by the recursive algorithm)
# sharepath(dataFrame - optionnal): current flows dataFrame from the initial starting point to the current starting point (used by the recursive algorithm)
# shares(dataFrame - optionnal): total flows dataFrame (accumulates the different paths)
# minshare(Series - optionnal): minimum share under which the algorithm stops the current path
# include_losses(bool - optionnal): whether losses should be included in the flows or not (default False)
# normalised(bool - optionnal): whether shares should be expressed in absolute values or on a base 1 of the initial consumption (default False)
def shares_from_node(flows, node, end_nodes=[], direction='backwards', share=pd.Series(dtype=float), sharepath=pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=('Source','Target','Type'))), shares=pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=('Source','Target','Type'))), minshare=pd.Series(dtype=float), include_losses=False, normalised=False):
    final_nodes = end_nodes[:] # makes sure we make a copy of the list and not actually change it, cf. https://stackoverflow.com/a/8744133/7496027
    # Algorithm initialisation, cf. https://stackoverflow.com/a/33573262/7496027
    if share.empty:
        share = node_consumption(flows,node,direction).squeeze() # We start from the node consumption
        if normalised: share.loc[:] = 1
        if(share.empty or share.max() == 0): return shares
        minshare=share/1E4 # below a certain threshold, we drop the path to avoid infinite loops - threshold should be a compromise between calculation time and accuracy
        if node in final_nodes: final_nodes.remove(node) # removing initial node from end nodes, otherwise the exploration doesn't start
    field = 'Source' if direction == 'forward' else 'Target'
    connected_flows = flows.columns[flows.columns.get_level_values(field) == node]
    if connected_flows.size == 0 or node in final_nodes: # Leaf node, we merge/sum the path with the existing ones
        shares = pd.concat([shares,sharepath], axis=1).groupby(axis=1,level=[0,1,2]).sum()
    else:
        total_flow = flows[connected_flows].sum(axis=1)
        for flow in connected_flows:
            if flow[1] == 'per' and include_losses: continue
            share_ratio = (flows[flow] / total_flow).fillna(0)
            newshare = share.copy()
            newsharepath = sharepath.copy()
            # Adding the new 'share' item to the current sharepath
            newsharepath[flow] = newsharepath.get(flow,0) + newshare
            # Check if we need to add transformation and network losses
            loss_flow = (flow[0],'per',flow[2])
            if include_losses and loss_flow in flows.columns:
                # We select flows going out of the same source, of the same type/process, except losses
                output_flows_columns = (flows.columns.get_level_values('Source') == loss_flow[0]).astype('bool') * (flows.columns.get_level_values('Target') != 'per').astype('bool') * (flows.columns.get_level_values('Type') == loss_flow[2]).astype('bool')
                transformation_outputs = flows[flows.columns[output_flows_columns]].sum(axis=1)
                loss_ratio = (flows[loss_flow] / transformation_outputs).fillna(0)
                if direction == 'backwards':
                    newsharepath[loss_flow] = newsharepath.get(loss_flow,0) + newshare * loss_ratio
                else:
                    share_ratio *= (1 + loss_ratio)
            # Multiplying the whole path with the current flow share
            newsharepath = newsharepath.multiply(share_ratio, axis=0)
            newshare = newsharepath[flow]
            if direction == 'backwards' and include_losses: newshare += newsharepath.get(loss_flow,0)
            if newshare.gt(minshare).sum() > 0: # we keep exploring this path if the flow is not too small
                connected_node = flow[1] if direction == 'forward' else flow[0]
                shares = shares_from_node(flows,connected_node,final_nodes,direction,newshare,newsharepath,shares,minshare,include_losses)
    return shares

## Other misc. calculation functions
# Share of primary energy by category (ren/fos/nuk)
def share_primary_category(df, nodes):
    result=pd.DataFrame(index=df.index)
    for node in df:
        node_cat = nodes.loc[node,'Category']
        result[node_cat] = result.get(node_cat,0) + df[node]
    return result

def share_percent(df,base=1):
    result=df.copy()
    total = result.sum(axis=1)
    total = total.where(total != 0, np.nan)
    result = result.div(total,axis=0).fillna(0) * base
    return result

def consistency_check(first_df,second_df,title,warning_treshold=0,unit='%'):
    first_series = first_df if isinstance(first_df, pd.Series) else first_df.sum(axis=1)
    second_series = second_df if isinstance(second_df, pd.Series) else second_df.sum(axis=1)
    if unit == '%':
        delta = round(100*(first_series.round(6) / second_series.round(6) - 1).abs().max(),2)
    else:
        delta = round((first_series - second_series).abs().max(),2)
    if delta > warning_treshold: print(title+': '+str(delta)+' '+unit)
    
def reduction_rate(series,base=1):
    return base*(1 - series / series.iat[0])

## Visualisation functions
def title_to_output(title, id, template):
    result = "<h2>" if id == "" else "<h2 id=\""+id+"\">"
    result += title
    if template == 'raw': result += " - <a href=\"#top\">Back to top</a>"
    result += "</h2>"
    return result

# Final Plotly figure formatting (project logo & DRAFT watermark)
def format_chart(fig, chart_type, main_params):
    if main_params['DRAFT']: # Add DRAFT watermark
        fig.add_annotation(
            text="DRAFT",
            textangle=-30,
            opacity=0.1,
            font=dict(family="Verdana, sans-serif", color="black", size=100),
            xref="paper", yref="paper",
            x=0.5, y=0.5)
    logo = dict(source=main_params['PROJECT_LOGO'],
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=1,
        xanchor="center",
        yanchor="bottom",
        sizex=0.3,
        sizey=0.3,
        layer="below")
    if chart_type=="sankey":
        logo['y']=0.05
        logo['yanchor']="middle"
    # fig.add_layout_image(logo)
    fig.update_layout(
        font_color="#4A4949",
        title_font_family="Verdana, sans-serif",
        title_font_color="#4A4949")
    return fig

# Converts Plotly figure into an HTML string
def chart_to_output(fig):
    return '<div style="height:85%">'+fig.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo': False})+'</div>'

def add_data_to_output(xls_writer, title, unit, df):
    if xls_writer!=None:
        chart_name = 'Chart ' + str(len(xls_writer.sheets) + 1)
        df.to_excel(xls_writer, sheet_name=chart_name, startrow=2)
        xls_writer.sheets[chart_name]['A1'] = title + ' (' + re.sub(r"</?sub>", '', unit) + ')'
  
# Plotly single line chart
def create_line_chart(results, main_params, title='', xls_writer=None, unit='TWh/year'):
    df = results.round(main_params['DECIMALS']).rename(unit)
    add_data_to_output(xls_writer, title, unit, df)
    df = df.reset_index()
    fig = px.line(df, x='Year', y=unit, title=title)
    fig.update_layout(hovermode='x')
    fig.update_traces(hovertemplate='%{y:.1f}', line_width=2)
    format_chart(fig, "line", main_params)
    return fig
    
# Plotly area or line chart, based on nodes label & colors
def create_node_chart(results, NODES, main_params, type="area", title='', xls_writer=None, unit='TWh/year', targets={}):
    df = results.round(main_params['DECIMALS']).T # transposing results
    df = pd.concat([df, NODES.loc[results.columns,'Label']], axis=1).set_index('Label',drop=True) # adding indicator labels
    add_data_to_output(xls_writer, title, unit, df.T)
    df = pd.melt(df, ignore_index=False, var_name='Year').reset_index() # Unpivoting years for chart rendering
    color_map = NODES.loc[results.columns].set_index('Label',drop=True)['Color'].to_dict()
    if type == "area":
        fig = px.area(df, title=title, x="Year", y="value", hover_name="Label", color="Label", color_discrete_map=color_map)
        updatemenus = [dict(
            buttons=[dict(
                    args = [{"groupnorm": ''},{"yaxis":{"ticksuffix":'',"title":unit}}],
                    label = "Absolute",
                    method="update"),
                dict(
                    args = [{"groupnorm": 'percent'},{"yaxis":{"ticksuffix":'%',"title":'',"range":[1,100],"type":'linear'}}],
                    label = "Normalised",
                    method="update")],
            type = "buttons",
            direction="down")]
        line_width = 1
    else:
        fig = px.line(df, title=title, x="Year", y="value", hover_name="Label", color="Label", color_discrete_map=color_map)
        updatemenus = []
        line_width = 2
    fig.update_layout(hovermode='x',legend_title_text='',yaxis_title=unit)
    if len(targets)>0:
        fig.add_scatter(y=targets['y'], x=targets['x'], mode=targets['mode'], name=targets['title'], marker_size=15, marker_color='black')
    else:
        fig.update_layout(updatemenus=updatemenus)
    if type == "area": fig.add_scatter(y=results.sum(axis=1).to_list(), x=results.index.to_list(), mode='lines', name='Total', line_color="black")
    fig.update_traces(hovertemplate='%{y:.1f}', line_width=line_width)
    format_chart(fig, type, main_params)
    return fig

# Convert hex color code to RGBA (with opacity)
def hex_to_rgba(h,opacity):
    if re.match('#?[\dA-Fa-f]{6}',h) == None:
        return h
    else:
        h = h.replace("#", "")
        rgb = list(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
        rgb.append(str(opacity))
        return 'rgba('+','.join(rgb)+')'

# Sankey
# Create Sankeys with slider (every 'interval_year' years)
def create_sankey(flows, nodes, processes, main_params, interval_year=10, title="Sankey diagram"):
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

def create_carbon_sankey(flows_co2, nodes, processes, main_params, interval_year=10, title="Carbon Sankey diagram"):
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

def add_sankey_label(fig, x, title, y=1.05):
    fig.add_annotation(x=x, y=y, text="<b>"+title+"</b>", showarrow=False)
    
# Chloropleth maps
def create_map(results, countries, title, main_params, xls_writer=None, interval_year=10, unit='%', min_scale=np.NaN, mid_scale=np.NaN, max_scale=np.NaN, reverse=False):
    df = results.round(main_params['DECIMALS']).T # transposing results
    df = df.filter(countries.index,axis=0) # filtering countries
    df = pd.concat([df, countries], axis=1).reset_index() # adding country labels and moving country name from index to columns
    fig = go.Figure()
    steps = []
    years = range(results.index.min(), results.index.max() + 1, interval_year)
    for i, year in enumerate(years):
        # df['customdata'] = '<b>%{hovertext}</b>: %{z:.1f} %'
        colorscale = ["#ff4d00","#feda47","#008556"]
        zmin = df[year].min()
        if not np.isnan(min_scale): zmin = min(min_scale,zmin)
        zmax = df[year].max()
        if not np.isnan(max_scale): zmax = max(max_scale,zmax)
        if zmax != zmin and not np.isnan(zmax) and not np.isnan(mid_scale):
            midpoint = min(1,max(0,(mid_scale-zmin)/(zmax-zmin)))
            if midpoint == 0:
                zmin = mid_scale
                colorscale = [[0, colorscale[1]],[1, colorscale[2]]]
            elif midpoint == 1:
                zmax = mid_scale
                colorscale = [[0, colorscale[0]],[1, colorscale[1]]]
            else:
                colorscale = [[0, colorscale[0]],[midpoint, colorscale[1]],[1, colorscale[2]]]
        mp = go.Choropleth(name=year,
                            locations=df['ISO_Code'],
                            z = df[year],
                            text=df['Label'],
                            # customdata=df['customdata'],
                            hovertemplate='<b>%{text}</b>: %{z:.2f} '+unit, # %<br/>%{customdata}
                            colorscale = colorscale,
                            zmin = zmin,
                            zmax = zmax,
                            reversescale = reverse,
                            colorbar_title = unit,
                            visible = False,)
        fig.add_trace(mp)
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
        title_text = title+' in '+ str(year),
        geo_scope='europe', # limit map scope to Europe
        sliders=[dict(active = visible_step, currentvalue = {"visible": False}, steps=steps)],
    )
    fig.update_geos(fitbounds="locations", resolution=50) # Zoom to available data , visible=False
    add_data_to_output(xls_writer, 'Map of '+title, unit, results.round(main_params['DECIMALS']).filter(countries.index))
    format_chart(fig, "map", main_params)
    return fig

# Combine several charts into one (with buttons)
# combinations is a list of tuples (button_label,dataframe) or (button_label,dataframe,targets)
# chart_type can be 'map', 'sankey', 'areachart', 'linechart'
def combine_charts(combinations, main_params, description=pd.DataFrame(), title='', chart_type='map', xls_writer=None, unit='TWh/year', sk_proc=pd.DataFrame(), min_scale=np.NaN, mid_scale=np.NaN, max_scale=np.NaN, reverse=False):
    buttons = []
    for i,combination in enumerate(combinations):
        button_label = combination[0]
        df = combination[1]
        targets = combination[2] if len(combination)>2 else {}
        fig_title = title + ' ' + button_label
        button_args = [{}, {"title": {"text": fig_title}}]
        if chart_type == 'map':
            temp_fig = create_map(df, description, fig_title, main_params, xls_writer=xls_writer, unit=unit, min_scale=min_scale, mid_scale=mid_scale, max_scale=max_scale, reverse=reverse)
        elif chart_type == 'sankey':
            temp_fig = create_sankey(df, description, sk_proc, main_params, title=fig_title)
        elif chart_type == 'carbon sankey':
            temp_fig = create_carbon_sankey(df, description, sk_proc, main_params, title=fig_title)
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
                if chart_type == 'areachart': number_traces += 1 # For the "total" line
                visibility_toggles += [j==i] * number_traces
        button_args[0] = {"visible": visibility_toggles}
        buttons.append(dict(args = button_args,
            label=button_label[0].upper() + button_label[1:],
            method="update"))
    update_menus = [fig.layout.updatemenus[0]] if len(fig.layout.updatemenus) > 0 else [] # We keep existing update menus (in the case of areacharts for ex)
    if len(buttons)>1: update_menus += [dict(buttons=buttons, x=1,y=1.15)]
    fig.update_layout(updatemenus=update_menus)
    return chart_to_output(fig)