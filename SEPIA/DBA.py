# -*- coding: utf-8 -*-
"""
DBA - DashBoard Analysis
"""
import pandas as pd # Read/analyse data
import numpy as np
import re # Regex
import plotly.graph_objects as go
import os # File system
import warnings # to manage user warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl') # Disable warning from openpyxl

DIRNAME = os.path.dirname(__file__)
file = pd.ExcelFile(os.path.join(DIRNAME, 'DBA_config.xlsx'))
CONFIG = pd.read_excel(file, ["COUNTRIES","CALCULATED"], index_col=0)
COUNTRIES = CONFIG["COUNTRIES"]
COUNTRIES = COUNTRIES[COUNTRIES['Input_File'].notna()]
CALCULATED = CONFIG["CALCULATED"]
YEARS = [2015,2030,2040,2050]

tot_data = {}
for indicator_type in ['raw','pop','evol']:
    tot_data[indicator_type] = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Indicator','Country']))
indicator_desc = pd.DataFrame()

def eval_formula(df,string):
    # Replacing indicator codes by calls to the relevant index/columns in df
    regex = re.compile(r"([a-z]+)")
    indicators = re.findall(regex,string)
    if not set(indicators).issubset(df.index):
        print('Warning, unable to find indicators :', ", ".join(indicators))
        return pd.Series(dtype='float64')
    else:
        formula = re.sub(regex, r"df.loc['\1',YEARS]", string)
        return eval(formula)

for country in COUNTRIES.index:
    print("||| "+country+" |||")
    country_input_file = COUNTRIES.loc[country,'Input_File']+'.xlsx'
    file = pd.ExcelFile(os.path.join(DIRNAME,'Inputs',country_input_file))
    sheets = ["Macro","Residential","Tertiary","Transport","Agriculture","AFOLUB (Solagro figures)","Energy"]
    sheets += ["Industry_Results"] if "Industry_Results" in file.sheet_names else ["Industry"]
    df = pd.read_excel(file, sheets, index_col=2, usecols="E:G,W,AB:AH")
    for sheet in df:
        df[sheet] = df[sheet].rename(columns={df[sheet].columns.values[0]:'Description'})
        # Indicator calculation
        for calc_ind in CALCULATED.loc[CALCULATED['Sheet']==sheet].index:
            indicator = CALCULATED.loc[calc_ind]
            rs = eval_formula(df[sheet],indicator['Formula1'])
            rs['Unit'] = indicator['Unit']
            rs['Description'] = indicator['Description'] +'<br /><span style="font-size:11px">Formula: ' + indicator['Formula1'] + '</span>'
            if indicator.notna()['Formula2']:
                rs[rs.isna()] = eval_formula(df[sheet],indicator['Formula2'])
                rs['Description'] += '<br /><span style="font-size:11px">Alternative: '+ indicator['Formula2'] + '</span>'
            df[sheet].loc[calc_ind] = rs
        df[sheet].index = pd.MultiIndex.from_product([[sheet],df[sheet].index,[country]], names=['Sheet','Indicator','Country'])
    df = pd.concat(df.values())
    df = df[df.index.get_level_values('Indicator').notnull()] # Remove lines without "indicator code"
    if df.index.has_duplicates:
        # print("Warning, duplicate indicators", df.index[df.index.duplicated()])
        df = df[~df.index.duplicated(keep='first')] # Remove duplicate indicators
    if indicator_desc.empty :
        indicator_desc = df.iloc[:, [0,1]].droplevel(['Sheet','Country'])
        indicator_desc = indicator_desc[~indicator_desc.index.duplicated(keep='first')]
    df = df.drop(columns=['Description','Unit'])
    for year in YEARS:
        df[year] = pd.to_numeric(df[year], errors='coerce') # Cast as float
    df = df.T # Transpose
    df.index = df.index.rename('Year')
    df = df[df.index.isin(YEARS)] # Filter years
    
    # Per capita & evolution vs. 2015 dataframes
    data = {}
    data['raw'] = df
    data['pop'] = 1000 * df.divide(df.xs('pop', level='Indicator', axis=1).squeeze(),axis=0).reorder_levels(df.columns.names,axis=1)
    data['evol'] = 100 * df.divide(df.iloc[0].replace(0,np.nan),axis=1)

    # Adding country results to total results dataframes
    for indicator_type in data:
        tot_data[indicator_type] = pd.concat([tot_data[indicator_type], data[indicator_type]], axis=1)

# Bar charts
def create_bar_chart(df,indicator_type,indicators):
    fig_data = []
    selected_indicators = indicators[indicators.notnull()]
    if selected_indicators.empty: return False
    selected_values = df[indicator_type].droplevel('Sheet', axis=1)[list(zip(selected_indicators,selected_indicators.index))]
    title = indicator_desc.loc[selected_indicators,'Description'][0]
    unit = indicator_desc.loc[selected_indicators,'Unit'][0]
    if indicator_type != 'raw' and unit == '%':
        return None
    else:
        if selected_values.columns.get_level_values('Indicator').nunique() > 1: # If indicators are not the same for all countries, we add it to the labels
            legend_title = ''
            x_labels = [country+' ('+ind+')' for (ind,country) in selected_values.columns]
        else:
            legend_title = selected_values.columns.get_level_values('Indicator')[0]
            x_labels = selected_values.columns.get_level_values('Country')
        for year in df[indicator_type].index:
            year_values = selected_values.loc[year]
            fig_data += [go.Bar(name=year, x=x_labels, y=year_values.to_list())]
        fig = go.Figure(data=fig_data)
        if indicator_type == 'pop':
            unit += '/Mcap.'
        elif indicator_type == 'evol':
            unit = '% (base 2015)'
        fig.update_layout(barmode='group', title=title, yaxis_title=unit, legend_title_text=legend_title)
        return fig

print("||| Charts |||")
results_file = open(os.path.join(DIRNAME,'Results','DBA_charts.html'), 'w')
charts = COUNTRIES.filter(regex=r'Chart\d+', axis=1)
for chart in charts.columns:
    results_file.write('<h1>'+chart+'</h1>')
    fig = go.Figure()
    buttons = []
    for (i,indicator_type) in enumerate(data):
        temp_fig = create_bar_chart(tot_data,indicator_type,charts[chart])
        if temp_fig:
            if indicator_type=='raw':
                title = 'Absolute'
                fig = temp_fig
            else:
                if indicator_type=='pop':
                    title = 'Per capita'
                else:
                    title = 'Evolution vs. 2015'
                for trace in temp_fig.data:
                    trace.visible = False
                    fig.add_trace(trace)
            visibility_toggles = [False] * len(data) * len(YEARS)
            for j in range(len(YEARS)*i,len(YEARS)*(i+1)): visibility_toggles[j] = True
            buttons.append(dict(args = ({"visible": visibility_toggles},{"yaxis": {"title": {"text": temp_fig.layout.yaxis.title.text}}}), label=title, method="update"))
    if len(buttons)>1: fig.update_layout(updatemenus=[dict(buttons=buttons, type="buttons")])
    if len(fig.data)>0: results_file.write(fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displaylogo': False})+'<br/><br/>')

# Excel outputs
print("||| File writing |||")
with pd.ExcelWriter(os.path.join(DIRNAME,'Results','DBA_data.xlsx')) as writer:
    for indicator_type in data:
        df = tot_data[indicator_type]
         # Sorting by indicators (keeping initial indicator order)
        indicators = df.columns.droplevel('Country').unique()
        df = df.reindex(columns=[(sheet,indicator,country) for (sheet,indicator) in indicators for country in COUNTRIES.index])
        df.T.to_excel(writer, sheet_name=indicator_type, merge_cells=False)