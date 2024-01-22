#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:50:52 2023

"""
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import os
with open("../../config/config.yaml") as file:
    config = yaml.safe_load(file)

scenario = 'bau'



colors = config["plotting"]["tech_colors"] 
colors["methane"] = "orange"
colors["Non-energy demand"] = "black" 
colors["hydrogen for industry"] = "cornflowerblue"
colors["agriculture electricity"] = "royalblue"
colors["agriculture and industry heat"] = "lightsteelblue"
colors["agriculture oil"] = "darkorange"
colors["electricity demand of residential and tertairy"] = "navajowhite"
colors["gas for Industry"] = "forestgreen"
colors["electricity for Industry"] = "limegreen"
colors["aviation oil demand"] = "black"
colors["land transport EV"] = "lightcoral"
colors["land transport hydrogen demand"] = "mediumpurple"
colors["oil to transport demand"] = "thistle"
colors["low-temperature heat for industry"] = "sienna"
colors["naphtha for non-energy"] = "sandybrown"
colors["shipping methanol"] = "lawngreen"
colors["shipping hydrogen"] = "gold"
colors["shipping oil"] = "turquoise"
colors["solid biomass for Industry"] = "paleturquoise"
colors["Residential and tertiary DH demand"] = "gray"
colors["Residential and tertiary heat demand"] = "pink"
colors["electricity demand for rail network"] = "blue"
colors["H2 for non-energy"] = "violet" 
colors["solid biomass"] = "greenyellow"
colors["electricity"] = "midnightblue"
colors["hydrogen"] = "violet"
colors["oil"] = "gray"
    
mapping = {
        "hydrogen for industry": "hydrogen",
        "H2 for non-energy": "Non-energy demand",
        "shipping hydrogen": "hydrogen",
        "shipping oil": "oil",
        "agriculture electricity": "electricity",
        "agriculture heat": "heat",
        "agriculture oil": "oil",
        "electricity demand of residential and tertairy": "electricity",
        "gas for Industry": "methane",
        "electricity for Industry": "electricity",
        "aviation oil demand": "oil",
        "land transport EV": "electricity",
        "land transport hydrogen demand": "hydrogen",
        "oil to transport demand": "oil",
        "low-temperature heat for industry": "heat",
        "naphtha for non-energy": "Non-energy demand",
        "electricity demand for rail network": "electricity",
        "Residential and tertiary DH demand": "heat",
        "Residential and tertiary heat demand": "heat",
        "solid biomass for Industry": "solid biomass",
        "NH3":"hydrogen",
}
countries = ['BE', 'DE', 'FR', 'GB', 'NL']    
for country in countries:
        data_1 = pd.read_excel(f"../../results/bau/sepia/inputs{country}.xlsx", index_col=0)
        columns_to_drop = ['source', 'target']
        data_1 = data_1.drop(columns=columns_to_drop)
        data_1 = data_1.groupby(data_1.index).sum()

        # Apply your mapping to the data
        data_1 = data_1[data_1.index.isin(mapping.keys())]
        data_1.index = pd.MultiIndex.from_tuples([(mapping[i], i) for i in data_1.index])
        data_bau = data_1.groupby(level=0).sum()
            
        data_2 = pd.read_excel(f"../../results/ncdr/sepia/inputs{country}.xlsx", index_col=0)
        columns_to_drop = ['source', 'target', '2020']
        data_2 = data_2.drop(columns=columns_to_drop)
        data_2 = data_2.groupby(data_2.index).sum()

        # Apply your mapping to the data
        data_2 = data_2[data_2.index.isin(mapping.keys())]
        data_2.index = pd.MultiIndex.from_tuples([(mapping[i], i) for i in data_2.index])
        data_ncdr = data_2.groupby(level=0).sum()
            
        nf = data_bau.loc[:,'2020']
        mf1 = data_bau.loc[:,'2030']
        mf2 = data_bau.loc[:,'2040']
        mf3 = data_bau.loc[:,'2050']
        pf1 = data_ncdr.loc[:,'2030']
        pf2 = data_ncdr.loc[:,'2040']
        pf3 = data_ncdr.loc[:,'2050']

        x= [-0.85,-0.58,-0.3, -0.02, 0.26, 0.54,0.82]
        x_labels = ['Reff', 'BAU-2030','BAU-2040', 'BAU-2050','Suff-2030','Suff-2040','Suff-2050']
        order = [
          "electricity",
          "heat",
          "oil",
         "solid biomass",
         "methane",
         "hydrogen",
         "Non-energy demand",
          ]

        fig, ax = plt.subplots(figsize=(10, 13), )
        pd.DataFrame(nf.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=4.8,
        width=0.2,
        stacked=True,
        color=colors,
        edgecolor="k",
        legend=True,
        )
        pd.DataFrame(mf1.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=3.4,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
    
        )
        pd.DataFrame(mf2.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=2,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
    
        )
        pd.DataFrame(mf3.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=0.6,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
        )
        pd.DataFrame(pf1.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=-0.8,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
    
        )
        pd.DataFrame(pf2.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=-2.2,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
    
        )
        pd.DataFrame(pf3.groupby(level=0).sum().loc[order]).T.plot.bar(
        ax=ax,
        position=-3.6,
        width=0.2,
        stacked=True,
        color=colors,
        label= "Ref",
        edgecolor="k",
        legend=False,
    
        )

        plt.ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)

        plt.xlim(-1, 1)
        plt.ylim(0, 1000)
        plt.xticks(x, x_labels, rotation='vertical')
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)

        a = ax.get_legend_handles_labels()
        b = {l:h for h,l in zip(*a)}       
        c = [*zip(*b.items())]              
        d = c[::-1]                        
        plt.legend(*d, ncol=1, fontsize=14)
        save_folder = f"../../results/{scenario}/plots"  # Specify the desired folder path for each country
        if not os.path.exists(save_folder):
          os.makedirs(save_folder)

        fn = f"{country}_energy_demand_bar_plot.png"
        fn_path = os.path.join(save_folder, fn)
        plt.savefig(fn_path, dpi=300, bbox_inches="tight")
        plt.close()


        nf = data_1.loc[:,'2020']
        mf1 = data_1.loc[:,'2030']
        mf2 = data_1.loc[:,'2040']
        mf3 = data_1.loc[:,'2050']
        pf1 = data_2.loc[:,'2030']
        pf2 = data_2.loc[:,'2040']
        pf3 = data_2.loc[:,'2050']    
        
        fig, ax = plt.subplots(figsize=(30, 15))

        x= [0.075,0.19,0.3,0.43,0.53,0.64,0.75,1.075,1.19,1.3,1.43,1.53,1.64,1.75,2.075,2.19,2.3,2.43,2.53,2.64,2.75,3.075,3.19,3.3,3.43,3.53,3.64,3.75,4.075,4.19,4.3,4.43,4.53,4.64,4.75,5.075,5.19,5.3,5.43,5.53,5.64,5.75,6.075,6.19,6.3,6.43,6.53,6.64,6.75]

        x_labels = ['Reff', 'BAU-2030','BAU-2040', 'Electricity       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Heat       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Oil       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Biomass       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Methane       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Hydrogen       BAU-2050','Suff-2030','Suff-2040','Suff-2050','Reff', 'BAU-2030','BAU-2040', 'Non-energy       BAU-2050','Suff-2030','Suff-2040','Suff-2050']
        nf.unstack().loc[order].loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-0.5, width=.075,legend=False,color=colors,
          )
        mf1.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-2, width=0.075,legend=True,color=colors,
          )
        mf2.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-3.5, width=0.075,legend=False,color=colors,
          )
        mf3.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-5, width=0.075,legend=False,color=colors,
          )
        pf1.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-6.5, width=0.075,legend=False,color=colors,
          )
        pf2.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-8, width=0.075,legend=False,color=colors,
          )
        pf3.unstack().loc[order].plot.bar(
          ax=ax, stacked=True, edgecolor="k",position=-9.5, width=0.075,legend=False,color=colors,
          )
        plt.ylabel("Final energy and non-energy demand [TWh/a]", fontsize=20)
        plt.xlim(0, 7)


        plt.xticks(x, x_labels, rotation='vertical', fontsize=18)
        plt.rcParams['legend.fontsize'] = '15'
        plt.rc('ytick', labelsize=20) 
        fn = f"{country}_sector_demand_bar_plot.png"
        fn_path = os.path.join(save_folder, fn)
        plt.savefig(fn_path, dpi=300, bbox_inches="tight")
        plt.close()


        

