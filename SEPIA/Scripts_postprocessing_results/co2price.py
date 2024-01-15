#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:23:09 2023

"""
import pypsa
import matplotlib.pyplot as plt
import numpy as np

n=pypsa.Network("../../results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2020.nc")

m_1=pypsa.Network("../../results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2030.nc")
m_2=pypsa.Network("../../results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2040.nc")
m_3=pypsa.Network("../../results/bau/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2050.nc")

r_1=pypsa.Network("../../results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2030.nc")
r_2=pypsa.Network("../../results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2040.nc")
r_3=pypsa.Network("../../results/ncdr/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2050.nc")

aa=n.global_constraints
bb=m_1.global_constraints
cc=m_2.global_constraints
dd=m_3.global_constraints
hh=r_1.global_constraints
ii=r_2.global_constraints
jj=r_3.global_constraints

countries = ['BE', 'DE', 'FR', 'GB', 'NL']

for country in countries:
 a=aa.loc[f"co2_limit_per_country{country}", "mu"]
 b=bb.loc[f"co2_limit_per_country{country}", "mu"]
 c=cc.loc[f"co2_limit_per_country{country}", "mu"]
 d=dd.loc[f"co2_limit_per_country{country}", "mu"]
 e=hh.loc[f"co2_limit_per_country{country}", "mu"]
 f=ii.loc[f"co2_limit_per_country{country}", "mu"]
 g=jj.loc[f"co2_limit_per_country{country}", "mu"]

 a=abs(a)
 b=abs(b)
 c=abs(c)
 d=abs(d)
 e=abs(e)
 f=abs(f)
 g=abs(g)





 colors = ['black']
 labels =['Reff','BAU-2030', 'BAU-2040', 'BAU-2050', 'Suff-2030','Suff-2040','Suff-2050']
 values = [a, b, c, d, e, f, g]
 fig_size = (8, 6)  # Adjust the width and height as needed

 # Create the figure with the specified size
 plt.figure(figsize=fig_size)
 x = np.arange(len(labels))
 bar_width = 0.6

 # Create the bar plot
 plt.bar(x, values, width=bar_width, align='center', alpha=0.7, color=colors)

 # Optionally, set the labels for the x-axis ticks
 plt.xticks(x, labels, fontsize=20, rotation='vertical')

 # Add labels to the axes
 plt.xlabel('')
 plt.ylabel('Costs [Euro/ton(CO2)', fontsize=20)
 plt.yticks(fontsize=20)

 # Add a title to the plot (optional)
 plt.title('CO2 Price', fontsize=20)
 plt.show()

 fig_size = (10, 7)  # Adjust the width and height as needed

 # Create the figure with the specified size
 plt.figure(figsize=fig_size)
 x = range(7)

 # Define your values
 values = [a, b, c, d, e, f, g]

 # Define colors for the scatter plot (assuming you have 10 colors)
 colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown']

 # Create a scatter plot
 plt.scatter(x, values,s=30,alpha=0.7, color=colors, label='Reff')

 # Create lines connecting b, c, d, and e
 plt.plot(x[1:4], values[1:4], linestyle='-', marker='s', color='black', markersize=10,label='BAU')

 # Create lines connecting f, g, h, i, and j
 plt.plot(x[4:7], values[4:7], linestyle='-', marker='o',markersize=10, color='green', label='Suff')

 for i, txt in enumerate(['Reff', '2030', '2040', '2050', '2030', '2040', '2050']):
    plt.text(x[i], values[i], txt, fontsize=15, ha='center', va='bottom', color='black')
 ax = plt.gca()
 # Label your axes and add a legend
 ax.grid(True, linestyle='--', color='gray', alpha=0.5)

 plt.ylabel('CO2 Price [Eur/ton]', fontsize=20)
 w = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
 u = {l:h for h,l in zip(*w)}        # b = {l1:h1, l2:h2}             unique
 v = [*zip(*u.items())]              # c = [(l1 l2) (h1 h2)]
 z = v[::-1]                        
 plt.legend(*z, ncol=1, fontsize=16)
 plt.rc('ytick',labelsize=20)
 plt.gca().get_xaxis().set_visible(False)
 plt.show()

