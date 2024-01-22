#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:08:37 2023

"""

import pypsa
import yaml
import pandas as pd

scenario = 'bau'


file = f'../../results/{scenario}/postnetworks/elec_s_6_lvopt_EQ0.70c_1H-T-H-B-I-A-dist1_2020.nc'

n= pypsa.Network(file)


with open("../../config/config.yaml") as file:
    config = yaml.safe_load(file)

# DC lines:
idx = n.links[(n.links.carrier=='DC') & (n.links.bus1 =='BE1 0') ].index
dclines = pd.DataFrame(index=idx,columns=['from','to','capacity (MW)'])
dclines['from'] = n.links.bus0[idx]
dclines['to'] = n.links.bus1[idx]
dclines['capacity (MW)'] = n.links.p_nom_opt[idx]
print(dclines)

# AC lines:
idx = n.lines[n.lines.bus0=='BE1 0'].index
aclines = pd.DataFrame(index=idx,columns=['from','to','capacity (MW)'])
aclines['from'] = n.lines.bus0[idx]
aclines['to'] = n.lines.bus1[idx]
aclines['capacity (MW)'] = n.lines.s_nom_opt[idx]
print(aclines)

a = 1