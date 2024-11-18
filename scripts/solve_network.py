# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""
import importlib
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from prepare_sector_network import determine_emission_sectors

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n, planning_horizons, config):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n, planning_horizons, config)
    else:
        _add_land_use_constraint(n)


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values

    return n


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        extendable_i = (n.generators.carrier == carrier) & n.generators.p_nom_extendable
        n.generators.loc[extendable_i, "p_nom_min"] = 0

        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n, planning_horizons, config):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    grouping_years = config["existing_capacities"]["grouping_years_power"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            {i.split(sep=" ")[0] + " " + i.split(sep=" ")[1] for i in existing.index}
        )

        previous_years = [
            str(y)
            for y in set(planning_horizons + grouping_years)
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_co2_sequestration_limit(n, foresight, config):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """
    if not n.investment_periods.empty:
        periods = n.investment_periods
        names = pd.Index([f"co2_sequestration_limit-{period}" for period in periods])
    else:
        periods = [np.nan]
        names = pd.Index(["co2_sequestration_limit"])
    if foresight == "overnight":
      limit = config["sector"]["co2_sequestration_potential"]
    else:
      current_horizon = int(snakemake.wildcards.planning_horizons)
      limit = config["sector"]["co2_sequestration_potential"][current_horizon]
    n.madd(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n):
    """
    Add maximum growth rates for different carriers.
    """

    opts = snakemake.params["sector"]["limit_max_growth"]
    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period

    return n


def add_retrofit_gas_boiler_constraint(n, snapshots):
    """
    Allow retrofitting of existing gas boilers to H2 boilers.
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    p_gas = dispatch.sel(Link=gas_i)
    p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 3000  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=3000,  # Eur/kWh
            p_nom=1e6,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n, planning_horizons, config)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)
        if snakemake.params["sector"]["limit_max_growth"]["enable"]:
            n = add_max_growth(n)

    if n.stores.carrier.eq("co2 sequestered").any():
        # limit = co2_sequestration_potential
        add_co2_sequestration_limit(n, foresight, config)

    return n

def imposed_values_genertion(n, foresight, config):
    ''' This funtion impse values for generation technologies. For example the
    wind offshore, onshore, solar and nuclear capacities are constraint for 
    Belgium for year 2030 considering the values from ELIA. Also it considers
    that after 2030 the gas storage site at Loenhout will be a H2 hydrogen site.'''
    if foresight == "myopic":
     country = config["imposed_values"]["country"]
     suffix = "1 0"
     if f"{country}{suffix} nuclear-1975" in n.links.index:
        
      #getting values from config file
      
      onwind_max = config["imposed_values"]["onwind"]
      offwind_ac_max = config["imposed_values"]["offshore_ac"]
      offwind_dc_max = config["imposed_values"]["offshore_dc"]
      solar_max = config["imposed_values"]["solar"]
      nuclear_max = config["imposed_values"]["nuclear"]
    
      # preparing data for technoligies considering already installed capacities excluding 2030
      onwind = n.generators[
        n.generators.index.str.contains(country) & 
        n.generators.index.str.contains('onwind') & 
        ~n.generators.index.str.contains('-2030')
      ].p_nom.sum()
    
      offwind_ac = n.generators[
        n.generators.index.str.contains(country) & 
        n.generators.index.str.contains('offwind-ac') & 
        ~n.generators.index.str.contains('-2030')
      ].p_nom.sum()
    
      offwind_dc = n.generators[
        n.generators.index.str.contains(country) & 
        n.generators.index.str.contains('offwind-dc') & 
        ~n.generators.index.str.contains('-2030')
      ].p_nom.sum()
    
      solar = n.generators[
        n.generators.index.str.contains(country) & 
        n.generators.index.str.contains('solar') & 
        ~n.generators.index.str.contains('-2030')
      ].p_nom.sum()
      
      #imposing values in the model for year 2030
      n.generators.loc[f"{country}{suffix} solar-2030", "p_nom_max"] = solar_max - solar
      n.generators.loc[f"{country}{suffix} onwind-2030", "p_nom_max"] = onwind_max - onwind
      n.generators.loc[f"{country}{suffix} offwind-ac-2030", "p_nom_max"] = offwind_ac_max - offwind_ac
      n.generators.loc[f"{country}{suffix} offwind-dc-2030", "p_nom_max"] = offwind_dc_max - offwind_dc
     
      #nuclear is grouped by grouping years so imposing value in last grouping year
      n.links.loc[f"{country}{suffix} nuclear-1975", "p_nom"] = nuclear_max
      #Imposing no underground H2 storage potential for Belgium in 2030
      # n.stores.loc[f"{country}{suffix} H2 Store-2030", "e_nom_max"] = 0.0
     
      # Imposing rooftop potential values gor Belgium based on Energyville BREGILAB project for 
     solar_max_pot = config["imposed_values"]["solar_max"]
     if f"{country}{suffix} solar-2040" in n.generators.index:
          n.generators.loc[f"{country}{suffix} solar-2040", "p_nom_max"] = solar_max_pot
          offwind_max_val = config["imposed_values"]["offshore_max"]
          offwind_val = n.generators[
            n.generators.index.str.contains(country) & 
            n.generators.index.str.contains('offwind')].p_nom_opt.sum()
          offwind_max = offwind_max_val - offwind_val
          n.generators.loc[f"{country}{suffix} offwind-dc-2040", "p_nom_max"] = offwind_max
          #Imposing no underground gas storage potential for Belgium in 2040 considering it would be converted into H2
          # n.stores.loc[f"{country}{suffix} gas Store", "e_nom_min"] = 0.0
          # n.stores.loc[f"{country}{suffix} gas Store", "e_nom_max"] = 0.0
     if f"{country}{suffix} solar-2050" in n.generators.index:
          n.generators.loc[f"{country}{suffix} solar-2050", "p_nom_max"] = solar_max_pot
          offwind_max_val = config["imposed_values"]["offshore_max"]
          offwind_val = n.generators[
            n.generators.index.str.contains(country) & 
            n.generators.index.str.contains('offwind')].p_nom_opt.sum()
          offwind_max = offwind_max_val - offwind_val
          n.generators.loc[f"{country}{suffix} offwind-dc-2050", "p_nom_max"] = offwind_max
          # n.stores.loc[f"{country}{suffix} gas Store", "e_nom_min"] = 0.0
          # n.stores.loc[f"{country}{suffix} gas Store", "e_nom_max"] = 0.0
       
    return n       

def imposed_values_sensitivity_offshore(n, foresight, config):
  ''' This funtion impse values for offshore technologies for Belgium considering additional
    capacity in northsea'''
  if "sensitivity_analysis_offshore_northsea" in config["run"]["name"]: 
    if foresight == "myopic":
     country = config["imposed_values"]["country"]
     suffix = "1 0"
     if f"{country}{suffix} offwind-dc-2040" in n.generators.index:
         n.generators.loc[f"{country}{suffix} offwind-dc-2040", "p_nom_max"] += config["sensitivity_analysis"]["additional_capacity"][2040]
     if f"{country}{suffix} offwind-dc-2050" in n.generators.index:
         n.generators.loc[f"{country}{suffix} offwind-dc-2050", "p_nom_max"] += config["sensitivity_analysis"]["additional_capacity"][2050]
         
  return n

def imposed_values_sequestration(n, config):
  ''' This funtion impse values for carbon sequestration for Belgium for ref scenario.'''
  if config["run"]["name"] == "ref":
      country = config["imposed_values"]["country"]
      suffix = "1 0"
      if f"{country}{suffix} co2 sequestered-2030" in n.stores.index:
          n.stores.loc[f"{country}{suffix} co2 sequestered-2030", "e_nom_max"] = config["sequestration_potentia_BE"][2030] * 1e6
      if f"{country}{suffix} co2 sequestered-2040" in n.stores.index:
          n.stores.loc[f"{country}{suffix} co2 sequestered-2040", "e_nom_max"] = config["sequestration_potentia_BE"][2040] * 1e6
      if f"{country}{suffix} co2 sequestered-2050" in n.stores.index:
          n.stores.loc[f"{country}{suffix} co2 sequestered-2050", "e_nom_max"] = config["sequestration_potentia_BE"][2050] * 1e6 

  return n


def imposed_TYNDP(n, foresight, config):
   ''' This funtion impse values for TYNDP for transmissions lines'''
   tyndp_values_mapping = {
      ("AT1 0", "CH1 0"): {"s_nom": "at_ch", "s_nom_min": "at_ch"},
      ("AT1 0", "CZ1 0"): {"s_nom": "at_cz", "s_nom_min": "at_cz"},
      ("AT1 0", "DE1 0"): {"s_nom": "at_de", "s_nom_min": "at_de"},
      ("AT1 0", "HU1 0"): {"s_nom": "at_hu", "s_nom_min": "at_hu"},
      ("AT1 0", "IT1 0"): {"s_nom": "at_it", "s_nom_min": "at_it"},
      ("AT1 0", "SI1 0"): {"s_nom": "at_si", "s_nom_min": "at_si"},
      ("BE1 0", "FR1 0"): {"s_nom": "be_fr", "s_nom_min": "be_fr"},
      ("BE1 0", "LU1 0"): {"s_nom": "be_lu", "s_nom_min": "be_lu"},
      ("BE1 0", "NL1 0"): {"s_nom": "be_nl", "s_nom_min": "be_nl"},
      ("BG1 0", "HR1 0"): {"s_nom": "bg_hr", "s_nom_min": "bg_hr"},
      ("BG1 0", "RO1 0"): {"s_nom": "bg_ro", "s_nom_min": "bg_ro"},
      ("CH1 0", "DE1 0"): {"s_nom": "ch_de", "s_nom_min": "ch_de"},
      ("CH1 0", "FR1 0"): {"s_nom": "ch_fr", "s_nom_min": "ch_fr"},
      ("CH1 0", "IT1 0"): {"s_nom": "ch_it", "s_nom_min": "ch_it"},
      ("CZ1 0", "DE1 0"): {"s_nom": "cz_de", "s_nom_min": "cz_de"},
      ("CZ1 0", "PL1 0"): {"s_nom": "cz_pl", "s_nom_min": "cz_pl"},
      ("CZ1 0", "SK1 0"): {"s_nom": "cz_sk", "s_nom_min": "cz_sk"},
      ("DE1 0", "DK1 0"): {"s_nom": "de_dk", "s_nom_min": "de_dk"},
      ("DE1 0", "FR1 0"): {"s_nom": "de_fr", "s_nom_min": "de_fr"},
      ("DE1 0", "LU1 0"): {"s_nom": "de_lu", "s_nom_min": "de_lu"},
      ("DE1 0", "NL1 0"): {"s_nom": "de_nl", "s_nom_min": "de_nl"},
      ("DE1 0", "PL1 0"): {"s_nom": "de_pl", "s_nom_min": "de_pl"},
      ("DK2 0", "SE2 0"): {"s_nom": "dk_se", "s_nom_min": "dk_se"},
      ("EE6 0", "LV6 0"): {"s_nom": "ee_lv", "s_nom_min": "ee_lv"},
      ("ES1 0", "FR1 0"): {"s_nom": "es_fr", "s_nom_min": "es_fr"},
      ("ES1 0", "PT1 0"): {"s_nom": "es_pt", "s_nom_min": "es_pt"},
      ("FI2 0", "SE2 0"): {"s_nom": "fi_se", "s_nom_min": "fi_se"},
      ("FR1 0", "IT1 0"): {"s_nom": "fr_it", "s_nom_min": "fr_it"},
      ("GR1 0", "HR1 0"): {"s_nom": "gr_hr", "s_nom_min": "gr_hr"},
      ("HR1 0", "HU1 0"): {"s_nom": "hr_hu", "s_nom_min": "hr_hu"},
      ("HR1 0", "RO1 0"): {"s_nom": "hr_ro", "s_nom_min": "hr_ro"},
      ("HR1 0", "SI1 0"): {"s_nom": "hr_si", "s_nom_min": "hr_si"},
      ("HU1 0", "RO1 0"): {"s_nom": "hu_ro", "s_nom_min": "hu_ro"},
      ("HU1 0", "SK1 0"): {"s_nom": "hu_sk", "s_nom_min": "hu_sk"},
      ("IT1 0", "SI1 0"): {"s_nom": "it_si", "s_nom_min": "it_si"},
      ("LT6 0", "LV6 0"): {"s_nom": "lt_lv", "s_nom_min": "lt_lv"},
      ("NO2 0", "SE2 0"): {"s_nom": "no_se", "s_nom_min": "no_se"},
      ("BE1 0", "DE1 0"): {"s_nom": "be_de", "s_nom_min": "be_de"},
      ("PL1 0", "SK1 0"): {"s_nom": "pl_sk", "s_nom_min": "pl_sk"},}
   if foresight == "overnight":
      
      for index, row in n.lines.iterrows():
       key = (row["bus0"], row["bus1"])
       if key in tyndp_values_mapping:
        values = tyndp_values_mapping[key]
        n.lines.loc[index, "s_nom"] = config["TYNDP_values"][values["s_nom"]]
        n.lines.loc[index, "s_nom_min"] = config["TYNDP_values"][values["s_nom_min"]]
    
  
       n.links.loc["T6", "p_nom"] = 1000
       n.links.loc["14801", "p_nom"] = 1000
      # n.links.loc["T19", "p_nom"] = config["TYNDP_values"]["T19"] 
      # n.links.loc["T21", "p_nom"] =config["TYNDP_values"]["T21"] 
      # n.links.loc["T22", "p_nom"] = config["TYNDP_values"]["T22"] 
      # n.links.loc["T33", "p_nom"] = config["TYNDP_values"]["T33"]
      # n.links.loc["T34", "p_nom"] = config["TYNDP_values"]["T34"] 
   else:        
      country = config["imposed_values"]["country"]
      suffix = "1 0"
      if f"{country}{suffix} nuclear-1975" in n.links.index:
      
       for index, row in n.lines.iterrows():
        key = (row["bus0"], row["bus1"])
        if key in tyndp_values_mapping:
         values = tyndp_values_mapping[key]
         n.lines.loc[index, "s_nom"] = config["TYNDP_values"][values["s_nom"]]
         n.lines.loc[index, "s_nom_min"] = config["TYNDP_values"][values["s_nom_min"]]
      if config["TYNDP_values"]["expansion_limit"] == True:  
       n.lines["s_nom_max"] = n.lines["s_nom"] * config["TYNDP_values"]["max_expansion"]
       condition = (n.links['carrier'] == 'DC')
       n.links.loc[condition, "p_nom_max"] = n.links.loc[condition, "p_nom"] * config["TYNDP_values"]["max_expansion"]
         
   return n
   
def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24h]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"], index_col=[0, 1]
    )
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier])
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, level, by_country, config):
    """
    Add equity constraints to the network.
    The equity option specifies a certain level x of equity (as in
    EQx) where x is a number between 0 and 1. When this option is set,
    each node in the network is required to produce at least a
    fraction of x of its energy demand locally. For example, when
    EQ0.7 is set, each node is required to produce at least 70% of its
    energy demand locally.
    Locally produced energy includes local renewable generation and
    (some) conventional generation (nuclear, coal, geothermal).
    How conventional generation is dealt with depends on whether the
    model is run in electricity-only mode or is sector-coupled. In
    electricity-only mode, all conventional generation is considered
    local production.
    In the sector-coupled model, however, gas and oil are endogenously
    modelled. Oil is not spatially resolved, meaning that any use of
    oil is considered an import, but any production of oil
    (Fischer-Tropsch) is considered an export. When gas is not
    spatially resolved, it functions the same. When, however, a gas
    network is modelled, imports and exports are instead calculated
    using gas network flow. For now, even locally extracted natural
    gas is considered "imported" for the purposes of this equation.
    For other conventional generation (coal, oil, nuclear) the fuel is
    not endogenously modelled, and this generation is considered local
    (even though implementation-wise nodes have to "import" the fuel
    from a copper-plated "EU" node).
    Optionally the EQ option may be suffixed by the letter "c", which
    makes the equity constraint act on a country level instead of a
    node level.
    Regardless, the equity constraint is only enforced on average over
    the whole year.
    In a sector-coupled network, energy production is generally
    greater than consumption because of efficiency losses in energy
    conversions such as hydrogen production (whereas heat pumps
    actually have an "efficiency" greater than 1). Ignoring these
    losses would lead to a weakening of the equity constraint (i.e. if
    1.5TWh of electricity needs to be produced to satisfy a final
    demand of 1 TWh of energy, even an equity constraint of 100% would
    be satisfied if 1TWh of electricity is produced locally).
    Therefore, for the purpose of the equity constraint, efficiency
    losses in a sector-coupled network are effectively counted as
    demand, and the equity constraint is enforced on the sum of final
    demand and efficiency losses.
    Again in the sector-coupled model, some energy supply and energy
    demand are copper-plated, meaning that they are not spatially
    resolved by only modelled europe-wide. For the purpose of the
    equity constraint in a sector-coupled model, energy supplied from
    a copper-plated carrier (supplied from the "european node") is
    counted as imported, not locally produced. Similarly, energy
    demand for a copper-plated carrier (demanded at the "european
    node") is counted as exported, not locally demanded.
    Parameters
    ----------
    -------
    scenario:
        opts: [Co2L-EQ0.7-24H]
    """
    # TODO: Does this work for myopic optimisation?

    # Implementation note: while the equity constraint nominally
    # specifies that a minimum fraction demand be produced locally, in
    # the implementation we enforce a minimum ratio between local
    # production and net energy imports. This because
    #     local_production + net_imports = demand + efficiency_losses
    # for each node (or country), so we can convert a constraint of the form
    #     local_production >= level * (demand + efficiency_losses)
    # to the equivalent:
    #     net_imports <= (1 / level - 1) * local_production
    # or, putting all variables on the right hand side and constants
    # on the left hand side:
    #     (1 - 1 / level) * local_production + net_imports <= 0
    #
    # While leading to an equivalent constraint, we choose this
    # implementation because it allows us to avoid having to calculate
    # efficiency losses explicitly; local production and net imports
    # are slightly easier to deal with.
    #
    # Notes on specific technologies. Locally produced energy comes
    # from the following sources:
    # - Variable renewables (various types of solar, onwind, offwind,
    #   etc.), implemented as generators.
    # - Conventional sources (gas, coal, nuclear), implemented as
    #   either generators or links (depending on whether or not the
    #   model is sector-coupled).
    # - Biomass, biogass, if spatially resolved, implemented as stores.
    # - Hydro, implemented as storageunits.
    # - Ambient heat used in heat pumps, implemented as links.
    # Imports can come in the following forms:
    # - Electricity (AC & DC), implemented as lines and links.
    # - Gas pipelines, implemented as links.
    # - H2 pipelines, implemented as links.
    # - Gas imports (pipeline, LNG, production), implemented as generators.

    # if config["foresight"] != "overnight":
    #     logging.warning(
    #         "Careful! Equity constraint is only tested for 'overnight' "
    #         f"foresight models, not '{config['foresight']}' foresight"
    #     )

    # While we need to group components by bus location in the
    # sector-coupled model, there is no "location" column in the
    # electricity-only model.
    location = (
        n.buses.location
        if "location" in n.buses.columns
        else pd.Series(n.buses.index, index=n.buses.index)
    )

    def group(df, b="bus"):
        """
        Group given dataframe by bus location or country.
        The optional argument `b` allows clustering by bus0 or bus1 for
        lines and links.
        """
        if by_country:
            return df[b].map(location).map(n.buses.country).to_xarray()
        else:
            return df[b].map(location).to_xarray()

    # Local production by generators. Note: the network may not
    # actually have all these generators (for instance some
    # conventional generators are implemented as links in the
    # sector-coupled model; heating sector might not be turned on),
    # but we list all that might be in the network.
    local_gen_carriers = list(
        set(
            config["electricity"]["extendable_carriers"]["Generator"]
            + config["electricity"]["conventional_carriers"]
            + config["electricity"]["renewable_carriers"]
            + [c for c in n.generators.carrier if "solar thermal" in c]
            + ["solar rooftop", "wave"]
        )
    )
    local_gen_i = n.generators.loc[
        n.generators.carrier.isin(local_gen_carriers)
        & (n.generators.bus.map(location) != "EU")
    ].index
    local_gen_p = (
        n.model["Generator-p"]
        .loc[:, local_gen_i]
        .groupby(group(n.generators.loc[local_gen_i]))
        .sum()
    )
    local_gen = (local_gen_p * n.snapshot_weightings.generators).sum("snapshot")

    # Hydro production; the only local production from a StorageUnit.
    local_hydro_i = n.storage_units.loc[n.storage_units.carrier == "hydro"].index
    local_hydro_p = (
        n.model["StorageUnit-p_dispatch"]
        .loc[:, local_hydro_i]
        .groupby(group(n.storage_units.loc[local_hydro_i]))
        .sum()
    )
    local_hydro = (local_hydro_p * n.snapshot_weightings.stores).sum("snapshot")

    # Biomass and biogas; these are only considered locally produced
    # if spatially resolved, not if they belong to an "EU" node. They
    # are modelled as stores with initial capacity to model a finite
    # yearly supply; the difference between initial and final capacity
    # is the total local production.
    # local_bio_i = n.stores.loc[
    #     n.stores.carrier.isin(["biogas", "solid biomass"])
    #     & (n.stores.bus.map(location) != "EU")
    # ].index
    # #Building the following linear expression only works if it's non-empty
    # if len(local_bio_i) > 0:
    #     local_bio_first_e = n.model["Store-e"].loc[n.snapshots[0], local_bio_i]
    #     local_bio_last_e = n.model["Store-e"].loc[n.snapshots[-1], local_bio_i]
    #     local_bio_p = local_bio_first_e - local_bio_last_e
    #     local_bio = local_bio_p.groupby(group(n.stores.loc[local_bio_i])).sum()
    # else:
    #     local_bio = None

    # Conventional generation in the sector-coupled model. These are
    # modelled as links in order to take the CO2 cycle into account.
    # All of these are counted as local production even if the links
    # may take their fuel from an "EU" node, except for gas and oil,
    # which are modelled endogenously and is counted under imports /
    # exports.
    conv_carriers = config["sector"].get("conventional_generation", {})
    conv_carriers = [
        gen for gen, carrier in conv_carriers.items() if carrier not in ["gas", "oil"]
    ]
    if config["sector"].get("coal_cc") and not "coal" in conv_carriers:
        conv_carriers.append("coal")
    local_conv_gen_i = n.links.loc[n.links.carrier.isin(conv_carriers)].index
    if len(local_conv_gen_i) > 0:
        local_conv_gen_p = n.model["Link-p"].loc[:, local_conv_gen_i]
        # These links have efficiencies, which we multiply by since we
        # only want to count the _output_ of each conventional
        # generator as local generation for the equity balance.
        efficiencies = n.links.loc[local_conv_gen_i, "efficiency"]
        local_conv_gen_p = (
            (local_conv_gen_p * efficiencies)
            .groupby(group(n.links.loc[local_conv_gen_i], b="bus1"))
            .sum()
            .rename({"bus1": "Bus"})
        )
        local_conv_gen = (local_conv_gen_p * n.snapshot_weightings.generators).sum(
            "snapshot"
        )
    else:
        local_conv_gen = None

    #TODO: should we (in prepare_sector_network.py) model gas
    #pipeline imports from outside the EU and LNG imports separately
    #from gas extraction / production? Then we could model gas
    #extraction as locally produced energy.

    # #Ambient heat for heat pumps
    # heat_pump_i = n.links.filter(like="heat pump", axis="rows").index
    # if len(heat_pump_i) > 0:
    #     # To get the ambient heat extracted, we subtract 1 from the
    #     # efficiency of the heat pump (where "efficiency" is really COP
    #     # for heat pumps).
    #     from_ambient = n.links_t["efficiency"].loc[:, heat_pump_i] - 1
    #     local_heat_from_ambient_p = n.model["Link-p"].loc[:, heat_pump_i]
    #     local_heat_from_ambient = (
    #         (local_heat_from_ambient_p * from_ambient)
    #         .groupby(group(n.links.loc[heat_pump_i], b="bus1"))
    #         .sum()
    #         .rename({"bus1": "bus"})
    #     )
    #     local_heat_from_ambient = (
    #         local_heat_from_ambient * n.snapshot_weightings.generators
    #     ).sum("snapshot")
    # else:
    #     local_heat_from_ambient = None

    #Total locally produced energy
    local_energy = sum(
        e
        for e in [
            local_gen,
            # local_hydro,
            # local_bio,
            local_conv_gen,
            #local_heat_from_ambient,
        ]
        if e is not None
    )

    # Now it's time to collect imports: electricity, hydrogen & gas
    # pipeline, other gas, biomass, gas terminals & production.

    # Start with net electricity imports.
    lines_cross_region_i = n.lines.loc[
        (group(n.lines, b="bus0") != group(n.lines, b="bus1")).to_numpy()
    ].index
    # Build linear expression representing net imports (i.e. imports -
    # exports) for each bus/country.
    lines_in_s = (
    n.model["Line-s"]
    .loc[:, lines_cross_region_i]
    .groupby(group(n.lines.loc[lines_cross_region_i], b="bus1"))
    .sum()
    .rename({"bus1": "Bus"})
      ) - (
    n.model["Line-s"]
    .loc[:, lines_cross_region_i]
    .groupby(group(n.lines.loc[lines_cross_region_i], b="bus0"))
    .sum()
    .rename({"bus0": "Bus"})
      )
    line_imports = (lines_in_s * n.snapshot_weightings.generators).sum("snapshot")
    

    # Link net imports, representing all net energy imports of various
    # carriers that are implemented as links. We list all possible
    # link carriers that could be represented in the network; some
    # might not be present in some networks depending on the sector
    # configuration. Note that we do not count efficiencies here (e.g.
    # for oil boilers that import oil) since efficiency losses are
    # counted as "local demand".
    link_import_carriers = [
        # Pipeline imports / exports
        # "H2 pipeline",
        # "H2 pipeline retrofitted",
        # "gas pipeline",
        # "gas pipeline new",
        # # Solid biomass
        # "solid biomass transport",
        # DC electricity
        "DC"
        # # Oil (imports / exports between spatial nodes and "EU" node)
        # "Fischer-Tropsch",
        # "biomass to liquid",
        # "residential rural oil boiler",
        # "services rural oil boiler",
        # "residential urban decentral oil boiler",
        # "services urban decentral oil boiler",
        # "oil",  # Oil powerplant (from `prepare_sector_network.add_generation`)
        # # Gas (imports / exports between spatial nodes and "EU" node,
        # # only cross-region if gas is not spatially resolved)
        # "Sabatier",
        # "helmeth",
        # "SMR CC",
        # "SMR",
        # "biogas to gas",
        # "BioSNG",
        # "residential rural gas boiler",
        # "services rural gas boiler",
        # "residential urban decentral gas boiler",
        # "services urban decentral gas boiler",
        # "urban central gas boiler",
        # "urban central gas CHP",
        # "urban central gas CHP CC",
        # "residential rural micro gas CHP",
        # "services rural micro gas CHP",
        # "residential urban decentral micro gas CHP",
        # "services urban decentral micro gas CHP",
        # # "allam",
        # "OCGT",
        # "CCGT",
    ]
    links_cross_region_i = (
        n.links.loc[(group(n.links, b="bus0") != group(n.links, b="bus1")).to_numpy()]
        .loc[n.links.carrier.isin(link_import_carriers)]
        .index
    )
    # links_cross_region_i = links_cross_region_i[~links_cross_region_i.str.contains('-reversed', case=False)]
    # Build linear expression representing net imports (i.e. imports -
    # exports) for each bus/country.
    links_in_p = (
        n.model["Link-p"]
        .loc[:, links_cross_region_i]
        .groupby(group(n.links.loc[links_cross_region_i], b="bus1"))
        .sum()
        .rename({"bus1": "Bus"})
    ) - (
        n.model["Link-p"]
        .loc[:, links_cross_region_i]
        .groupby(group(n.links.loc[links_cross_region_i], b="bus0"))
        .sum()
        .rename({"bus0": "Bus"})
    )
    link_imports = (links_in_p * n.snapshot_weightings.generators).sum("snapshot")

    # Gas imports by pipeline from outside of Europe, LNG terminal or
    # gas production (all modelled as generators).
    # gas_import_i = n.generators.loc[n.generators.carrier == "gas"].index
    # if len(gas_import_i) > 0:
    #     gas_import_p = (
    #         n.model["Generator-p"]
    #         .loc[:, gas_import_i]
    #         .groupby(group(n.generators.loc[gas_import_i]))
    #         .sum()
    #     )
    #     gas_imports = (gas_import_p * n.snapshot_weightings.generators).sum("snapshot")
    # else:
    #     gas_imports = None

    imported_energy = sum(
        i for i in [line_imports, link_imports] if i is not None
    )

    # local_factor = 1 - level
    local_factor = level - 1
    n.model.add_constraints(
            local_factor * local_energy + imported_energy <= 0, name="equity_min"
    )


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum(
            "Generator"
        )

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * p_max_pu[ext_i]

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    backward_i = get_backward_i(forward_i)

    lhs = n.model["Link-p_nom"].loc[backward_i]
    rhs = n.model["Link-p_nom"].loc[forward_i]

    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")

def add_co2_atmosphere_constraint(n, snapshots):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")

def add_co2limit_country(n, limit_countries, nyears=1.0):
    """
    Add a set of emissions limit constraints for specified countries.
    The countries and emissions limits are specified in the config file entry 'co2_budget_country_{investment_year}'.
    Parameters
    ----------
    n : pypsa.Network
    config : dict
    limit_countries : dict
    nyears: float, optional
        Used to scale the emissions constraint to the number of snapshots of the base network.
    """
    logger.info(f"Adding CO2 budget limit for each country as per unit of 1990 levels")

    countries = n.config["countries"]

    # TODO: import function from prepare_sector_network? Move to common place?
    sectors = determine_emission_sectors(options)

    # convert Mt to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)
    #Consider non-energy emissions from agriculture in the carbon budget on country level
    ghg_emissions_agri= 1e6 * pd.read_csv(snakemake.input.ghg_emissions_agri, index_col=0)
    non_energy_ghg_agri = ghg_emissions_agri.loc[countries, 'Total net GHG emissions from non-energy sources in agriculture']
    non_energy_ghg_agri[non_energy_ghg_agri < 0] = 0
    co2_limit_countries = co2_totals.loc[countries, sectors].sum(axis=1)
    co2_limit_countries = co2_limit_countries.loc[
        co2_limit_countries.index.isin(limit_countries.keys())
    ]
    lulucf = co2_totals.loc[countries, 'LULUCF']
    lulucf[lulucf > 0] = 0
    lulucf = lulucf * -1
    co2_limit_countries *= co2_limit_countries.index.map(limit_countries) * nyears
    co2_limit_countries = (co2_limit_countries + lulucf) - non_energy_ghg_agri

    p = n.model["Link-p"]  # dimension: (time, component)

    # NB: Most country-specific links retain their locational information in bus1 (except for DAC, where it is in bus2, and process emissions, where it is in bus0)
    country = n.links.bus1.map(n.buses.location).map(n.buses.country)
    country_DAC = (
        n.links[n.links.carrier == "DAC"]
        .bus3.map(n.buses.location)
        .map(n.buses.country)
    )
    country[country_DAC.index] = country_DAC
    country_process_emissions = (
        n.links[n.links.carrier.str.contains("process emissions")]
        .bus0.map(n.buses.location)
        .map(n.buses.country)
    )
    country[country_process_emissions.index] = country_process_emissions

    lhs = []
    for port in [col[3:] for col in n.links if col.startswith("bus")]:
        if port == str(0):
            efficiency = (
                n.links["efficiency"].apply(lambda x: 1.0).rename("efficiency0")
            )
        elif port == str(1):
            efficiency = n.links["efficiency"]
        else:
            efficiency = n.links[f"efficiency{port}"]
        mask = n.links[f"bus{port}"].map(n.buses.carrier).eq("co2")

        idx = n.links[mask].index

        grouping = country.loc[idx]

        if not grouping.isnull().all():
            expr = (
                (p.loc[:, idx] * efficiency[idx])
                .groupby(grouping, axis=1)
                .sum()
                * n.snapshot_weightings.generators
            ).sum(dims="snapshot")
            lhs.append(expr)

    lhs = sum(lhs)  # dimension: (country)
    lhs = lhs.rename({list(lhs.dims)[0]: "snapshot"})
    rhs = pd.Series(co2_limit_countries)  # dimension: (country)
    for ct in lhs.indexes["snapshot"]:
        n.model.add_constraints(
            lhs.loc[ct] <= rhs[ct],
            name=f"GlobalConstraint-co2_limit_per_country{ct}",
        )
        n.add(
            "GlobalConstraint",
            f"co2_limit_per_country{ct}",
            constant=rhs[ct],
            sense="<=",
            type="",
        )

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)


    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)
    for o in opts:
        if "EQ" in o:
            EQ_regex = "EQ(0\.[0-9]+)(c?)"  # Ex.: EQ0.75c
            m = re.search(EQ_regex, o)
            if m is not None:
                level = float(m.group(1))
                level = level+0.1
                by_country = True if m.group(2) == "c" else False
                add_EQ_constraints(n, level, by_country, config)
            else:
                logging.warning(f"Invalid EQ option: {o}")
    if n.config["sector"]["co2_budget_national"]:
        # prepare co2 constraint
        nhours = n.snapshot_weightings.generators.sum()
        nyears = nhours / 8760
        investment_year = int(snakemake.wildcards.planning_horizons[-4:])
        limit_countries = snakemake.config["co2_budget_national"][investment_year]

        # add co2 constraint for each country
        logger.info(f"Add CO2 limit for each country")
        add_co2limit_country(n, limit_countries, nyears)
    if snakemake.params.custom_extra_functionality:
        source_path = snakemake.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config

    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )
    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        # labels = n.model.compute_infeasibilities()
        # logger.info(f"Labels:\n{labels}")
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            configfiles="../config/test/config.perfect.yaml",
            simpl="",
            opts="",
            clusters="37",
            ll="v1.0",
            sector_opts="CO2L0-1H-T-H-B-I-A-dist1",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    config = snakemake.config
    options = snakemake.params.sector
    foresight=snakemake.params.foresight
    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )
    n = imposed_values_genertion(
        n,
        config=snakemake.config,
        foresight=snakemake.params.foresight,)
    n = imposed_values_sensitivity_offshore(
        n,
        foresight=snakemake.params.foresight,
        config=snakemake.config,)
    n = imposed_values_sequestration(
        n,
        config=snakemake.config,)
    n = imposed_TYNDP(
        n,
        config=snakemake.config,
        foresight=snakemake.params.foresight,)
    
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        n = solve_network(
            n,
            config=snakemake.config,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")  

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
