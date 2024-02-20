# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for a specific country for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging

logger = logging.getLogger(__name__)


import pandas as pd
import pypsa
from prepare_sector_network import prepare_costs
idx = pd.IndexSlice





opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}
def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            if i == -1:
                c.df.loc[names, "location"] = ""
            else:
                c.df.loc[names, "location"] = names.str[:i]


def calculate_cfs(n, label, cfs, config):
    for c in n.iterate_components(
        n.branch_components
        | n.controllable_one_port_components ^ {"Load", "StorageUnit"}
    ):
        capacities_c = (
            c.df[opt_name.get(c.name, "p") + "_nom_opt"].groupby([c.df.carrier, c.df.location]).sum()
        )
        capacities_c = capacities_c.loc[capacities_c.index.get_level_values('location').str.contains(country)]
        
        if c.name in ["Link", "Line", "Transformer"]:
            p = c.pnl.p0.filter(like=country).abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.filter(like=country).abs().mean()
        else:
            p = c.pnl.p.filter(like=country).abs().mean()

        p_c = p.groupby(c.df.carrier).sum()

        cf_c = p_c / capacities_c
        cf_c = cf_c.fillna(0)

        cf_c = pd.concat([cf_c], keys=[c.list_name])

        cfs = cfs.reindex(cf_c.index.union(cfs.index))

        cfs.loc[cf_c.index, label] = cf_c

    return cfs


def calculate_costs(n, label, costs, config):
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        c.df["capital_costs"] = (
            c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        )
        capital_costs = c.df.groupby(["location", "carrier"])["capital_costs"].sum()
        
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "capital") + t for t in capital_costs.index.to_list()]
        )
        index = index[index.get_level_values(2).str.contains(country)]
        capital_costs = capital_costs.loc[capital_costs.index.get_level_values('location').str.contains(country)]
        costs = costs.reindex(index.union(costs.index))
        costs.loc[index, label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.filter(like=country).multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.filter(like=country).multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.filter(like=country).multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        c.df["marginal_costs"] = p * c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum()
        
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()]
        )
        index = index[index.get_level_values(2).str.contains(country)]
        marginal_costs = marginal_costs.loc[marginal_costs.index.get_level_values('location').str.contains(country)]
        costs = costs.reindex(index.union(costs.index))
        costs.loc[index, label] = marginal_costs.values

    return costs


def calculate_capacities(n, label, capacities, config):
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        capacities_c = c.df.groupby(["location", "carrier"])[
            opt_name.get(c.name, "p") + "_nom_opt"
        ].sum()
        index = pd.MultiIndex.from_tuples(
            [(c.list_name,) + t for t in capacities_c.index.to_list()]
        )
        index = index[index.get_level_values(1).str.contains(country)]
        capacities_c = capacities_c.loc[capacities_c.index.get_level_values('location').str.contains(country)]
        capacities = capacities.reindex(index.union(capacities.index))
        capacities.loc[index, label] = capacities_c.values

    return capacities



def calculate_curtailment(n, label, curtailment, config):
    avail = (
        n.generators_t.p_max_pu.filter(like=country).multiply(n.generators.p_nom_opt.filter(like=country))
        .sum()
        .groupby(n.generators.carrier)
        .sum()
    )
    used = n.generators_t.p.filter(like=country).sum().groupby(n.generators.carrier).sum()

    curtailment[label] = (((avail - used) / avail) * 100).round(3)

    return curtailment


def calculate_energy(n, label, energy, config):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.name in n.one_port_components:
            c_energies = (
                c.pnl.p.filter(like=country).multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.sign)
                .groupby(c.df.carrier)
                .sum()
            )
        else:
            c_energies = pd.Series(0.0, c.df.carrier.unique())
            for port in [col[len("bus"):] for col in c.df.columns if col.startswith("bus")]:
                totals = (
                    c.pnl["p" + port].filter(like=country)
                    .multiply(n.snapshot_weightings.generators, axis=0)
                    .sum()
                )
                # remove values where bus is missing (bug in nomopyomo)
                mask = (c.df["bus" + port] == "") & c.df.index.str.contains(country)
                totals[mask] = float(n.component_attrs[c.name].loc["p" + port, "default"])
                c_energies -= totals.groupby(c.df.carrier).sum()

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index.union(energy.index))

        energy.loc[c_energies.index, label] = c_energies

    return energy


def calculate_supply(n, label, supply, config):
    """
    Calculate the max dispatch of each component at the buses aggregated by
    carrier.
    """
    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = n.buses.carrier == i
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):
            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = (
                c.pnl.p[items].filter(like=country)
                .max()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"])
                .sum()
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply = supply.reindex(s.index.union(supply.index))
            supply.loc[s.index, label] = s

        for c in n.iterate_components(n.branch_components):
            for c in n.iterate_components(n.branch_components):
             for end in [col[len("bus"):] for col in c.df.columns if col.startswith("bus")]:
                items = c.df.index[(c.df["bus" + end].map(bus_map).fillna(False)) & c.df.index.str.contains(country)]


                if len(items) == 0:
                    continue
                # lots of sign compensation for direction and to do maximums
                items_to_keep = items.intersection(c.pnl["p" + end].columns)
                s = (-1) ** (1 - int(end)) * (
                    (-1) ** int(end) * c.pnl["p" + end][items_to_keep]
                ).filter(like=country).max().groupby(c.df.loc[items_to_keep, "carrier"]).sum()
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply = supply.reindex(s.index.union(supply.index))
                supply.loc[s.index, label] = s

    return supply


def calculate_supply_energy(n, label, supply_energy, config):
    """
    Calculate the total energy supply/consuption of each component at the buses
    aggregated by carrier.
    """
    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = n.buses.carrier == i
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):
            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = (
                c.pnl.p[items].filter(like=country)
                .multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"])
                .sum()
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s

        for c in n.iterate_components(n.branch_components):
            for end in [col[len("bus"):] for col in c.df.columns if col.startswith("bus")]:
                items = c.df.index[(c.df["bus" + str(end)].map(bus_map).fillna(False)) & c.df.index.str.contains(country)]

                if len(items) == 0:
                    continue
                items_to_keep = items.intersection(c.pnl["p" + end].columns)
                s = (-1) * c.pnl["p" + end][items_to_keep].filter(like=country).multiply(
                    n.snapshot_weightings.generators, axis=0
                ).sum().groupby(c.df.loc[items_to_keep, "carrier"]).sum()
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(
                    s.index.union(supply_energy.index)
                )

                supply_energy.loc[s.index, label] = s

    return supply_energy



def calculate_prices(n, label, prices, config):
    prices = prices.reindex(prices.index.union(n.buses.carrier.unique()))

    # WARNING: this is time-averaged, see weighted_prices for load-weighted average
    prices[label] = n.buses_t.marginal_price.filter(like=country).mean().groupby(n.buses.carrier).mean()

    return prices


def calculate_market_values(n, label, market_values, config):
    # Warning: doesn't include storage units

    carrier = "AC"

    buses = n.buses.filter(like=country).index[n.buses.carrier == carrier]

    ## First do market value of generators ##

    generators = n.generators.filter(like=country).index[n.buses.loc[n.generators.bus, "carrier"] == carrier]

    techs = n.generators.loc[generators, "carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index.union(techs))

    for tech in techs:
        gens = generators[n.generators.loc[generators, "carrier"] == tech]

        dispatch = (
            n.generators_t.p[gens].filter(like=country)
            .groupby(n.generators.loc[gens, "bus"], axis=1)
            .sum()
            .reindex(columns=buses, fill_value=0.0)
        )

        revenue = dispatch * n.buses_t.marginal_price[buses]

        market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    ## Now do market value of links ##

    for i in ["0", "1"]:
        all_links = n.links.filter(like=country).index[n.buses.loc[n.links["bus" + i], "carrier"] == carrier]

        techs = n.links.loc[all_links, "carrier"].value_counts().index

        market_values = market_values.reindex(market_values.index.union(techs))

        for tech in techs:
            links = all_links[n.links.loc[all_links, "carrier"] == tech]

            dispatch = (
                n.links_t["p" + i][links].filter(like=country)
                .groupby(n.links.loc[links, "bus" + i], axis=1)
                .sum()
                .reindex(columns=buses, fill_value=0.0)
            )

            revenue = dispatch * n.buses_t.marginal_price[buses]

            market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    return market_values


def calculate_price_statistics(n, label, price_statistics, config):
    price_statistics = price_statistics.reindex(
        price_statistics.index.union(
            pd.Index(["zero_hours", "mean", "standard_deviation"])
        )
    )

    buses = n.buses.filter(like=country).index[n.buses.carrier == "AC"]

    threshold = 0.1  # higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.0, columns=buses, index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.0

    price_statistics.at["zero_hours", label] = df.sum().sum() / (
        df.shape[0] * df.shape[1]
    )

    price_statistics.at["mean", label] = (
        n.buses_t.marginal_price[buses].unstack().mean()
    )

    price_statistics.at["standard_deviation", label] = (
        n.buses_t.marginal_price[buses].unstack().std()
    )

    return price_statistics


def make_summaries(networks_dict, config):
    outputs = [
        "cfs",
        "costs",
        "capacities",
        "curtailment",
        "energy",
        "supply",
        "supply_energy",
        "prices",
        "price_statistics",
        "market_values",
    ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=["cluster", "ll", "opt", "planning_horizon"]
    )

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)

    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        n = pypsa.Network(filename)

        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output], config)
    
    return df


def to_csv(df):
    for key in df:
       df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("make_country_summary")
        
    simpl = ""
    cluster = 6
    opt = "EQ0.70c"
    sector_opt = "1H-T-H-B-I-A-dist1"
    ll = "vopt"
    planning_horizons = [2020, 2030, 2040, 2050]
    study = snakemake.params.study
    networks_dict = {
            (cluster, ll, opt + sector_opt, planning_horizon): f"results/{study}" +
            f"/postnetworks/elec_s{simpl}_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
            for planning_horizon in planning_horizons
        }
    country = snakemake.params.country
    planning_horizons = snakemake.params.planning_horizons
    planning_horizons = int(''.join(map(str, planning_horizons)))
    logging.basicConfig(level=snakemake.config["logging"]["level"])
    Nyears = len(pd.date_range(freq="h", **snakemake.params.snapshots)) / 8760

    costs_db = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        Nyears,
    )
    config=snakemake.config
    df = make_summaries(networks_dict, config)


    to_csv(df)
