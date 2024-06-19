# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


if config["foresight"] != "perfect":

    rule plot_power_network_clustered:
        params:
            plotting=config_provider("plotting"),
        input:
            network=resources("networks/elec_s{simpl}_{clusters}.nc"),
            regions_onshore=resources(
                "regions_onshore_elec_s{simpl}_{clusters}.geojson"
            ),
        output:
            map=resources("maps/power-network-s{simpl}-{clusters}.pdf"),
        threads: 1
        resources:
            mem_mb=4000,
        benchmark:
            benchmarks("plot_power_network_clustered/elec_s{simpl}_{clusters}")
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network_clustered.py"

    rule plot_power_network:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_elec_s{simpl}_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_power_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_power_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network.py"
    if config["run"]["name"] != "reff":
     rule plot_hydrogen_network:
        params:
            plotting=config_provider("plotting"),
            foresight=config_provider("foresight"),
        input:
            network=RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_elec_s{simpl}_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-h2_network_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_hydrogen_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_hydrogen_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_hydrogen_network.py"

    rule plot_gas_network:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_elec_s{simpl}_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-ch4_network_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_gas_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_gas_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_gas_network.py"


if config["foresight"] == "perfect":

    def output_map_year(w):
        return {
            f"map_{year}": RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_"
            + f"{year}.pdf"
            for year in config_provider("scenario", "planning_horizons")(w)
        }

    rule plot_power_network_perfect:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_brownfield_all_years.nc",
            regions=resources("regions_onshore_elec_s{simpl}_{clusters}.geojson"),
        output:
            unpack(output_map_year),
        threads: 2
        resources:
            mem_mb=10000,
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network_perfect.py"
        
        
rule make_summary:
    params:
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        scenario=config_provider("scenario"),
        RDIR=RDIR,
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            allow_missing=True,
        ),
        costs=lambda w: (
            ("data/costs_{}.csv".format(config_provider("costs", "year")(w)))
            if config_provider("foresight")(w) == "overnight"
            else (
                "data/costs_{}.csv".format(
                    config_provider("scenario", "planning_horizons", 0)(w)
                )
            )
        ),
        ac_plot=expand(
            resources("maps/power-network-s{simpl}-{clusters}.pdf"),
            **config["scenario"],
            allow_missing=True,
        ),
        costs_plot=expand(
            RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"],
            allow_missing=True,
        ),
        h2_plot=lambda w: expand(
            (
                RESULTS
                + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-h2_network_{planning_horizons}.pdf"
                if config_provider("sector", "H2_network")(w)
                else []
            ),
            **config["scenario"],
            allow_missing=True,
        ),
        ch4_plot=lambda w: expand(
            (
                RESULTS
                + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-ch4_network_{planning_horizons}.pdf"
                if config_provider("sector", "gas_network")(w)
                else []
            ),
            **config["scenario"],
            allow_missing=True,
        ),
    output:
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        nodal_cfs=RESULTS + "csvs/nodal_cfs.csv",
        cfs=RESULTS + "csvs/cfs.csv",
        costs=RESULTS + "csvs/costs.csv",
        capacities=RESULTS + "csvs/capacities.csv",
        curtailment=RESULTS + "csvs/curtailment.csv",
        energy=RESULTS + "csvs/energy.csv",
        supply=RESULTS + "csvs/supply.csv",
        supply_energy=RESULTS + "csvs/supply_energy.csv",
        prices=RESULTS + "csvs/prices.csv",
        weighted_prices=RESULTS + "csvs/weighted_prices.csv",
        market_values=RESULTS + "csvs/market_values.csv",
        price_statistics=RESULTS + "csvs/price_statistics.csv",
        metrics=RESULTS + "csvs/metrics.csv",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/make_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary.py"


rule plot_summary:
    params:
        countries=config_provider("countries"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        emissions_scope=config_provider("energy", "emissions"),
        plotting=config_provider("plotting"),
        foresight=config_provider("foresight"),
        co2_budget=config_provider("co2_budget"),
        sector=config_provider("sector"),
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/costs.csv",
        energy=RESULTS + "csvs/energy.csv",
        balances=RESULTS + "csvs/supply_energy.csv",
        eurostat="data/eurostat/eurostat-energy_balances-april_2023_edition",
        co2="data/bundle-sector/eea/UNFCCC_v23.csv",
    output:
        costs=RESULTS + "graphs/costs.pdf",
        energy=RESULTS + "graphs/energy.pdf",
        balances=RESULTS + "graphs/balances-energy.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/plot_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary.py"


if "sensitivity_analysis" in config["run"]["name"]:
 rule sensitivity_results:
    params:
        countries=config_provider("countries"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        sector_opts=config_provider("scenario", "sector_opts"),
        plotting=config_provider("plotting"),
        scenario=config_provider("scenario"),
        study = config_provider("run", "name"),
        foresight=config_provider("foresight"),
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        balances=RESULTS + "graphs/balances-energy.pdf",
        sepia_config = "SEPIA/SEPIA_config.xlsx",
        template = "SEPIA/Template/pypsa.html",
        logo = "SEPIA/Template/logo.png",         
    output:
        htmlfile=expand(RESULTS + "htmls/{country}_sensitivity_chart.html",study = config["run"]["name"], country=config["countries"]),
    threads: 1
    log:
        RESULTS + "logs/sensitivity_results.log",
    benchmark:
        RESULTS + "benchmarks/sensitivity_results",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/sensitivity_results.py"

planning_horizons = [2020, 2030, 2040, 2050] 
local_countries = config["countries"].copy()
if "EU" not in local_countries:
    local_countries.append("EU")                      
rule prepare_sepia:
    params:
        countries=config_provider("countries"),
        planning_horizons=planning_horizons,
        sector_opts=config_provider("scenario", "sector_opts"),
        emissions_scope=config_provider("energy", "emissions"),
        eurostat_report_year=config_provider("energy", "eurostat_report_year"),
        plotting=config_provider("plotting"),
        scenario=config_provider("scenario"),
        study = config_provider("run", "name"),
        year = config_provider("energy", "energy_totals_year"),
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        costs = "data/costs_2050.csv", 
        summary = RESULTS + "graphs/costs.pdf",
    output:
        excelfile=expand(RESULTS + "sepia/inputs{country}.xlsx", country=local_countries),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/prepare_sepia.log",
    benchmark:
        RESULTS + "benchmarks/prepare_sepia",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/excel_generator.py"
        
rule generate_sepia:
    params:
        countries=config_provider("countries"),
        year = config_provider("energy", "energy_totals_year"),
        study = config_provider("run", "name"),
        planning_horizons=planning_horizons,
        cluster=config_provider("scenario","clusters"),
    input:
        countries = "SEPIA/COUNTRIES.xlsx",
        costs = "data/costs_2050.csv",
        sepia_config = "SEPIA/SEPIA_config.xlsx",
        template = "SEPIA/Template/CLEVER.html",
        #biomass_potentials = expand(resources("biomass_potentials_s{simpl}_{clusters}_{planning_horizons}.csv")),**config["scenario"],
        excelfile=expand(RESULTS + "sepia/inputs{country}.xlsx", country=local_countries),
        
    output:
        excelfile=expand(RESULTS + "htmls/ChartData_{country}.xlsx", country=local_countries),
        htmlfile=expand(RESULTS + "htmls/Results_{country}.html", country=local_countries),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/generate_sepia.log",
    benchmark:
        RESULTS + "benchmarks/generate_sepia",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/SEPIA.py"
        
rule make_country_summary:
    params:
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
        snapshots=config_provider("snapshots"),
        scenario=config_provider("scenario"),
        planning_horizons=planning_horizons,
        country = config_provider("country_summary"),
        study = config_provider("run", "name"),
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        costs=lambda w: (
            ("data/costs_{}.csv".format(config_provider("costs", "year")(w)))
            if config_provider("foresight")(w) == "overnight"
            else (
                "data/costs_{}.csv".format(
                    config_provider("scenario", "planning_horizons", 0)(w)
                )
            )
        ),
        plots=expand(
            RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"]
        ),
        htmlfile=expand(RESULTS + "htmls/Results_{country}.html", country=local_countries),
    output:
        cfs=RESULTS + "country_csvs/cfs.csv",
        costs=RESULTS + "country_csvs/costs.csv",
        capacities=RESULTS + "country_csvs/capacities.csv",
        curtailment=RESULTS + "country_csvs/curtailment.csv",
        energy=RESULTS + "country_csvs/energy.csv",
        supply=RESULTS + "country_csvs/supply.csv",
        supply_energy=RESULTS + "country_csvs/supply_energy.csv",
        prices=RESULTS + "country_csvs/prices.csv",
        price_statistics=RESULTS + "country_csvs/price_statistics.csv",
        market_values=RESULTS + "country_csvs/market_values.csv",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/make_country_summary.log",
    benchmark:
        RESULTS + "benchmarks/make_country_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_country_summary.py"

rule prepare_results:
    params:
        countries=config_provider("countries"),
        planning_horizons=planning_horizons,
        sector_opts=config_provider("scenario", "sector_opts"),
        plotting=config_provider("plotting"),
        scenario=config_provider("scenario"),
        study = config_provider("run", "name"),
        foresight=config_provider("foresight"),
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        market_values=RESULTS + "country_csvs/market_values.csv",
        excelfile=expand(RESULTS + "htmls/ChartData_{country}.xlsx", country=local_countries),
        costs = "data/costs_2050.csv",
        sepia_config = "SEPIA/SEPIA_config.xlsx",
        template = "SEPIA/Template/pypsa.html",
        logo = "SEPIA/Template/logo.png",         
    output:
        htmlfile=expand(RESULTS + "htmls/{country}_combined_chart.html",study = config["run"]["name"], country=local_countries),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/prepare_results.log",
    benchmark:
        RESULTS + "benchmarks/prepare_results",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/Pypsa_results.py"

rule prepare_dispatch_plots:
    params:
        countries=config_provider("countries"),
        planning_horizons=planning_horizons,
        sector_opts=config_provider("scenario", "sector_opts"),
        plotting=config_provider("plotting"),
        scenario=config_provider("scenario"),
        study = config_provider("run", "name"),
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        htmlfile=expand(RESULTS + "htmls/{country}_combined_chart.html",study = config["run"]["name"], country=config["countries"]),      
    output:
        powerfile=expand(RESULTS + "htmls/raw_html/Power_Dispatch-{country}_{planning_horizons}.html", country=config["countries"],planning_horizons=planning_horizons,),
        heatfile=expand(RESULTS + "htmls/raw_html/Heat_Dispatch-{country}_{planning_horizons}.html", country=config["countries"],planning_horizons=planning_horizons,),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/prepare_dispatch_plots.log",
    benchmark:
        RESULTS + "benchmarks/prepare_dispatch_plots",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/Dispatch_plots_weekly.py"
        
                       
STATISTICS_BARPLOTS = [
    "capacity_factor",
    "installed_capacity",
    "optimal_capacity",
    "capital_expenditure",
    "operational_expenditure",
    "curtailment",
    "supply",
    "withdrawal",
    "market_value",
]


rule plot_elec_statistics:
    params:
        plotting=config_provider("plotting"),
        barplots=STATISTICS_BARPLOTS,
    input:
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        **{
            f"{plot}_bar": RESULTS
            + f"figures/statistics_{plot}_bar_elec_s{{simpl}}_{{clusters}}_ec_l{{ll}}_{{opts}}.pdf"
            for plot in STATISTICS_BARPLOTS
        },
        barplots_touch=RESULTS
        + "figures/.statistics_plots_elec_s{simpl}_{clusters}_ec_l{ll}_{opts}",
    script:
        "../scripts/plot_statistics.py"
