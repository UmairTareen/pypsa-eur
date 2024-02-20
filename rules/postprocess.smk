# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


localrules:
    copy_config,
    copy_conda_env,


rule plot_network:
    params:
        foresight=config["foresight"],
        plotting=config["plotting"],
    input:
        network=RESULTS
        + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        regions=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
    output:
        map=RESULTS
        + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
        today=RESULTS
        + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}-today.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        (
            BENCHMARKS
            + "plot_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_network.py"


rule copy_config:
    params:
        RDIR=RDIR,
    output:
        RESULTS + "config.yaml",
    threads: 1
    resources:
        mem_mb=1000,
    benchmark:
        BENCHMARKS + "copy_config"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/copy_config.py"


rule make_summary:
    params:
        foresight=config["foresight"],
        costs=config["costs"],
        snapshots=config["snapshots"],
        scenario=config["scenario"],
        RDIR=RDIR,
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        costs="data/costs_{}.csv".format(config["costs"]["year"])
        if config["foresight"] == "overnight"
        else "data/costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
        plots=expand(
            RESULTS
            + "maps/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"]
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
        LOGS + "make_summary.log",
    benchmark:
        BENCHMARKS + "make_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary.py"
        
rule plot_summary:
    params:
        countries=config["countries"],
        planning_horizons=config["scenario"]["planning_horizons"],
        sector_opts=config["scenario"]["sector_opts"],
        emissions_scope=config["energy"]["emissions"],
        eurostat_report_year=config["energy"]["eurostat_report_year"],
        plotting=config["plotting"],
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/costs.csv",
        energy=RESULTS + "csvs/energy.csv",
        balances=RESULTS + "csvs/supply_energy.csv",
        eurostat=input_eurostat,
        co2="data/bundle-sector/eea/UNFCCC_v23.csv",
    output:
        costs=RESULTS + "graphs/costs.pdf",
        energy=RESULTS + "graphs/energy.pdf",
        balances=RESULTS + "graphs/balances-energy.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        LOGS + "plot_summary.log",
    benchmark:
        BENCHMARKS + "plot_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary.py"
        

                   
planning_horizons = [2020, 2030, 2040, 2050] 
local_countries = config["countries"].copy()
if "EU" not in local_countries:
    local_countries.append("EU")                      
rule prepare_sepia:
    params:
        countries=config["countries"],
        planning_horizons=planning_horizons,
        sector_opts=config["scenario"]["sector_opts"],
        emissions_scope=config["energy"]["emissions"],
        eurostat_report_year=config["energy"]["eurostat_report_year"],
        plotting=config["plotting"],
        scenario=config["scenario"],
        study = config["run"]["name"],
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
        LOGS + "prepare_sepia.log",
    benchmark:
        BENCHMARKS + "prepare_sepia",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/excel_generator.py"


rule generate_sepia:
    params:
        countries=config["countries"],
    input:
        countries = "SEPIA/COUNTRIES.xlsx",
        costs = "data/costs_2050.csv",
        sepia_config = "SEPIA/SEPIA_config.xlsx",
        template = "SEPIA/Template/CLEVER.html",
        excelfile=expand(RESULTS + "sepia/inputs{country}.xlsx", country=local_countries),
        
    output:
        excelfile=expand(RESULTS + "htmls/ChartData_{country}.xlsx", country=local_countries),
        htmlfile=expand(RESULTS + "htmls/Results_{country}.html", country=local_countries),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        LOGS + "generate_sepia.log",
    benchmark:
        BENCHMARKS + "generate_sepia",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/SEPIA.py"
        
rule make_country_summary:
    params:
        foresight=config["foresight"],
        costs=config["costs"],
        snapshots=config["snapshots"],
        scenario=config["scenario"],
        planning_horizons=planning_horizons,
        country = config["country_summary"],
        study = config["run"]["name"],
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        costs="data/costs_{}.csv".format(config["costs"]["year"])
        if config["foresight"] == "overnight"
        else "data/costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
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
        LOGS + "make_country_summary.log",
    benchmark:
        BENCHMARKS + "make_country_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_country_summary.py"

rule plot_country_summary:
    params:
        planning_horizons=config["scenario"]["planning_horizons"],
        sector_opts=config["scenario"]["sector_opts"],
        emissions_scope=config["energy"]["emissions"],
        eurostat_report_year=config["energy"]["eurostat_report_year"],
        plotting=config["plotting"],
        RDIR=RDIR,
    input:
        balances=RESULTS + "country_csvs/supply_energy.csv",
    output:
        balances=RESULTS + "country_graphs/balances-oil.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        LOGS + "plot_country_summary.log",
    benchmark:
        BENCHMARKS + "plot_country_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_country_summary.py"


rule prepare_results:
    params:
        countries=config["countries"],
        planning_horizons=planning_horizons,
        sector_opts=config["scenario"]["sector_opts"],
        plotting=config["plotting"],
        scenario=config["scenario"],
        study = config["run"]["name"],
    input:
        networks=expand(
            RESULTS
            + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"]
        ),
        balances=RESULTS + "country_graphs/balances-oil.pdf",
        excelfile=expand(RESULTS + "htmls/ChartData_{country}.xlsx", country=local_countries),
        costs = "data/costs_2050.csv",
        sepia_config = "SEPIA/SEPIA_config.xlsx",
        template = "SEPIA/Template/pypsa.html",
        logo = "SEPIA/Template/logo.png",         
    output:
        htmlfile=expand(RESULTS + "htmls/{country}_combined_chart.html",study = config["run"]["name"], country=config["countries"]),
    threads: 1
    resources:
        mem_mb=10000,
    log:
        LOGS + "prepare_results.log",
    benchmark:
        BENCHMARKS + "prepare_results",
    conda:
        "../envs/environment.yaml"
    script:
        "../SEPIA/Pypsa_results.py"
        
rule prepare_dispatch_plots:
    params:
        countries=config["countries"],
        planning_horizons=planning_horizons,
        sector_opts=config["scenario"]["sector_opts"],
        plotting=config["plotting"],
        scenario=config["scenario"],
        study = config["run"]["name"],
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
        LOGS + "prepare_dispatch_plots.log",
    benchmark:
        BENCHMARKS + "prepare_dispatch_plots",
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
        plotting=config["plotting"],
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
