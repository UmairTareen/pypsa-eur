# SEPIA
Simplified Energy Prospective and Interterritorial Analysis tool v1.8
***
This tool is used as part of the CLEVER project, led by négaWatt Association - www.negawatt.org


## Description of outputs
Running the "SEPIA.py" script will create several output files in the "Results" folder :
- Static html files, one per country (Results_X.html). Additionnal aggregated results are also produced : "EU" corresponds to the official EU27 perimeter, and includes comparison with official objectives (and ad  hoc calculations), "ALL" corresponds to the total EU30 perimet.
- Excel files with data shown in the graphs of the html files (ChartData_Y.xlsx)
- A CSV debug file (Debug.csv)
- A raw CSV output file (Results_bigtable.csv)

This last file is organised as follows:
- One line represents values for a given year and a given country (or EU / ALL perimeter)
- One column represents an indicator

These indicators are named as "MainIndicator.SubIndicator(.SubSubIndicator)". List of "MainIndicators":
- **fec**: final energy consumption, split by energy OR final demand sector. Energies and demand sectors are described in SEPIA_config.xlsx ("NODES" sheet). This indicator is even subdivided by energy x demand sector, eg. "fec.agr.elc_fe" is the final consumption of electricity for the agriculture sector.
- **fec_EU**: final energy consumption as per EU objective definitions (without ambient heat, non-energy consumption and the energy sector - except blast furnaces).
- **fec_bkdn**: breakdown of final energy consumption, by primary energy source.
- **gfec_bkdn**: gross final energy consumption, same as final energy consumption, with network losses (electricity, gas, district heating). Breakdown by type of primary energy source (renewable, fossil, nuclear).
- **ren_cov_ratio**: coverage of final energy consumption by renewable energy sources, by type of final energy (total is the weighted average).
- **ren_cov_ratios_EU**: renewable energy share in gross final energy consumption, as per EU definitions for electricity, transportation, heating & cooling.
- **cov_ratio**: local coverage ratios (local production / total uses) for different energies. These energies are listed as sub indicator. Example : "cov_ratio.cms_pe" is the local coverage ratio of primary coal.
- **pec**: primary energy consumption, by energy source.
- **pec_uses**: final and internal uses of fossil primary energies.
- **pec_EU**: primary energy consumption as per EU objective definitions (without ambient heat and non-energy consumption / feedstock).
- **sec_mix**: production mix of secondary energies (electricity, gas, heat...).
- **sec_uses**: final and internal uses of secondary energies (+ solid biomass).
- **ghg_sector**: all GHG emissions (all types of GHG) by sector. The CRF nomenclature of these sectors is given in SEPIA_config.xlsx ("NODES" sheet, "GHG_Code" column).
- **ghg_sector_2**: same as above, but with power & heat sector emissions allocated to other sectors (according to their energy consumption). Caution: sector emissions are then no longer consistent with the CRF nomenclature.
- **ghg_sector_EU**: same as ghg_sector, but on the EU perimeter (international aviation included, no international maritime).
- **ghg_en**: energy-related GHG emissions (all types of GHG). This indicator is subdivided by primary energy source x demand sector, eg. "ghg_en.cms_pe.agr" is the GHG emissions associated with coal combustion of the agriculture sector.
- **ghg_source**: all GHG emissions (all types of GHG) by primary energy source.
- **ghg_CO2**: same but for CO2 only GHG emissions.
- **ghg_CH4**: same but for CO2 only GHG emissions.
- **ghg_nes**: non energy-related GHG emissions (all types of GHG).
- **ghgco2_nes**: non energy-related GHG emissions (CO2 only).
- **ghgch4_nes**: non energy-related GHG emissions (CH4 only).
- **pop**: population.