# PyPSA-Eur: Use of pypsa-eur for sufficiency scenario studies
This repository contains modified scripts to use PyPSA-Eur for sufficiency scenario studies. The demand data used in the scenarios are based on the CLEVER scenario (https://clever-energy-scenario.eu/). The data folder in the repository contains CSV files considered for demands in the scenarios, which can be freely used and utilized for reproduction purposes or further improvement of sufficiency scenarios. The scripts folder also includes a script to convert CLEVER sufficiency data for 28 countries into csv files which can also be freely utilized. The Config folder also contains config files used for scenarios in the current study. For further information, please feel free to contact Sylvain Quoilin (squoilin@uliege.be) and Muhammad Umair Tareen (muhammadumair.tareen@uliege.be).

**Quick Usage**:
- Download repository
- Install the pypsa environment and activate pypsa-eur:
> conda env create -f envs/environment.yaml

> conda activate pypsa-eur
   
- A solver is also needed for optimisation, Gurobi is recommended but Cplex can also be used:

> conda install -c gurobi gurobi

**First run of the model**:
During the first run, all data bundles must be downloaded. In the config file, set the retrieve options as such:
- retrieve: auto
- prepare_links_p_nom: true
- retrieve_databundle: true
- retrieve_sector_databundle: true
- retrieve_cost_data: true
- build_cutout: false
- retrieve_cutout: true
- build_natura_raster: false
- retrieve_natura_raster: true
- custom_busmap: false

Be aware that, depending on your connection speed, download time may be several hours!

You can then run:
> snakemake -s {Snakefile} -call all

After running the whole snakemake, the options can be set back to:
- retrieve: auto
- prepare_links_p_nom: false
- retrieve_databundle: false
- retrieve_sector_databundle: false
- retrieve_cost_data: false
- build_cutout: false
- retrieve_cutout: false
- build_natura_raster: false
- retrieve_natura_raster: true
- custom_busmap: false

**Package to be added to the environment**
> conda install plotly

> pip install -U kaleido

**Selection of the scenario**:
- To run the default workflow, activate pypsa-eur and run, this will run all scenarios in a sequence:
> snakemake -s Snakefile_master -call run_all_scenarios

- Tu run a different scenario/workflow, use the dedicated Snakefile, e.g:
> snakemake -s Snakefile_suff -call all

- Currently there are 2 sensitivity analysis one on nuclear reactors price and one on assuming additional offshore capacity
  excess only for Belgium in the Northsea. The config files of sensitivity analysis canbe found in config folder.
  To run the sensitivity scenarios, use the dedicated Snakefile, e.g: 
> snakemake -s Snakefile_sensitivity_nuclear -call run_all_sensitivity_scenarios 

> snakemake -s Snakefile_sensitivity_offshore -call run_all_sensitivity_scenarios 

- The Sankey codes and all post processing results are automatically generated for all scenarios are included in the repository.

**myopic scenarios**:
- The myopic scenarios perform the optimization for successive years defined in the config file. They can be run using the dedicated Snakefile.

<!--
SPDX-FileCopyrightText: 2017-2023 The PyPSA-Eur Authors
SPDX-License-Identifier: CC-BY-4.0
-->

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pypsa/pypsa-eur?include_prereleases)
[![Build Status](https://github.com/pypsa/pypsa-eur/actions/workflows/ci.yaml/badge.svg)](https://github.com/PyPSA/pypsa-eur/actions)
[![Documentation](https://readthedocs.org/projects/pypsa-eur/badge/?version=latest)](https://pypsa-eur.readthedocs.io/en/latest/?badge=latest)
![Size](https://img.shields.io/github/repo-size/pypsa/pypsa-eur)
[![Zenodo PyPSA-Eur](https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg)](https://doi.org/10.5281/zenodo.3520874)
[![Zenodo PyPSA-Eur-Sec](https://zenodo.org/badge/DOI/10.5281/zenodo.3938042.svg)](https://doi.org/10.5281/zenodo.3938042)
[![Snakemake](https://img.shields.io/badge/snakemake-≥7.7.0-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/pypsa-eur)](https://api.reuse.software/info/github.com/pypsa/pypsa-eur)
[![Stack Exchange questions](https://img.shields.io/stackexchange/stackoverflow/t/pypsa)](https://stackoverflow.com/questions/tagged/pypsa)

# PyPSA-Eur: A Sector-Coupled Open Optimisation Model of the European Energy System

PyPSA-Eur is an open model dataset of the European energy system at the
transmission network level that covers the full ENTSO-E area. The model is suitable both for operational studies and generation and transmission expansion planning studies.
The continental scope and highly resolved spatial scale enables a proper description of the long-range
smoothing effects for renewable power generation and their varying resource availability.




The model is described in the [documentation](https://pypsa-eur.readthedocs.io)
and in the paper
[PyPSA-Eur: An Open Optimisation Model of the European Transmission
System](https://arxiv.org/abs/1806.01613), 2018,
[arXiv:1806.01613](https://arxiv.org/abs/1806.01613).
The model building routines are defined through a snakemake workflow.
Please see the [documentation](https://pypsa-eur.readthedocs.io/)
for installation instructions and other useful information about the snakemake workflow.
The model is designed to be imported into the open toolbox
[PyPSA](https://github.com/PyPSA/PyPSA).

**WARNING**: PyPSA-Eur is under active development and has several
[limitations](https://pypsa-eur.readthedocs.io/en/latest/limitations.html) which
you should understand before using the model. The github repository
[issues](https://github.com/PyPSA/pypsa-eur/issues) collect known topics we are
working on (please feel free to help or make suggestions). The
[documentation](https://pypsa-eur.readthedocs.io/) remains somewhat patchy. You
can find showcases of the model's capabilities in the Joule paper [The potential
role of a hydrogen network in
Europe](https://doi.org/10.1016/j.joule.2023.06.016), another [paper in Joule
with a description of the industry
sector](https://doi.org/10.1016/j.joule.2022.04.016), or in [a 2021 presentation
at EMP-E](https://nworbmot.org/energy/brown-empe.pdf). We do not recommend to
use the full resolution network model for simulations. At high granularity the
assignment of loads and generators to the nearest network node may not be a
correct assumption, depending on the topology of the underlying distribution
grid, and local grid bottlenecks may cause unrealistic load-shedding or
generator curtailment. We recommend to cluster the network to a couple of
hundred nodes to remove these local inconsistencies. See the discussion in
Section 3.4 "Model validation" of the paper.


![PyPSA-Eur Grid Model](doc/img/elec.png)

The dataset consists of:

- A grid model based on a modified [GridKit](https://github.com/bdw/GridKit)
  extraction of the [ENTSO-E Transmission System
  Map](https://www.entsoe.eu/data/map/). The grid model contains 6763 lines
  (alternating current lines at and above 220kV voltage level and all high
  voltage direct current lines) and 3642 substations.
- The open power plant database
  [powerplantmatching](https://github.com/FRESNA/powerplantmatching).
- Electrical demand time series from the
  [OPSD project](https://open-power-system-data.org/).
- Renewable time series based on ERA5 and SARAH, assembled using the [atlite tool](https://github.com/FRESNA/atlite).
- Geographical potentials for wind and solar generators based on land use (CORINE) and excluding nature reserves (Natura2000) are computed with the [atlite library](https://github.com/PyPSA/atlite).

A sector-coupled extension adds demand
and supply for the following sectors: transport, space and water
heating, biomass, industry and industrial feedstocks, agriculture,
forestry and fishing. This completes the energy system and includes
all greenhouse gas emitters except waste management and land use.

This diagram gives an overview of the sectors and the links between
them:

![sector diagram](graphics/multisector_figure.png)

Each of these sectors is built up on the transmission network nodes
from [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur):

![network diagram](https://github.com/PyPSA/pypsa-eur/blob/master/doc/img/base.png?raw=true)

For computational reasons the model is usually clustered down
to 50-200 nodes.

Already-built versions of the model can be found in the accompanying [Zenodo
repository](https://doi.org/10.5281/zenodo.3601881).

# Contributing and Support
We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.
-   In case of code-related **questions**, please post on [stack overflow](https://stackoverflow.com/questions/tagged/pypsa).
-   For non-programming related and more general questions please refer to the [mailing list](https://groups.google.com/group/pypsa).
-   To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [discord server](https://discord.com/invite/AnuJBk23FU).
-   For **bugs and feature requests**, please use the [PyPSA-Eur Github Issues page](https://github.com/PyPSA/pypsa-eur/issues).

# Licence

The code in PyPSA-Eur is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see `LICENSE.txt`.
However, different licenses and terms of use may apply to the various
input data.
