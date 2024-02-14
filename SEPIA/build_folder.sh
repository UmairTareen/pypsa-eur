#!/bin/bash

mkdir -p ../results/html_folder/other_countries

cp -R ../results/bau/country_csvs ../results/html_folder/csvs_bau
cp -R ../results/ncdr/country_csvs ../results/html_folder/csvs_suff
cp ../results/ncdr/htmls/BE_combined_chart.html ../results/html_folder/BE_combined_chart_suff.html
cp ../results/bau/htmls/BE_combined_chart.html ../results/html_folder/BE_combined_chart_bau.html
cp ../results/ncdr/htmls/Results_BE.html ../results/html_folder/BE_results_suff.html
cp ../results/bau/htmls/Results_BE.html ../results/html_folder/BE_results_bau.html

cp ../results/ncdr/htmls/FR_combined_chart.html ../results/html_folder/other_countries/FR_combined_chart_suff.html
cp ../results/bau/htmls/FR_combined_chart.html ../results/html_folder/other_countries/FR_combined_chart_bau.html
cp ../results/ncdr/htmls/Results_FR.html ../results/html_folder/other_countries/FR_results_suff.html
cp ../results/bau/htmls/Results_FR.html ../results/html_folder/other_countries/FR_results_bau.html

cp ../results/ncdr/htmls/DE_combined_chart.html ../results/html_folder/other_countries/DE_combined_chart_suff.html
cp ../results/bau/htmls/DE_combined_chart.html ../results/html_folder/other_countries/DE_combined_chart_bau.html
cp ../results/ncdr/htmls/Results_DE.html ../results/html_folder/other_countries/DE_results_suff.html
cp ../results/bau/htmls/Results_DE.html ../results/html_folder/other_countries/DE_results_bau.html

cp ../results/ncdr/htmls/GB_combined_chart.html ../results/html_folder/other_countries/GB_combined_chart_suff.html
cp ../results/bau/htmls/GB_combined_chart.html ../results/html_folder/other_countries/GB_combined_chart_bau.html
cp ../results/ncdr/htmls/Results_GB.html ../results/html_folder/other_countries/GB_results_suff.html
cp ../results/bau/htmls/Results_GB.html ../results/html_folder/other_countries/GB_results_bau.html

cp ../results/ncdr/htmls/NL_combined_chart.html ../results/html_folder/other_countries/NL_combined_chart_suff.html
cp ../results/bau/htmls/NL_combined_chart.html ../results/html_folder/other_countries/NL_combined_chart_bau.html
cp ../results/ncdr/htmls/Results_NL.html ../results/html_folder/other_countries/NL_results_suff.html
cp ../results/bau/htmls/Results_NL.html ../results/html_folder/other_countries/NL_results_bau.html

cp ../results/ncdr/htmls/Results_EU.html ../results/html_folder/other_countries/EU_results_suff.html
cp ../results/bau/htmls/Results_EU.html ../results/html_folder/other_countries/EU_results_bau.html

cp ../results/scenario_results/BE_combined_chart.html ../results/html_folder/BE_combined_scenario_chart.html

cp ../results/scenario_results/DE_combined_chart.html ../results/html_folder/other_countries/DE_combined_scenario_chart.html
cp ../results/scenario_results/NL_combined_chart.html ../results/html_folder/other_countries/NL_combined_scenario_chart.html
cp ../results/scenario_results/FR_combined_chart.html ../results/html_folder/other_countries/FR_combined_scenario_chart.html
cp ../results/scenario_results/GB_combined_chart.html ../results/html_folder/other_countries/GB_combined_scenario_chart.html








