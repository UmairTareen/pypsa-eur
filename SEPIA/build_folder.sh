#!/bin/bash

mkdir -p ../results/html_folder/other_countries
mkdir -p ../results/html_folder/other_countries/dispatch_plots/

cp -R ../results/ref/country_csvs ../results/html_folder/csvs_ref
cp -R ../results/suff/country_csvs ../results/html_folder/csvs_suff
cp ../results/suff/htmls/BE_combined_chart.html ../results/html_folder/BE_combined_chart_suff.html
cp ../results/ref/htmls/BE_combined_chart.html ../results/html_folder/BE_combined_chart_ref.html
cp ../results/suff/htmls/Results_BE.html ../results/html_folder/BE_results_suff.html
cp ../results/ref/htmls/Results_BE.html ../results/html_folder/BE_results_ref.html

cp ../results/suff/htmls/EU_combined_chart.html ../results/html_folder/EU_combined_chart_suff.html
cp ../results/ref/htmls/EU_combined_chart.html ../results/html_folder/EU_combined_chart_ref.html
cp ../results/suff/htmls/Results_EU.html ../results/html_folder/EU_results_suff.html
cp ../results/ref/htmls/Results_EU.html ../results/html_folder/EU_results_ref.html

cp ../results/suff/htmls/FR_combined_chart.html ../results/html_folder/other_countries/FR_combined_chart_suff.html
cp ../results/ref/htmls/FR_combined_chart.html ../results/html_folder/other_countries/FR_combined_chart_ref.html
cp ../results/suff/htmls/Results_FR.html ../results/html_folder/other_countries/FR_results_suff.html
cp ../results/ref/htmls/Results_FR.html ../results/html_folder/other_countries/FR_results_ref.html

cp ../results/suff/htmls/DE_combined_chart.html ../results/html_folder/other_countries/DE_combined_chart_suff.html
cp ../results/ref/htmls/DE_combined_chart.html ../results/html_folder/other_countries/DE_combined_chart_ref.html
cp ../results/suff/htmls/Results_DE.html ../results/html_folder/other_countries/DE_results_suff.html
cp ../results/ref/htmls/Results_DE.html ../results/html_folder/other_countries/DE_results_ref.html

cp ../results/suff/htmls/GB_combined_chart.html ../results/html_folder/other_countries/GB_combined_chart_suff.html
cp ../results/ref/htmls/GB_combined_chart.html ../results/html_folder/other_countries/GB_combined_chart_ref.html
cp ../results/suff/htmls/Results_GB.html ../results/html_folder/other_countries/GB_results_suff.html
cp ../results/ref/htmls/Results_GB.html ../results/html_folder/other_countries/GB_results_ref.html

cp ../results/suff/htmls/NL_combined_chart.html ../results/html_folder/other_countries/NL_combined_chart_suff.html
cp ../results/ref/htmls/NL_combined_chart.html ../results/html_folder/other_countries/NL_combined_chart_ref.html
cp ../results/suff/htmls/Results_NL.html ../results/html_folder/other_countries/NL_results_suff.html
cp ../results/ref/htmls/Results_NL.html ../results/html_folder/other_countries/NL_results_ref.html

cp ../results/suff/htmls/Results_EU.html ../results/html_folder/other_countries/EU_results_suff.html
cp ../results/ref/htmls/Results_EU.html ../results/html_folder/other_countries/EU_results_ref.html

cp ../results/scenario_results/BE_combined_scenario_chart.html ../results/html_folder/BE_combined_scenario_chart.html
cp ../results/scenario_results/EU_combined_scenario_chart.html ../results/html_folder/EU_combined_scenario_chart.html

cp ../results/scenario_results/DE_combined_scenario_chart.html ../results/html_folder/other_countries/DE_combined_scenario_chart.html
cp ../results/scenario_results/NL_combined_scenario_chart.html ../results/html_folder/other_countries/NL_combined_scenario_chart.html
cp ../results/scenario_results/FR_combined_scenario_chart.html ../results/html_folder/other_countries/FR_combined_scenario_chart.html
cp ../results/scenario_results/GB_combined_scenario_chart.html ../results/html_folder/other_countries/GB_combined_scenario_chart.html

cp ../results/suff/htmls/raw_html/Power_Dispatch-BE_2050.html ../results/html_folder/other_countries/dispatch_plots/Power_Dispatch-BE_weekly_suff.html
cp ../results/suff/htmls/raw_html/Heat_Dispatch-BE_2050.html ../results/html_folder/other_countries/dispatch_plots/Heat_Dispatch-BE_weekly_suff.html

cp ../results/ref/htmls/raw_html/Power_Dispatch-BE_2050.html ../results/html_folder/other_countries/dispatch_plots/Power_Dispatch-BE_weekly_ref.html
cp ../results/ref/htmls/raw_html/Heat_Dispatch-BE_2050.html ../results/html_folder/other_countries/dispatch_plots/Heat_Dispatch-BE_weekly_ref.html



