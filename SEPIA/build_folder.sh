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


# Now synchronize with the FTP server.

# Configuration
FTP_HOST="boucan.domainepublic.net"
FTP_USER="negawattbe_syl"
FTP_PORT=21
LOCAL_FOLDER="../results/html_folder"
REMOTE_BASE="/www/prod.negawatt.be/scenario"

# Get current date in YYMMDD format
DATE_FOLDER=$(date '+%y%m%d')
REMOTE_FOLDER="$REMOTE_BASE/$DATE_FOLDER"

# Check if local folder exists
if [ ! -d "$LOCAL_FOLDER" ]; then
    echo "Error: Local folder '$LOCAL_FOLDER' not found"
    exit 1
fi

# Prompt for password
read -s -p "Enter FTP password: " FTP_PASS
echo

# Execute lftp with commands via heredoc
lftp -p $FTP_PORT -u "$FTP_USER,$FTP_PASS" "ftp://$FTP_HOST" << EOF
# Debug and connection settings
debug 3
set ssl:verify-certificate no
set ftp:ssl-force true
set ftp:ssl-protect-data true
set ssl:priority "NORMAL:+VERS-TLS1.2:+VERS-TLS1.1:+VERS-TLS1.0:-VERS-SSL3.0"
set ftp:ssl-allow true
set ftp:ssl-protect-list yes
set ftp:ssl-force-auth TLS
set net:timeout 15
set net:max-retries 2
set net:reconnect-interval-base 5

# Create remote directory (ignore error if exists)
mkdir -p "$REMOTE_FOLDER"

# Navigate to remote directory
cd "$REMOTE_FOLDER"

# Upload local folder contents recursively
mirror -R "$LOCAL_FOLDER" .

# Verify upload
echo "Verifying upload..."
ls -R

quit
EOF

# Check if lftp command was successful
if [ $? -eq 0 ]; then
    echo "Upload completed successfully to $REMOTE_FOLDER"
else
    echo "Error: Upload failed"
    exit 1
fi





