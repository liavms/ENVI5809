# ENVI5809: Final Project
Andy Papadopoulos and Liav Meoded Stern

## The impacts of Cyclone Debbie on the Great Barrier Reef
**Main Objective**: the main aim of the analysis is to investigate the spatio-temporal impacts of cyclone Debbie (2017) on oceanographic parameters that are significant for coral health, to better understand the immediate impacts on coral reefs. 
**Parameters that were examined**: Sea Surface Temperature, Chlorophyll-a and water clarity as a Secchi disk depth. 

## Research Questions and Hypothesis 
**Research Questions:**

(1) What is the impact of Cyclone Debbie on water temperature, primary production and turbidity?

(2) What is the extent of this impact? i.e., what is the envelope of these variables during the cyclonic activity, and afterwards?

**Hypothesis:** Cyclone Debbie will cause significant changes in chlorophyll-a concentrations, turbidity and water temperature along its path

## Analysis:
**Data used**: 
eReef biogeochemical dataset in 4km resolution: (https://thredds.ereefs.aims.gov.au/thredds/catalog/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd.html?dataset=GBR4_H2p0_B3p1_Cq3b_Dhnd-daily)

The path of Cyclone Debbie from the Bureau of Meteorology (BOM): (http://www.bom.gov.au/cyclone/history/tracks/beta/)

The planned analysis:

(1) **Visualisation:** generate GIFs in order to visualise the parameters of interest during the cyclone period to examine whether there is a difference in the distribution

(2) **Extracting data:** using KD-Tree to find the biogeochemical data that exist within 50 km radius of the cyclone's positions for three timeframes: during the cyclone, one month before, and one month after. 

(3) **Variables' envelope**: generate graphs of the maximum, minimum and mean values of Sea Surface Temperatures, chlorophyll-a and turbidity (as Secchi). 

(4) **Comparison to non-cyclone conditions:** comparing the variables' envelopes that were found in the previous steps to the monthly averages values of March 2016, March 2017 and March 2018. 
