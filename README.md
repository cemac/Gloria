This is the code to calcluate the CO2 footprint of industries by extending the Leontief matrix method, (I-A).X=Y using coefficients to turn the monetary value into a footprint. The leontief equation can be expressed as X=L.Y where Y is the inv(I-A). To turn this into a carbon footprint we need to do e.L*y for y being a column in Y and e being the co2 stressor coefficient.
This has been adapted from code given to us by SRI and was using very large matrices which contained many 0s.
This code has been modified to create new versions of functions to just use the required parts of the large matrix in an attempt to reduce both time and memory use.

The code should be run as:

python calculate_emission_gloria <config_file> <start_year> <end_year> [-n -v]

where -n means run in the new way of processing, -v means verbose

The gloria environment needs openpyxl, pandas, matplotlib, numpy and scipy. If memory profiling is required also install memory_profiler

If memory_profiler is not available profile.py takes its place and does nothing
To do line by line memory profiling the MPROFILE_Lbl environment variable must be set to 1, otherwise set toit to 0. In linux use export MPROFILE_Lbl=1, in Anaconda powershell use conda env config vars set MPROFILE_Lbl=1 and reactivate the conda environment.

To do cProfile timing the @profile decoration in calculate_emissions_functions.py needs to be removed as does the memory profiling import statements.
