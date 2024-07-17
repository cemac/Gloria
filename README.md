This is the code to calcluate the CO2 footprint of industries using the Leontief matrix method, (I-A).X=e where we are solving for X, i.e. X=L.e where L is inv(I-A)
This has been adapted from code given to us by SRI and was using very large matrices which contained many 0s.
This code has been modified to create new versions of functions to just use the required parts of the large matrix in an attempt to reduce both time and memory use.
The old code used e.L in determining X but it should have used L.e.

The code should be run as:

python calculate_emission_gloria start_year end_year [-n -Le -S -v -t]

where -n means run in the new way of processing, -Le means calculate the dot product as L.e not e.L if using the old code, -S means use small data rather than large, -v means verbose and -t means do timing

The gloria environment needs openpyxl, pandas, matplotlib, numpy and scipy. If memory profiling is required also install memory_profiler
If memory_profiler is not available profile.py takes its place and does nothing
