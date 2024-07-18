# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pandas as pd
from sys import argv
import numpy as np
import time
from calculate_emissions_functions import *
try:
    from memory_profiler import profile
except ImportError:
    # use dummy profile
    from profile import *
    pass



#-------------------------------------------------------------------------------------
# This is where the code starts
# This should be run as python calculate_emission_gloria <config> <start_year> <end_year> [-n -Le -v -t]
# where config is a config file defining pathnames of files to read and output directory
#       start_year and end_year define the years to read
#       -n means run using minimum data (otherwise do it the old way with big matrices)
#       -Le means use L.e in the old code rather than e.L
#       -v = verbose
#       -t = do timing
#       -m = memory profiling
#-------------------------------------------------------------------------------------
def main():

    new=False # run in the old way
    fextra='_old' # used for me to output files for testing
    fout_extra='' # used for the footprint file to indicate we used the new/old way to do this
    verbose=False
    do_timing=False
    # in the old code it calculated e.L but should do L.e use this flag to define whether we should make the correction
    # we will always use L.e in the new code
    use_Le = False
    if len(argv)<4:
        print('Useage: python', argv[0], '<config> <start_year> <end_year> [-n -Le -t -v]')
        print('where -n means use new way to process data,\n-Le means use L.e in old way,\n-t means do timing,\n-v means verbose')
        exit()

    config_file=argv[1]
    start_year=int(argv[2])
    end_year=int(argv[3])
    for i in range(4,len(argv)):
        if argv[i]=='-n':
            new=True
            print('NEW')
            fextra='_new'
            fout_extra='_New'
        elif argv[i]=='-Le':
            use_Le=True
            fout_extra='_Le'
        elif argv[i]=='-v':
            verbose=True
        elif argv[i]=='-t':
            do_timing=True


    # read config file to get filenames
    mrio_filepath, outdir, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname = read_config(config_file)

    if do_timing:
        time00=time.time()
    z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows=get_metadata_indices(mrio_filepath,labels_fname, lookup_fname)
    if do_timing:
        time1=time.time()
        print('TIME: get_metadata_indices', time1-time00)

    stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'" # use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis

    if new:
        # JAC work out which row stressor_cat is on
        stressor_row = pd.Index(sat_rows).get_loc(stressor_cat)

    # define sample year, normally this is: range(2010, 2019)
    # here years is now determined from inputs,
    # it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
    # years = [2016]
    for year in range(start_year,end_year+1):

        # set up filepaths
        # file name changes from 2017, so define this here
        if year < 2017:
            date_var = '20230314'
        else:
            date_var = '20230315'

        split=Z_fname.split('%')
        if len(split)>1:
            z_filepath=mrio_filepath+date_var+split[1]+str(year)+split[2]
        else:
            z_filepath=mrio_filepath+Z_fname

        split=Y_fname.split('%')
        if len(split)>1:
            y_filepath=mrio_filepath+date_var+split[1]+str(year)+split[2]
        else:
            y_filepath=mrio_filepath+Y_fname

        split=co2_fname.split('%')
        if len(split)>1:
            co2_filepath=mrio_filepath+split[0]+str(year)+split[1]
        else:
            co2_filepath=mrio_filepath+co2_fname

        outfile=outdir+'Gloria_CO2_' + str(year) + fout_extra+'.csv'

        if do_timing:
            time0=time.time()    
        if new:
            S, U, Y, stressor=read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)


        else:    
            S, U, Y, stressor=read_data_old(z_filepath, y_filepath, co2_filepath, z_idx,industry_idx, product_idx, y_cols, sat_rows, stressor_cat)

        if do_timing:
            time1=time.time()
            print('TIME: read_data', time1-time0)

        if verbose:
            print('DBG: size S, U', S.shape, U.shape)
            print('DBG: size Y', Y.shape, Y.to_numpy().shape)
            print('DBG: size stressor', stressor.shape)
            #np.save('Y_'+fextra+'.npy', Y.to_numpy())

            print('Data loaded for ' + str(year))

        if do_timing:
            time0=time.time()    
        if new:    
            footprint=indirect_footprint_SUT_new(S, U, Y, stressor,verbose, do_timing)    
        else:
            footprint=indirect_footprint_SUT(S, U, Y, stressor, use_Le, verbose, do_timing)    

        if do_timing:
            time1=time.time()
            print('TIME: indirect_footprint_SUT', time1-time0)

        if verbose:
            print('Footprint calculated for ' + str(year))
    
        footprint.to_csv(outfile)
        print('Footprint saved for ' + str(year))

    print('Gloria Done')
    if do_timing:
        time_end=time.time()
        print('TIME: whole time', time_end-time00)

    #my_profiler.disable()
    # MRI write information to file in specified order
    #stats = pstats.Stats( my_profiler ).sort_stats('tottime')
    #stats.dump_stats('Profiling/gloria_smaller_ref.dat')
    #stats.print_stats()

if __name__ == '__main__':
    main()
