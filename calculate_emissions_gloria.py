# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pandas as pd
from sys import platform, argv
import calculate_emissions_functions as cef
import numpy as np
import time

try:
    from memory_profiler import profile
except ImportError:
    # use dummy profile
    from profile import *
    pass

#####################
## Gloria Metadata ##
#####################
#----------------------------------------------------------------
# this function will split the strings into a list of <country>+ (<code>), a list of <code> and a list of <some_str>,
# where <some_str> is the remainder of the string
# inputs:
#   labels is a list of strings containing <country> (<code>) <some_str>
#   valid_country_codes is a list of valid country codes
# returns:
#   country_full - list of <country>+(<code>)
#   country_code - list of <code>
#   remainder - list of <some_str> 
#------------------------------------------------------
def split_country_and_code(labels, valid_country_codes):

    nl=len(labels)
    country_full = ['NA']*nl
    country_code=['']*nl
    remainder=['']*nl
    n=0
    for cs in labels:
        items=cs.split('(')
        # items[0] is the country, items[1] should be <code>) <some_str> but if there are () in some_str this will also have been split
        items2=items[1].split(')')
        # items2[0] is <code>, items2[1] is remainder of the string
        if items2[0] in valid_country_codes:
            country_full[n]=items[0]+'('+items2[0]+')'
            country_code[n]=items2[0]
            remainder[n]=cs.replace(country_full[n], '')
        n+=1


    # end code if there is a mismatch in labels, this is done to test that the code does what it should
    if 'NA' in country_full:
        print('Error: Missing country labels')
        raise SystemExit 
        
    return country_full, country_code, remainder


#----------------------------------------------------------------
# This function reads the excel files containing the indices in Z
# mrio_filepath is the directory where they exist
#----------------------------------------------------------------
def get_metadata_indices(mrio_filepath):
    # read metadata which is used for column and row labels later on
    readme = mrio_filepath + 'GLORIA_ReadMe_057_small.xlsx' # MRI mod for "small dataset"
    labels = pd.read_excel(readme, sheet_name=None)

    # get lookup to fix labels
    # MRI modify for "small dataset"
    lookup = pd.read_excel(mrio_filepath + 'mrio_lookup_sectors_countries_finaldemand_small.xlsx', sheet_name=None)
    # get list of countries in dataset
    lookup['countries'] = lookup['countries'][['gloria', 'gloria_code']].drop_duplicates().dropna()
    lookup['countries']['gloria_combo'] = lookup['countries']['gloria'] + ' (' + lookup['countries']['gloria_code'] + ') '
    # get list of sectors in dataset
    lookup['sectors'] = lookup['sectors'][['gloria']].drop_duplicates().dropna()

    # fix Z labels
    # This helps beng able to split the 'country' from the 'sector' component in the label later on
    # Essentially this removes some special characters from the country names, making it easier to split the country from the secro by a specific charater later on
    t_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_regionSector_labels']).drop_duplicates(); t_cats.columns = ['label']
    # remove special characters frm country names -JAC this seems to be only removing sector
    # JAC replace loop with call to function
    valid_country_codes=lookup['countries']['gloria_code'].tolist()
    country_full, country_code, remainder=split_country_and_code(t_cats['label'], valid_country_codes)        
    t_cats['country_full'] = country_full
    t_cats['country'] = country_code
    t_cats['sector'] = remainder

    # fix final demand labels (this is in the Y dataframe later)
    # it follows the same logic as the Z label fix above
    fd_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0)); fd_cats.columns = ['label']
    # remove special characters frm country names
    # JAC replace loop below with call to function
    country_full, country_code, remainder=split_country_and_code(fd_cats['label'], valid_country_codes)        
    fd_cats['country_full'] = country_full
    fd_cats['country'] = country_code
    fd_cats['fd'] = remainder

    # split lables by ndustries vs products (these later get split into the S and U dataframes)
    t_cats['ind'] = t_cats['label'].str[-8:]
    industries = t_cats.loc[t_cats['ind'] == 'industry']
    products = t_cats.loc[t_cats['ind'] == ' product']
    sector_type=np.asarray(t_cats['ind'])
    iix=np.where(sector_type=='industry')[0]
    pix=np.where(sector_type==' product')[0]

    # make index labels
    z_idx = pd.MultiIndex.from_arrays([t_cats['country'], t_cats['sector']]) # labels for Z dataframe
    industry_idx = pd.MultiIndex.from_arrays([industries['country'], industries['sector']]) # labels used to split Z into S and U
    product_idx = pd.MultiIndex.from_arrays([products['country'], products['sector']]) # labels used to split Z into S and U
    y_cols = pd.MultiIndex.from_arrays([fd_cats['country'], fd_cats['fd']]) # labels for Y dataframe

    sat_rows = labels['Satellites']['Sat_indicator'] # labels for CO2 dataframe
    # clear space in variable explorer to free up memory
    # del t_cats, temp_c, cs, a, c, temp_s, i, temp, fd_cats, industries, products, item
    
    # running the code to this point normally takes around 15 seconds
    return z_idx, industry_idx, product_idx, iix,pix, y_cols, sat_rows

###########################################################################################
## Gloria Emissions
## I now have 2 versions of the code that reads the data and processes into the footprint
## The old version that uses big matrices and a new version that just uses the components
## that are actually useful, ie non zero
## the read_data_new/old do the following:
##     reads S and U from Z (z_filepath)
##     reads the product rows from Y (y_filepath)
##     read the industry columns and stressor_row from co2_filepath.
##
## In read_data_new, only load required rows and cols from everything
## There are 2 version of cef.indirect_footprint_SUT to calcluate the footprint
## cef.indirect_footprint_SUT_new calculates the footprint using only the data given
## without constructing big matrices again
###########################################################################################

@profile
#--------------------------------------------------------------------
# function read_data_new
# This reads only the relevant rows and columns from Z, Y and co2file
# this is the new way to get the data using much less memory
# inputs:
#     z_filepath - the filepath from where to read the big matrix Z
#     y_filepath - the filepath from where to read Y
#     co2_filepath - the filepath from where to read the stressor data
#     iix -  these are the indices (0,1....) where the industry rows/columns are
#     pix - these are the indices (0,1...) where the product rows/columns are
#     industry_idx - this is the multiIndex to set in S/U/stressor
#     product_idx -  this is the multiIndex to set in S/U/Y
#     y_cols - the multiIndex to set in Y columns
#     stressor_row - the row to pick out from co2_filepath
# returns:
#     S - the [industry_idx, product_idx] part of Z
#     U - the [product_idx, industry_idx] part of Z
#     Y - the product_idx rows of Y
#     stressor - the [stressor_row, industry_idx] part of co2_filepath (NB this is a single row)
#--------------------------------------------------------------------
def read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row):

    # read S and U directly from Z by specifying specific rows and cols
    S=pd.read_csv(z_filepath, header=None, index_col=None, skiprows = lambda x: x not in iix, usecols=pix)
    S.index=industry_idx; S.columns=product_idx
    U=pd.read_csv(z_filepath, header=None, index_col=None,skiprows = lambda x: x not in pix, usecols=iix)
    U.index=product_idx; U.columns=industry_idx

    # read product rows of Y and rename index and column
    # JAC just read the required rows directly and set up the indices for the columns
    Y = pd.read_csv(y_filepath, header=None, index_col=None, skiprows = lambda x: x not in pix)
    Y.index=product_idx; Y.columns=y_cols
    
    # import stressor (co2) data
    # JAC just read the required row and columns directly from the csv
    stressor = pd.read_csv(co2_filepath, header=None, index_col=None, nrows=1, skiprows=stressor_row, usecols=iix)
    stressor.columns=industry_idx

    return S, U, Y, stressor

@profile
#--------------------------------------------------------------------
# function read_data_old
# This reads the relevant rows and columns from Z, Y and co2file by reading
# the whole lot and then selecting the required rows/columns
# this is the old way this worked
# inputs:
#     z_filepath - the filepath from where to read the big matrix Z
#     y_filepath - the filepath from where to read Y
#     co2_filepath - the filepath from where to read the stressor data
#     z_idx - this is the multiIndex for the whole of Z
#     industry_idx - this is the multiIndex to select the relevant parts for S/U
#     product_idx -  this is the multiIndex to select the relevants parts for S/U
#     y_cols - the multiIndex to set in Y columns
#     sat_rows - the indices of all the rows of stressor in co2_filepath
#     stressor_cat - used to find the particular row needed from co2_filepath
# returns:
#     S - the [industry_idx, product_idx] part of Z
#     U - the [product_idx, industry_idx] part of Z
#     Y - the product_idx rows of Y
#     stressor - the [stressor_row, industry_idx] part of co2_filepath (NB this is a single row)
#--------------------------------------------------------------------
def read_data_old(z_filepath,y_filepath,co2_filepath,z_idx,industry_idx, product_idx, y_cols, sat_rows, stressor_cat):

    # import Z file to make S and U tables
    # This is not necessar, but it then follows the same structure as the other datasets we use, which is why this is currenlty done
    # Mainly splitting the Z here just allows us to use a single function for multiple datasets, from what I can tell this is not the part taking super long, so I have kept it like this
    # but it's possible that just making a new function for the Gloria data would be better than reorganising this data twice as
    # S and U are later combined into Z again, but with slightly different index and column order
    Z = pd.read_csv(z_filepath, header=None, index_col=None)
    Z.index = z_idx; Z.columns = z_idx
    S = Z.loc[industry_idx, product_idx]
    U = Z.loc[product_idx, industry_idx]
    del Z # remove Z to clear memory

    # import Y and rename index and column
    # again, it's matched to the strutcure of other datasets we analyse
    Y = pd.read_csv(y_filepath, header=None, index_col=None)
    Y.index = z_idx; Y.columns = y_cols
    Y = Y.loc[product_idx]
    
    # import stressor (co2) data
    # again, it's matched to the structure of other datasets we analyse
    stressor = pd.read_csv(co2_filepath, header=None, index_col=None)
    stressor.index = sat_rows; stressor.columns = z_idx
    stressor = stressor.loc[stressor_cat, industry_idx]

    return S, U, Y, stressor



#-------------------------------------------------------------------------------------
# This is where the code starts
# This should be run as python calculate_emission_gloria start_year end_year -n -S -v
# where -n means run using minimum data (otherwise do it the old way with big matrices)
#       -Le means use L.e in the old code rather than e.L
#       -S means use small data rather than large data - defines directories from which to read/write data
#       -v = verbose
#       -t = do timing
#-------------------------------------------------------------------------------------
@profile
def main():

    print(platform)

    # set working directory 
    # make different path depending on operating system
    # -> not needed for test, but as Anne sometimes works on a Mac and I work on a windows machine this makes sure it runs everywhere
    if platform[:3] == 'win':
    ###mri    wd = 'O://' # This is how Anne and I have named the A72 drive when mapping it
        wd = '../' # MRI hack for local data on my Win10 system
    elif platform =='linux':
        wd='../' # JAC hack for running on foe-linux
    else:
        wd = r'/Volumes/a72/' 
    

    new=False # run in the old way
    fextra='_old' # used for me to output files for testing
    fout_extra='' # used for the footprint file to indicate we used the new/old way to do this
    verbose=False
    do_timing=False
    small=False # define whether to use Small or Large data
    # in the old code it calculated e.L but should do L.e use this flag to define whether we should make the correction
    # we will always use L.e in the new code
    use_Le = False
    if len(argv)<2:
        print('Useage: python', argv[0], 'start_year end_year [-n -Le -S -t -v]')
        print('where -n means use new way to process data,\n-Le means use L.e in old way,\n-S means use Small data,\n-t means do timing,\n-v means verbose')
        exit()

    start_year=0
    end_year=0
    for i in range(1,len(argv)):
        if argv[i]=='-n':
            new=True
            print('NEW')
            fextra='_new'
            fout_extra='_New'
        elif argv[i]=='-Le':
            use_Le=True
            fout_extra='_Le'
        elif argv[i]=='-S':
            small=True
        elif argv[i]=='-v':
            verbose=True
        elif argv[i]=='-t':
            do_timing=True
        else:
            if start_year==0:
                start_year=int(argv[i])
            else:
                end_year=int(argv[i])

    # define filepaths
    if small:
        mrio_filepath = wd + 'GloriaSmallData/' # Changed to match sample folder
        outdir='../Output/GloriaSmall/'
        print('using Small data')
    else:
        mrio_filepath = wd + 'GloriaLargeData/' # Changed to match sample folder
        outdir='../Output/GloriaLarge/'
        print('using Large data')

    if do_timing:
        time0=time.time()
    z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows=get_metadata_indices(mrio_filepath)
    if do_timing:
        time1=time.time()
        print('TIME: get_metadata_indices', time1-time0)

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
        # file name ending changes from 2017, so define this here
        if year < 2017:
            date_var = '20230314'
        else:
            date_var = '20230315'
        # define filenames to make script cleaner
###mri     z_filepath = (mrio_filepath + date_var + '_120secMother_AllCountries_002_T-Results_' + str(year) + '_057_Markup001(full).csv') 
###mri     y_filepath = (mrio_filepath + '_120secMother_AllCountries_002_Y-Results_' + str(year) + '_057_Markup001(full).csv') 
###mri     co2_filepath = (mrio_filepath + '20230727_120secMother_AllCountries_002_TQ-Results_' + str(year) + '_057_Markup001(full).csv') 
        z_filepath = (mrio_filepath + 'Z_small.csv')
        y_filepath = (mrio_filepath + 'Y_small.csv')
        co2_filepath = (mrio_filepath + 'stressor_small.csv')
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
            footprint=cef.indirect_footprint_SUT_new(S, U, Y, stressor,verbose, do_timing)    
        else:
            footprint=cef.indirect_footprint_SUT(S, U, Y, stressor, use_Le, verbose, do_timing)    

        if do_timing:
            time1=time.time()
            print('TIME: cef.indirect_footprint_SUT', time1-time0)

        if verbose:
            print('Footprint calculated for ' + str(year))
    
        footprint.to_csv(outfile)
        print('Footprint saved for ' + str(year))

    print('Gloria Done')

if __name__ == '__main__':
    main()
