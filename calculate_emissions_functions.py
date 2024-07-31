# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki

updated by Julia Crook (CEMAC) to optimize the code by only reading what is necessary from csv files
and by computing footprint using the components rather than a big matrix
I have added read_config to get the required parameters to run either on Large data or Small data. 
Also it makes the code more configurable in where the files are stored.

"""

import pandas as pd
import numpy as np
#import scipy.linalg as sla


#-----------------------------------------
# to handle memory profiling line by line
#-----------------------------------------
import os

mprof=os.environ['MPROFILE_Lbl']
print('MPROFILE_Lbl', mprof)
if mprof==1 or mprof=='1':
    print('cef: importing mprofiler')
    from memory_profiler import profile
else:
    from profile import *

#----------------------------------------------------------------
# read config file to get indir, outdir and filenames which are different
# for small and large data
#----------------------------------------------------------------
def read_config(cfname):

    cf = open(cfname, "r")
    lines=cf.readlines()
    if len(lines)<7:
        print('invalid config file')
        raise SystemExit
    # need to remove '\n' from line
    indir=lines[0][:-1]
    outdir=lines[1][:-1]
    labels_fname=lines[2][:-1]
    lookup_fname=lines[3][:-1]
    Z_fname=lines[4][:-1]
    Y_fname=lines[5][:-1]
    co2_fname=lines[6][:-1]
    cf.close()

    return indir, outdir, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname


#######################################################################################
## Gloria Metadata
## The next two funtions are for reading the lookup and labels excel files to determine
## the multiIndices for Z and Y
#######################################################################################
#----------------------------------------------------------------
# this function will split the strings into a list of <country>+ (<code>), 
# and <some_str>, where <some_str> is the remainder of the string
# inputs:
#   labels is a list of strings containing <country> (<code>) <some_str>
#       note that the string before the code may also contain ()
#   valid_country_codes is a list of valid country codes
# returns:
#   country - list of <country>
#   country_code - list of <code>
#   remainder - list of <some_str> 
#------------------------------------------------------
def split_country_and_code(labels, valid_country_codes):

    nl=len(labels)
    country = ['NA']*nl
    country_code=['']*nl
    remainder=['']*nl
    n=0
    for cs in labels:
        items=cs.split('(')
        # find which part has the country code
        for item in items:
            items2=item.split(')')
            if items2[0] in valid_country_codes:
                country_code[n]=items2[0]
                # country is the string before the country code and remainder is the part after
                items3=cs.split('('+items2[0]+')')
                country[n]=items3[0]
                remainder[n]=items3[1]
        if country[n]=='NA':
            print('Could not find country code in', cs)
        n+=1


    # end code if there is a mismatch in labels, this is done to test that the code does what it should
    if 'NA' in country:
        print('Error: Missing country labels')
        raise SystemExit 
        
    return country, country_code, remainder


#----------------------------------------------------------------
# This function reads the excel files containing the indices in Z
# mrio_filepath is the directory where they exist
# labels_fname is the filename for the labels file
# lookup_fname is the filename for the lookup file
#----------------------------------------------------------------
@profile
def get_metadata_indices(mrio_filepath, labels_fname, lookup_fname):
    # read metadata which is used for column and row labels later on
    readme = mrio_filepath + labels_fname
    labels = pd.read_excel(readme, sheet_name=None)

    # get lookup to fix labels
    lookup = pd.read_excel(mrio_filepath + lookup_fname, sheet_name=None)
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
    country, country_code, remainder=split_country_and_code(t_cats['label'], valid_country_codes)        
    #t_cats['country'] = country # dont really need this as not used
    t_cats['country_code'] = country_code
    t_cats['sector'] = remainder

    # fix final demand labels (this is in the Y dataframe later)
    # it follows the same logic as the Z label fix above
    fd_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0)); fd_cats.columns = ['label']
    # remove special characters frm country names
    # JAC replace loop below with call to function
    country, country_code, remainder=split_country_and_code(fd_cats['label'], valid_country_codes)        
    #fd_cats['country'] = country # dont really need this as not used
    fd_cats['country_code'] = country_code
    fd_cats['fd'] = remainder

    # split lables by industries vs products (these later get split into the S and U dataframes)
    t_cats['ind'] = t_cats['label'].str[-8:]
    industries = t_cats.loc[t_cats['ind'] == 'industry']
    products = t_cats.loc[t_cats['ind'] == ' product']
    sector_type=np.asarray(t_cats['ind'])
    iix=np.where(sector_type=='industry')[0]
    pix=np.where(sector_type==' product')[0]

    # make index labels
    z_idx = pd.MultiIndex.from_arrays([t_cats['country_code'], t_cats['sector']]) # labels for Z dataframe
    industry_idx = pd.MultiIndex.from_arrays([industries['country_code'], industries['sector']]) # labels used to split Z into S and U
    product_idx = pd.MultiIndex.from_arrays([products['country_code'], products['sector']]) # labels used to split Z into S and U
    y_cols = pd.MultiIndex.from_arrays([fd_cats['country_code'], fd_cats['fd']]) # labels for Y dataframe

    sat_rows = labels['Satellites']['Sat_indicator'] # labels for CO2 dataframe
    # clear space in variable explorer to free up memory
    # del t_cats, temp_c, cs, a, c, temp_s, i, temp, fd_cats, industries, products, item
    
    # running the code to this point normally takes around 15 seconds
    return z_idx, industry_idx, product_idx, iix,pix, y_cols, sat_rows

###########################################################################################
## Reading Data
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
@profile
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
@profile
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

###########################################################################################
## Gloria Emissions calculations
## The following functions are for calcluating the footprint
###########################################################################################
@profile
def make_x(Z, Y, verbose):
    
    x = np.sum(Z, 1)+np.sum(Y, 1)
    x[x == 0] = 0.000000001
    if verbose:
        print("DBG: X shape is ", x.shape)
    return x

# equivalent function of make_x but does it as components
@profile
def make_x_comp_new(S, U, Y, verbose):
    # components of what was x
    sumS=np.sum(S,1)  # this is x1
    sumS[sumS == 0] = 0.000000001 # do this so dividing by 0 does not happen
    sumU=np.sum(U,1) 
    sumY=np.sum(Y,1)
    sumUY=sumU+sumY   # this is x2
    sumUY[sumUY==0] = 0.000000001 # do this so dividing by 0 does not happen

    if verbose:
        print('DBG: sumS, sumUY shape is ', sumS.shape, sumUY.shape)

    return sumS, sumUY

@profile
def make_L(Z, x, verbose):
    
    bigX = np.zeros(shape = (len(Z)))    
    bigX = np.tile(np.transpose(x), (len(Z), 1))

    A = np.divide(Z, bigX)
    #np.save('A_old.npy', A)    
    I_minus_A=np.identity(len(Z))-A
    L = np.linalg.inv(I_minus_A)
    #np.save('L_old.npy', L)
    if verbose:
        print('DBG: bigX shape is ', bigX.shape)
        print('DBG: A shape is ', A.shape)
        print('DBG: L shape is ', L.shape)

    return L

# equivalent of make_L but does it as components
@profile
def make_L_comp_new(S, U, sumS, sumUY, verbose):

    bigSumS = np.tile(np.transpose(sumS), (S.shape[0],1))
    bigSumUY = np.tile(np.transpose(sumUY), (U.shape[0],1))

    # use elementwise divide as was done in make_L to get A
    Snorm=np.divide(S,bigSumUY)
    Unorm=np.divide(U,bigSumS)
    #np.save('Snorm.npy', Snorm)
    #np.save('Unorm.npy', Unorm)

    # in equation [I-A]X=D where I-A top is  [I,-Snorm] and I-A bottom is [-Unorm, I], Dtop is 0 and Dbottom is Y
    # assume X has X1 as the top part and X2 as the bottom part from which we get
    # 1. X1-Snorm.X2 = 0 and
    # 2. -Unorm.X1 + X2 = Y
    #   
    # from 1. we get 3. X1=Snorm.X2
    # insert into 2. X2 - Unorm.(Snorm.X2) = Y
    # so (I-Unorm.Snorm).X2 = Y
    # so X2 = inv(I-Unorm.Snorm).Y
    # If we say L has components [Ltl, Ltr]
    #                            [Lbl, Lbr]
    # then Ltl.0+Ltr.Y=X1, i.e. Ltr.Y=X1=Snorm.X2=Snorm.inv(I-Unorm.Snorm).Y
    # so Ltr=Snorm.inv(I-Unorm.Snorm)
    # and Lbl.0+Lbr.Y=X2=inv(I-Unorm.Snorm).Y
    # so Lbr=inv(I-Unorm.Snorm)
    # we cannot say anything about Ltl and Lbl from these equations
    # we will be using L to do e.L where the bottom part of e is 0. This means we will need Ltl and Ltr
    # We also know that [Ltl, Ltr] [I, -Snorm] = [I, 0]
    #                   [Lbl, Lbr] [-Unorm, I]   [0, I]
    # so Ltl-Ltr.Unorm=I, ie Ltl =I+Ltr.Unorm
    # and Lbl-Lbr.Unorm=0  ie Lbl=Lbr.Unorm

    I=np.identity(S.shape[0])
    L=np.linalg.inv(I-np.matmul(Unorm,Snorm))
    # use sci version - faster? Test this on big data on machine with multiple cores
    #L = sla.inv(I-np.matmul(Unorm, Snorm))
    #np.save('L_new.npy', L)
    if verbose:
        print('DBG: Unorm and Snorm shape', Unorm.shape, Snorm.shape)
        print('DBG: L shape is ', L.shape)
    
    # As we are going to do e.L where the second half of e is 0 we only need the Ltl and Ltr components
    # but then we are going to multiply by Y where Y is 0 in top half so we only need Ltr
    Ltr=np.dot(Snorm, L)
    #Ltl=I+np.dot(Ltr, Unorm)

    return Ltr

def make_e(stressor, x):
    # MRI not used in this model for some reason
    e = np.zeros(shape = (1, np.size(x)))
    e[0, 0:np.size(stressor)] = np.transpose(stressor)
    e = e/x

@profile
def make_Z_from_S_U(S, U, verbose):
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
    if verbose:
        print('DBG: make Z from S and U', Z.size, Z.shape )

    return Z

# there is no equivalent of make_Z_from_S_U for the new way as we just work with S and U

# I have pulled out the for loop that creates the footprint so I can time it
@profile
def calculate_footprint(bigY, eL, y_cols, su_idx, u_cols):

    footprint = np.zeros(shape = bigY.shape).T

    for a in range(np.size(bigY, 1)):
        footprint[a] = np.dot(eL, np.diag(bigY[:, a]))
    
    old_shape=footprint.shape
    footprint = pd.DataFrame(footprint, index=y_cols, columns=su_idx)
    footprint = footprint[u_cols]
    return footprint

@profile
def indirect_footprint_SUT(S, U, Y, stressor, verbose):
    # make column names
    s_cols = S.columns.tolist()
    u_cols = U.columns.tolist()
    su_idx = pd.MultiIndex.from_arrays([[x[0] for x in s_cols] + [x[0] for x in u_cols],
                                        [x[1] for x in s_cols] + [x[1] for x in u_cols]])
    y_cols = Y.columns

    # calculate emissions
    Z = make_Z_from_S_U(S, U,verbose)
    # clear memory
    del S, U
    
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y 

    x = make_x(Z, bigY,verbose)
    L = make_L(Z, x, verbose)

    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    #np.save('e_old.npy', e)

    eL = np.dot(e, L)
    #np.save('eL_old.npy', eL)

    if verbose:
        print('DBG: bigY shape', bigY.shape)
        print('DBG: e shape is ', e.shape, 'big_stressor is ', bigstressor.shape)
        print('DBG: eL shape is ', eL.shape)

    footprint = calculate_footprint(bigY, eL, y_cols, su_idx, u_cols)
    if verbose:
         print('DBG: footprint shape is', footprint.shape)
 
    return footprint

# I have pulled out the calculation of the footprint so I can time it
@profile
def calculate_footprint_new(eL2,Y,y_cols,u_cols):

    # for each column in Y the code used to take the diagonal of bigY to find the dot product with eL
    # as bigY was 0 in the top half, only the bottom half of eL would have been valid
    # therefore we only need to use the eL2 part.
    
    Y2=Y.to_numpy()
    footprint=np.asarray([eL2*Y2[:,a] for a in range(Y.shape[1])])
    footprint = pd.DataFrame(footprint, index=y_cols, columns=u_cols)
    return footprint

# equivalent of indirect_footprint_SUT but does it as components
@profile
def indirect_footprint_SUT_new(S, U, Y, stressor,verbose):
    # calculate emissions
    sumS, sumUY=make_x_comp_new(S,U,Y,verbose)

    # stressor has 1 row also may be different indexing which messes up np.divide so just look at array
    stress=stressor.to_numpy()[0,:]
    e1=np.divide(stress, sumS) # lower part of e is 0 as bigstressor only had stressor in top part
    e2=0
    if verbose:
        print('DBG: e1 shape', e1.shape)
    #np.save('e1_new.npy', e1)

    Ltr = make_L_comp_new(S, U, sumS, sumUY, verbose)
    #eL1=np.dot(e1,Ltl)
    eL2=np.dot(e1,Ltr)
    if verbose:
        print('DBG: Ltr shape', Ltr.shape)
    #np.save('eL2_new.npy', eL2)
    
    y_cols = Y.columns
    u_cols=U.columns
    footprint=calculate_footprint_new(eL2,Y,y_cols,u_cols)
    if verbose:
        print('DBG: footprint shape is',footprint.shape)

    return footprint
