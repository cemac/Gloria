# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
import numpy as np
import time
#import scipy.linalg as sla

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
# a list of <code> and a list of <some_str>, where <some_str> is the remainder of the string
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
# labels_fname is the filename for the labels file
# lookup_fname is the filename for the lookup file
#----------------------------------------------------------------
def get_metadata_indices(mrio_filepath, labels_fname, lookup_fname):
    # read metadata which is used for column and row labels later on
    readme = mrio_filepath + labels_fname
    labels = pd.read_excel(readme, sheet_name=None)

    # get lookup to fix labels
    # MRI modify for "small dataset"
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

#@profile
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

#@profile
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

###########################################################################################
## Gloria Emissions calculations
## The following functiosn are for calcluating the footprint
###########################################################################################
#@profile
def make_x(Z, Y, verbose):
    
    x = np.sum(Z, 1)+np.sum(Y, 1)
    x[x == 0] = 0.000000001
    if verbose:
        print("DBG: X shape is ", x.shape)
    return x

#@profile
# equivalent function of make_x but does it as components
def make_x_comp_new(S, U, Y, verbose):
    # components of what was x
    sumS=np.sum(S,1)
    sumS[sumS == 0] = 0.000000001 # this is x1
    sumU=np.sum(U,1) 
    sumY=np.sum(Y,1)
    sumUY=sumU+sumY
    sumUY[sumUY==0] = 0.000000001 # this is x2

    if verbose:
        print("DBG: sumS, sumUY shape is ", sumS.shape, sumUY.shape)

    return sumS, sumUY

#@profile
def make_L(Z, x, verbose, do_timing):
    
    bigX = np.zeros(shape = (len(Z)))    
    # bigX is as big as Z yet another wasteful storage (delete it? or chose other method)
    bigX = np.tile(np.transpose(x), (len(Z), 1))
    # MRI A is doubling the storage requirement (Z) and then L also adds same again
    A = np.divide(Z, bigX)
    #np.save('A_old.npy', A)    
    I_minus_A=np.identity(len(Z))-A
    if do_timing:
        time0=time.time()
    L = np.linalg.inv(I_minus_A)
    #L = sla.inv(np.identity(len(Z))-A) Marks new bit
    if do_timing:
        time1=time.time()
        print('TIME: inverting I-A matrix', time1-time0)

    #np.save('L_old.npy', L)
    if verbose:
        print("DBG: bigX shape is ", bigX.shape)
        print("DBG: A shape is ", A.shape)
        print("DBG: L shape is ", L.shape)

    return L, I_minus_A

#@profile
# equivalent of make_L but does it as components
def make_L_comp_new(S, U, sumS, sumUY, verbose, do_timing):

    bigSumS = np.tile(np.transpose(sumS), (S.shape[0],1))
    bigSumUY = np.tile(np.transpose(sumUY), (U.shape[0],1))

    # use elementwise divide as was done in make_L to get A
    scaledS=np.divide(S,bigSumUY)
    scaledU=np.divide(U,bigSumS)
    #np.save('scaledS.npy', scaledS)
    #np.save('scaledU.npy', scaledU)

    # in equation [I-A]X=D where I-A top is  [I,-scaledS] and I-A bottom is [-scaledU, I], Dtop is e1 and Dbottom=0
    # assume X is [X1,X2] from which we get
    # 1. X1-scaledS.X2 = e1 and
    # 2. -scaledU.X1 + X2 = 0
    #   
    # from 2. we get 3. X2=scaledU.X1
    # insert into 1. X1 - scaledS.(scaledU.X1) = e1
    # (I-scaledS.scaledU).X1 = e1
    # so X1 = inv(I-scaledS.scaledU).e1
    # then use X2=scaledU.X1

    I=np.identity(S.shape[0])
    if do_timing:
        time0=time.time()
    L=np.linalg.inv(I-np.matmul(scaledS,scaledU))
    # use sci version - faster? Test this on big data on machine with multiple cores
    #L = sla.inv(I-np.matmul(scaledS,scaledU))
    if do_timing:
         time1=time.time()
         print('TIME: inverting matrix', time1-time0)
    #np.save('L_new.npy', L)
    if verbose:
        print('scaledU and U shape', scaledU.shape, scaledS.shape)
        print("DBG: L shape is ", L.shape)

    return L, scaledU, scaledS

def make_e(stressor, x):
    # MRI not used in this model for some reason
    e = np.zeros(shape = (1, np.size(x)))
    e[0, 0:np.size(stressor)] = np.transpose(stressor)
    e = e/x

#@profile
def make_Z_from_S_U(S, U, verbose):
    # MRI this makes Z a numpy array and fills with zeroes
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
    if verbose:
        print("DBG: make Z from S and U", Z.size, Z.shape )

    return Z

#@profile
def indirect_footprint_SUT(S, U, Y, stressor, use_Le, verbose, do_timing):
    # make column names
    s_cols = S.columns.tolist()
    u_cols = U.columns.tolist()
    su_idx = pd.MultiIndex.from_arrays([[x[0] for x in s_cols] + [x[0] for x in u_cols],
                                        [x[1] for x in s_cols] + [x[1] for x in u_cols]])
    y_cols = Y.columns

    # calculate emissions
    if do_timing:
        time0=time.time()
    Z = make_Z_from_S_U(S, U,verbose)
    if do_timing:
        time1=time.time()
        print('TIME: make_Z_from_S_U', time1-time0)
    # clear memory
    del S, U
    
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    
    footprint = np.zeros(shape = bigY.shape).T
    footprint_Le = np.zeros(shape = bigY.shape).T

    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y 
    x = make_x(Z, bigY,verbose)
    if do_timing:
        time0=time.time()
    L,I_minus_A = make_L(Z, x, verbose, do_timing)
    if do_timing:
        time1=time.time()
        print('TIME: make_L', time1-time0)

    #np.save('L_old.npy', L)
    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    #np.save('e_old.npy', e)

    if use_Le:
        Le=np.dot(L,e)
        #np.save('Le_old.npy', Le)
        dot_prod=Le
        dot_prod_str='Le'
    else:
        eL = np.dot(e, L)
        #np.save('eL_old.npy', eL)
        dot_prod=eL
        dot_prod_str='eL'

    if verbose:
        print('DBG: bigY shape', bigY.shape)
        print("DBG: e shape is ", e.shape, "big_stressor is ", bigstressor.shape)
        print("DBG: "+dot_prod_str+" shape is ", dot_prod.shape)
        # check it works the other way
        exp_e=np.dot(I_minus_A,dot_prod)
        diff=abs(exp_e-e)
        ix=np.where(diff>0.00001)
        print(len(ix[0]), 'diffs exp_e and e using', dot_prod_str)

    if do_timing:
        time0=time.time()
    for a in range(np.size(Y, 1)):
        footprint[a] = np.dot(dot_prod, np.diag(bigY[:, a]))
    
    old_shape=footprint.shape
    footprint = pd.DataFrame(footprint, index=y_cols, columns=su_idx)
    footprint = footprint[u_cols]
    if do_timing:
         time1=time.time()
         print('TIME: make footprint', time1-time0)
    if verbose:
         print('DBG: full,u_cols footprint shape is',old_shape, footprint.shape)
 
    return footprint

#@profile
def indirect_footprint_SUT_new(S, U, Y, stressor,verbose, do_timing):
    # calculate emissions
    sumS, sumUY=make_x_comp_new(S,U,Y,verbose)

    # stressor has 1 row also may be different indexing which messes up np.divide so just look at array
    stress=stressor.to_numpy()[0,:]
    e1=np.divide(stress, sumS) # lower part of e is 0 as bigstressor only had stressor in top part
    e2=0
    if verbose:
        print('DBG: e1 shape', e1.shape)
    #np.save('e1_new.npy', e1)

    if do_timing:
        time0=time.time()
    L, scaledU, scaledS = make_L_comp_new(S, U, sumS, sumUY, verbose, do_timing)
    if do_timing:
        time1=time.time()
        print('TIME: make_L_comp_new', time1-time0)

    # It should be L dot e not e dot L - in fact I have shown it does not work in reverse if you use e dot L
    X1=np.dot(L,e1)
    X2=np.dot(scaledU, X1)
    Le=np.zeros(len(X1)*2)
    Le[:len(X1)]=X1
    Le[len(X1):]=X2
    if verbose:
        # check it works the other way
        exp_e1=X1-np.dot(scaledS, X2)
        exp_e2=X2-np.dot(scaledU,X1)
        diff=exp_e1-e1
        ix=np.where(abs(diff)>0.000001)
        if len(ix[0])>0:
            print('L.e calc not reversible!')

        print('DBG: X1, X2 and Le shape', X1.shape, X2.shape, Le.shape)
    #np.save('Le_new.npy', Le)
    
    # then for each column in Y the code used to take the diagonal of bigY to find the dot product with eL
    # as bigY was 0 in the top half only the bottom half of eL would have been valid
    # therefore we only need to use the X2 part
    
    footprint = np.zeros(shape = Y.shape).T
    Y2=Y.to_numpy()
    if do_timing:
        time0=time.time()
    for a in range(np.size(Y2, 1)):
        footprint[a] = np.dot(X2, np.diag(Y2[:, a]))

    y_cols = Y.columns
    u_cols=U.columns
    footprint = pd.DataFrame(footprint, index=y_cols, columns=u_cols)
    if do_timing:
         time1=time.time()
         print('TIME: make footprint', time1-time0)
    if verbose:
        print('DBG: footprint shape is',footprint.shape)

    return footprint
