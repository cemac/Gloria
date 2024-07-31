# -*- coding: utf-8 -*-
"""
Created on 23/7/2024

@author: Julia Crook (earjacr), CEMAC

Reads the timing (cProfile) dump for new and old way of running the code and looks for functions of interest
It then plots the time taken in each of these functions as a bar for both new and old ways so we can compare times

"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import pstats
import io
from sys import argv

# read a dump file (tfname) and get the cumtime and ncalls for each function in func_names
def get_timing(tfname, func_names):
    cumtimes=np.zeros(len(func_names))+np.nan
    ncalls_all=np.zeros(len(func_names))+np.nan
    s = io.StringIO()
    timing=pstats.Stats(tfname,stream=s)
    for n in range(len(func_names)):
        if func_names[n]=='':
            continue
        #s.seek(0) # set cursor to start of string
        s.truncate(0)
        timing.print_stats(func_names[n])
        this_dump=s.getvalue()
        lines=this_dump.split('\n')
        # expect 8 lines of header then line required then three empty lines so our line is nlines -4
        nlines=len(lines)
        print(func_names[n],nlines)
        split=lines[nlines-4].split() # split by spaces
        if len(split)<=5:
            print(split)
            pdb.set_trace()
            continue
        try:
            ncalls=int(split[0])
        except:
            ncalls=split[0].split('/')[0]
        tottime=float(split[1])
        percall_tot=float(split[2])
        cumtime=float(split[3])
        percall_cum=float(split[4])
        location=split[5]
        loc_split=location.split('/')
        func_name=loc_split[-1].split('(')[-1][:-1] # find the function name in brackets and remove last )
        if func_names[n]!=func_name:
            pdb.set_trace()
        else:
            print('found', ncalls, tottime, percall_tot, cumtime, percall_cum, func_name)
            cumtimes[n]=cumtime
            ncalls_all[n]=ncalls
    return cumtimes, ncalls_all

def main():

    if len(argv)<3:
        print('Useage: python', argv[0], '<outdir> <test> where test=0/1/2')
        exit()

    outdir=argv[1]
    test=argv[2]

    if test=='0':
        # plotting profiling for running whole of gloria
        # set up the function names we are interested in - these need to be the same size even though for new there is no equivalent of make_Z_from_S_U
        func_names_old=np.asarray(['read_config', 'get_metadata_indices', 'read_data_old','indirect_footprint_SUT', 'make_Z_from_S_U','make_x', 'make_L', 'inv', 'calculate_footprint'])
        func_names_new=np.asarray(['read_config', 'get_metadata_indices', 'read_data_new','indirect_footprint_SUT_new', '', 'make_x_comp_new', 'make_L_comp_new', 'inv','calculate_footprint_new'])
        dump_fname_old='gloria_timing.dat'
        dump_fname_new='gloria_timing_New.dat'
        outfname='timing_gloria.png'
        title=outdir+' Gloria Timing'
    elif test=='1':
        # for timing of read SU only use these functions
        func_names_old=np.asarray(['read_config', 'get_metadata_indices','read_SU_old', 'read_csv'])
        func_names_new=np.asarray(['read_config', 'get_metadata_indices','read_SU_new', 'read_csv'])
        dump_fname_old='read_SU_timing.dat'
        dump_fname_new='read_SU_timing_New.dat'
        outfname='timing_read_SU.png'
        title=outdir+' read_SU Timing'
    else:
        # for timing of performing matrix inversion with numpy or scipy only use these functions
        func_names_old=np.asarray(['read_config', 'get_metadata_indices','read_SUY', 'make_L_comp'])
        func_names_new=np.asarray(['read_config', 'get_metadata_indices','read_SUY', 'make_L_comp'])
        dump_fname_old='linalg_inv_timing.dat'
        dump_fname_new='linalg_inv_timing_New.dat'
        outfname='timing_linalg_inv.png'
        title=outdir+' linalg.inv Timing'

    # get the data from the dump files
    cumtimes_old, ncalls_old=get_timing(outdir+dump_fname_old, func_names_old)
    cumtimes_new, ncalls_new=get_timing(outdir+dump_fname_new, func_names_new)
    if test=='0':
        print('Total time for old', np.sum(cumtimes_old[0:4]))
        print('Total time for new', np.sum(cumtimes_new[0:4]))
        pdb.set_trace()
    print('plotting')
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.title(title)
    ax.set_ylabel('cum time (s)')
    ax.set_xlabel('function')
    positions=np.arange(len(func_names_old))*3+1
    bars=plt.bar(positions, cumtimes_old)
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    rot=90
    for p in range(len(positions)):
        if func_names_old[p]!='':
            ax.text(positions[p],1, func_names_old[p]+' ({n:d})'.format(n=int(ncalls_old[p])), fontsize=8, rotation=rot)
        bars[p].set_color('cyan')
    bars=plt.bar(positions+1, cumtimes_new)
    for p in range(len(positions)):
        if func_names_new[p]!='' :
            ax.text(positions[p]+1,1, func_names_new[p]+' ({n:d})'.format(n=int(ncalls_new[p])), fontsize=8, rotation=rot)
        bars[p].set_color('red')

    outfile=outdir+outfname
    plt.savefig(outfile,dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()


