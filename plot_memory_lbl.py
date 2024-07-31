# -*- coding: utf-8 -*-
"""
Created on 23/7/2024

@author: Julia Crook (earjacr), CEMAC

Reads the memory useage output for new and old way of running the code and identifies  the different functions that were profiled. For each function profiled it then plots the memory used for each line

"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from sys import argv

class FunctionMemory:
    def __init__(self, name,init_line_no, init_memory):
        self.func_name=name
        self.init_memory=init_memory
        self.memory=[]
        self.line_no=[]
        self.loc=[]
        self.add_memory(init_memory, init_line_no, 'init')

    def add_memory(self, memory, line_no, loc):
        self.memory.append(memory)
        self.line_no.append(line_no)
        self.loc.append(loc)

    def nlines(self):
        return len(self.memory)

# read a memory output file (mfname) and get memory per line number for each function identified
def get_memory(mfname):

    func_memories=[]
    mf = open(mfname, "r")
    lines=mf.readlines()
    header_found=False
    func_name_found=False
    func_memory=None
    for line in lines:

        split=line.split() # split by spaces
        if header_found and func_name_found==False:
            if len(split)==7 and split[-1]=='@profile':
                init_line_no=int(split[0])
                init_memory=float(split[1]) # next is 'MiB'
            if len(split)>=3:
                if split[1]=='def':
                    func_name=split[2].split('(')[0]
                    print('function', func_name, 'found')
                    func_name_found=True
                    func_memory=FunctionMemory(func_name, init_line_no, init_memory)
        elif func_name_found:
            if len(split)==0:
                # thats the end of the function
                header_found=False
                func_name_found=False
                func_memories.append(func_memory)
                func_memory=None
                init_line_no=-1
                init_memory=0
            elif len(split)>=10:
                if split[2]=='MiB':
                    line_no=int(split[0])
                    memory=float(split[1]) # next is 'MiB'
                    increment=float(split[3]) # next is 'MiB' then occurences
                    loc=split[6]
                    #print('found memory', line_no,memory,loc)
                    func_memory.add_memory(memory, line_no,loc)
        else:
            # find a line with the headers on
            # ie 'Line #    Mem usage    Increment  Occurrences   Line Contents'
            if len(split)==8 and split[0]=='Line':
                # this is the header so now look for func_name
                header_found=True
                print('found header')

    return func_memories

def main():

    if len(argv)<4:
        print('Useage: python', argv[0], '<outdir>, <mfname_old>, <mfname_new>')
        exit()

    outdir=argv[1]
    mfname_old=argv[2]
    mfname_new=argv[3]
    outfname=outdir+mfname_old[:-8]+'.png'

    # get the data from the files given
    func_memory_old=get_memory(outdir+mfname_old)
    func_memory_new=get_memory(outdir+mfname_new)

    nfunc_old=len(func_memory_old)
    nfunc_new=len(func_memory_new)
    nfunc=np.max([nfunc_old, nfunc_new])

    print('plotting', nfunc_old, nfunc_new)
    nrows=1
    if nfunc>6:
        nrows=3
    elif nfunc>2:
        nrows=2
    ncols=int(np.ceil((nfunc+1)/nrows))
    print(nrows, ncols)
    fig=plt.figure(figsize=(12, 13*nrows/ncols))
    nold=0
    nnew=0
    overall_mem_old=[]
    overall_mem_new=[]
    for n in range(nfunc):
        ax=fig.add_subplot(nrows,ncols,n+1)
        if func_memory_old[nold].func_name[:6]==func_memory_new[nnew].func_name[:6]:
            plot_old=True
            plot_new=True
        else:
            plot_old=False
            plot_new=False
            if func_memory_old[nold].func_name[:6]==func_memory_new[nnew+1].func_name[:6]:
                plot_new=True
            elif func_memory_old[nold+1].func_name[:6]==func_memory_new[nnew].func_name[:6]:
                plot_old=True
            print('plot old new', plot_old, plot_new, nold, nnew)

        title=''
        if plot_old:
            title=func_memory_old[nold].func_name
            if plot_new:
                title=title+'/'
        if plot_new:
            title=title+func_memory_new[nnew].func_name
        plt.title(title)
        ax.set_ylabel('memory (Mb)')
        #ax.set_xlabel('line')
        if plot_old:
            memory=np.asarray(func_memory_old[nold].memory)
            x=np.arange(len(memory))
            ax.plot(x, memory, color='k',linestyle='solid', label='Old')
            nold=nold+1
            if nfunc> 4:
                overall_mem_old.append(memory)
        else:
            if nfunc> 4:
                overall_mem_old.append([])

        if plot_new:
            memory=np.asarray(func_memory_new[nnew].memory)
            x=np.arange(len(memory))
            ax.plot(x, memory, color='k',linestyle='dotted',label='New')
            nnew=nnew+1
            if nfunc> 4:
                overall_mem_new.append(memory)
        else:
            if nfunc> 4:
                overall_mem_new.append([])

    if nfunc>4:
        ax=fig.add_subplot(nrows,ncols,nrows*ncols)
        plt.title('overall')
        ax.set_ylabel('memory (Mb)')
        ax.set_xlabel('line')
        nmem=len(overall_mem_old)
        old_l=0
        new_l=0
        for n in range(nfunc-1):
           if len(overall_mem_old[n])>0:
               line=np.arange(len(overall_mem_old[n]))+old_l
               ax.plot(line, overall_mem_old[n], color='k',linestyle='solid',label='Old')
               old_l=old_l+len(line)
           if len(overall_mem_new[n])>0:
               line=np.arange(len(overall_mem_new[n]))+new_l
               ax.plot(line, overall_mem_new[n], color='k',linestyle='dotted', label='New')
               new_l=new_l+len(line)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=len(labels))
    plt.suptitle(outdir+' '+mfname_old[:-8])
    plt.gcf().subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(outfname,dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()


