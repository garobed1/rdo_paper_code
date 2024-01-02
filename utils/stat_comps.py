import numpy as np
from utils.sutils import convert_to_smt_grads
import copy
from mpi4py import MPI
from math import ceil
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




def _mu_sigma_comp(func_handle, N, tx, xlimits, scales, pdf_list, tf = None, weights=None):

    dim = tx.shape[1]
    dim_u = 0
    for j in range(dim):
        if not isinstance(pdf_list[j], float):
            dim_u += 1

    vals = np.zeros([N,1])
    dens = np.ones([N,1])
    summand = np.zeros([N,1])    

    N_act = 1

    # uniform importance sampled monte carlo integration
    #NOTE: REMEMBER TO CITE/MENTION THIS

    # split by dim/size
    split = ceil(dim_u/size)
    
    arrs = np.array_split(tx, split)
    l1 = 0
    l2 = 0
    # import pdb; pdb.set_trace()
    for k in range(split):
        l2 += arrs[k].shape[0]
        if(tf is not None): #data
            vals[l1:l2,:] = tf[l1:l2,:]
        else:   #function
            vals[l1:l2,:] = func_handle(arrs[k])
        for j in range(dim):
            if not isinstance(pdf_list[j], float):
                if weights is None:
                    dens[l1:l2,:] = np.multiply(dens[l1:l2,:], pdf_list[j].pdf(arrs[k][:,j]).reshape((l2-l1, 1))) #TODO: loc must be different for certain dists
                else:
                    dens[l1:l2,:] = np.atleast_2d(weights[l1:l2]).T
        summand[l1:l2,:] = dens[l1:l2,:]*vals[l1:l2,:]
        l1 += arrs[k].shape[0]

    area = 1
    if weights is None:
        area = np.prod(scales) #just a hypercube
        N_act = copy.deepcopy(N)
    mean = area*sum(summand)/N_act
    #stdev = np.sqrt(((area*sum(summand*vals))/N_act - (mean)**2))#/N
    A = (sum(dens)/N_act)*(area)
    stdev = np.sqrt(((area*sum(summand*vals))/N_act - (2-A)*(mean)**2 ))#/N
    return (mean, stdev), vals


#TODO: need to not recompute functions if not needed, right now it will rerun analyses already completed
def _mu_sigma_grad(func_handle, N, tx, xlimits, scales, static_list, pdf_list, tf, tg = None, weights=None):

    dim = tx.shape[1]
    dim_u = 0
    for j in range(dim):
        if not isinstance(pdf_list[j], float):
            dim_u += 1

    dim_d = len(static_list)
    
    grads = np.zeros([N,dim])
    vals = np.zeros([N,1])
    dens = np.ones([N,1])
    summand = np.zeros([N,1])
    gsummand = np.zeros([N,dim_d])
    
    N_act = 1

    split = ceil(dim_u/size)
    arrs = np.array_split(tx, split)
    l1 = 0
    l2 = 0
    for k in range(split):
        l2 += arrs[k].shape[0]
        
        # tf is needed
        vals[l1:l2,:] = tf[l1:l2,:]
        
        if tg is not None:
            grads[l1:l2,:] = tg[l1:l2,:]
        else:
            # grads[l1:l2,:] = convert_to_smt_grads(func_handle, arrs[k])
            for ki in range(dim):
                grads[l1:l2,ki] = func_handle(arrs[k], kx=ki)[:,0]
        for j in range(dim):
            #import pdb; pdb.set_trace()
            if not isinstance(pdf_list[j], float):
                if weights is None:
                    dens[l1:l2,:] = np.multiply(dens[l1:l2,:], pdf_list[j].pdf(arrs[k][:,j]).reshape((l2-l1, 1))) #TODO: loc must be different for certain dists
                else:
                    dens[l1:l2,:] = np.atleast_2d(weights[l1:l2]).T
        summand[l1:l2,:] = dens[l1:l2,:]*vals[l1:l2,:]
        for j in range(dim_d):
            gsummand[l1:l2,j] = dens[l1:l2,:][:,0]*grads[l1:l2,static_list[j]]
        l1 += arrs[k].shape[0]

    area = 1
    if weights is None:
        area = np.prod(scales) #just a hypercube
        N_act = copy.deepcopy(N)
    mean = area*sum(summand)/N_act
    gmean = (area/N_act)*np.sum(gsummand, axis=0 )
    
    #stdev = np.sqrt(((area*sum(summand*vals))/N - (mean)**2))#/N
    A = (sum(dens)/N_act)*(area)
    stdev = np.sqrt(((area*sum(summand*vals))/N_act - (2-A)*(mean)**2 ))#/N
    work = 0.5*(1./stdev)
    work2 = 2.*(2.-A)*mean*gmean
    work3 = np.multiply(2*summand, grads[:,static_list])
    work3 = (area/N)*np.sum(work3, axis=0)
    gstdev = work*(work3 - work2)
    # import pdb; pdb.set_trace()
    #return full gradients, but gmean and gstdev are only with respect to dvs
    return (gmean, gstdev), grads

