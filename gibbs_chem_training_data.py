from Chempy.parameter import ModelParameters
a=ModelParameters()
import multiprocessing as mp
import numpy as np
import tqdm
import time
from Chempy.cem_function import run_Chempy_sample


# First load parameter dataset:
training_params=np.load("APOGEE Training Data.npz")
param_grid=training_params['param_grid']
norm_grid = training_params['norm_grid']
N = len(param_grid) # number of samples
    

def runner(index):
    """Function to compute the Chempy predictions for each parameter set"""
    b=ModelParameters()
    params=param_grid[index]
    norm_params=norm_grid[index]
    abun,els=run_Chempy_sample(params,b);
    del b;
    return abun,params,norm_params

if __name__=='__main__':
    init_time=time.time()
    
    # Compute elements by running code once:
    _,els=run_Chempy_sample(param_grid[0],a)
    
    # Now run multiprocessing
    cpus=mp.cpu_count()
    p=mp.Pool(min(16,cpus))
    output=list(tqdm.tqdm(p.imap_unordered(runner,range(N)),total=N))
    abuns=[o[0] for o in output]
    pars=[o[1] for o in output]
    n_pars=[o[2] for o in output]
    
    end_time=time.time()
    
    print("multiprocessing complete after %d seconds"%(end_time-init_time));
    
    # Now save output
    np.savez("APOGEE_Training_Predictions",abundances=abuns,elements=els,params=pars,norm_params=n_pars);
