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
N = len(param_grid) # number of samples
    

def runner(index):
    """Function to compute the Chempy predictions for each parameter set"""
    params=param_grid[index]
    abun,els=run_Chempy_sample(params,a);
    return abun

if __name__=='__main__':
    init_time=time.time()
    
    # Compute elements by running code once:
    _,els=run_Chempy_sample(param_grid[0],a)
    
    # Now run multiprocessing
    cpus=mp.cpu_count
    p=mp.Pool(min(10,cpu_count))
    output=list(tqdm.tqdm(p.imap(runner,range(N)),total=N))
    
    end_time=time.time()
    
    print("multiprocessing complete after %d seconds"%(end_time-init_time));
    
    # Now save output
    np.savez("APOGEE Training Predictions",abundances=output,elements=els);
