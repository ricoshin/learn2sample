import numpy as np

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
      return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
      return min(1.0, float(step/x0))
    elif anneal_function == None:
      return 1.0
    else:
      print('kl annealing function error\n')