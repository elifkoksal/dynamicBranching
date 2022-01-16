"""
@author: Elif Koksal Ersoz

This file is for running simulations in simu_E_g.py file for a given combination.  
    """
import numpy as np
import simu_E_g
import time
import os
from parameterLists import parameters

def runSimulation(p, eta, tau, u, mu, G):
    
    for n in range(0, int(p.simulationspercombination)):
        simu_E_g.simulation(p.branch_type, p.N, p.simTime, eta, tau, u, mu, G, p.I, n)

def main():
    p = parameters()

    for eta in p.liste_eta:
        for rho in p.liste_rho:
            for tau in p.liste_tau:             
                for G in p.liste_G:
                    for mu in p.liste_mu:
                        u = rho/tau
                        print('Combination : ',[eta, tau, u, G, mu])
                        time_start1 = time.time()                                                 
                        runSimulation(p, eta, tau, u, mu, G)
                        time_end1 = time.time()
                    print("Completed in {} sec".format(time_end1 - time_start1))

if __name__ == '__main__':
    main()
    
