import os
import numpy as np

class parameters:
    def __init__(self):
        self.simulationspercombination = 10 # simulations per combination

        self.branch_type = 2 # for Case 1: branching
        # self.branch_type = 3 # for Case 2: cycling

        self.distance = np.array([0, 1., 4.])
        self.N=10
        self.branch_unit = [3.0] # the unit from which the sequence bifurcates 

        self.liste_eta = np.array([0.04])
        self.liste_rho = np.array([1.2])
        self.liste_tau = np.array([300]) 
        self.liste_G = np.array([0.600])
        self.liste_mu = np.array([0.4])
        self.mu2 = 0.4
        self.I=0.0

        self.simTime = 5000

        self.patternLegend=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.distanceLegend=['-6','-5','-4','-3','-2','-1','0','1', '2', '3', '4', '5', '6']