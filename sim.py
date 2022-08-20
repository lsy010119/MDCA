#!/usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cp
import time
# matplotlib.rcParams['text.usetex'] = True

from uav import UAV
from mdca import MDCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes



class Simulator:
    '''
    Notations

    K   : total number of uavs
    i,j : index of two arbitrary uav

    N : total number of waypoints of each uav
    k : index of an waypoint

    '''

    def __init__(self, delt, UAVs, v_min, v_max, d_safe, visuallize = True):
        
        self.delt = delt

        self.v_min = v_min

        self.v_max = v_max

        self.d_safe = d_safe

        self.collision_points = {}

        self.traj = {}

        self.total_time = 0
        
        self.mdca = MDCA(UAVs,v_min,v_max,d_safe)

        self.UAVs = UAVs # list of UAVs

        self.visuallize = visuallize





    def run(self):

        self.mdca.run(avoidance=True)

        for uav in self.UAVs:

            uav.calculate_trajec(self.delt)

        if self.visuallize:

            self.visuallize()



if __name__ == "__main__":


    wp1 = np.array([[0,0], [6,5], [8,8], [15,0]])
    wp2 = np.array([[0,5], [4,0], [9,3], [15,5]])
    # wp3 = np.array([[15,3], [6,0], [3,6], [0,2]])

    wp1 = np.array([[0,0], [4,0], [4,5], [10,5]])
    wp2 = np.array([[0,6], [3,6], [3,1], [10,1]])

    # wp1 = np.array([[0,0], [10,10]])
    # wp2 = np.array([[0,10], [10,0]])

    uav1 = UAV(1,wp1)
    uav2 = UAV(2,wp2)
    # uav3 = UAV(3,wp3)

    
    UAVs = [uav1,uav2]

    SIM = Simulator(0.1,UAVs,3,5,1)
    SIM.run()