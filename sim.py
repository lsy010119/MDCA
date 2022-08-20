#!/usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cp
import time
# matplotlib.rcParams['text.usetex'] = True

from lib.uav import UAV
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes



class Simulator:


    def __init__(self, delt, UAVs, v_min, v_max, d_safe, visuallize = True):
        
        self.delt = delt

        self.v_min = v_min

        self.v_max = v_max

        self.d_safe = d_safe

        self.collision_points = {}

        self.traj = {}

        self.total_time = -1
        
        self.solver = Solver(delt,UAVs,v_min,v_max, d_safe)

        self.UAVs = UAVs # list of UAVs

        self.visuallize = visuallize



    def calculate_trajec(self,uav):

        traj = np.zeros((2,1))

        for i in range(uav.n - 1):

            wp_i1 = uav.wp[i]
            wp_i2 = uav.wp[i+1]

            del_xi = ( wp_i2[0] - wp_i1[0] ) / ( uav.t[i] / self.delt )
            del_yi = ( wp_i2[1] - wp_i1[1] ) / ( uav.t[i] / self.delt )

            traj_xi = np.arange( wp_i1[0], wp_i2[0], del_xi ) + wp_i1[0]
            traj_yi = np.arange( wp_i1[1], wp_i2[1], del_yi ) + wp_i1[1]

            traj_i = np.vstack(( traj_xi, traj_yi ))

            traj = np.hstack((traj,traj_i))

        traj = traj[:,1:]

        uav.traj = traj
        


    def run(self):

        self.solver.run()

        for uav in self.UAVs:

            self.calculate_trajec(uav)

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

    SIM = Simulator(0.1,UAVs,0,0,0)
    SIM.run()