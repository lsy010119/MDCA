import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes

from lib.uav             import UAV
from lib.waypoint        import *
from lib.waypoint_insert import *
from lib.admm            import ADMM


class MDCA:

    def __init__(self,UAVs,v_min,v_max,d_safe,split_interval):

        self.UAVs = UAVs

        self.v_min = v_min
        self.v_max = v_max
        
        self.d_safe = d_safe

        self.split_interval = split_interval

        self.t_st = np.zeros((len(self.UAVs),1))

        self.c_set = []

        self.N = 0
        self.N_c = 0


    def run(self, avoidance=True, simul_arr=True):

        K = len(self.UAVs)                                            # total number of uavs

        self.UAVs, self.N_c = insert_collision_point(self.UAVs)       # insert collision points as waypoint
        
        # self.UAVs = split_segment(self.UAVs,self.split_interval)      # split segments with given interval

        for i in range(K):  
            
            self.UAVs[i].rearange()                                   # rearange the indeces of waypoints

            self.N += self.UAVs[i].N                                  # count the number of waypoints


        admm = ADMM(self.UAVs,self.v_min,self.v_max,self.d_safe,self.N,self.N_c,self.t_st)


        for i in range(K):
    
            wps = np.array([0,0])
    
            for n in range(len(self.UAVs[i].wp)):

                wps = np.vstack((wps,self.UAVs[i].wp[n].loc))

            plt.scatter(wps[1:,0],wps[1:,1])
            plt.plot(wps[1:,0],wps[1:,1])
        
        plt.show()




if __name__ == "__main__":

    wp1 = np.array([[0,0], [6,5], [8,8], [15,0]])
    wp1 = multiple_insert(wp1,0)
    uav1 = UAV(0,wp1)

    wp2 = np.array([[0,5], [4,1], [9,3], [15,5]])
    wp2 = multiple_insert(wp2,1)
    uav2 = UAV(1,wp2)
    
    wp3 = np.array([[0,9], [5,6], [6,0], [15,3]])
    wp3 = multiple_insert(wp3,2)
    uav3 = UAV(2,wp3)

    uavs = [uav1,uav2,uav3]

    mdca = MDCA(uavs,1,10,2,0.5)

    mdca.run()