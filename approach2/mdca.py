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
        # self.t_st = np.array([[1],[2],[0]])

        self.c_set = []

        self.K = len(self.UAVs)
        self.N = 0
        self.N_c = 0


    def rearange(self):
        '''
        Rearranging after waypoint insertions
        '''

        for i in range(self.K):  
            
            self.UAVs[i].rearange()                                   # rearange the indeces of waypoints

            self.N += self.UAVs[i].N                                  # count the number of waypoints
            
        # aranging collision points
        for i in range(self.K):

            UAV_i = self.UAVs[i]

            for wpi in UAV_i.wp:

                if wpi.is_cp:

                    loc = wpi.loc

                    UAV_j = self.UAVs[wpi.collide_with]

                    if i < wpi.collide_with:

                        for wpj in UAV_j.wp:

                            if np.all(loc == wpj.loc):

                                collision_pair = (i,\
                                                wpi.collide_with,\
                                                wpi.idx,\
                                                wpj.idx)

                                self.c_set.append(collision_pair)
                    

    def run(self, avoidance=True, simul_arr=True):


        self.UAVs, self.N_c = insert_collision_point(self.UAVs)       # insert collision points as waypoint
        
        self.UAVs = split_segment(self.UAVs,self.split_interval)      # split segments with given interval

        self.rearange()


        admm = ADMM(self.UAVs,self.v_min,self.v_max,self.d_safe,self.N,self.N_c,self.t_st,self.c_set)

        t = admm.run(1000)

        # for i in range(self.K):
    
        #     wps = np.array([0,0])
    
        #     for n in range(len(self.UAVs[i].wp)):

        #         wps = np.vstack((wps,self.UAVs[i].wp[n].loc))

        #     plt.scatter(wps[1:,0],wps[1:,1])
        #     plt.plot(wps[1:,0],wps[1:,1])
        
        # plt.show()


        N_temp = 0

        for id, uav in enumerate(self.UAVs):

            t_i = t[N_temp:N_temp + uav.N,0]

            t_i_ = np.append(np.zeros(1) ,t_i[:-1] )

            v_i = uav.d / ( t_i[1:] - t_i_[1:] )
            
            self.UAVs[id].t = t_i
            self.UAVs[id].del_t = t_i - t_i_
            self.UAVs[id].v = v_i

            N_temp += uav.N


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