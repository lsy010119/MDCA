
from cmath import inf
import struct
from matplotlib import projections
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes

from uav_wp_insert import UAV


class WP:

    def __init__(self,loc,id,is_cp=False,id_collide=None):
        
        self.loc = loc
        self.id = id
        self.is_cp = is_cp
        self.id_collide = id_collide


def multiple_insert(wps,num):

    wplist = []

    for i in range(len(wps)):

        wp = WP(wps[i],num)

        wplist.append(wp)

    return wplist


class MDCA:

    def __init__(self,UAVs,v_min,v_max,d_safe,split_interval):

        self.UAVs = UAVs

        self.v_min = v_min
        self.v_max = v_max
        
        self.d_safe = d_safe

        self.split_interval = split_interval


    def check_intersection(self, wp1_1, wp1_2, wp2_1, wp2_2):

        # L1 = ( 1 - t )*wp1_1 + t*wp1_2 
        # L2 = ( 1 - s )*wp2_1 + s*wp2_2

        wp1_1 = wp1_1.loc
        wp1_2 = wp1_2.loc
        wp2_1 = wp2_1.loc
        wp2_2 = wp2_2.loc

        x1 = wp1_1[0]
        y1 = wp1_1[1]
        x2 = wp1_2[0]
        y2 = wp1_2[1]
        
        x3 = wp2_1[0]
        y3 = wp2_1[1]
        x4 = wp2_2[0]
        y4 = wp2_2[1]

        if (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) == 0: # parallel

            return np.zeros(1)
        
        else:

            t = ( (x4 - x3)*(y1- y3) - (y4 - y3)*(x1 - x3) ) / ( (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) )
            s = ( (x2 - x1)*(y1- y3) - (y2 - y1)*(x1 - x3) ) / ( (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) )
            
            if ( t < 0 or 1 < t ) or ( s < 0 or 1 < s ):

                return np.zeros(1)

            else:

                wp_c = np.zeros((2))

                wp_c[0] = x1 + t*(x2 -x1)
                wp_c[1] = y1 + t*(y2 -y1)

                return wp_c
    

    def check_collision_point(self):

        K = len(self.UAVs)                  # total number of UAVs

        collision_points = {}               # contains 'i,j, ik,jk'

        for i in range(K):

            for j in range(K-i-1):

                uavi = self.UAVs[i]         # ith uav
                uavj = self.UAVs[i+j+1]     # jth uav

                for ik in range(len(uavi.wp)-1):  # for N_i waypoints of uav_i

                    wpi_1 = uavi.wp[ik] 
                    wpi_2 = uavi.wp[ik+1]

                    for jk in range(len(uavj.wp)-1): # for N_j waypoints of uav_j

                        wpj_1 = uavj.wp[jk]
                        wpj_2 = uavj.wp[jk+1]

                        wp_c = self.check_intersection(wpi_1,wpi_2,wpj_1,wpj_2)

                        if len(wp_c) != 1:

                            d_ci = np.linalg.norm(wpi_1.loc-wp_c)
                            d_cj = np.linalg.norm(wpj_1.loc-wp_c)

                            collision_points[(i,i+j+1,ik,jk,d_ci,d_cj)] = wp_c
        
        # indexes of collided uav set : i,j
        # indexes of waypoint passing through : ik, jk
        return collision_points


    def insert_collision_points(self,collision_points):

        K = len(self.UAVs)              # total number of uavs

        for i in range(K):

            cp_count = 0
            cp_i = []

            for indeces, loc in collision_points.items():

                i_,j_,n,m,dc_i,dc_j = indeces

                if i == i_:

                    wp_ci = WP(loc,i,is_cp=True,id_collide=j_)

                    cp_i.append([wp_ci,n,dc_i])

                elif i == j_:

                    wp_ci = WP(loc,i,is_cp=True,id_collide=i_)

                    cp_i.append([wp_ci,m,dc_j])

            cp_i_dist_sorted = sorted(cp_i, key = lambda key : key[2])
            cp_i_segm_sorted = sorted(cp_i_dist_sorted, key = lambda key : key[1])

            for cp in cp_i_segm_sorted:

                self.UAVs[i].wp.insert(cp[1]+1+cp_count, cp[0])

                cp_count += 1

            self.UAVs[i].calculate_dist()


    def split_segment(self,interval):

        K = len(self.UAVs)

        for i in range(K):

            self.UAVs[i].calculate_dist()

            ip_count = 0

            for n in range(len(self.UAVs[i].wp)-1):

                ip_num = int(self.UAVs[i].d[n]/interval)

                for j in range(1,ip_num):

                    insert_loc = (1-j/ip_num)*self.UAVs[i].wp[n+ip_count].loc + (j/ip_num)*self.UAVs[i].wp[n+1+ip_count].loc

                    insert_point = WP(insert_loc, i)

                    self.UAVs[i].wp.insert(n+1+ip_count,insert_point)

                    ip_count += 1

            self.UAVs[i].calculate_dist()

            self.UAVs[i].N = len(self.UAVs[i].wp)


    def run(self, avoidance=True, simul_arr=True):

        K = len(self.UAVs)              # total number of uavs

        self.split_segment(self.split_interval)

        # collision_points = self.check_collision_point()

        # self.insert_collision_points(collision_points)        


        # for i in range(K):
    
        #     wps = np.array([0,0])
    
        #     for n in range(len(self.UAVs[i].wp)):

        #         wps = np.vstack((wps,self.UAVs[i].wp[n].loc))

        #     plt.scatter(wps[1:,0],wps[1:,1])
        #     plt.plot(wps[1:,0],wps[1:,1])
        
        # plt.show()


        t = []                          # t = [t^1, t^2, ... , t^K]
        d = []                          # d = [d^1, d^2, ... , d^K]
        N = []                          # N = [N^1, N^2, ... , N^K]
        t_c = []                        # t_c = [t_c_1, t_c_2, ... , t_c_n ]
        
        obj   = 0                       # objective function
        const = []                      # constraints

        # t_opt = []                      # optimal time set      : [t_opt^1, t_opt^2, ... , t_opt^K]
        # v_opt = []                      # optimal velocity set  : [v_opt^1, v_opt^2, ... , v_opt^K]


        ''' Initializing Variables & Objective Function '''

        for uav in self.UAVs: # for i'th drone

            ti = cp.Variable( uav.N - 1 )          # t^i = [t^i_1, t^i_2, ... , t^i_{N^i-1}]
            di = uav.d                             # d^i = [d^i_1, d^i_2, ... , d^i_{N^i-1}]

            t.append(ti)                           # t = [t^1, t^2, ... , t^K]
            d.append(di)                           # d = [d^1, d^2, ... , d^K]
            N.append(uav.N)                        # N = [N^1, N^2, ... , N^K]

            obj +=  ti[-1]                         # cost = Sum of arrival time of UAVs

        if simul_arr: # additional cost function : simultaneus arrival cost

            for i in range(K):
                for j in range(K-i-1):

                    t_arr_i = t[i] 
                    t_arr_j = t[i+j+1]

                    obj += 100*cp.sum_squares(  (t_arr_i[-1] - t_arr_j[-1])  )   # cost = sum( |t_i - t_j|^2 )




        

        ''' Constraints #1 : Velocity constraints

                d_k/v_max < t_k < d_k/v_min
        
        '''

        for i in range(K):

            ## Selection Matrix ###
            S = np.eye( N[i]-1 )
            
            for j in range(N[i] - 2):
                
                S[j+1,j] = -1

            ### Velocity Constraints ###
            const += [ S@t[i] - d[i]/self.v_min <= 0 ]
            const += [ d[i]/self.v_max - S@t[i] <= 0 ]


        if avoidance:

            ''' Constraints #2 : Collision avoidance constraints
            
                    | t^i_c - t^j_c | > t_safety

            '''

            ### check collision risk point ###
            collision_points = self.check_collision_point()   # set of collision points

            x = []
            
            ### appending collision avoidance constraints ###
            for indexes, collision_point in collision_points.items():
                
                # For the point where the i-th uav and j-th uav can collide.

                # the i'th uav and j'th uav was coming from
                # ik'th waypoint and j'th waypoint respectively

                i =  indexes[0]             
                j =  indexes[1]
                ik = indexes[2]-1 # index of waypoint where i'th uav came from
                jk = indexes[3]-1 # index of waypoint where j'th uav came from


                # defining d^i_c and d^j_c

                if ik < 0: # if i'th uav came from initial waypoint
                    
                    ti = t[i]                       # t^i     : t set of i'th uav
                    ti_n1 = 0                       
                    ti_n2 = ti[0]                   # t^i_1   : t_0 of i'th uav                
                    
                    di_n = self.UAVs[i].d[0]        # d^i_1   : d_0 of i'th uav
                    di_c = indexes[4]               # d^i_c   : d_col of i'th uav

                    # sum_di_c = self.UAVs[i].d[0] + di_c
                    sum_di_c = np.sum(self.UAVs[i].d[:1])
                    # print(f"sum_di_c : {sum_di_c}")

                else:
                    ti = t[i]                       # t^i     : t set of i'th uav
                    ti_n1 = ti[ik]                  # t^i_n   : t_n of i'th uav
                    ti_n2 = ti[ik+1]                # t^i_n+1 : t_n+1 of i'th uav

                    di_n = self.UAVs[i].d[ik+1]     # d^i_n   : d_n of i'th uav
                    di_c = indexes[4]               # d^i_c   : d_col of i'th uav

                    # sum_di_c = np.sum(self.UAVs[i].d[:ik+1]) + di_c
                    sum_di_c = np.sum(self.UAVs[i].d[:ik+2])
                    # print(f"sum_di_c : {sum_di_c}")

                if jk < 0:

                    tj = t[j]                       # t^j     :  t set of j'th uav
                    tj_m1 = 0                       
                    tj_m2 = tj[0]                   # t^j_m   :  t_0 of j'th uav

                    dj_m = self.UAVs[j].d[0]        # d^j_m   :  d_0 of j'th uav
                    dj_c = indexes[5]               # d^j_c   :  d_col of j'th uav

                    # sum_dj_c = self.UAVs[j].d[0] + dj_c
                    sum_dj_c = np.sum(self.UAVs[j].d[:jk+2])
                    # print(f"sum_dj_c : {sum_dj_c}")

                else:

                    tj = t[j]                       # t^j     : t set of j'th uav
                    tj_m1 = tj[jk]                  # t^j_m   : t_m of j'th uav
                    tj_m2 = tj[jk+1]                # t^j_m+1 : t_m+1 of j'th uav

                    dj_m = self.UAVs[j].d[jk+1]     # d^j_m   : d_m of j'th uav
                    dj_c = indexes[5]               # d^j_c   : d_col of j'th uav                

                    # sum_dj_c = np.sum(self.UAVs[j].d[:jk+1]) + dj_c
                    sum_dj_c = np.sum(self.UAVs[j].d[:jk+2])
                    # print(f"sum_dj_c : {sum_dj_c}")

                # total distance from start point to collision point

                ### t_safety definition ###
                t_safety = (self.d_safe/di_n)*(ti_n2-ti_n1)
                t_safety = (self.d_safe/dj_m)*(tj_m2-tj_m1)

                ### ti_c, tj_c definition ###
                ti_c = (di_c/di_n)*(ti_n2-ti_n1) + ti_n1
                tj_c = (dj_c/dj_m)*(tj_m2-tj_m1) + tj_m1


                ''' Approach #1 '''
                # const += [ t_safety - ti_c + tj_c <= 0 ]
                # const += [ t_safety + ti_c - tj_c <= 0 ]


                ''' Approach #2 '''
                if sum_di_c >= sum_dj_c:

                    const += [ t_safety - (ti_c - tj_c) <= 0 ]

                else:

                    const += [ t_safety + (ti_c - tj_c) <= 0 ]


                ''' Approach #3 '''
                # x_n = cp.Variable(1)

                # x.append(x_n)

                # const += [ cp.abs(ti_c - tj_c) <= (2**0.5)*x_n ]
                # const += [ t_safety - x_n <= 0 ]
                # const += [ x_n <= 100 ]




        ''' Solve '''
        cp.Problem( cp.Minimize(obj), const ).solve(solver=cp.ECOS, verbose=True)

    
        for i in range(K):

            ti_opt = t[i].value 

            ti_1 = ti_opt
            ti_2 = np.append(np.zeros(1) ,ti_opt[:-1] )

            vi_opt = d[i] / ( ti_1 - ti_2)
            
            (self.UAVs[i]).v_set = vi_opt


            self.UAVs[i].t = ti_opt
            self.UAVs[i].del_t = ti_1 - ti_2
            self.UAVs[i].v = vi_opt

        


        ''' for printing result data '''

        for uav in self.UAVs: # for i'th drone

            ti = cp.Variable( uav.N - 1 )          # t^i = [t^i_1, t^i_2, ... , t^i_{N^i-1}]
            di = uav.d                             # d^i = [d^i_1, d^i_2, ... , d^i_{N^i-1}]

            t.append(ti)                           # t = [t^1, t^2, ... , t^K]
            d.append(di)                           # d = [d^1, d^2, ... , d^K]
            N.append(uav.N)                        # N = [N^1, N^2, ... , N^K]

            obj +=  ti[-1]                         # cost = Sum of arrival time of UAVs

        if simul_arr: # additional cost function : simultaneus arrival cost

            for i in range(K):
                for j in range(K-i-1):

                    t_arr_i = t[i] 
                    t_arr_j = t[i+j+1]

                    obj += 100*cp.sum_squares(  (t_arr_i[-1] - t_arr_j[-1])  )   # cost = sum( |t_i - t_j|^2 )


        print(f"Total cost :")



        num = 0
        collision_points = self.check_collision_point()   # set of collision points

        print(f"==== Time differences at collision Points ====")

        for indexes, collision_point in collision_points.items():
            
            # For the point where the i-th uav and j-th uav can collide.

            # the i'th uav and j'th uav was coming from
            # ik'th waypoint and j'th waypoint respectively

            i =  indexes[0]             
            j =  indexes[1]
            ik = indexes[2]-1 # index of waypoint where i'th uav came from
            jk = indexes[3]-1 # index of waypoint where j'th uav came from


            # defining d^i_c and d^j_c

            if ik < 0: # if i'th uav came from initial waypoint
                
                ti = self.UAVs[i].t             # t^i     : t set of i'th uav
                ti_n1 = 0                       
                ti_n2 = ti[0]                   # t^i_1   : t_0 of i'th uav                
                
                di_n = self.UAVs[i].d[0]        # d^i_1   : d_0 of i'th uav
                di_c = indexes[4]               # d^i_c   : d_col of i'th uav

                sum_di_c = self.UAVs[i].d[0] + di_c

            else:
                ti = self.UAVs[i].t             # t^i     : t set of i'th uav
                ti_n1 = ti[ik]                  # t^i_n   : t_n of i'th uav
                ti_n2 = ti[ik+1]                # t^i_n+1 : t_n+1 of i'th uav

                di_n = self.UAVs[i].d[ik+1]     # d^i_n   : d_n of i'th uav
                di_c = indexes[4]               # d^i_c   : d_col of i'th uav

                sum_di_c = np.sum(self.UAVs[i].d[:ik+1]) + di_c

            if jk < 0:

                tj = self.UAVs[j].t             # t^j     :  t set of j'th uav
                tj_m1 = 0                       
                tj_m2 = tj[0]                   # t^j_m   :  t_0 of j'th uav

                dj_m = self.UAVs[j].d[0]        # d^j_m   :  d_0 of j'th uav
                dj_c = indexes[5]               # d^j_c   :  d_col of j'th uav

                sum_dj_c = self.UAVs[j].d[0] + dj_c

            else:

                tj = self.UAVs[j].t             # t^j     : t set of j'th uav
                tj_m1 = tj[jk]                  # t^j_m   : t_m of j'th uav
                tj_m2 = tj[jk+1]                # t^j_m+1 : t_m+1 of j'th uav

                dj_m = self.UAVs[j].d[jk+1]     # d^j_m   : d_m of j'th uav
                dj_c = indexes[5]               # d^j_c   : d_col of j'th uav                

                sum_dj_c = np.sum(self.UAVs[j].d[:jk+1]) + dj_c

            # total distance from start point to collision point

            ### t_safety definition ###
            t_safety = (self.d_safe/di_n)*(ti_n2-ti_n1)
            t_safety = (self.d_safe/dj_m)*(tj_m2-tj_m1)

            ### ti_c, tj_c definition ###
            ti_c = (di_c/di_n)*(ti_n2-ti_n1) + ti_n1
            tj_c = (dj_c/dj_m)*(tj_m2-tj_m1) + tj_m1
            

            print(f"==== Collision Point of ( {i+1}th UAV, {j+1}th UAV ) ====")
            print("Collision point arrival time difference | ti_c - tj_c | \n: ",abs(ti_c - tj_c))
            print("t_safety calculated \n: ",t_safety)
        
            num += 1

        print(f"==== Arrival time differences ====")

        for i in range(K):
            for j in range(K-i-1):

                print(f" {i+1}th UAV, {i+j+2}th UAV : ",abs(self.UAVs[i].t[-1]-self.UAVs[i+j+1].t[-1]))




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

    mdca = MDCA(uavs,1,10,2)

    mdca.run()