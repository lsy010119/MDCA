#!/usr/bin/python3
from tabnanny import verbose
from turtle import position
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cp
import time
# matplotlib.rcParams['text.usetex'] = True

from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes


class UAV:

    def __init__(self, num=np.array([]), wp=np.array([])):

        self.num = num
        self.wp = wp
        self.v_set = np.array([])
        self.d_set = np.array([])


        if len(self.wp) != 0:        
            
            for n in range( len(wp) - 1 ):

                d_n = np.linalg.norm( wp[n+1] - wp[n])

                self.d_set = np.append( self.d_set, d_n )





class Solver:

    def __init__(self, delt, UAVs, v_min, v_max, d_safe):

        self.delt = delt # discrete-time interval

        self.UAVs = UAVs # list of UAVs

        self.v_min = v_min

        self.v_max = v_max

        self.d_safe = d_safe



    def check_intersection(self, wp1_1, wp1_2, wp2_1, wp2_2):

        # L1 = ( 1 - t )*wp1_1 + t*wp1_2 
        # L2 = ( 1 - s )*wp2_1 + s*wp2_2

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

        N = len(self.UAVs)  # total number of UAVs

        collision_points = {}

        for i in range(N):

            for j in range(N-i-1):

                uavi = self.UAVs[i]         # ith uav
                uavj = self.UAVs[i+j+1]     # jth uav

                for ik in range(len(uavi.wp)-1):

                    wpi_1 = uavi.wp[ik] 
                    wpi_2 = uavi.wp[ik+1]

                    for jk in range(len(uavj.wp)-1):

                        wpj_1 = uavj.wp[jk]
                        wpj_2 = uavj.wp[jk+1]

                        wp_c = self.check_intersection(wpi_1,wpi_2,wpj_1,wpj_2)

                        if len(wp_c) != 1:

                            d_ci = np.linalg.norm(wpi_1-wp_c)
                            d_cj = np.linalg.norm(wpj_1-wp_c)

                            collision_points[(i,i+j+1,ik,jk,d_ci,d_cj)] = wp_c
        
        return collision_points


    def run(self):

        K = len(self.UAVs)              # total number of uavs

        t = []                          # [ t^(1) t^(2) ... t^(K-1)     ]
        t_c = []                        # [ t_c^(1) t_c^(2) ... t_c^(n) ]
        delt_c = []                     # [ t_c^(1) t_c^(2) ... t_c^(n) ]
        d = []                          # [ d^(1) d^(2) ... d^(K)       ]
        N = []                          # [ N_1   N_2   ... N_K         ]
        
        obj   = 0                       # objective function
        const = []                      # constraints

        t_opt = []                      # optimal time set
        v_opt = []                      # optimal velocity set

        for k in range(K): # index of wp of k th drone

            t.append(cp.Variable( (len(self.UAVs[k].wp))-1 ))
            d.append((self.UAVs[k]).d_set)
            N.append( (len(self.UAVs[k].wp)) )  

            obj += (t[k])[-1]           # cost += arrival time

        

        '''Constraints #1 : Velocity constraints'''
        for k in range(K):

            ## Selection Matrix S ###
            S = np.eye( N[k]-1 )
            
            for j in range(N[k] - 2):
                
                S[j+1,j] = -1

            ### Velocity Constraints ###
            const += [ S@t[k] - d[k]/self.v_min <= 0 ]
            const += [ d[k]/self.v_max - S@t[k] <= 0 ]


        '''Constraints #2 : Collision avoidance constraints'''
        cp_set = self.check_collision_point()

        i_c = 0

        for indexes, collision_point in cp_set.items():
            
            delt_c.append(cp.Variable(1))
            
            i =  indexes[0]
            j =  indexes[1]
            ik = indexes[2]-1
            jk = indexes[3]-1

            if ik < 0:
                
                ti = t[i]                       # t set of i'th uav
                ti_n1 = 0                       # t_init of i'th uav
                ti_n2 = ti[0]                   # t_0 of i'th uav                
                
                di_n = self.UAVs[i].d_set[0]    # d_0 of i'th uav
                di_c = indexes[4]               # d_col of i'th uav

            else:
                ti = t[i]                       # t set of i'th uav
                ti_n1 = ti[ik]                  # t_n of i'th uav
                ti_n2 = ti[ik+1]                # t_n+1 of i'th uav

                di_n = self.UAVs[i].d_set[ik+1]   # d_n of i'th uav
                di_c = indexes[4]               # d_col of i'th uav

            if jk < 0:

                tj = t[j]                       # t set of j'th uav
                tj_m1 = 0                       # t_init of j'th uav
                tj_m2 = tj[0]                   # t_0 of j'th uav

                dj_m = self.UAVs[j].d_set[0]    # d_0 of j'th uav
                dj_c = indexes[5]               # d_col of j'th uav

            else:

                tj = t[j]                       # t set of j'th uav
                tj_m1 = tj[jk]                  # t_m of j'th uav
                tj_m2 = tj[jk+1]                # t_m+1 of j'th uav

                dj_m = self.UAVs[j].d_set[jk+1] # d_m of j'th uav
                dj_c = indexes[5]               # d_col of j'th uav                


            ### t_safety definition ###
            t_safety = (self.d_safe/di_n)*(ti_n2-ti_n1)
            t_safety = (self.d_safe/dj_m)*(tj_m2-tj_m1)


            ### ti_c, tj_c definition ###
            ti_c = (di_c/di_n)*(ti_n2-ti_n1) + ti_n1
            tj_c = (dj_c/dj_m)*(tj_m2-tj_m1) + tj_m1

            # const += [t_safety <= ti_c - tj_c]
            # const += [ti_c - tj_c <= 100]
            const += [-100 <= ti_c - tj_c]
            const += [ti_c - tj_c <= -t_safety]
            # const += [ delt_c[i_c] - cp.abs(ti_c - tj_c) == 0]
            # const += [ delt_c[i_c] - 100 <= 0 ]
            # const += [ -delt_c[i_c] + t_safety <= 0 ]

            print( ((ti_c - tj_c)**2).is_dcp() )
            print( ( (2/1)*cp.log(1+cp.exp(1*(ti_c-tj_c))) - (ti_c-tj_c) - (2/1)*cp.log(2) <= 10 ).is_dcp() )
            print( ( (2/1)*cp.log(1+cp.exp(1*(ti_c-tj_c))) - (ti_c-tj_c) - (2/1)*cp.log(2) <= 10 ).is_dcp() )
                               

            i_c += 1


        ''' Solve '''
        cp.Problem( cp.Minimize(obj), const ).solve(verbose=False)

    
        for k in range(K):

            tk_opt = np.round( t[k].value , 5) 
            t_opt.append(tk_opt)

            tk_1 = tk_opt
            tk_2 = np.append(np.zeros(1) ,tk_opt[:-1] )
            del_t = tk_1 - tk_2

            vk_opt = np.round( d[k] / del_t, 5 )
            v_opt.append(vk_opt)

            (self.UAVs[k]).v_set = vk_opt



        for indexes, collision_point in cp_set.items():

            
            i =  indexes[0]
            j =  indexes[1]
            ik = indexes[2]-1
            jk = indexes[3]-1


            if ik < 0:
                
                ti = t_opt[i]                   # t set of i'th uav
                ti_n1 = 0                       # t_init of i'th uav
                ti_n2 = ti[0]                   # t_0 of i'th uav                
                
                di_n = self.UAVs[i].d_set[0]    # d_0 of i'th uav
                di_c = indexes[4]               # d_col of i'th uav

            else:
                ti = t_opt[i]                       # t set of i'th uav
                ti_n1 = ti[ik]                  # t_n of i'th uav
                ti_n2 = ti[ik+1]                # t_n+1 of i'th uav

                di_n = self.UAVs[i].d_set[ik+1]   # d_n of i'th uav
                di_c = indexes[4]               # d_col of i'th uav

            if jk < 0:

                tj = t_opt[j]                       # t set of j'th uav
                tj_m1 = 0                       # t_init of j'th uav
                tj_m2 = tj[0]                   # t_0 of j'th uav

                dj_m = self.UAVs[j].d_set[0]    # d_0 of j'th uav
                dj_c = indexes[5]               # d_col of j'th uav

            else:

                tj = t_opt[j]                       # t set of j'th uav
                tj_m1 = tj[jk]                  # t_m of j'th uav
                tj_m2 = tj[jk+1]                # t_m+1 of j'th uav

                dj_m = self.UAVs[j].d_set[jk+1]   # d_m of j'th uav
                dj_c = indexes[5]               # d_col of j'th uav                


            ### t_safety definition ###
            t_safety = (self.d_safe/di_n)*(ti_n2-ti_n1)
            t_safety = (self.d_safe/dj_m)*(tj_m2-tj_m1)

            # print((self.d_safe/di_n)*(ti_n2-ti_n1))
            # print((self.d_safe/dj_m)*(tj_m2-tj_m1))


            ### ti_c, tj_c definition ###
            ti_c = (di_c/di_n)*(ti_n2-ti_n1) + ti_n1
            tj_c = (dj_c/dj_m)*(tj_m2-tj_m1) + tj_m1

            t_c.append([ti_c,tj_c])
            print('safety',t_safety)
            print('ti_c - tj_c',ti_c-tj_c)
            print(ti_c)
            print(tj_c)

        return t_c, t_opt, v_opt



            









class Simulator:

    def __init__(self, delt, UAVs, v_min, v_max, d_safe, viz = True):
        
        self.delt = delt # discrete-time interval

        self.UAVs = UAVs # list of UAVs

        self.v_min = v_min

        self.v_max = v_max

        self.d_safe = d_safe

        self.viz = viz

        self.collision_points = {}

        self.solver = Solver(delt,UAVs,v_min,v_max, d_safe)

        self.traj = []

        self.total_time = []

        self.t_c = None

    def run(self):

        self.collision_points = self.solver.check_collision_point()
        self.t_c, t_opt, v_opt = self.solver.run()


        for k,uav in enumerate(self.UAVs):

            pose_set = np.zeros((2,1))

            t_set = t_opt[k]

            for i,t_i in enumerate(t_set): 

                wp_1 = uav.wp[i]
                wp_2 = uav.wp[i+1]

                
                pose_x = np.linspace(wp_1[0],wp_2[0],int(t_i/self.delt))
                pose_y = np.linspace(wp_1[1],wp_2[1],int(t_i/self.delt))


                pose = np.vstack((pose_x,pose_y))

                pose_set = np.hstack(( pose_set, pose ))

            self.traj.append(pose_set[:,1:])
            self.total_time.append(len(pose_set[0])-1)

        if self.viz:
            print("Vel : ",v_opt)
            self.visuallize()



    def visuallize(self):

        fig = plt.figure("Constraints : Dynamics")


        ''' Trajectory Plot '''
        traj = fig.add_subplot(2,1,1)
        traj.set_title(r"$\bf figure 1$  UAV trajectory")
        traj.set_xlabel(r"$x$",fontsize=10)
        traj.set_ylabel(r"$y$",fontsize=10)

        for uav in self.UAVs:
    
            traj.plot( uav.wp[:,0], uav.wp[:,1] , '--', color = np.random.rand(3,) , label='UAV #%d'%(uav.num))

            for k in range(len(uav.wp)):

                traj.scatter( uav.wp[k,0], uav.wp[k,1], marker = 'o', color='black', s=20 )
                traj.text( uav.wp[k,0], uav.wp[k,1] + 0.2, r'$ WP^{(%d)}_{%d}$'%(uav.num,k) )
    
        for indexes,col_pos in self.collision_points.items():

            i, j, uavi_k, uavj_k = indexes[0],indexes[1],indexes[2],indexes[3]

            traj.scatter( col_pos[0], col_pos[1], marker = '*', color='red', s=100 )
            traj.text( col_pos[0], col_pos[1] + 0.2, r'$\bf CP^{(%d,%d)}$'%(i+1,j+1), color = 'red' )


        traj.axis('equal')
        traj.legend()


        ''' Relative Position '''
        relpos = fig.add_subplot(2,1,2)
        relpos.set_title(r"$\bf figure 2$  Relative Distance")
        relpos.set_xlabel(r"$time$",fontsize=10)
        relpos.set_ylabel(r"$relative distance$",fontsize=10)


        end_time = max(self.total_time)
        
        for i,cp in enumerate(self.t_c):

            ti_c = cp[0]/self.delt
            tj_c = cp[1]/self.delt
            relpos.vlines( ti_c, 0,10, color='red')
            relpos.vlines( tj_c, 0,10, color='blue')


        for k,pose_set in enumerate(self.traj):

            waiting_time = end_time - len(pose_set[0])


            pose_expanded = np.zeros((2,end_time))
            pose_expanded[0] = np.hstack((pose_set[0],pose_set[0,-1]*np.ones(waiting_time)))
            pose_expanded[1] = np.hstack((pose_set[1],pose_set[1,-1]*np.ones(waiting_time)))

            self.traj[k] = pose_expanded
            

            # print(pose_expanded[0,int(ti_c)])
            # print(pose_expanded[1,int(ti_c)])

        relpos.plot( np.arange(0,end_time,1), np.linalg.norm(self.traj[0]-self.traj[1],2,axis=0),'-', color = np.random.rand(3,) , label='UAV #1 & UAV #2')
        # relpos.text( uav.wp[k,0], uav.wp[k,1] + 0.2, r'$\bf WP^{(%d)}_{%d}$'%(uav.num,k) )
        relpos.hlines( self.d_safe, 0, end_time)

        relpos.legend()
        print("Minimal Distance : ",np.min(np.linalg.norm(self.traj[0]-self.traj[1],2,axis=0)))
        plt.show()






if __name__ == "__main__":

    wp1 = np.array([[0,0], [6,5], [8,8], [15,0]])
    wp2 = np.array([[0,5], [4,0], [9,3], [15,5]])
    # wp3 = np.array([[15,3], [6,0], [3,6], [0,2]])

    wp1 = np.array([[0,0], [4,0], [4,5], [10,5]])
    wp2 = np.array([[0,6], [3,6], [3,1], [10,1]])

    wp1 = np.array([[0,0], [10,10]])
    wp2 = np.array([[0,10], [10,0]])

    uav1 = UAV(1,wp1)
    uav2 = UAV(2,wp2)
    # uav3 = UAV(3,wp3)

    
    UAVs = [uav1,uav2]

    Sim = Simulator(0.0001, UAVs, 0.1, 5, 2, viz=False)
    Sim.run()