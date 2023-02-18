#!/usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cp
import time
# matplotlib.rcParams['text.usetex'] = True

from lib.uav import UAV
from lib.waypoint import *
from mdca import MDCA

from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


class Simulator:
    '''
    Notations

    K   : total number of uavs
    i,j : index of two arbitrary uav

    N : total number of waypoints of each uav
    k : index of an waypoint

    '''

    def __init__(self, delt, UAVs, v_min, v_max, d_safe, split_interval, t_st, \
                 avoidance=True, simul_arr=True, viz = [True,True,True],params = [1e-3,1.5,1e-3]):
        
        ### Environment ###

        self.delt = delt                        # time interval for simulator

        self.v_min = v_min                      # minimum velocity

        self.v_max = v_max                      # maximum velocity

        self.d_safe = d_safe                    # safety distance
        
        self.split_interval = split_interval    # segment split interval

        ### MDCA Initialize ###
        self.mdca = MDCA(UAVs,v_min,v_max,d_safe,split_interval,t_st,params)

        ### UAVs ###
        self.UAVs = UAVs                        # list of UAVs

        ### Options ###

        self.viz = viz                          # apply visualize

        self.avoidance = avoidance              # apply avoidance constraints

        self.simul_arr = simul_arr              # apply simultaneous arrival



    def visuallize(self):
        
        t_set = []
        uav_pose_xmax = []
        uav_pose_ymax = []
        uav_pose_xmin = []
        uav_pose_ymin = []
        endtime       = 0

        for uav in self.UAVs:

            t_set.append(len(uav.traj[0]))
            uav_pose_xmax.append(np.max(uav.traj[0]))
            uav_pose_ymax.append(np.max(uav.traj[1]))
            uav_pose_xmin.append(np.min(uav.traj[0]))
            uav_pose_ymin.append(np.min(uav.traj[1]))

            if uav.t[-1] > endtime:
                endtime = uav.t[-1]

        total_timesteps = max(t_set)

        for uav in self.UAVs:

            waiting_time = total_timesteps - len(uav.traj[0])

            traj_expanded    = np.zeros((2,total_timesteps))

            traj_expanded[0] = np.hstack((uav.traj[0],uav.traj[0,-1]*np.ones(waiting_time)))
            traj_expanded[1] = np.hstack((uav.traj[1],uav.traj[1,-1]*np.ones(waiting_time)))

            uav.traj = traj_expanded


        if self.viz[0]:

            fig1 = plt.figure("Trajectory")

            ''' Trajectory Plot '''

            traj = fig1.add_subplot(1,1,1)

            # traj.set_title(r"$\bf figure 1$  UAV trajectory")
            traj.set_xlabel(r"$\bf x(m)$",fontsize=10)
            traj.set_ylabel(r"$\bf y(m)$",fontsize=10)
            traj.set_xlim(-1,11)
            traj.set_ylim(-1,11)

            traj.set_xlim(min(uav_pose_xmin),max(uav_pose_xmax))
            traj.set_ylim(min(uav_pose_ymin),max(uav_pose_ymax))
            traj.grid()
            traj.axis('equal')


            for uav in self.UAVs:
                
                color = np.random.rand(3)

                traj.plot(uav.traj[0],uav.traj[1],'--',label=r'$UAV^{(%d)}$'%(uav.num),color = color)

                for wp in uav.wp:
                    
                    traj.plot(wp.loc[0],wp.loc[1],'o',color = color)


            uav_position = traj.plot([],[],'k*',markersize=10)[0]


            def update(timestep):

                traj_x = []
                traj_y = []

                for uav in self.UAVs:

                    traj_x.append(uav.traj[0,timestep])
                    traj_y.append(uav.traj[1,timestep])

                uav_position.set_data(traj_x,traj_y)
            

            sim = FuncAnimation(fig=fig1, func=update, frames=total_timesteps, interval=0.1) 
            # sim.save('Senario3_app2.gif', fps=30, dpi=100)

            traj.legend()
            plt.show()


        if self.viz[1]:
            
            fig2 = plt.figure("Relative Distances")

            ''' Relative Position Plot '''

            rel = fig2.add_subplot(1,1,1)

            rel.set_title(r"$\bf figure 2$  Relative Position")
            rel.set_xlabel(r"$\bf time\;(s)$",fontsize=10)
            rel.set_ylabel(r"$\bf Relative\;Distance\;(m)$",fontsize=10)
                

            for i in range(len(self.UAVs)):
                for j in range(len(self.UAVs) - i - 1):

                    rel.plot( np.arange(0,total_timesteps,1), \
                              np.linalg.norm(self.UAVs[i].traj-self.UAVs[i+j+1].traj,2,axis=0),'-',\
                              color = np.random.rand(3,) ,\
                              label=r'$UAV^{(%d)}$ & $UAV^{(%d)}$'%(self.UAVs[i].num,self.UAVs[j+i+1].num))


            rel.hlines( self.d_safe, 0, total_timesteps, label=r"$d_{safety}$")
            rel.legend()
            plt.show()


        if self.viz[2]:

            fig3 = plt.figure("Velocity Data",figsize=(20,20))


            for uav in self.UAVs:

                vel = fig3.add_subplot(len(self.UAVs),1,uav.num+1)
                vel.set_ylim(self.v_min-4,self.v_max+4)
                vel.set_xlim(-1, endtime + 2)

                vel.set_title(r"$\bf UAV^{(%d)}$"%(uav.num), loc="right",fontsize=15)
                vel.set_xlabel(r"$\bf time\;(s)$",fontsize=10)
                vel.set_ylabel(r"$\bf Velocity\;(m/s)$",fontsize=15)

                t_set = uav.t 
                v_set = uav.v 

                for i in range(len(t_set)):
    
                    try:

                        t_set = np.insert(t_set,2*i,t_set[2*i])
                        v_set = np.insert(v_set,2*i,v_set[2*i])

                    except:
                        
                        pass


                t_set = t_set[:-1]
                t_set = np.insert(t_set,0,-1)

                # if t_set[-1] < total_timesteps*self.delt:

                t_set = np.append( t_set, np.array([t_set[-1],endtime + 2]) )

                v_set = np.append( v_set, np.array([0,0]) )

                v_set = np.append( np.array([0,0]), v_set )

                vel.vlines(uav.t[-1], -10, 20, color="red", linewidth=1, linestyles='-', label=r"Arrival Time")
                vel.vlines(uav.t[0], -10, 20, color="green", linewidth=1, linestyles='-', label=r"Starting Time")

                vel.hlines(self.v_max, -1, t_set[-1], color="red", linewidth=2, linestyles='--', label=r"Max velocity")
                vel.hlines(self.v_min, -1, t_set[-1], color="green", linewidth=2, linestyles='--', label=r"Min velocity")
                vel.plot(t_set, v_set, color="black", linewidth=4, label=r"Velocity")
                vel.legend(loc='lower left',prop={'size':10})
                vel.grid()
                plt.tick_params(axis='both', which='major', labelsize=10)
            
            # plt.savefig("vel",dpi=300)
            plt.show()



    def run(self):

        self.mdca.run(avoidance=self.avoidance, simul_arr=self.simul_arr)

        for uav in self.UAVs:

            uav.calculate_trajec(self.delt)

        if self.viz:

            self.visuallize()



if __name__ == "__main__":

    '''senario #1'''
    # wp1 = np.array([[0,0], [10,10]])
    # wp2 = np.array([[0,10], [10,0]])

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)

    # UAVs = [uav1,uav2]



    '''test #1'''
    # wp1 = np.array([[0,0], [18,0], [20,10]])
    # wp2 = np.array([[0,10], [5,10], [10,10], [18,10], [20,0]])

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)

    # UAVs = [uav1,uav2]



    '''senario #2'''
    # wp1 = np.array([[0,0], [4,3], [6,4], [9,5], [11,6], [14,7], [15,7]])
    # wp2 = np.array([[0,9], [3,8], [7,5], [13,3], [15,2]])

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)

    # UAVs = [uav1,uav2]



    '''senario #3'''
    # wp1 = np.array([[0,0], [6,5], [8,8], [15,0]])
    # wp1 = multiple_insert(wp1,0)
    # uav1 = UAV(0,wp1)

    # # wp2 = np.array([[0,5], [4,1], [9,3], [15,5]])
    # wp2 = np.array([ [15,5], [9,3], [4,1], [0,5]])
    # wp2 = multiple_insert(wp2,1)
    # uav2 = UAV(1,wp2)
    
    # # wp3 = np.array([[0,9], [5,6], [6,0], [15,3]])
    # wp3 = np.array([ [15,3], [6,0], [5,6], [0,9]])
    # wp3 = multiple_insert(wp3,2)
    # uav3 = UAV(2,wp3)

    # UAVs = [uav1,uav2,uav3]


    '''senario #4'''
    # wp1 = np.array([[0,0], [2,0.5], [3,1], [5,4], [6,6], [8,7.5], [10,8], [12,8.2]])
    # wp1 = multiple_insert(wp1,0)
    # uav1 = UAV(0,wp1)

    # wp2 = np.array([[0,3], [4,2], [5,1.5], [6,1.5], [7,2], [8,3], [12,6]])
    # wp2 = multiple_insert(wp2,1)
    # uav2 = UAV(0,wp2)    
    
    # wp3 = np.array([[0,9], [4,6], [8,1], [10,0.5], [12,1]])
    # wp3 = multiple_insert(wp3,2)
    # uav3 = UAV(0,wp3)
        
    # wp4 = np.array([[0,6], [4,4], [10,2], [12,2]])
    # wp4 = multiple_insert(wp4,3)
    # uav4 = UAV(0,wp4)

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)
    # uav3 = UAV(3,wp3)
    # uav4 = UAV(4,wp4)

    # UAVs = [uav1,uav2,uav3,uav4]


    '''senario #5'''
    # wp1 = np.array([[0,0], [2,2], [4,4], [6,6]])
    # wp1 = multiple_insert(wp1,0)
    # uav1 = UAV(0,wp1)

    # wp2 = np.array([[3,6], [3,4], [3,2], [3,0]])
    # wp2 = multiple_insert(wp2,1)
    # uav2 = UAV(1,wp2)

    # wp3 = np.array([[0,6], [2,4], [4,2], [6,0]])
    # wp3 = multiple_insert(wp3,2)
    # uav3 = UAV(2,wp3)


    # UAVs = [uav1,uav2,uav3]


    '''senario #6'''
    wp1 = np.array([[0,0], [2,2], [4,4], [6,6]])
    wp1 = multiple_insert(wp1,0)
    uav1 = UAV(0,wp1)

    wp2 = np.array([[0,3], [3,2], [4,1], [6,0]])
    wp2 = multiple_insert(wp2,1)
    uav2 = UAV(1,wp2)

    wp3 = np.array([[0,8], [3,5], [6,2]])
    wp3 = multiple_insert(wp3,2)
    uav3 = UAV(2,wp3)


    UAVs = [uav1,uav2,uav3]



    ### visualize ###
    Traj=False
    Reldist=False
    Vel=False
    #################
    
    ### input parameters ###
    delt            = 0.01
    v_min           = 1
    v_max           = 10
    d_safe          = 3
    split_interval  = 1
    t_st            = np.array([[0],[0],[0]])
    avoidance       = True
    simul_arr       = True
    param_adjthold  = 1e-1
    param_adjratio  = 1
    param_stopcrit  = 1e-5

    SIM = Simulator(delt,UAVs,v_min,v_max,d_safe,split_interval,t_st,\
                    avoidance=True,simul_arr=True,viz=[Traj,Reldist,Vel],\
                    params=[param_adjthold,param_adjratio,param_stopcrit])
    SIM.run()
