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
from matplotlib.animation import FuncAnimation


class Simulator:
    '''
    Notations

    K   : total number of uavs
    i,j : index of two arbitrary uav

    N : total number of waypoints of each uav
    k : index of an waypoint

    '''

    def __init__(self, delt, UAVs, v_min, v_max, d_safe, viz = True):
        
        self.delt = delt

        self.v_min = v_min

        self.v_max = v_max

        self.d_safe = d_safe
        
        self.mdca = MDCA(UAVs,v_min,v_max,d_safe)

        self.UAVs = UAVs # list of UAVs

        self.viz = viz




    def visuallize(self):
        
        fig1 = plt.figure("Trajectory")
        # fig2 = plt.figure("Relative Distances")
        # fig3 = plt.figure("Velocity Data")

        ''' Trajectory Plot '''

        traj = fig1.add_subplot(1,1,1)

        # traj.set_title(r"$\bf figure 1$  UAV trajectory")
        traj.set_xlabel(r"$\bf x(m)$",fontsize=10)
        traj.set_ylabel(r"$\bf y(m)$",fontsize=10)
        traj.set_xlim(-1,11)
        traj.set_ylim(-1,11)



        t_set = []
        uav_pose_xmax = []
        uav_pose_ymax = []
        uav_pose_xmin = []
        uav_pose_ymin = []

        for uav in self.UAVs:

            t_set.append(len(uav.traj[0]))
            uav_pose_xmax.append(np.max(uav.traj[0]))
            uav_pose_ymax.append(np.max(uav.traj[1]))
            uav_pose_xmin.append(np.min(uav.traj[0]))
            uav_pose_ymin.append(np.min(uav.traj[1]))

        total_timesteps = max(t_set)
        traj.set_xlim(min(uav_pose_xmin),max(uav_pose_xmax))
        traj.set_ylim(min(uav_pose_ymin),max(uav_pose_ymax))
        traj.axis('equal')


        for uav in self.UAVs:

            waiting_time = total_timesteps - len(uav.traj[0])

            traj_expanded    = np.zeros((2,total_timesteps))

            traj_expanded[0] = np.hstack((uav.traj[0],uav.traj[0,-1]*np.ones(waiting_time)))
            traj_expanded[1] = np.hstack((uav.traj[1],uav.traj[1,-1]*np.ones(waiting_time)))

            uav.traj = traj_expanded
            
            traj.plot(uav.traj[0],uav.traj[1],'--',label=r'$UAV^{(%d)}$'%(uav.num))

        uav_position = traj.plot([],[],'k*',markersize=10)[0]

        def update(timestep):

            traj_x = []
            traj_y = []

            for uav in self.UAVs:

                traj_x.append(uav.traj[0,timestep])
                traj_y.append(uav.traj[1,timestep])

            uav_position.set_data(traj_x,traj_y)
        
        sim = FuncAnimation(fig=fig1, func=update, frames=total_timesteps, interval=0.1) 
        
        traj.legend()
        plt.show()



    def run(self):

        self.mdca.run(avoidance=True)

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


    '''senario #2'''
    # wp1 = np.array([[0,0], [4,0], [4,5], [10,5]])
    # wp2 = np.array([[0,6], [3,6], [3,1], [10,1]])

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)

    # UAVs = [uav1,uav2]


    '''senario #3'''
    wp1 = np.array([[0,0], [15,15]])
    wp2 = np.array([[0,3], [15,3]])
    wp3 = np.array([[0,6], [15,6]])
    wp4 = np.array([[0,9], [15,9]])
    wp5 = np.array([[0,12], [15,12]])

    uav1 = UAV(1,wp1)
    uav2 = UAV(2,wp2)
    uav3 = UAV(3,wp3)
    uav4 = UAV(3,wp4)
    uav5 = UAV(3,wp5)

    UAVs = [uav1,uav2,uav3,uav4,uav5]


    '''senario #4'''
    # wp1 = np.array([[0,0], [6,5], [8,8], [15,0]])
    # wp2 = np.array([[0,5], [4,1], [9,3], [15,5]])
    # wp3 = np.array([[0,9], [5,6], [6,0], [15,3]])

    # uav1 = UAV(1,wp1)
    # uav2 = UAV(2,wp2)
    # uav3 = UAV(3,wp3)
    # uav4 = UAV(3,wp3)

    # UAVs = [uav1,uav2,uav3]
    

    SIM = Simulator(0.01,UAVs,5,10,2)
    SIM.run()