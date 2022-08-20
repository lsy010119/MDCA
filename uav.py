import numpy as np



class UAV:

    def __init__(self, num, waypoints):


        self.num = num              # index of UAV

        self.wp = waypoints         # [wp1,wp2,...,wpn]
        self.N  = len(waypoints)    # number of the waypoints

        self.d = []                 # distance between wpi,wpi+1 
        self.t = []                 # time at wpi
        self.del_t = []             # time intervals between wpi,wpi+1
        self.v = []                 # velo city at wpi ~ wpi+1

        self.traj = None            # trajcetory of the uav

        ### distance calculation ###
        for k in range( self.N - 1 ):

            d_k = np.linalg.norm( self.wp[k+1] - self.wp[k])

            self.d = np.append( self.d, d_k )

            

    def calculate_trajec(self, delt):

        traj = np.zeros((2,1))

        for k in range(self.N - 1):

            wp_k1 = self.wp[k]
            wp_k2 = self.wp[k+1]

            del_xk = ( wp_k2[0] - wp_k1[0] ) / ( self.del_t[k] / delt )
            del_yk = ( wp_k2[1] - wp_k1[1] ) / ( self.del_t[k] / delt )


            if del_xk != 0 and del_yk == 0: # for case moving horizontally

                traj_xk = np.arange( wp_k1[0], wp_k2[0], del_xk ) 
                traj_yk = np.zeros(len(traj_xk)) + wp_k1[1]


            elif del_xk == 0 and del_yk != 0: # for case moving vertically

                traj_yk = np.arange( wp_k1[1], wp_k2[1], del_yk ) 
                traj_xk = np.zeros(len(traj_yk)) + wp_k1[0]


            else: # for other cases
                traj_xk = np.arange( wp_k1[0], wp_k2[0], del_xk ) 
                traj_yk = np.arange( wp_k1[1], wp_k2[1], del_yk ) 


            traj_k = np.vstack(( traj_xk, traj_yk ))

            traj = np.hstack((traj,traj_k))

        traj = traj[:,1:]

        self.traj = traj