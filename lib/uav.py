import numpy as np






class UAV:

    def __init__(self, k, waypoints):


        self.k = k                  # k'th UAV

        self.wp = waypoints         # [wp1,wp2,...,wpn]
        self.n  = len(waypoints)  # number of the waypoints

        self.d = []                 # distance between wpi,wpi+1 
        self.t = []                 # time at wpi
        self.v = []                 # velo city at wpi ~ wpi+1

        self.traj = None            # trajcetory of the uav

        ### distance calculation ###
        for i in range( self.n - 1 ):

            d_i = np.linalg.norm( self.wp[i+1] - self.wp[i])

            self.d = np.append( self.d, d_i )

            
        