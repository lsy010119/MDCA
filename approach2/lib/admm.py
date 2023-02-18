import numpy as np
from numpy import array,zeros,zeros_like,ones
from numpy.linalg import inv, norm

from copy import deepcopy
import time

import matplotlib.pyplot as plt

class ADMM:

    def __init__(self, UAVs, v_min, v_max, d_safe, N, N_c, t_st, c_set, params):

        ### UAVs ###
        self.UAVs   = UAVs

        ### Constraints ###
        self.v_min  = v_min
        self.v_max  = v_max
        self.d_safe = d_safe
        self.t_safe = d_safe/v_max

        ### Numbers ###
        self.K      = len(UAVs)
        self.N      = N
        self.N_c    = N_c

        ### Variables ###
        self.t      = zeros((self.N,1))
        self.s      = zeros((self.N-self.K,1))
        self.x      = zeros((self.N_c,1))
        self.lam    = 10*ones((self.N+self.N_c,1))
        self.p      = 1000

        ### Constants ###
        self.d      = zeros((self.N-self.K,1))
        self.c_set  = c_set
        self.t_st   = t_st

        ### Parameters ###
        self.param_adjthold     = params[0]
        self.param_adjratio     = params[1]
        self.param_stopcrit     = params[2]



        ### Coefficient Matrices ###

        # Cost function
        self.q      = zeros((self.N,1))
        self.Q      = zeros((int(self.K*(self.K-1)/2),self.N))
        self.rho    = 1000

        # Constraints 
        self.S_1    = zeros((self.N-self.K,self.N))
        self.S_2    = zeros((self.N_c,self.N))
        self.S_3    = zeros((self.K,self.N))

        # Augmented Lagrangian
        

        ### Initializing ###

        N_temp      = 0
        N_temp2     = 0
        N_temp3     = 0

        # t_init, q, d, S_1, S_3, Q
        for id,uav in enumerate(UAVs):

            # t_init
            dt_init = 2*uav.d/(self.v_max + self.v_min)
            ti_init = self.t_st[id]*np.ones(uav.N)+np.tril(np.ones(uav.N),k=-1)[:,:-1]@dt_init

            self.t[N_temp2:N_temp2+uav.N,0] = ti_init

            # q
            self.q[N_temp2+uav.N-1,0] = 1

            # d
            self.d[N_temp:N_temp+uav.N-1,0] = uav.d

            # S_1
            self.S_1[N_temp:N_temp+uav.N-1, N_temp2:N_temp2+uav.N-1] += np.diag(-np.ones(uav.N-1))
            self.S_1[N_temp:N_temp+uav.N-1, N_temp2+1:N_temp2+uav.N] += np.diag(np.ones(uav.N-1))

            # S_3
            self.S_3[id,N_temp2] = 1

            N_temp  += uav.N-1
            N_temp2 += uav.N
            N_temp4 =  0

            # Q
            for id2 in range(self.K-id-1):

                N_temp4 += self.UAVs[id+id2+1].N

                self.Q[N_temp3 + id2 ,N_temp2 - 1] = 1

                self.Q[N_temp3 + id2 ,N_temp2 + N_temp4 - 1] = -1

            N_temp3 += self.K - id - 1

        # S_2
        for idx, collision_pair in enumerate(self.c_set):
            
            N_temp5i = 0
            N_temp5j = 0

            id_i,id_j,idx_n,idx_m = collision_pair

            for i in range(id_i): N_temp5i += self.UAVs[i].N
            for j in range(id_j): N_temp5j += self.UAVs[j].N

            self.S_2[idx,N_temp5i+idx_n] = 1
            self.S_2[idx,N_temp5j+idx_m] = -1

        # s_init
        self.s = self.d/self.v_min - self.S_1@self.t

        # x_init
        self.x = self.S_2@self.t

        # P
        self.P = self.rho*self.Q.T @ self.Q + self.q@self.q.T

        # A
        self.A = np.block([[self.S_1],[self.S_2],[self.S_3]])

        # B
        self.B = np.block([[np.eye(self.N-self.K)],[np.zeros((self.N_c+self.K,self.N-self.K))]])
        
        # C
        self.C = np.block([[np.zeros((self.N-self.K,self.N_c))],[-np.eye(self.N_c)],[np.zeros((self.K,self.N_c))]])

        # a
        self.a = np.block([[self.d/self.v_min],[np.zeros((self.N_c,1))],[t_st]])


        ### Constant Matrices & Inverses ###

        # t update step : R & R^{-1}
        self.Rt = self.P + (self.p/2)*self.A.T@self.A
        self.Rt_inv = inv(self.Rt)

        # s update step : R & R^{-1}
        self.Rs = (self.p/2)*np.eye((self.N-self.K))
        self.Rs_inv = (2/self.p)*np.eye((self.N-self.K))

        self.S_1_inv = self.S_1.T@inv(self.S_1@self.S_1.T)

        # x update step : R & R^{-1}
        self.Rx = (self.p/2)*np.eye((self.N_c))
        self.Rx_inv = (2/self.p)*np.eye((self.N_c))



    def update_t(self):

        s,x,lam,A,B,C,a,p,q = self.s,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        r = (1/2)*( lam.T@A + p*( s.T@B.T@A + x.T@C.T@A - a.T@A ) ).T

        self.t = -self.Rt_inv@r

    
    def update_s(self):

        t,x,lam,A,B,C,a,p,q = self.t,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        r = (1/2)*( lam.T@B + p*( t.T@A.T@B + x.T@C.T@B - a.T@B ) ).T

        self.s = -self.Rs_inv@r

        self.proj_S()

    
    def update_x(self):

        t,s,lam,A,B,C,a,p,q = self.t,self.s,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        r = (1/2)*( lam.T@C + p*( t.T@A.T@C + s.T@B.T@C - a.T@C ) ).T

        self.x = -self.Rx_inv@r

        self.proj_D()


    def update_lam(self):

        t,s,x,lam,A,B,C,a,p,q = self.t,self.s,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        self.lam = lam + p*( A@t + B@s + C@x - a )


    def update_p(self,t_prev,s_prev,x_prev):

        t_curr,s_curr,x_curr,p_prev,A,B,C,a = self.t,self.s,self.x,self.p,self.A,self.B,self.C,self.a

        C_prev = A@t_prev + B@s_prev + C@x_prev - a
        C_curr = A@t_curr + B@s_curr + C@x_curr - a


        if 0 < norm(C_prev) - norm(C_curr) < self.param_stopcrit*norm(C_prev):

            self.p = p_prev*self.param_adjratio


        else:

            self.p = p_prev


    def proj_S(self):

        for i in range(self.N-self.K):

            s_i = self.s[i]

            if s_i < 0:

                self.s[i] = 0

            elif s_i > self.d[i]/self.v_min - self.d[i]/self.v_max:

                self.s[i] = self.d[i]/self.v_min - self.d[i]/self.v_max


    def proj_D(self):

        for i in range(self.N_c):

            x_i = self.x[i]

            idx_i, idx_j = np.where(self.S_2[i] == 1)[0][0], np.where(self.S_2[i] == -1)[0][0]

            if abs(x_i) < self.t_safe and x_i >= 0:

                self.x[i] = self.t_safe

            elif abs(x_i) < self.t_safe and x_i < 0:

                self.x[i] = -self.t_safe


    def cost_function(self):

        J = (self.t.T@(self.Q.T @ self.Q)@self.t + self.q.T@self.t)[0,0]

        return J


    def constraints(self):

        C = self.A@self.t + self.B@self.s + self.C@self.x - self.a

        return C


    def run(self,N_iter):

        J_list = np.zeros(N_iter)
        S1_list = np.zeros((self.N-self.K,N_iter))
        S2_list = np.zeros((self.N_c,N_iter))
        S3_list = np.zeros((self.K,N_iter))

        s_list = np.zeros((self.N - self.K,N_iter))
        x_list = np.zeros((self.N_c,N_iter))

        for i in range(N_iter):

            t_prev   = deepcopy(self.t)
            s_prev   = deepcopy(self.s)
            x_prev   = deepcopy(self.x)
            lam_prev = deepcopy(self.lam)
            J_prev   = self.cost_function()


            self.update_t()
            self.update_x()
            self.update_s()
            self.update_lam()
            self.update_p(t_prev,s_prev,x_prev)

            J = self.cost_function()
            C = self.constraints()

            J_list[i]       = J
            S1_list[:,i]    = C[:self.N-self.K,0]
            S2_list[:,i]    = C[self.N-self.K:self.N-self.K + self.N_c,0]
            S3_list[:,i]    = C[-self.K:,0]
            s_list[:,i]     = self.s[:,0]
            x_list[:,i]     = self.x[:,0]

            if 0 < norm(J_prev) - norm(J) < self.param_stopcrit*norm(J_prev):

                print("converged")

                break


        J_list = J_list[:i]
        S1_list = S1_list[:,:i]
        S2_list = S2_list[:,:i]
        S3_list = S3_list[:,:i]
        s_list = s_list[:,:i]
        x_list = x_list[:,:i]
        

        print((self.t.T@(self.Q.T @ self.Q)@self.t + self.q.T@self.t)[0,0])
        print(self.Q@self.t)
        print(self.S_1@self.t)
        print(self.S_2@self.t)
        print(self.t)

        # fig1 = plt.figure(figsize=(30,10))
        # fig2 = plt.figure(figsize=(15,5))
        # fig3 = plt.figure(figsize=(15,5))
        
        # costplot = fig1.add_subplot(1,1,1)
        # constplot1 = fig2.add_subplot(1,1,1)
        # constplot2 = fig3.add_subplot(1,1,1)

        # costplot.plot(np.arange(i),J_list,label=r'Cost : $\sum^{K}_{i=1}t^{(i)}_{N^{(i)}}+\rho \sum^{K}_{i,j}\sum^{K}_{i\neq j} (t^{(i)}_{N^{(i)}} - t^{(j)}_{N^{(j)}} )^2$')
        # costplot.set_xlim(0,i)
        # costplot.set_xlabel('Iterations')

        # constplot1.plot(np.arange(i),S1_list.T,color='red')
        # constplot1.plot(0,0,color='red',label=r"Equality Constraints #1 : $S_1t+s-d/V_{\min}$")
        # constplot1.plot(np.arange(i),S2_list.T,color='green')
        # constplot1.plot(0,0,color='green',label=r"Equality Constraints #2 : $S_2t-x$")
        # constplot1.plot(np.arange(i),S3_list.T,color='blue')
        # constplot1.plot(0,0,color='blue',label=r"Equality Constraints #3 : $S_3t-t_{\rm safety}$")
        # constplot1.set_xlim(0,i)
        # constplot1.set_ylim(-0.6,0.6)
        # constplot1.set_xlabel('Iterations')
        
        # constplot2.plot(np.arange(i),x_list.T,label=r'$t^{(i)}_j - t^{(j)}_i$')
        # constplot2.hlines(self.t_safe,0,i,linestyles='--',colors='blue',linewidth=1,label=r'$t_{\rm safety}$')
        # constplot2.hlines(-self.t_safe,0,i,linestyles='--',colors='blue',linewidth=1)
        # constplot2.axhspan(self.t_safe, 0.6, xmin=0, xmax=i, alpha=0.06, color='blue')
        # constplot2.axhspan(-self.t_safe, -0.6, xmin=0, xmax=i, alpha=0.06, color='blue')
        # constplot2.text(i/2,self.t_safe+0.1,r"Feasible Area $D$",fontsize=15, alpha=0.1,color='blue')
        # constplot2.set_xlim(0,i)
        # constplot2.set_ylim(-0.6,0.6)
        # constplot2.set_xlabel('Iterations')
        # constplot2.set_ylabel('Time differences')

        # costplot.legend()
        # plt.show()



        return self.t
        

