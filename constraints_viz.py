#!/usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cvxpy as cp
import time
# matplotlib.rcParams['text.usetex'] = True

from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.axes import Axes


x_space = np.linspace(-100,100,50)
y_space = np.linspace(-100,100,50)

x,y = np.meshgrid(x_space,y_space)

t_safety = 30

# z = np.clip( z_lim - np.abs(x-y) ,None,0) + z_lim

d2 = 30


def t_c(d_c,d,t_1,t_2):

    return (d_c/d)*(t_2-t_1)+t_1


def f(t_c1,t_c2):

    return np.clip( t_safety - np.abs(t_c1-t_c2) ,None,0)

def f1(x,y):

    return np.abs(x-y)

def f2(x,y):

    k = 0.1
    return (2/k)*np.log(1+np.exp(k*(x-y))) - (x-y) - (2/k)*np.log(2)
    

t1_1 = x
t1_2 = y

t2_1 = x
t2_2 = y

d1_c = 10
d1 = 20

d2_c = 20

t1_c = t_c(d1_c,d1,t1_1,t1_2)
t2_c = t_c(d2_c,d2,t2_1,t2_2)

abs_error = f(t1_c,t2_c)

###############################

fig = plt.figure("constraints")
fig.suptitle("Constraints")
const3d = fig.add_subplot(1,1,1,projection='3d')
const3d.set_title(r"$\bf figure 1$  Constraints in 3-D")
# constymx = fig.add_subplot(1,3,2)
# constymx.set_title(r"$\bf figure 2$  Constraints projected to $t^{(i)}_c = - t^{(j)}_c$ plane")
# constcontour = fig.add_subplot(1,2,2)
# constcontour.set_title(r"$\bf figure 2$  Constraints in 2-D")


# const3d.plot_surface(x,y,f(x,y),cmap="plasma",rstride=3, cstride=3,edgecolors="black",lw=0.1,alpha=1)
const3d.plot_surface(x,y,f2(x,y),cmap="plasma",rstride=1, cstride=1,edgecolors="black",lw=0.1,alpha=0.5)
const3d.plot_surface(x,y,f1(x,y),cmap="plasma",rstride=1, cstride=1,edgecolors="black",lw=0.1,alpha=0.5)
const3d.set_xlabel(r"$t^{(i)}_c$",fontsize=10)
const3d.set_ylabel(r"$t^{(j)}_c$",fontsize=10)
const3d.set_zlabel(r"$t_{safety}-|t^{(i)}_c - t^{(j)}_c|$",fontsize=10,rotation=90)
const3d.tick_params(labelsize=5)

const3d.set_xlim(-100,100)
const3d.set_ylim(-100,100)
const3d.set_zlim(0,200)


# constymx.plot(x_space,f(x_space,-x_space),'k-')
# constymx.set_xlabel(r"$t^{(i)}_c = - t^{(j)}_c$",fontsize=10)
# constymx.set_ylabel(r"$t_{safety}-|t^{(i)}_c - t^{(j)}_c|$",fontsize=10)


# constcontour.contour(x,y,f(x,y),levels=100,cmap="plasma")
# constcontour.set_xlim(-100,100)
# constcontour.set_ylim(-100,100)
# constcontour.set_xlabel(r"$t^{(i)}_c$",fontsize=10)
# constcontour.set_ylabel(r"$t^{(j)}_c$",fontsize=10)

#############################

# fig = plt.figure("constraints")
# fig.suptitle("Constraints")
# const3d = fig.add_subplot(2,1,1,projection='3d')
# const3d.set_title(r"$\bf figure 1$  Constraints in 3-D")
# constcontour = fig.add_subplot(2,1,2)
# constcontour.set_title(r"$\bf figure 3$  Constraints in 2-D")


# const3d.plot_surface(t1_c,t2_c,abs_error,cmap="plasma",rstride=3, cstride=3,edgecolors="black",lw=0.1,alpha=1)
# const3d.set_xlabel(r"$t^{(i)}_c$",fontsize=10)
# const3d.set_ylabel(r"$t^{(j)}_c$",fontsize=10)
# const3d.set_zlabel(r"$t_{safety}-|t^{(i)}_c - t^{(j)}_c|$",fontsize=10,rotation=90)
# const3d.tick_params(labelsize=5)

# constcontour.contour(t1_c,t2_c,abs_error,levels=100,cmap="plasma")
# constcontour.set_xlim(-100,100)
# constcontour.set_ylim(-100,100)
# constcontour.set_xlabel(r"$t^{(i)}_c$",fontsize=10)
# constcontour.set_ylabel(r"$t^{(j)}_c$",fontsize=10)

###############################


plt.show()