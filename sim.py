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

class Simulator:

    def __init__(self, delt, UAVs, visuallize=True):

        