import numpy as np

'''
The methods here will be used to solve the 1D burgers equation, with optional
phase uncertainty for sinusoidal initial data. See Barth and Sukys 2018

d_t u + d_x u^2/2 = 0
u(x,0) = Asin(\phi (x + \omega))

'''

