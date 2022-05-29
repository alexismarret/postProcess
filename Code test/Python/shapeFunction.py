#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:37:49 2022

@author: alexis
"""

#--------------------------------------------------------------
def shapeFunction(self, kind, pos):

    if kind=="cubic":

        #need position of macroparticles
        if   self.ndim==1:
            # gi  = pos/self.meshsize

            # dgi = gi - int(gi)

            relx = np.abs(pos[...,None] - self.getAxis("x")[None,...])

            cond1X =  (relx <= self.meshsize[0])

            cond2X =  (relx >  self.meshsize[0]) & (relx <= 2*self.meshsize[0])

            S1 = 2/3 * (1 - 3/2*(relx/self.meshsize[0])**2 +
                            3/4*(relx/self.meshsize[0])**3)

        elif self.ndim==2:
            # gi = pos[0]/self.meshsize[0]
            # gj = pos[1]/self.meshsize[1]

            # dgi = gi - int(gi)
            # dgj = gj - int(gj)

            relx = np.abs(pos[0][...,None] - self.getAxis("x")[None,...])
            rely = np.abs(pos[1][...,None] - self.getAxis("y")[None,...])

            cond1X = (relx <= self.meshsize[0])
            cond1Y = (rely <= self.meshsize[1])

            cond2X =  (relx >  self.meshsize[0]) & (relx <= 2*self.meshsize[0])
            cond2Y =  (rely >  self.meshsize[1]) & (rely <= 2*self.meshsize[1])

        elif self.ndim==3:
            gi = pos[0]/self.meshsize[0]
            gj = pos[1]/self.meshsize[1]
            gk = pos[2]/self.meshsize[2]

            dgi = gi - int(gi)
            dgj = gj - int(gj)
            dgk = gk - int(gk)

            relx = np.abs(pos[0][...,None] - self.getAxis("x")[None,...])
            rely = np.abs(pos[1][...,None] - self.getAxis("y")[None,...])
            relz = np.abs(pos[2][...,None] - self.getAxis("z")[None,...])

            cond1X = (relx <= self.meshsize[0])
            cond1Y = (rely <= self.meshsize[1])
            cond1Z = (relz <= self.meshsize[2])

            cond2X =  (relx >  self.meshsize[0]) & (relx <= 2*self.meshsize[0])
            cond2Y =  (rely >  self.meshsize[1]) & (rely <= 2*self.meshsize[1])
            cond2Z =  (relz >  self.meshsize[2]) & (relz <= 2*self.meshsize[2])


    return
