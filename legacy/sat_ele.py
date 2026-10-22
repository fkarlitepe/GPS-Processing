# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:24:15 2023

@author: Furkan
"""

import numpy as np
import pandas as pd



def calculate_Cll(result_gps, sat3_df):
    xyz0_apprx = pd.read_excel('xyz0_apprx.xlsx')
    xyz0_apprx=xyz0_apprx.drop(xyz0_apprx.columns[0], axis=1)
    
    Cll = np.eye(36)  # *1e-3 #öncül varyans ayarlanması
    Cll[:3, :3] = Cll[:3, :3] * (1e+2) ** 2  # position initial varyans-covaryans
    Cll[3:4, 3:4] = Cll[3:4, 3:4] * (1e+5) ** 2  # receiver clock initial varyans-covaryans
    Cll[4:, 4:] = Cll[4:, 4:] * (2e+1) ** 2  # integer ambiguity initial varyans-covaryans

    zenit_1 = np.array([])

    for j in result_gps['epok'].unique():
        sat3 = sat3_df[sat3_df['epok'] == j]
        a = 6378137  # GRS80 elipsoid büyük yarı eksen
        b = 6356752.314140347  # GRS80 elipsoid küçük yarı eksen
        nxyz = np.array([xyz0_apprx.iloc[0] / (a ** 2), xyz0_apprx.iloc[1] / (a ** 2), xyz0_apprx.iloc[2] / (b ** 2)]).astype('float')
        nxyz_sqrt = np.sqrt(np.dot(nxyz.T, nxyz))

        zenit = np.array([])
        
        #zenit açısına göre 10 derece 170 derece aralığını al
        for i in range(len(sat3)):
            D_xyz = np.array([sat3.iloc[i, 3:6]]) - xyz0_apprx.values.T
            D_xyz_sqrt = np.sqrt(np.dot(D_xyz, D_xyz.T))
            comp = np.degrees(
                np.arccos(np.dot(D_xyz, nxyz) / (D_xyz_sqrt * nxyz_sqrt)))
            zenit = np.append(zenit, comp)

        zenit = 90 - zenit

        for i in range(len(sat3)):
            if 10 <= zenit[i] <= 170:
                zenit_1 = np.append(zenit_1, zenit[i])

    s0 = ((0.003 ** 2) / np.sin(np.deg2rad(zenit_1)))
    s0 = s0.min()

    Cll = Cll / s0
    
    cll=pd.DataFrame(Cll)
    cll=cll.to_excel('cll.xlsx')

    return Cll

