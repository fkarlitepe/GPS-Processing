# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:28:35 2023

@author: Furkan
"""

import pandas as pd
import numpy as np
#########################################l1, l2 ve l5 sinyalle,nin dalga boylarının ayarlanaması
#https://en.wikipedia.org/wiki/GPS_signals
c=299792458# Metre/sn
l1=1575420000 # Sinyal Hertz   
l2=1227600000 # Sinyal Hertz
l5=1176450000 # Sinyal Hertz
lamda_l1=c/l1 #l1 dalga boyu
lamda_l2=c/l2 #l2 dalga boyu
lamda_l5=c/l5 #l5 dalga boyu
lamda_IF=77*lamda_l1/(77**2 - 60**2) #l1/l2 PC ve LC Ionophere free combination wavelength
lamda_WL=c/(l1-l2) #wide-lane dalga boyu
lamda_NL=c/(l1+l2) #narrow-lane dalga boyu

def process_gps_file(rinex_file_path):
    gps_data = pd.read_table(rinex_file_path, header=None)
    k = int(np.where(gps_data =='                                                            END OF HEADER')[0][0])
    
    xyz0_gps = gps_data.iloc[:k]
    xyz0_gps.columns = ["bilgi"]
    xyz01 = xyz0_gps['bilgi'].str.endswith('APPROX POSITION XYZ', na=False)
    aprx = xyz0_gps[xyz01]
    aprx = aprx.iloc[0][0][2:42].split('  ')
    xyz0_gps = np.array(pd.DataFrame(aprx)).astype('float')

    gps_1 = gps_data.iloc[k+1:]
    
    mask = gps_1[0].str.startswith('G') # g ile başlayan satırları filtreledik ve alt satırdaki kod ile topladık
    obs2 = gps_1[mask]
    obs2_gps = obs2[0].str.replace('G', '', 1)
    obs3_gps=pd.DataFrame([])
    obs3_gps['gps'] = obs2_gps.apply(lambda x: x[0:2])
    obs3_gps['c1c'] = obs2_gps.apply(lambda x: x[4:16])
    obs3_gps['l1c'] = obs2_gps.apply(lambda x: x[19:32])
    obs3_gps['c1w'] = obs2_gps.apply(lambda x: x[52:64])
    obs3_gps['l1w'] = obs2_gps.apply(lambda x: x[67:80])
    obs3_gps['c2x'] = obs2_gps.apply(lambda x: x[100:112])
    obs3_gps['l2x'] = obs2_gps.apply(lambda x: x[115:128])
    obs3_gps['c2w'] = obs2_gps.apply(lambda x: x[148:160])
    obs3_gps['l2w'] = obs2_gps.apply(lambda x: x[163:176])
    obs3_gps['c5x'] = obs2_gps.apply(lambda x: x[196:208])
    obs3_gps['l5x'] = obs2_gps.apply(lambda x: x[211:224])
    obs3_gps['c1x'] = obs2_gps.apply(lambda x: x[244:256])
    obs3_gps['l1x'] = obs2_gps.apply(lambda x: x[259:272])
    obs3_gps = obs3_gps.iloc[:].replace('             ', '0',)
    obs3_gps = obs3_gps.iloc[:].replace('            ', '0',)
    obs3_gps = obs3_gps.iloc[:].replace('', '0',)
    obs3_gps=obs3_gps.astype('float')
    
    obs3_gps['l1c'] = obs3_gps['l1c'] * lamda_l1
    obs3_gps['l1w'] = obs3_gps['l1w'] * lamda_l1
    obs3_gps['l1x'] = obs3_gps['l1x'] * lamda_l1

    obs3_gps['l2x'] = obs3_gps['l2x'] * lamda_l2
    obs3_gps['l2w'] = obs3_gps['l2w'] * lamda_l2
    
    obs3_gps['l5x'] = obs3_gps['l5x'] * lamda_l5
#ionophere free combination
    alfa_IF = (l1**2) / (l1**2 - l2**2)
    beta_IF = -(l2**2) / (l1**2 - l2**2)
    Pc = obs3_gps['c1w'] * alfa_IF + obs3_gps['c2w'] * beta_IF
    Lc = obs3_gps['l1w'] * alfa_IF + obs3_gps['l2w'] * beta_IF
    obs_ion_free_gps = np.c_[Pc, Lc, obs3_gps['gps']]
#observation time
    mask = gps_1[0].str.startswith('> 20', na=False)
    epok_time = gps_1[mask]
    epok_time = epok_time[0].str.split(' ', expand=True)
    epok_time[6][epok_time[6]=='']=epok_time[7][epok_time[7]=='0.0000000']
    epok_time = epok_time[4].astype('float') * 60 * 60 + epok_time[5].astype('float') * 60 + epok_time[6].astype('float')

    epok = gps_1[mask].index
    say = np.array([])
    for i in range(len(epok)-1):
        gp1 = gps_1.loc[epok[i]:epok[i+1]]
        gp2 = gp1[gp1[0].str.startswith('G')].shape[0]
        say = np.append(say, gp2)
    gp3 = gps_1.loc[epok[-1]:]
    gp4 = gp3[gp3[0].str.startswith('G')].shape[0]
    say = np.r_[say, gp4]
    say = np.c_[say, epok_time]

    yeni = np.zeros((len(obs_ion_free_gps), 4))
    start = 0
    for idx, i in enumerate(say):
        yeni[start:start + int(i[0]), :3] = obs_ion_free_gps[start:start + int(i[0])]
        yeni[start:start + int(i[0]), 3] = np.ones(int(i[0])) * int(i[1])
        start = start + int(i[0])
    obs_ion_free_gps = pd.DataFrame(yeni)

#melbourne wübbena combination
    alfa_WL = l1 / (l1 - l2)
    beta_WL = -l2 / (l1 - l2)
    alfa_NL = l1 / (l1 + l2)
    beta_NL = l2 / (l1 + l2)

    L4 = np.array([])
    P6 = np.array([])
    A4 = np.array([])

    A4 = (obs3_gps['l1w'] * alfa_WL + obs3_gps['l2w'] * beta_WL) - (obs3_gps['c1w'] * alfa_NL + obs3_gps['c2w'] * beta_NL)
    mw_ion_gps = np.c_[obs_ion_free_gps, A4]
    mw_ion_gps = pd.DataFrame(mw_ion_gps, columns=['Pc', 'Lc', 'sat', 'epok', 'A4'])
    mw_ion_gps.to_excel('mw_ion_gps.xlsx')
    obs3_gps.to_excel('obs3_gps.xlsx')
    pd.DataFrame(xyz0_gps).to_excel('xyz0_apprx.xlsx')

    return mw_ion_gps

