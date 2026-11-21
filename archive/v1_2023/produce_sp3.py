# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:30:54 2023

@author: Furkan

çalışma gününe ait SP3 uzantılı dosyayı bir bün önceki ve bir gün sonrakinide ekleyerek 
dosya yoluna koy. Örneğin 327. gün için çalışıyorsan 326. gün ve 328. güne ait SP3 dosya
ları bulundur. Başka dosya olmasın. SP3 büyük harfle yazılacak.
"""

#SP3 dosyları alma işlemi
import numpy as np
import pandas as pd
import os

# process_sp3_file fonksiyonu ile sp3 içini uygun hale getiriyoruz.
def process_sp3_file(file_path):
    sp3 = pd.read_csv(file_path, header=None)
    sp3_time = sp3[0].str.startswith('*  202')
    sp3_time = sp3[sp3_time]
    sp3_time1 = int(sp3_time.iloc[1][0][16:20]) * 60
    sp3_1 = sp3[0].str.startswith('PG')
    sp3_1 = sp3[sp3_1]
    
    for i in range(len(sp3_1)):    
        sp3_1.iloc[i] = sp3_1.iloc[i].str.replace('     ', ' ')
        sp3_1.iloc[i] = sp3_1.iloc[i].str.replace('    ', ' ')
        sp3_1.iloc[i] = sp3_1.iloc[i].str.replace('   ', ' ')
        sp3_1.iloc[i] = sp3_1.iloc[i].str.replace('  ', ' ')
        sp3_1.iloc[i] = sp3_1.iloc[i].str.replace('PG', '')

    sp3_2 = sp3_1[0].str.split(' ', expand=True, n=4)
    sp3_2_ = sp3_2[4].str.split(' ', expand=True)
    sp3_2[4] = sp3_2_[0]
    sp3_2.columns = ['Sat', 'x', 'y', 'z', 't']
    sp3 = sp3_2.astype(float)

    sp3_time_ = np.array([])
    for i in range(len(sp3_time)):
        sp3_time_ = np.append(sp3_time_,
                              int(sp3_time[0].iloc[i][14:16]) * 60 * 60 + int(
                                  sp3_time[0].iloc[i][17:20]) * 60 + float(
                                  sp3_time[0].iloc[i][21:25]))

    sp3 = np.c_[sp3, np.zeros((len(sp3), 1))]

    start = np.arange(0, len(sp3), 32)
    for i in range(len(start) - 1):
        sp3[start[i]:start[i + 1], 5] = sp3_time_[i]

    sp3 = pd.DataFrame(sp3, columns=['sat', 'x', 'y', 'z', 'time', 'epok'])
    sp3.iloc[3040:, 5] = 85500

    return sp3


#sp3 dosyaları ölçüm günü, önceki gün ve sonraki gün birleştiriliyor.
def combine_sp3_data(sp3_before_path, sp3_day_path, sp3_after_path):
    sp3_before = process_sp3_file(sp3_before_path)
    sp3_day = process_sp3_file(sp3_day_path)
    sp3_after = process_sp3_file(sp3_after_path)

    sp3_combined = pd.concat([sp3_before.iloc[-32 * 6:], sp3_day, sp3_after[:32 * 6]], ignore_index=True)

    sp3_combined.iloc[0:32, 5] = -5400
    sp3_combined.iloc[32:64, 5] = -4500
    sp3_combined.iloc[64:96, 5] = -3600
    sp3_combined.iloc[96:128, 5] = -2700
    sp3_combined.iloc[128:160, 5] = -1800
    sp3_combined.iloc[160:192, 5] = -900

    sp3_combined.iloc[3264:3296, 5] = 85500 + 900
    sp3_combined.iloc[3296:3328, 5] = 85500 + 900 + 900
    sp3_combined.iloc[3328:3360, 5] = 85500 + 900 + 900 + 900
    sp3_combined.iloc[3360:3392, 5] = 85500 + 900 + 900 + 900 + 900
    sp3_combined.iloc[3392:3424, 5] = 85500 + 900 + 900 + 900 + 900 + 900
    sp3_combined.iloc[3424:3456, 5] = 85500 + 900 + 900 + 900 + 900 + 900 + 900

    return sp3_combined

