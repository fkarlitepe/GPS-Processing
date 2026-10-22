# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:01:20 2023

@author: Furkan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

########################################################################
########################################################################
########################################################################
#SP3 dosyları alma işlemi
from produce_sp3 import combine_sp3_data #produce_sp3.py dosyasından combine_sp3_data fonksiyonunu çağırdık

dosya_yolu = os.getcwd() #çalışılan dosya yolunu bulduk.
try:
    # Dosya yolundaki .sp3 uzantılı dosyaları listele
    sp3_dosya_listesi = [dosya for dosya in os.listdir(dosya_yolu) if dosya.endswith(".SP3")]
    # Sonuçları yazdır
    print(".sp3 Uzantılı Dosyalar:")
    for sp3_dosya in sp3_dosya_listesi:
        print(sp3_dosya)
except OSError as e:
    print(f".SP3 uzantılı Dosya Bulunamadı: {e}")
    
sp3_before_path = sp3_dosya_listesi[0]
sp3_day_path = sp3_dosya_listesi[1]
sp3_after_path = sp3_dosya_listesi[2]

combined_sp3_data = combine_sp3_data(sp3_before_path, sp3_day_path, sp3_after_path)
combined_sp3_data.to_excel('combine_sp3.xlsx')
########################################################################
########################################################################
########################################################################
from observation_rinex3 import process_gps_file
# Usage example:
file_path = 'MADR00ESP_R_20233270000_01D_30S_MO.RNX'
result_gps = process_gps_file(file_path)

########################################################################
########################################################################
########################################################################
from sat_regression import process_data
result_gps, sat3_df = process_data(combined_sp3_data, result_gps)
########################################################################
########################################################################
########################################################################
from sat_ele import calculate_Cll
cll_result = calculate_Cll(result_gps, sat3_df)
########################################################################
########################################################################
########################################################################
from PPP_process import process_result_gps
xyz_result, n3_amb, dx_1 = process_result_gps(result_gps, sat3_df, cll_result)

result_gps.to_excel('result_gps.xlsx')
cll_result.to_excel('cll_result.xlsx')
result_gps.to_excel('sat3_df.xlsx')

dx_1.to_excel('dx_1.xlsx')
n3_amb.to_excel('n3_amb.xlsx')
xyz_result.to_excel('xyz_result.xlsx')
########################################################################
########################################################################
########################################################################