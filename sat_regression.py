import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import lagrange

#regression_data fonksiyonunda önce dt_sat zamanı predicte edilir ardından uydu xyz koordinataları predicte edilir.
def regression_data(combined_sp3_data, result_gps, c=299792458):

    ###uydu zamanını tahminleme için kullanılan fonksiyon
    def predict_dt_sat(result_gps, combined_sp3_data):
        dt_sat_list = []

        for j in result_gps['epok'].unique():
            observation = result_gps[result_gps['epok'] == j]
            kk7 = observation['sat'].astype(int).tolist()

            for jj in kk7:
                sat1 = combined_sp3_data[combined_sp3_data['sat'] == jj]

                lin_reg = LinearRegression()
                lin_reg.fit(sat1[['epok']], sat1[['epok']])
                sat2_epok = lin_reg.predict([[j]])

                rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
                rf_reg.fit(sat1[['epok']], sat1[['time']].values.ravel())
                sat2_t = rf_reg.predict([[j]])

                sat2 = pd.DataFrame({'epok': [sat2_epok[0][0]], 'sat': [jj], 'dt_sat': [sat2_t[0]]})
                dt_sat_list.append(sat2)

        dt_sat = pd.concat(dt_sat_list, ignore_index=True)
        dt_sat['epok'] = dt_sat['epok'].round()

        return dt_sat

    dt_sat = predict_dt_sat(result_gps, combined_sp3_data)#fonksiyonu çalıştırma
    result_gps['dt_sat'] = dt_sat['dt_sat']#uydu zamanını result_gps dosyasına ekleme
    
   
    obs3_gps = pd.read_excel('obs3_gps.xlsx') #gözlem dosyasını çağırıyoruz. 
    C1 = obs3_gps['c1c'].reset_index(drop=True)# sinyal seyahat süresi hesaplıyoruz. Epok değeri- sinyal seyahet süresi yapılacak
    t_ems = result_gps['epok'] - ((C1 / c) + result_gps['dt_sat'] * 10 ** (-6))
    result_gps['t_ems'] = t_ems


    #Lagrange ile uydu xyz koordinatlarını tahminleme için kullanılan fonksiyon
    
    def lagrange_rf_prediction(result_gps, combined_sp3_data):
        sat3_list = []

        for j in result_gps['epok'].unique():
            observation = result_gps[result_gps['epok'] == j]
            kk7 = observation['sat'].astype(int)
            kk8 = observation['t_ems']

            for jj, tem in zip(kk7, kk8):
                sat1 = combined_sp3_data[combined_sp3_data['sat'] == jj]
                sat1 = sat1.reset_index(drop=True)
                uu = sat1[sat1['epok'] >= j].index[0]

                sc1 = StandardScaler()
                x_olcekli = sc1.fit_transform(sat1[['x']])
                sc2 = StandardScaler()
                y_olcekli = sc2.fit_transform(sat1[['y']])
                sc3 = StandardScaler()
                z_olcekli = sc3.fit_transform(sat1[['z']])

                sat1_scaler = pd.DataFrame({'x': x_olcekli.flatten(), 'y': y_olcekli.flatten(), 'z': z_olcekli.flatten()})

                lin_reg = LinearRegression()
                lin_reg.fit(sat1[['epok']], sat1[['epok']])
                sat2_epok = lin_reg.predict([[tem]])

                sat2_coords = {}
                for col in ['x', 'y', 'z']:
                    x_values = sat1.loc[uu - 4:uu + 5, 'epok']
                    y_values = sat1_scaler.loc[uu - 4:uu + 5, col]
                    xi, yi = x_values.values, y_values.values
                    x_values_filtered = xi[xi != xi[0]]
                    lag_poly = lagrange(x_values_filtered, yi[xi != xi[0]])
                    sat2_coords[col] = lag_poly(tem)

                sat2_coords['x'] = sc1.inverse_transform(np.array([sat2_coords['x']]).reshape(-1, 1)) * 1000
                sat2_coords['y'] = sc2.inverse_transform(np.array([sat2_coords['y']]).reshape(-1, 1)) * 1000
                sat2_coords['z'] = sc3.inverse_transform(np.array([sat2_coords['z']]).reshape(-1, 1)) * 1000

                rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
                rf_reg.fit(sat1[['epok']], sat1['time'])
                sat2_t = rf_reg.predict([[tem]])

                sat2 = pd.DataFrame({'epok': [j], 't_ems': [sat2_epok[0][0]], 'sat': [jj], 'x': [sat2_coords['x'][0][0]],
                                     'y': [sat2_coords['y'][0][0]], 'z': [sat2_coords['z'][0][0]], 'time': [sat2_t[0]]})

                sat3_list.append(sat2)

        return sat3_list

    #Lagrange işlemi yapan fonksiyonu çalıştırılır.
    sat3_df = pd.concat(lagrange_rf_prediction(result_gps, combined_sp3_data), ignore_index=True)

    # Emission time düzeltme
    We = 7.2921151467 * 10 ** -5  # rad/saniye
    We_dt = We * (-obs3_gps['c1c'] / c)  # afyn['Lc']/c birimi saniye

    for i in range(len(We_dt)):
        R = np.array([[np.cos(We_dt.iloc[i]), np.sin(We_dt.iloc[i]), 0],
                      [-np.sin(We_dt.iloc[i]), np.cos(We_dt.iloc[i]), 0], [0, 0, 1]])
        sat3_df.iloc[i, 3:6] = np.dot(R, sat3_df.iloc[i, 3:6])
    
    sat3_df.to_excel('sat3_df.xlsx')
    return result_gps, sat3_df
