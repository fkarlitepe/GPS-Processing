import numpy as np
import pandas as pd

def process_result_gps(result_gps, sat3_df, cll_result):
    xyz0_apprx = pd.read_excel('xyz0_apprx.xlsx')
    xyz0_apprx = xyz0_apprx.drop(xyz0_apprx.columns[0], axis=1)

    n3_amb = np.array([])

    for j in result_gps['epok'].unique():
        sat3 = sat3_df[sat3_df['epok'] == j]
        result_gps_1 = result_gps[result_gps['epok'] == j]
        
        #uydu yükseklik açısı hesaplama adımı (zenit)
        a = 6378137
        b = 6356752.314140347
        nxyz = np.array([xyz0_apprx.iloc[0] / (a ** 2), xyz0_apprx.iloc[1] / (a ** 2), xyz0_apprx.iloc[2] / (b ** 2)])
        nxyz_sqrt = np.sqrt(np.dot(nxyz.T, nxyz))
        zenit = np.array([])
        zenit_1 = np.array([])
        gps1=np.array([])

        for i in range(len(sat3)):
            D_xyz = np.array([sat3.iloc[i, 3:6]]) - xyz0_apprx.values.T
            D_xyz_sqrt = np.sqrt(
                np.dot(np.array([sat3.iloc[i, 3:6]]) - xyz0_apprx.values.T,
                       np.array(np.array([sat3.iloc[i, 3:6]]) - xyz0_apprx.values.T).T))
            comp = np.degrees(
                np.arccos(np.dot(D_xyz, nxyz) / (D_xyz_sqrt * nxyz_sqrt)))
            zenit = np.append(zenit, comp)
            
        # sat4=np.array([])
        zenit = 90 - zenit
        for i in range(len(sat3)):
            if 10 <= zenit[i] <= 170:
                zenit_1 = np.append(zenit_1, zenit[i])
                result_gps_1.iloc[i,:]=result_gps_1.iloc[i,:]
                
                
        #         sat4=np.append(sat4,sat3.iloc[i,3:6])
        
        # sat4=pd.DataFrame(sat4.reshape(int(len(sat4)/3),3),columns=('x','y','z'))        

        s0 = ((0.003 ** 2) / np.sin(np.deg2rad(zenit_1)))
        s0 = s0.min()

        pseudorange = np.sqrt(((sat3['x'] - xyz0_apprx.iloc[0,0]) ** 2) + ((sat3['y'] - xyz0_apprx.iloc[1,0]) ** 2) + (
                    (sat3['z'] - xyz0_apprx.iloc[2,0]) ** 2))
        dt_sat = result_gps_1['dt_sat'] / 10 ** 6
        c = 299792458
        code = pseudorange - c * (dt_sat)

        gps1 =  np.arange(1,33)
        a_matrix = np.zeros((len(gps1) * 2, 3 + 1 + len(gps1)))
        prefit = np.zeros((len(gps1) * 2, 1))
        p_matrix = np.eye(len(gps1) * 2)

        gps1.sort()
        start = -1
        for q in range(len(gps1)):
            start = +1
            for i in range(len(result_gps_1['sat'])):
                if gps1[q] == result_gps_1.iloc[i, 2]:
                    for k in range(3):
                        a_matrix[q, k] = (xyz0_apprx.iloc[k] - sat3.iloc[i, k + 3]) / pseudorange.iloc[i]
                        a_matrix[q + len(gps1), k] = (xyz0_apprx.iloc[k] - sat3.iloc[i, k + 3]) / pseudorange.iloc[i]

                    a_matrix[q, 3] = 1
                    a_matrix[q + len(gps1), 3] = 1

                    a_matrix[q, 3 + start + q] = 0
                    a_matrix[q + len(gps1), 3 + start + q] = 1

                    prefit[q, 0] = result_gps_1.iloc[i, 0] - code.iloc[i]
                    prefit[q + len(gps1), 0] = result_gps_1.iloc[i, 1] - code.iloc[i]

                    if zenit[i] < 30:
                        p_matrix[q, q] = s0 / ((3 ** 2) / (np.sin(np.deg2rad(zenit[i]))) ** 2)
                        p_matrix[q + len(gps1), q + len(gps1)] = s0 / (
                                    (0.003 ** 2) / (np.sin(np.deg2rad(zenit[i]))) ** 2)
                    elif 30< zenit[i] :
                        p_matrix[q, q] = s0 / ((3 ** 2) / np.sin(np.deg2rad(zenit[i])))
                        p_matrix[q + len(gps1), q + len(gps1)] = s0 / ((0.003 ** 2) / np.sin(np.deg2rad(zenit[i])))

        n_cofac = np.dot(np.dot(a_matrix.T, p_matrix), a_matrix) + np.linalg.inv(cll_result)
        n_short = np.dot(np.dot(a_matrix.T, p_matrix), prefit)
        dx_1 = np.dot(np.linalg.inv(n_cofac), n_short)
        dx_1 = dx_1.reshape(len(gps1) + 4, 1)
        cll_result = np.linalg.pinv(np.dot(np.dot(a_matrix.T, p_matrix), a_matrix) + np.linalg.inv(cll_result))

        xyz_result = np.zeros((3, 1))
        xyz_result[:3, 0:1] = xyz0_apprx[:].values + dx_1[:3].reshape(3, 1)

        n3_amb = np.append(n3_amb, dx_1[4:])
        
    n3_amb = np.c_[n3_amb, np.zeros((len(n3_amb)))]
    bb = np.array([])
    for ii in np.arange(0, len(n3_amb), 32):
        n3_amb[ii:ii + 32, 1] = np.arange(1, 33)
        bb = np.append(bb, 32)
    n3_amb = pd.DataFrame(n3_amb, columns=['N3', 'sat'])

    say = np.c_[bb, result_gps['epok'].unique()]
    yeni = np.zeros((len(n3_amb), 3))
    start = 0
    for idx, i in enumerate(say):
        yeni[start:start + int(i[0]), :2] = n3_amb[start:start + int(i[0])]
        yeni[start:start + int(i[0]), 2] = np.ones(int(i[0])) * int(i[1])
        start = start + int(i[0])
    n3_amb = pd.DataFrame(yeni)
    n3_amb.columns = ['N3', 'sat', 'epok']

    xyz_result=pd.DataFrame(xyz_result.reshape(1,3),columns = ['x', 'y', 'z'])

    

    return xyz_result, n3_amb, dx_1
