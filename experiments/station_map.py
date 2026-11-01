"""Map of MADR and the 50 IGS stations of the FCB network (PyGMT).

    ~/micromamba/envs/gmt/bin/python experiments/station_map.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pygmt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/'results'/'experiments'/'figures'/'station_map.png'


def geodetic(x, y, z):
    a, f = 6378137.0, 1/298.257223563
    e2 = f*(2-f)
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p*(1-e2))
    for _ in range(5):
        n = a/np.sqrt(1-e2*np.sin(lat)**2)
        lat = np.arctan2(z+e2*n*np.sin(lat), p)
    return lon, np.degrees(lat)


stations = pd.read_csv(HERE/'network_stations.csv')
stations['lon'], stations['lat'] = geodetic(stations.x.to_numpy(), stations.y.to_numpy(), stations.z.to_numpy())
user = stations[stations.code == 'MADR'].iloc[0]
network = stations[stations.code != 'MADR']
SKIPPED = ('GOP6', 'SFER')          # RINEX 4 files, not read by the software
used, skipped = network[~network.code.isin(SKIPPED)], network[network.code.isin(SKIPPED)]

fig = pygmt.Figure()
pygmt.config(FONT_ANNOT_PRIMARY='9p,Helvetica', FONT_LABEL='10p,Helvetica', MAP_FRAME_TYPE='plain')
region = [-30, 30, 25, 60]
fig.grdimage(grid='@earth_relief_03m', region=region, projection='M16c', cmap='geo', shading=True)
fig.coast(shorelines='0.25p,gray30', borders='1/0.2p,gray50', frame=['WSne', 'xa10f5', 'ya5f5'])
for radius in (500, 1000, 2000):
    fig.plot(x=[user.lon], y=[user.lat], style=f'E-{2*radius}k', pen='0.6p,black,-')
    fig.text(x=user.lon, y=user.lat+radius/111.2, text=f'{radius} km', font='7p,Helvetica,black', offset='0/0.15c')
fig.plot(x=used.lon, y=used.lat, style='t0.30c', fill='red', pen='0.4p,black', label=f'FCB network ({len(used)} stations)')
fig.plot(x=skipped.lon, y=skipped.lat, style='t0.30c', fill='gray60', pen='0.4p,black', label='Not processed (RINEX 4)')
fig.text(x=network.lon, y=network.lat, text=network.code.tolist(), font='6p,Helvetica-Bold,black', offset='0/0.28c')
fig.plot(x=[user.lon], y=[user.lat], style='a0.55c', fill='yellow', pen='0.6p,black', label='MADR (user station)')
fig.text(x=user.lon, y=user.lat, text='MADR', font='9p,Helvetica-Bold,black', offset='0.55c/-0.25c', fill='white@30')
fig.legend(position='jTL+o0.2c', box='+gwhite@20+p0.5p')
with fig.inset(position='jBR+o0.1c', box='+gwhite+p0.5p', region=[-180, 180, -60, 80], projection='Q5c'):
    fig.coast(land='gray70', water='white', shorelines='0.1p')
    fig.plot(x=[region[0], region[1], region[1], region[0], region[0]], y=[region[2], region[2], region[3], region[3], region[2]], pen='0.8p,red')
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(OUT, len(network), 'network stations')
