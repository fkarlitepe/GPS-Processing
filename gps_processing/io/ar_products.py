"""Strict normalized research product contract; not an automatic BIA/DC converter."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# IPC (Laurichesse et al. 2008) uses the same three satellite corrections as DCM: code clock,
# phase clock and MW (wide-lane) bias (Shi & Gao 2014, Table 1); only the user WL step differs.
COMPATIBLE={'SDBS':{'SDBS'},'DCM':{'DCM'},'IPC':{'IPC','DCM'}}

class ARProducts:
    def __init__(self,path,method,date):
        data=json.loads(Path(path).read_text())
        required={'schema','method','date','signals','source','orbit_product','convention','records'}
        if not required <= data.keys(): raise ValueError('AR ürün metadata alanları eksik.')
        if data['schema'] != 'togu-ar-1' or data['method'] not in COMPATIBLE.get(method,{method}):
            raise ValueError('AR ürün şeması veya yöntemi eşleşmiyor.')
        if data['date'] != date.strftime('%Y-%m-%d'):
            raise ValueError('AR ürün tarihi gözlem tarihiyle eşleşmiyor.')
        if data['convention'] != 'add_to_observation_m_positive_ambiguity':
            raise ValueError('AR ürün işaret/birim sözleşmesi desteklenmiyor.')
        if len(data['signals'])!=4 or not data['source'] or not data['orbit_product']:
            raise ValueError('AR sinyal/ürün kaynağı metadata geçersiz.')
        self.meta=data
        self.frame=pd.DataFrame(data['records'])
        cols=['epok','sat','code_m','phase_m','mw_m']
        if not set(cols)<=set(self.frame) or self.frame.empty:
            raise ValueError('AR ürün kayıtları eksik.')
        if not np.isfinite(self.frame[cols].to_numpy(dtype=float)).all():
            raise ValueError('AR ürününde geçersiz sayısal değer.')
        if self.frame.duplicated(['epok','sat']).any(): raise ValueError('Tekrarlı AR ürün kaydı.')
    def apply(self,observations):
        # Exact epoch matching avoids interpolation across clock/bias discontinuities.
        result=observations.merge(self.frame,on=['epok','sat'],how='left',validate='one_to_one')
        if result[['code_m','phase_m','mw_m']].isna().any().any():
            raise ValueError('AR ürünleri tüm gözlem epok/uydularını kapsamıyor.')
        expected=' '.join(self.meta['signals'])
        if 'signals' not in result:
            raise ValueError('AR ürün sinyalleri seçilen RINEX kod/faz sinyalleriyle eşleşmiyor.')
        matches=result.signals.eq(expected)
        if 'code_bias_aligned' in result:
            # Codes were shifted to the product codes with Bias-SINEX DSB/OSB; the phase only has
            # to be on the same frequency (tracking-mode phase offsets are common to all satellites).
            band=lambda s:' '.join(t if t[0]=='C' else t[:2] for t in s.split())
            matches|=result.code_bias_aligned&(result.signals.map(band)==band(expected))
        if not matches.all():
            raise ValueError('AR ürün sinyalleri seçilen RINEX kod/faz sinyalleriyle eşleşmiyor '
                             '(C1C/C2L gibi kodlar için Bias-SINEX DSB/OSB dosyası verin).')
        return result
