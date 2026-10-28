"""Strict normalized research product contract; not an automatic BIA/DC converter."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

class ARProducts:
    def __init__(self,path,method,date):
        data=json.loads(Path(path).read_text())
        required={'schema','method','date','signals','source','orbit_product','convention','records'}
        if not required <= data.keys(): raise ValueError('AR ürün metadata alanları eksik.')
        if data['schema'] != 'togu-ar-1' or data['method'] != method:
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
        if 'signals' not in result or not result.signals.eq(expected).all():
            raise ValueError('AR ürün sinyalleri seçilen RINEX kod/faz sinyalleriyle eşleşmiyor.')
        return result
