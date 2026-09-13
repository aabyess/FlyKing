# 당(설탕) 맛 감각뉴런 23개를 150Hz로 1초 자극 → 뇌 전체에서 가장 크게 반응한 뉴런을 본다.
# (Shiu et al. 2024 Nature의 대표 실험. v783 ID는 eon-fly-brain 예제에서 가져옴)
import os, sys
# Shiu 모델 저장소(model.py·utils.py·데이터)는 git 밖 ~/flybrain에 있다 — 그 폴더로 들어가 상대경로 그대로 쓴다.
SHIU = os.path.expanduser('~/flybrain/shiu-brain-model')
sys.path.insert(0, SHIU)
os.chdir(SHIU)

import time, pandas as pd
from brian2 import ms
from model import run_exp, default_params
import utils as utl
sugar = [720575940616885538,720575940630233916,720575940639332736,720575940632889389,720575940617000768,720575940632425919,
         720575940637568838,720575940629176663,720575940621502051,720575940638202345,720575940612670570,720575940611875570,
         720575940621754367,720575940633143833,720575940613601698,720575940630797113,720575940639198653,720575940639259967,
         720575940624963786,720575940640649691,720575940610788069,720575940623172843,720575940628853239]
params = dict(default_params); params['t_run'] = 1000*ms; params['n_run'] = 1
t0 = time.time()
run_exp(exp_name='sugar_1s', neu_exc=sugar, path_res='./results/test', path_comp='./Completeness_783.csv',
        path_con='./Connectivity_783.parquet', params=params, n_proc=-1, force_overwrite=True)
print('elapsed s:', round(time.time()-t0, 1))
df = pd.read_parquet('./results/test/sugar_1s.parquet')
rate, _ = utl.get_rate(df, t_run=params['t_run'], n_run=params['n_run'])
top = rate.iloc[:, 0].sort_values(ascending=False)
ann = pd.read_csv('../embodied-fly-brain/data/flywire_annotations.tsv', sep='\t', usecols=['root_id','super_class','cell_class','cell_type'], low_memory=False).set_index('root_id')
out = top.head(15).to_frame('Hz').join(ann, how='left')
print('active neurons:', int((top > 0).sum()))
print(out.to_string())
