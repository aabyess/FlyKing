# 모델이 이 맥에서 만들어지고 도는지만 본다 — 뉴런 하나를 0.1초 자극, 1회.
import os, sys
# Shiu 모델 저장소(model.py·데이터)는 git 밖 ~/flybrain에 있다 — 그 폴더로 들어가 상대경로 그대로 쓴다.
SHIU = os.path.expanduser('~/flybrain/shiu-brain-model')
sys.path.insert(0, SHIU)
os.chdir(SHIU)

import time, pandas as pd
from brian2 import ms
from model import run_exp, default_params
comp = pd.read_csv('Completeness_783.csv', index_col=0)
neu = [int(comp.index[0])]
params = dict(default_params); params['t_run'] = 100*ms; params['n_run'] = 1
t0 = time.time()
run_exp(exp_name='smoke', neu_exc=neu, path_res='./results/smoke', path_comp='./Completeness_783.csv',
        path_con='./Connectivity_783.parquet', params=params, n_proc=1, force_overwrite=True)
print('neurons:', len(comp), 'elapsed s:', round(time.time()-t0, 1))
