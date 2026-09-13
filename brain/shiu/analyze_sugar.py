# 저장된 결과에서 자극한 23개를 빼고, 그 신호를 받아 반응한 뉴런(하류)을 본다.
import os, sys
# Shiu 모델 저장소(utils.py·결과)는 git 밖 ~/flybrain에 있다 — 그 폴더로 들어가 상대경로 그대로 쓴다.
SHIU = os.path.expanduser('~/flybrain/shiu-brain-model')
sys.path.insert(0, SHIU)
os.chdir(SHIU)

import pandas as pd
from brian2 import ms
import utils as utl
sugar = set([720575940616885538,720575940630233916,720575940639332736,720575940632889389,720575940617000768,720575940632425919,
         720575940637568838,720575940629176663,720575940621502051,720575940638202345,720575940612670570,720575940611875570,
         720575940621754367,720575940633143833,720575940613601698,720575940630797113,720575940639198653,720575940639259967,
         720575940624963786,720575940640649691,720575940610788069,720575940623172843,720575940628853239])
df = pd.read_parquet('./results/test/sugar_1s.parquet')
rate, _ = utl.get_rate(df, t_run=1000*ms, n_run=1)
r = rate.iloc[:, 0]
down = r[~r.index.isin(sugar)].sort_values(ascending=False)
ann = pd.read_csv('../embodied-fly-brain/data/flywire_annotations.tsv', sep='\t', usecols=['root_id','super_class','cell_class','cell_type','side'], low_memory=False).set_index('root_id')
out = down.head(20).to_frame('Hz').join(ann, how='left')
print('downstream active:', int((down > 0).sum()))
print(out.to_string())
