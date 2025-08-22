
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for double-perturbation expression prediction (Enhanced model only).

Changes in this version:
- Adds a LONG-format output `predictions_long.csv` with columns: gene, perturbation, expression.
- Optional flag `--long_only` to emit only the long output (skip wide CSVs).

Inputs:
  --ckpt       : Path to model.pt saved by double_perturb_train_plus.py
  --train_csv  : Wide matrix CSV used for training (provides g+ctrl & baseline)
  --test_csv   : Single-column CSV listing target pairs (one per line): gA+gB

Outputs (to --out_dir):
  - predictions_long.csv    : long-format predictions (gene, perturbation, expression)
  - predictions_matrix.csv  : wide matrix (optional; skipped if --long_only)
  - predictions_binary.csv  : wide 0/1 labels vs baseline using tau (optional; skipped if --long_only)
  - infer_report.json       : diagnostics
"""

import argparse, os, json, re
from typing import List, Dict, Tuple, Optional
import numpy as np, pandas as pd, torch, torch.nn as nn

def set_seed(seed:int=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def parse_base(c:str)->str: return re.sub(r"\.\d+$", "", c)
def split_pair(c:str)->Tuple[str,str]: return tuple(c.split('+',1))

def ensure_index(df:pd.DataFrame)->pd.DataFrame:
    first=df.columns[0]
    if not np.issubdtype(df[first].dtype,np.number) and df[first].is_unique: df=df.set_index(first)
    return df

def collapse_reps(df:pd.DataFrame)->pd.DataFrame:
    df=ensure_index(df); groups={}
    for c in df.columns: groups.setdefault(parse_base(c),[]).append(c)
    out={b:(df[cols].astype(np.float32).mean(axis=1) if len(cols)>1 else df[cols[0]].astype(np.float32)) for b,cols in groups.items()}
    return pd.DataFrame(out,index=df.index)

def find_baseline_ctrl(df:pd.DataFrame)->Tuple[np.ndarray,str]:
    if 'ctrl+ctrl' in df.columns: return df['ctrl+ctrl'].values.astype(np.float32),'ctrl+ctrl'
    sc=[c for c in df.columns if c.endswith('+ctrl') or c.startswith('ctrl+')]
    if sc: return df[sc].median(axis=1).values.astype(np.float32),'median_single_ctrl'
    return df.median(axis=1).values.astype(np.float32),'global_median'

def load_test_pairs(path:str)->List[str]:
    df=pd.read_csv(path,header=None); pairs=df.iloc[:,0].dropna().astype(str).str.strip().tolist()
    seen=set(); out=[]
    for p in pairs:
        if '+' in p and p not in seen: seen.add(p); out.append(p)
    return out

def summarize_train(df:pd.DataFrame)->Dict[str,str]:
    singles={c for c in df.columns if '+ctrl' in c or 'ctrl+' in c}; m={}
    for c in singles:
        a,b=c.split('+',1)
        if a!='ctrl' and b=='ctrl': m[a]=c
        elif a=='ctrl' and b!='ctrl': m[b]=c
    return m

class PerGeneMLP(nn.Module):
    def __init__(self,in_dim:int,hidden:int=128,depth:int=3,dropout:float=0.1):
        super().__init__(); layers=[]; d=in_dim
        for _ in range(max(depth-1,1)):
            layers+=[nn.Linear(d,hidden),nn.GELU()]; 
            if dropout>0: layers.append(nn.Dropout(dropout)); d=hidden
        layers.append(nn.Linear(d,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): B,G,F=x.shape; return self.net(x.reshape(B*G,F)).reshape(B,G)

def make_flags(features:str)->Dict[str,bool]:
    f={'baseline':False,'interact':False}
    for t in [t.strip().lower() for t in features.split('+') if t.strip()]:
        if t=='baseline': f['baseline']=True
        elif t=='interact': f['interact']=True
    return f

def build_features(xA,xB,baseline,flags): 
    feats=[xA,xB]
    if flags['baseline'] and baseline is not None: feats.append(baseline)
    if flags['interact']: feats+=[np.abs(xA-xB),xA*xB,0.5*(xA+xB)]
    return np.stack(feats,axis=-1)

def wide_to_long(pred_df: pd.DataFrame) -> pd.DataFrame:
    # pred_df: rows=genes (index), cols=perturbations
    long_df = pred_df.stack().reset_index()
    long_df.columns = ['gene', 'perturbation', 'expression']
    # Ensure dtypes are as expected
    long_df['gene'] = long_df['gene'].astype(str)
    long_df['perturbation'] = long_df['perturbation'].astype(str)
    long_df['expression'] = long_df['expression'].astype(float)
    return long_df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--train_csv',default='data/train_set.csv')
    ap.add_argument('--test_csv', default='data/test_set.csv')
    ap.add_argument('--out_dir',default='./prediction'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size',type=int,default=16); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--long_only', action='store_true', default=True, help='Only save predictions_long.csv (skip wide CSVs)')
    args=ap.parse_args(); set_seed(args.seed); os.makedirs(args.out_dir,exist_ok=True)

    # 显式指定 weights_only=False：我们需要从自有 checkpoint 字典中读取 tau/args 元数据
    ckpt=torch.load(args.ckpt, map_location='cpu', weights_only=False); a=ckpt['args']; tau=float(ckpt['tau']); features=a['features']; flags=make_flags(features)
    feat_dim=ckpt['feat_dim']; hidden=int(a['hidden']); depth=int(a['depth']); dropout=float(a['dropout'])
    model=PerGeneMLP(feat_dim,hidden,depth,dropout); model.load_state_dict(ckpt['model_state']); model.to(args.device).eval()

    train=collapse_reps(pd.read_csv(args.train_csv, index_col=0, low_memory=False)); genes=train.index.to_numpy()
    baseline,src=find_baseline_ctrl(train); gene2single=summarize_train(train)
    targets=load_test_pairs(args.test_csv)

    pred=np.zeros((len(genes),len(targets)),np.float32); pred_bin=np.zeros_like(pred,np.int64); miss=[]
    base_t=torch.from_numpy(baseline).to(args.device).unsqueeze(0)
    batch=[]; idxs=[]
    for j,p in enumerate(targets):
        a,b=split_pair(p); ac,bc=gene2single.get(a),gene2single.get(b)
        if ac is None or bc is None: miss.append(p); continue
        XA, XB=train[ac].values.astype(np.float32),train[bc].values.astype(np.float32)
        X=build_features(XA,XB,baseline,flags); batch.append(X); idxs.append(j)
        if len(batch)==args.batch_size or j==len(targets)-1:
            Xb=torch.from_numpy(np.stack(batch)).to(args.device)
            with torch.no_grad(): Y=model(Xb)
            Ynp=Y.cpu().numpy().astype(np.float32); pred[:,idxs]=Ynp.T
            Ybin=(torch.abs(Y-base_t)>=tau).int().cpu().numpy(); pred_bin[:,idxs]=Ybin.T
            batch=[]; idxs=[]

    # Build DataFrames
    pred_df = pd.DataFrame(pred, index=genes, columns=targets)
    long_df = wide_to_long(pred_df)

    # Save outputs
    long_path = os.path.join(args.out_dir,'prediction.csv')
    long_df.to_csv(long_path, index=False)

    if not args.long_only:
        pred_df.to_csv(os.path.join(args.out_dir,'predictions_matrix.csv'))
        pd.DataFrame(pred_bin, index=genes, columns=targets).to_csv(os.path.join(args.out_dir,'predictions_binary.csv'))

    print(f"Saved LONG predictions to: {long_path}")
    if not args.long_only:
        print("Also saved wide predictions and binary labels.")

if __name__=='__main__': main()
