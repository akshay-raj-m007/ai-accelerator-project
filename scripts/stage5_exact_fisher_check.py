import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, torch, json
from fr import data as fdata, stage2 as S2
from fr.config import Config
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
import torch.nn.functional as Fn
cfg=Config()
model,ckpt=load_model('models/scratch/best.pth'); model.eval()
norm=fdata.resolve_normalization(cfg.normalize,"checkpoint",ckpt.get("config")); vs,sd=split_params(ckpt,cfg)
Xt,yt=load_tensors(fdata.get_calibration_loader(cfg.data_path(),norm,128,vs,sd,batch_size=128),'cpu')
Xv,yv=load_tensors(fdata.get_val_loader(cfg.data_path(),norm,500,vs,sd,0,False,subset=128),'cpu')
mods=quantizable_modules(model); params=[m.weight for _,m in mods]
for p in params: p.requires_grad_(True)
def exact_fisher(X):
    F=[torch.zeros_like(p,dtype=torch.float64) for p in params]; n=X.shape[0]; ex=0.0
    for i in range(n):
        lp=torch.log_softmax(model(X[i:i+1]).float(),1); p=lp.exp().detach().view(-1)
        for c in range(10):
            if p[c]<1e-7: continue
            g=torch.autograd.grad(-lp[0,c],params,retain_graph=True)
            for f,gi in zip(F,g): f+= float(p[c])*gi.double()**2
    return [f/n for f in F]
Ft=exact_fisher(Xt); Fv=exact_fisher(Xv)
tt=sum(f.sum().item() for f in Ft); tv=sum(f.sum().item() for f in Fv)
print("EXACT-expectation Fisher trace (no label sampling): train 128 imgs %.4e  val 128 imgs %.4e  ratio %.1f"%(tt,tv,tv/tt))
base=[p.detach().clone() for p in params]; codecs=S2.make_codecs(True)
for fmt in ('int4_pt','int8_pt','fp8_e4m3fn'):
    dirs=[]
    for p in base:
        w=p.numpy(); dirs.append(torch.from_numpy(np.ascontiguousarray(codecs[fmt].decode(codecs[fmt].encode(w)).reshape(w.shape)))-p)
    print(fmt,"1/2 dFd exact-train %.3e  exact-val %.3e"%(0.5*sum((f*d.double()**2).sum().item() for f,d in zip(Ft,dirs)),0.5*sum((f*d.double()**2).sum().item() for f,d in zip(Fv,dirs))))
