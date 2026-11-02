import os, json, time, random, hashlib, argparse, importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path(__file__).resolve().parent

def write_json(path,value):
 p=Path(str(path)+'.tmp'); p.write_text(json.dumps(value,indent=2)); p.replace(path)

def load(path):
 return torch.load(path,map_location='cpu',weights_only=False)

class MaskedSubgraphs(torch.utils.data.Dataset):
 def __init__(self,x,pairs,labels,A):
  self.x=x; self.pairs=pairs; self.labels=labels; self.A=A
 def __len__(self): return len(self.labels)
 def __getitem__(self,i):
  a,b=map(int,self.pairs[:,i])
  # Both roots are always present. Excluding their mutual edge before traversal
  # yields this same union for a one-hop enclosing subgraph.
  na=self.A.indices[self.A.indptr[a]:self.A.indptr[a+1]]
  nb=self.A.indices[self.A.indptr[b]:self.A.indptr[b+1]]
  nodes=np.unique(np.concatenate(([a,b],na[na!=b],nb[nb!=a])))
  sub=self.A[nodes,:][:,nodes].tocoo()
  u,v=sub.row,sub.col; ia=int(np.searchsorted(nodes,a)); ib=int(np.searchsorted(nodes,b))
  keep=~(((u==ia)&(v==ib))|((u==ib)&(v==ia)))
  u,v=u[keep],v[keep]
  assert not np.any(((u==ia)&(v==ib))|((u==ib)&(v==ia)))
  z=torch.full((len(nodes),),2,dtype=torch.long); z[ia]=1; z[ib]=1
  return Data(x=self.x[nodes],edge_index=torch.from_numpy(np.stack((u,v))).long(),z=z,node_id=torch.from_numpy(nodes).long(),y=self.labels[i:i+1])

def build_adjacency(train,n):
 a,b=train.numpy(); A=sp.csr_matrix((np.ones(2*len(a),dtype=np.int8),(np.concatenate((a,b)),np.concatenate((b,a)))),shape=(n,n))
 A.data[:]=1; A.setdiag(0); A.eliminate_zeros(); A.sort_indices(); return A

def make_dataset(payload,s,A):
 p=payload['split'][s]; pairs=torch.cat([p['edge'],p['edge_neg']],dim=1)
 y=torch.cat([torch.ones(p['edge'].shape[1]),torch.zeros(p['edge_neg'].shape[1])])
 return MaskedSubgraphs(payload['x'],pairs,y,A)

def audit(payload,A):
 n=payload['num_nodes']; sets={}
 for s,p in payload['split'].items():
  for name,e in p.items():
   a,b=e.numpy(); keys=np.minimum(a,b)*n+np.maximum(a,b)
   assert len(keys)==len(np.unique(keys)),(s,name,'duplicates')
   sets[s+'_'+name]=set(map(int,keys))
   if s!='train' and name=='edge': assert np.count_nonzero(np.asarray(A[a,b]))==0
 names=list(sets)
 for i,a in enumerate(names):
  for b in names[i+1:]: assert not sets[a].intersection(sets[b]),(a,b,'overlap')
 # Positive target removal, including reverse direction, on a triangle.
 toy=build_adjacency(torch.tensor([[0,0,1],[1,2,2]]),3)
 g=MaskedSubgraphs(torch.zeros(3,2560),torch.tensor([[0],[1]]),torch.ones(1),toy)[0]
 assert g.edge_index.shape[1]==4
 for s in ['train','valid','test']:
  d=make_dataset(payload,s,A)
  for i in [0,len(d)//2,len(d)-1]: d[i]
 return {'split_overlap':0,'heldout_positive_edges_in_support':0,'synthetic_target_mask_test':'passed','sampled_real_target_mask_tests':'passed','support_directed_entries':int(A.nnz),'self_positive_counts':{s:int((v['edge'][0]==v['edge'][1]).sum()) for s,v in payload['split'].items()}}

def create_model(payload,device):
 from src.baselines.dgcnn import model as mod
 emb=torch.nn.Embedding.from_pretrained(payload['x'].clone(),freeze=False)
 return mod.DGCNN(32,3,10000,0.6,None,False,use_feature=True,node_embedding=emb).to(device)

def focal(logits,y):
 prob=logits.sigmoid(); pt=torch.where(y==1,prob,1-prob); alpha=torch.where(y==1,0.25,0.75)
 return (-alpha*(1-pt)**2*torch.log(pt+1e-8)).mean()

@torch.no_grad()
def evaluate(model,loader,device):
 model.eval(); scores=[]; labels=[]
 for b in loader:
  b=b.to(device); p=model(b.z,b.edge_index,b.batch,b.x,None,b.node_id).view(-1).sigmoid()
  scores.append(p.cpu()); labels.append(b.y.cpu())
 p=torch.cat(scores).numpy(); y=torch.cat(labels).numpy()
 return {'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p))},p,y
