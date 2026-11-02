"""Portable frozen-data reproduction; outputs never overwrite reference artifacts."""
import argparse,hashlib,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from src.baselines.dgcnn.core import load,write_json,build_adjacency,make_dataset,create_model,evaluate,focal,audit

def main():
 a=argparse.ArgumentParser()
 a.add_argument('mode',choices=['train','evaluate','smoke'])
 a.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'fixed_data.pt')
 a.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'best.pt')
 a.add_argument('--output',type=Path,required=True)
 a.add_argument('--device',default='cuda:0');a.add_argument('--workers',type=int,default=4)
 a.add_argument('--microbatch',type=int,default=128);a.add_argument('--accumulate',type=int,default=2)
 a.add_argument('--epochs',type=int,default=20);a.add_argument('--seed',type=int,default=42)
 args=a.parse_args();assert args.microbatch>0 and args.accumulate>0 and args.workers>=0
 if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Output directory must be empty to preserve prior results')
 args.output.mkdir(parents=True,exist_ok=True);o=args.output
 torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
 payload=load(args.data);A=build_adjacency(payload['split']['train']['edge'],payload['num_nodes'])
 write_json(o/'graph_audit.json',audit(payload,A))
 active=torch.unique(torch.cat([e.flatten() for d in payload['split'].values() for e in d.values()]))
 assert torch.isfinite(payload['x'][active]).all() and (payload['x'][active].norm(dim=1)>0).all()
 dev=torch.device(args.device);model=create_model(payload,dev)
 ds={s:make_dataset(payload,s,A) for s in ['train','valid','test']}
 kw={'num_workers':args.workers,'pin_memory':dev.type=='cuda'}
 if args.workers:kw['prefetch_factor']=2
 def loader(s,shuffle=False):return DataLoader(ds[s],batch_size=args.microbatch,shuffle=shuffle,**kw)
 checksum=hashlib.sha256(args.data.read_bytes()).hexdigest()
 write_json(o/'config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'data_sha256':checksum,'effective_batch':args.microbatch*args.accumulate})
 if args.mode=='evaluate':
  state=load(args.checkpoint);model.load_state_dict(state.get('model',state),strict=True)
  metrics,scores,labels=evaluate(model,loader('test'),dev)
  np.savez_compressed(o/'test_predictions.npz',pairs=ds['test'].pairs.numpy(),scores=scores,labels=labels)
  write_json(o/'result.json',{'test':metrics,'data_sha256':checksum});print(metrics);return
 opt=torch.optim.Adam(model.parameters(),lr=1e-4)
 if args.mode=='smoke':
  model.train();b=next(iter(loader('train'))).to(dev);loss=focal(model(b.z,b.edge_index,b.batch,b.x,None,b.node_id).view(-1),b.y);assert torch.isfinite(loss);loss.backward();opt.step();write_json(o/'smoke.json',{'loss':float(loss),'finite':True});return
 best=-1;history=[];train_loader=loader('train',True);eff=args.microbatch*args.accumulate
 for epoch in range(1,args.epochs+1):
  model.train();total=0;seen=0
  for step,b in enumerate(train_loader):
   if step%args.accumulate==0:opt.zero_grad(set_to_none=True);denom=min(eff,len(ds['train'])-step*args.microbatch)
   b=b.to(dev);loss=focal(model(b.z,b.edge_index,b.batch,b.x,None,b.node_id).view(-1),b.y)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   (loss*b.num_graphs/denom).backward()
   if (step+1)%args.accumulate==0 or step+1==len(train_loader):opt.step()
   total+=float(loss)*b.num_graphs;seen+=b.num_graphs
   if step%100==0:write_json(o/'status.json',{'stage':'training','epoch':epoch,'microbatch_step':step+1,'loss':total/seen})
  write_json(o/'status.json',{'stage':'validation','epoch':epoch})
  val,_,_=evaluate(model,loader('valid'),dev);row={'epoch':epoch,'train_loss':total/seen,'valid':val};history.append(row);write_json(o/'history.json',history)
  state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'valid':val,'data_sha256':checksum}
  torch.save(state,o/'last.pt')
  if val['auprc']>best:best=val['auprc'];torch.save(state,o/'best.pt')
  print(json.dumps(row),flush=True)
 state=load(o/'best.pt');model.load_state_dict(state['model']);write_json(o/'status.json',{'stage':'test','best_epoch':state['epoch']})
 metrics,scores,labels=evaluate(model,loader('test'),dev)
 np.savez_compressed(o/'test_predictions.npz',pairs=ds['test'].pairs.numpy(),scores=scores,labels=labels)
 write_json(o/'result.json',{'best_epoch':state['epoch'],'validation':state['valid'],'test':metrics,'data_sha256':checksum})
 write_json(o/'status.json',{'stage':'complete'})
if __name__=='__main__':main()
