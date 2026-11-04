"""Train a binary MNIST classifier and measure boundary mass during training."""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU())
        self.head=nn.Linear(128,2)
    def forward(self,x):return self.head(self.features(x))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--eps',type=float,default=.1)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets=[]
    for train in [True,False]:
        ds=MNIST(args.data_dir,train=train,download=True)
        mask=(ds.targets==3)|(ds.targets==5)
        x=ds.data[mask].unsqueeze(1).float()/255
        y=(ds.targets[mask]==5).long()
        datasets.append(TensorDataset(x,y))
    train=DataLoader(datasets[0],batch_size=64,shuffle=True)
    test=DataLoader(datasets[1],batch_size=512,shuffle=False)
    experiment_ids=torch.randperm(len(datasets[1]),generator=torch.Generator().manual_seed(args.seed))[:512]
    ax,ay=[v[experiment_ids].to(device) for v in datasets[1].tensors]
    beta=min(ay.float().mean().item(),1-ay.float().mean().item())
    model=BinaryCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rows=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        train_correct=0
        for x,y in train:
            optimizer.zero_grad(set_to_none=True)
            train_logits=model(x.to(device))
            loss=nn.functional.cross_entropy(train_logits,y.to(device))
            train_correct+=(train_logits.argmax(1).detach().cpu()==y).sum().item()
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            correct=sum((model(x.to(device)).argmax(1)==y.to(device)).sum().item() for x,y in test)
            logits=model(ax)
            gap=logits[:,1]-logits[:,0]
            w=model.head.weight[1]-model.head.weight[0]
            dist_feature=gap/w.norm().clamp_min(1e-12)
            dist_logit=gap/np.sqrt(2.)
            row={'epoch':epoch,'train_accuracy':train_correct/len(datasets[0]),'accuracy':correct/len(datasets[1]),'test_error':1-correct/len(datasets[1])}
            for name,dist in [('feature',dist_feature),('logit',dist_logit)]:
                row['psi_hard_'+name]=float((dist.abs()<=args.eps).float().mean().item()/(args.eps*beta))
                row['psi_kernel_'+name]=float(torch.exp(-.5*(dist/args.eps)**2).mean().item()/(args.eps*np.sqrt(2*np.pi)*beta))
            rows.append(row)
            print(json.dumps(row),flush=True)
    with (args.output_dir/'epochs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    correlations={}
    for metric in ['psi_hard_feature','psi_kernel_feature','psi_hard_logit','psi_kernel_logit']:
        for target in ['accuracy','test_error']:
            a=np.array([r[metric] for r in rows]);b=np.array([r[target] for r in rows])
            correlations[metric+'_vs_'+target]=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
    summary={
             'model_settings':{'digits':[3,5],'conv_filters':[32,64],'fc_units':128,'optimizer':'Adam','lr':1e-3,'batch_size':64,'experiment_test_examples':512,'epochs_shown_in_figure':10},
             'settings':{'seed':args.seed,'epochs':args.epochs,'eps':args.eps,'kernel_size':3,'padding':1,'pooling':'2x2 after each conv','input_normalization':'[0,1]','fixed_test_experiment_subset':True,'class_balance':'true labels'},
             'correlations':correlations,'final':rows[-1]}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    torch.save({'model_state':model.state_dict(),'args':vars(args)},args.output_dir/'last.pt')
    fig,a=plt.subplots(figsize=(7,4));b=a.twinx()
    a.plot([r['epoch'] for r in rows],[r['psi_hard_feature'] for r in rows],'r-o',label='Feature-space hard tube boundary mass')
    b.plot([r['epoch'] for r in rows],[r['train_accuracy'] for r in rows],'b--s',label='Train accuracy')
    b.plot([r['epoch'] for r in rows],[r['accuracy'] for r in rows],'g--^',label='Test accuracy')
    a.set(xlabel='Epoch',ylabel='Boundary mass (hard tube estimate)',title='MNIST 3 vs 5')
    b.set_ylabel('Accuracy');fig.legend(loc='upper center',bbox_to_anchor=(.5,.90),fontsize=7)
    fig.tight_layout();fig.savefig(args.output_dir/'mnist_training.png',dpi=200);plt.close(fig)


if __name__=='__main__':main()
