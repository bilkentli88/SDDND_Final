#!/usr/bin/env python3
"""Obtain the exact historical/third-party sources required by E1 and E4."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'nsddnew_e1':('https://github.com/bilkentli88/NSDDNew.git','ee1014a80760dd720e680c7597a3b985f0f4b609'),
 'torchdde':('https://github.com/onofriosmr/torchdde.git','4888c63dc84e966f6c6553277adc3851a2cca4fc'),
 'learnable_delays':('https://github.com/onofriosmr/learnable_delays.git','9b573c4bed5efee9fe9c6b765044dbd506b5bf07'),
}

def checkout(name,cache):
    url,commit=SOURCES[name];path=cache/name
    if not path.exists():
        subprocess.run(['git','clone','--no-checkout',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:
        raise RuntimeError(f'{path} is at {actual}, expected {commit}; use a new cache directory.')
    dirty=subprocess.check_output(['git','-C',str(path),'diff','HEAD','--name-only'],text=True).strip()
    if dirty:raise RuntimeError(f'Tracked dependency files have changed: {path}')
    print(f'{name}: verified {commit}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('studies',nargs='+',choices=['E1','E4'])
    p.add_argument('--cache',type=Path,default=ROOT/'.repro-deps')
    args=p.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    names=[]
    if 'E1' in args.studies:names.append('nsddnew_e1')
    if 'E4' in args.studies:names+=['torchdde','learnable_delays']
    for name in names:checkout(name,args.cache.resolve())

if __name__=='__main__':main()
