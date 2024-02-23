#!/usr/bin/env python
import pandas as pd 
import numpy as np
import metis
import sys
import itertools

class Cluster():
   def  __init__(self, core = None, neigbors = None):
      self.core = set() if core is None else core
      self.neigbors = set() if neigbors is None else neigbors

class Writer():
   def __init__(self, prefix):
      self.prefix = prefix
      self.countf = 0
   def write(self, V, c):
      with open(self.prefix + str(self.countf).zfill(3) + '.part', 'w') as f:
          for v in c.core:
            f.write(f'0\t{V[v]}\n')
          for v in c.neigbors:
            if v not in c.core: 
              f.write(f'1\t{V[v]}\n')
      self.countf += 1
      

if __name__ == "__main__":
  prefix = ''
  if len(sys.argv) >= 2:
    prefix = sys.argv[1]
  k = 20
  if len(sys.argv) >= 3:
    threshold = int(sys.argv[2])
  df = pd.read_table(sys.stdin, header=None, names=['source', 'target'])
  V = np.union1d(df['source'].unique(), df['target'].unique())
  e2id = {element: id for id, element in enumerate(V)}
  adjlst = [[] for _ in range(len(V))]
  for row in df.itertuples(index=False):
    adjlst[e2id[row.source]].append(e2id[row.target])
    adjlst[e2id[row.target]].append(e2id[row.source])
  G = metis.adjlist_to_metis(adjlst)
  (edgecuts, parts) = metis.part_graph(G, k)
  zeta = dict()
  for k, g in itertools.groupby(sorted(enumerate(parts), key=lambda x : x[1]), lambda x: x[1]):
    if k not in zeta:
      zeta[k] = set()
    for v in g:
      zeta[k].add(v[0])
  print(f'Edgecuts: {edgecuts}')
  writer = Writer(f'clusters/{prefix}')
  for z in zeta.values():
    cluster = Cluster()  
    for v in z:
      cluster.core.add(v)
    for v in z:
      for u in adjlst[v]:
        if u not in cluster.core:
          cluster.neigbors.add(u)
    writer.write(V, cluster)

      
