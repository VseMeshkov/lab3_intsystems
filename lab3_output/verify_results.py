"""Проверки математики, отсутствия утечки test и сохранённых графов."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
import lab3_core as c

z=np.array([.2,-.3,.1,.4,-.2]);costs=np.array([1.,2.,1.,1.,1.,1.]);h=1e-5
numeric=np.array([(c.objective(z+h*np.eye(5)[i],costs)[0]-
                   c.objective(z-h*np.eye(5)[i],costs)[0])/(2*h) for i in range(5)])
assert np.allclose(numeric,c.objective(z,costs)[1],atol=1e-7)
assert np.isclose(c.objective(np.zeros(5),costs)[2][1],.5)
assert len(c.STATES)==6
for name,(x,y,sd) in c.datasets().items():
    tr,va,te=c.split(len(x))
    assert not np.any(tr&va|tr&te|va&te)
    assert np.all(tr|va|te)
    m,_=c.score_models(x,y,sd,tr,va)
    perturbed=y.copy();perturbed[te]+=1000
    other,_=c.score_models(x,perturbed,sd,tr,va)
    assert np.allclose([a['score'] for a in m],[a['score'] for a in other])
    for full in (False,True):
        assert c.compatible(next(o for o,b in c.STATES if b==full),full)
for path in sorted(Path('results').glob('*_graphs.csv')):
    df=pd.read_csv(path)
    assert len(df)==5760
    for row in df.itertuples():
        order=row.order.split(' > ');rank={n:i for i,n in enumerate(order)}
        edges=json.loads(row.edges)
        assert all(rank[i]<rank[j] for i,j in edges)
        assert edges==[[c.NODES[i],c.NODES[j]] for i,j in c.graph_edges(row.full)]
summary=pd.read_csv('results/summary.csv')
assert summary.chosen.tolist()==['Yeoh-1','Yeoh-2','Yeoh-2']
nb=nbformat.read('lab3_variant12.ipynb',as_version=4)
nbformat.validate(nb)
code=[cell for cell in nb.cells if cell.cell_type=='code']
assert all(cell.execution_count is not None for cell in code)
assert not any(o.output_type=='error' for cell in code for o in cell.outputs)
print('OK: gradient, equal structure prior, train/val/test isolation, 17 280 DAGs, controls, executed notebook.')
