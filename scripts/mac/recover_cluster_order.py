"""
Recover the utterance order the cluster used for train-clean-100, so Mac extractions line up with
the analogy spec built there (outputs/analogy/inputs/librispeech-train-clean-100/w2v2/state_space_spec.h5).

Why: the spec stores word tokens as frame positions in the stream of all utterances, in the order the
preprocessed dataset had ON THE CLUSTER. The LibriSpeech loader lists speaker/chapter folders in
filesystem order, which differs between the cluster and macOS, so the local dataset has the same
utterances (same total frames) in a different order. Mac extractions in local order put each word's
span on the wrong audio (only ~1% of spec words are found in the local utterance at the spec's
item_idx).

How: each spec item's words (in onset order) are matched as a subsequence of the local transcripts
(28,134 of 28,538 unique). The rest (360 ambiguous, 44 without spec words) are resolved by walking
the items in order and requiring each item's first word onset to match the local word alignment at
the running frame offset (backtracking search). Verified on every spec token: onset within 2 frames
of the local alignment, and the same total frame count.

Writes outputs/pooled_states/librispeech-train-clean-100/cluster_order.npy (local dataset index of
cluster item 0, 1, ...) and cluster_order_ids.txt (the utterance ids, same order), which
extract_pooled_states.py picks up. Runs in a few minutes, CPU only, from the repo root:

    python scripts/mac/recover_cluster_order.py
"""
from pathlib import Path
import pandas as pd, datasets, numpy as np, pyarrow.compute as pc
from collections import defaultdict
c=pd.read_hdf('outputs/analogy/inputs/librispeech-train-clean-100/w2v2/state_space_spec.h5','cuts').reset_index()
w=c[c.level=='phoneme'].groupby(['label','instance_idx']).agg(item=('item_idx','first'),on=('onset_frame_idx','min'),off=('offset_frame_idx','max')).reset_index()
print('spec items with words:', w.item.nunique(), 'max item', w.item.max())
ds=datasets.load_from_disk('outputs/preprocessed_data/librispeech-train-clean-100')
meta=ds.select_columns(['id','word_detail']).to_pandas()
L=np.array(pc.list_value_length(ds.data.table.column('input_values')).to_pylist())
def nf(x):
    x=(x-10)//5+1
    for _ in range(4): x=(x-3)//2+1
    for _ in range(2): x=(x-2)//2+1
    return x
F=nf(L); print('local total frames', F.sum())
words=[tuple(u for u in d['utterance'] if u) for d in meta.word_detail]
spec_seq=w.sort_values('on').groupby('item').label.apply(tuple)

from collections import Counter
inv=defaultdict(set)
for j,ws in enumerate(words):
    for x in ws: inv[x].add(j)
def subseq(a,b):
    it=iter(b); return all(x in it for x in a)
perm={}; ambc={}; amb=0; miss=0
for it,seq in spec_seq.items():
    cand=set.intersection(*(inv.get(x,set()) for x in sorted(set(seq),key=lambda x:len(inv.get(x,())))[:3]))
    cand=[j for j in cand if subseq(seq,words[j])]
    if len(cand)==1: perm[it]=cand[0]
    elif cand: amb+=1; ambc[it]=cand
    else: miss+=1
print('matched',len(perm),'ambiguous',amb,'missing',miss, 'unique targets', len(set(perm.values())))

N=len(meta)
def local_onsets(j, word):
    d=meta.word_detail.iloc[j]; r=F[j]/L[j]
    return [st*r for st,u in zip(d['start'],d['utterance']) if u==word]
first=w.sort_values('on').groupby('item').first()
def fits(j, k, Sk, tol=2.0):
    lab,on=first.loc[k,'label'], first.loc[k,'on']
    return any(abs((on-Sk)-x)<=tol for x in local_onsets(j,lab))
perm_targets=set(perm.values())
spec_targets=perm_targets|set(x for c in ambc.values() for x in c)

used=set(); order=[None]*N; Sv=np.zeros(N+1,dtype=np.int64)
def options(k, Sk):
    if k in perm: c=[perm[k]]
    elif k in ambc: c=list(ambc[k])
    else: return None
    return [j for j in c if j not in used and fits(j,k,Sk)]
def candidates(k, Sk):
    c=options(k,Sk)
    prev=order[k-1] if k>0 else 0
    if c is None:
        pool=[j for j in range(N) if j not in used and j not in perm_targets]
        if k+1<N: pool=[j for j in pool if options(k+1,Sk+F[j]) != []]
        c=sorted(pool, key=lambda j:(j in spec_targets, abs(j-prev)))
    else:
        c=sorted(c, key=lambda j: abs(j-prev))
    return c
stack=[]; k=0; backtracks=0
while k<N:
    if len(stack)==k: stack.append(candidates(k,Sv[k]))
    if stack[k]:
        j=stack[k].pop(0); order[k]=j; used.add(j); Sv[k+1]=Sv[k]+F[j]; k+=1
    else:
        stack.pop(); k-=1; backtracks+=1
        assert k>=0, 'no consistent order'
        used.discard(order[k]); order[k]=None
print('placed',sum(o is not None for o in order),'of',N,'backtracks',backtracks,'total frames',Sv[N])
res=[]
for lab,on,it in zip(w.label,w.on,w.item):
    xs=local_onsets(order[it],lab); res.append(min(abs((on-Sv[it])-x) for x in xs) if xs else 99)
res=np.array(res); print('word tokens checked',len(res),'onset within 1 frame',np.mean(res<=1),'within 2',np.mean(res<=2),'max',res.max())
assert len(set(order))==N
out=Path('outputs/pooled_states/librispeech-train-clean-100'); out.mkdir(parents=True, exist_ok=True)
np.save(out/'cluster_order.npy', np.array(order))
(out/'cluster_order_ids.txt').write_text('\n'.join(meta.id.iloc[order]) + '\n')
print('wrote', out/'cluster_order.npy', 'and', out/'cluster_order_ids.txt')
