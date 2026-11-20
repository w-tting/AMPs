import sys
from pathlib import Path
import pandas as pd
mode, src = sys.argv[1:3]
out = Path('../data/split_data'); out.mkdir(exist_ok=True)
if mode == 'prepare':
    d = pd.read_csv(src)[['sequence', 'value']].copy()
    d.insert(0, 'seq_id', [f'pep_{i:06d}' for i in range(len(d))])
    d['label'] = d['value'].eq(4).map({True: 'nonAMP', False: 'AMP'})
    d.to_csv(out/'indexed.csv', index=False)
    with (out/'curated.fasta').open('w', encoding='ascii') as f:
        for r in d.itertuples(index=False): 
            f.write(f'>{r.seq_id}|label={r.label}\n{r.sequence}\n')
    print('Prepared', len(d), 'sequences:', d['label'].value_counts().to_dict())
elif mode == 'export':
    d = pd.read_csv(out/'indexed.csv')
    a = pd.read_csv(src)[['AC', 'cluster']]
    if a['AC'].duplicated().any() or not set(a['AC']).issubset(set(d['seq_id'])): 
        raise ValueError('Invalid assignments')
    if not a['cluster'].isin([0, 1,]).all(): 
        raise ValueError('Invalid split IDs')
    z = d.merge(a, left_on='seq_id', right_on='AC', how='left', validate='one_to_one')
    for code, name in [(0, 'train'), (1, 'test'), ]:
        s = z.loc[z['cluster'] == code, ['sequence', 'value']]
        s.to_csv(out/f'{name}.csv', index=False)
        print(name, len(s), 'AMP', int(s['value'].ne(4).sum()))
    z.loc[z['cluster'].isna(), ['sequence', 'value']].to_csv(out/'excluded.csv', index=False)
    z.to_csv(out/'split_audit.csv', index=False)
    print('GraphPart excluded', int(z['cluster'].isna().sum()))
