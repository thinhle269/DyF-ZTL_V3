import io
import json
import re

AUX = 'D:/Samia/Round_02/Extension/DyF_ZTL_R2_v22_reconciled/DyF-ZTL_R2.aux'
B = chr(92)
aux = io.open(AUX, encoding='utf-8', errors='ignore').read()
pat = re.compile(B + B + 'newlabel' + B + '{([^}]+)' + B + '}' + B + '{' + B + '{([^}]*)' + B + '}' + B + '{([^}]*)' + B + '}')
m = {}
for lab, num, page in pat.findall(aux):
    if lab.startswith(('sec:', 'tab:', 'fig:', 'alg:', 'app:', 'eq:')):
        m[lab] = dict(num=num, page=page)
json.dump(m, io.open('D:/Samia/Round_02/Extension/drafts/their_label_map.json', 'w', encoding='utf-8'), indent=0)
by = {}
for lab, v in m.items():
    by.setdefault((lab.split(':')[0], v['num']), []).append(lab)
dup = {k: v for k, v in by.items() if len(v) > 1}
print('labels', len(m), '| duplicated (kind, number):', dup)
print('figures:', {k: v['num'] for k, v in m.items() if k.startswith('fig:')})
print('apps/algs:', {k: v['num'] for k, v in m.items() if k.startswith(('app:', 'alg:'))})
print('tables:', {k: v['num'] for k, v in m.items() if k.startswith('tab:')})

