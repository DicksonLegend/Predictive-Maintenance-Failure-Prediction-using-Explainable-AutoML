import json

with open('results/phase56_evidence/03_dashboard_post_fix.json', encoding='utf-8-sig') as f:
    d = json.load(f)

dash = d.get('dashboard', {})
panels = dash.get('panels', [])
print(f"Total panels found: {len(panels)}")
all_ok = True

for i, p in enumerate(panels):
    ds = p.get('datasource', {})
    ds_uid = ds.get('uid') if isinstance(ds, dict) else ds
    title = p.get('title')
    print(f"Panel {i+1} ({title}): datasource uid = {ds_uid}")
    if ds_uid != 'prometheus':
        all_ok = False
        print(f"  [FAIL] Panel {i+1} datasource uid is {ds_uid}")
    for j, t in enumerate(p.get('targets', [])):
        tds = t.get('datasource', {})
        tds_uid = tds.get('uid') if isinstance(tds, dict) else tds
        print(f"   Target {j+1}: datasource uid = {tds_uid}")
        if tds_uid != 'prometheus':
            all_ok = False
            print(f"  [FAIL] Target {j+1} datasource uid is {tds_uid}")

print(f"Dashboard time: {dash.get('time')}")
print(f"Dashboard refresh: {dash.get('refresh')}")
print(f"All panels assert passed: {all_ok}")
assert all_ok, 'Not all panel datasource uids are prometheus!'
