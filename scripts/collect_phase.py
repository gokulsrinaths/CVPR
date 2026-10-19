import glob, json, collections
ROOT="/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/experiments/pilot"
LV=["blur32","blur16","blur8","blur4","blur2","base"]
spec={"PMC-VQA":("dose_q3b_length*_seed*_2*",None),"A-OKVQA":("a21_q3b_length*_seed*_2*",None),"OmniMedVQA":("a29_q3b_length*_seed*_2*",None)}
out={}
for name,(pat,_) in spec.items():
    cells=collections.defaultdict(list)
    for d in glob.glob(f"{ROOT}/{pat}"):
        if "smoke" in d: continue
        f=d+"/summary_dose.json"
        try: s=json.load(open(f))
        except Exception: continue
        if s.get("shortcut")!="length" or s.get("family")!="q3b": continue
        for lv in LV:
            g=s["g"][lv]
            cells[(s["dose"],lv)].append((g["g"],g["ci95"][0], g["g"]>=2.0 and g["ci95"][0]>=1.0))
    doses=sorted({k[0] for k in cells})
    out[name]={"doses":doses,"levels":LV,"mean_g":{f"{d}|{lv}":sum(x[0] for x in cells[(d,lv)])/len(cells[(d,lv)]) for d in doses for lv in LV},
               "veto_all":{f"{d}|{lv}":all(x[2] for x in cells[(d,lv)]) for d in doses for lv in LV},"n_seeds":{str(d):len(cells[(d,"base")]) for d in doses}}
    print(name,doses,out[name]["n_seeds"])
json.dump(out,open("/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/tmp/phase_data.json","w"))
