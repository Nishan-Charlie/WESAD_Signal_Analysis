import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np

WESAD_PATH = r'c:\Users\nisha\OneDrive\Desktop\Quantum_Computing\MultiModal_Quantum_Fusion\WESAD'
all_subjects = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']
label_map = {1: 0, 3: 1, 2: 2}
names = {0: 'No Stress (Baseline)', 1: 'Low Stress (Amusement)', 2: 'High Stress (Stress)'}

splits = [('TRAIN', all_subjects[:10]), ('VAL', all_subjects[10:12]), ('TEST', all_subjects[12:])]

for split_name, subs in splits:
    counts = {0: 0, 1: 0, 2: 0}
    for sid in subs:
        fp = os.path.join(WESAD_PATH, sid, f"{sid}.pkl")
        with open(fp, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        labels = d['label'].flatten()
        for orig_lbl, mapped_lbl in label_map.items():
            counts[mapped_lbl] += int((labels == orig_lbl).sum())
    
    total = sum(counts.values())
    print(f"\n{split_name}: {subs}")
    for c in [0, 1, 2]:
        pct = counts[c] / total * 100 if total > 0 else 0
        print(f"  {names[c]}: {counts[c]:,} ({pct:.1f}%)")
    
    # Print suggested class weights
    weights = [total / (3 * counts[c]) if counts[c] > 0 else 1.0 for c in [0, 1, 2]]
    print(f"  => Suggested class weights: {[round(w,3) for w in weights]}")
