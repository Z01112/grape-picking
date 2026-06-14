#!/usr/bin/env python
"""Simple diagnosis: compare postprocessor picking_points with annotation picking_points."""
import json, os, torch, sys
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

# Load annotations
with open('datasets/test/_annotations.grape_point.json') as f:
    ann_data = json.load(f)

gt_by_img_bbox = defaultdict(dict)  # img_id -> bbox_id -> picking_point pixel coords
bbox_map = defaultdict(dict)        # img_id -> bbox_id -> annotation
for ann in ann_data['annotations']:
    bbox_map[ann['image_id']][ann['id']] = ann
    if ann.get('has_picking') or ann.get('has_picking_point'):
        pt = ann.get('picking_point', ann.get('keypoints', [0, 0])[:2])
        gt_by_img_bbox[ann['image_id']][ann['id']] = pt

print(f"Annotations: {sum(len(v) for v in bbox_map.values())} bboxes, {sum(len(v) for v in gt_by_img_bbox.values())} picking")

# Load model
from engine.core.yaml_config import YAMLConfig
cfg = YAMLConfig('configs/rtv4/_test_b1pca_temp.yml', 
                 resume='outputs/01_mainline_results/candidate_b1_point_cross_attn_new1804_fair100/best_composite.pth',
                 test_only=True)
model = cfg.model
model.load_state_dict(torch.load('outputs/01_mainline_results/candidate_b1_point_cross_attn_new1804_fair100/best_composite.pth', map_location='cpu', weights_only=False)['model'], strict=False)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
pp = cfg.postprocessor

from engine.data.dataset import GrapePointCocoDetection
from engine.data.transforms._transforms import Resize, ConvertPILImage
from engine.data.transforms.container import Compose
from engine.data.dataloader import batch_image_collate_fn
from torchvision.ops import box_iou

ds = GrapePointCocoDetection('datasets/test', 'datasets/test/_annotations.grape_point.json',
                             transforms=Compose([Resize(size=[640,640]), ConvertPILImage(dtype='float32', scale=True)]),
                             return_masks=False, regenerate_point_offsets=False,
                             point_offset_mode='top_center', point_top_anchor_ratio=0.12)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, collate_fn=batch_image_collate_fn)

# Run inference
all_preds, all_targets = [], []
with torch.no_grad():
    for samples, targets in loader:
        samples = samples.to(device)
        outputs = model(samples, targets=targets)
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        preds = pp(outputs, orig_sizes)
        all_preds.extend(preds)
        all_targets.extend(targets)

# Analyze
records = []
fp_recs = []

for pred, target in zip(all_preds, all_targets):
    img_id = target['image_id'].item()
    gt_img_anns = bbox_map.get(img_id, {})
    ann_ids = list(gt_img_anns.keys())
    
    pboxes = pred['boxes'].cpu().numpy()
    ppoints = pred['picking_points'].cpu().numpy()
    php = pred['has_picking'].cpu().numpy()
    phps = pred['has_picking_scores'].cpu().numpy()
    
    # GT boxes in pixel xywh -> xyxy for IoU matching
    gt_boxes_raw = np.array([ann['bbox'] for ann in gt_img_anns.values()])  # xywh in pixels
    gt_xyxy = np.stack([gt_boxes_raw[:,0], gt_boxes_raw[:,1], 
                         gt_boxes_raw[:,0]+gt_boxes_raw[:,2], gt_boxes_raw[:,1]+gt_boxes_raw[:,3]], axis=1)
    
    # Match: pred bbox to GT bbox via IoU
    matched = {}
    if len(pboxes) > 0 and len(gt_xyxy) > 0:
        ious = box_iou(torch.tensor(pboxes), torch.tensor(gt_xyxy))
        for p in range(len(pboxes)):
            bg = ious[p].argmax().item()
            if ious[p, bg] > 0.5 and bg not in {v[0] for v in matched.values()}:
                matched[p] = (bg, ann_ids[bg])
    
    # Per picking GT
    for p, (g_idx, ann_id) in matched.items():
        ann = gt_img_anns[ann_id]
        is_pick_gt = bool(ann.get('has_picking') or ann.get('has_picking_point'))
        if not is_pick_gt:
            continue
        
        gt_pt = ann.get('picking_point', ann.get('keypoints', [0,0])[:2])
        rec = {
            'img_id': img_id, 'ann_id': ann_id,
            'bbox_area': ann['area'],
            'bbox_w': ann['bbox'][2], 'bbox_h': ann['bbox'][3],
            'hp_score': float(phps[p]), 'pred_pick': bool(php[p]),
            'is_fn': not bool(php[p]), 'is_tp': bool(php[p]),
        }
        if ppoints.shape[0] > p:
            pp = ppoints[p]
            l2 = float(np.sqrt((pp[0]-gt_pt[0])**2 + (pp[1]-gt_pt[1])**2))
            rec['l2_px'] = l2; rec['dx_px'] = float(abs(pp[0]-gt_pt[0])); rec['dy_px'] = float(abs(pp[1]-gt_pt[1]))
            rec['is_bad_l2'] = l2 > 30
        records.append(rec)
    
    # Unmatched FNs
    matched_ann_ids = {v[1] for v in matched.values()}
    for ann_id, ann in gt_img_anns.items():
        if (ann.get('has_picking') or ann.get('has_picking_point')) and ann_id not in matched_ann_ids:
            records.append({'img_id': img_id, 'ann_id': ann_id, 'bbox_area': ann['area'],
                           'bbox_w': ann['bbox'][2], 'bbox_h': ann['bbox'][3],
                           'has_match': False, 'is_fn': True, 'is_tp': False})
    
    # FPs
    matched_ps = {k for k in matched.keys()}
    for p in range(len(pboxes)):
        if p < len(php) and php[p]:
            if p in matched_ps:
                g_idx = matched[p][0]
                ann = gt_img_anns[ann_ids[g_idx]]
                if not (ann.get('has_picking') or ann.get('has_picking_point')):
                    fp_recs.append({'img_id': img_id, 'hp_score': float(phps[p]), 'type': 'matched_nonpick'})
            else:
                fp_recs.append({'img_id': img_id, 'hp_score': float(phps[p]), 'type': 'unmatched'})

# ── Report ──
tps = [r for r in records if r.get('is_tp')]
fns = [r for r in records if r.get('is_fn')]
with_l2 = [r for r in records if 'l2_px' in r]
bad_l2 = [r for r in with_l2 if r['l2_px'] > 30]

print(f"\n{'='*50}")
print(f"TEST SET DIAGNOSIS - B1+PointCA")
print(f"{'='*50}")
print(f"GT picking: {len(records)}")
print(f"TP: {len(tps)} ({100*len(tps)/len(records):.1f}%)")
print(f"FN: {len(fns)} ({100*len(fns)/len(records):.1f}%)")
print(f"FP: {len(fp_recs)}")
if with_l2:
    l2_vals = [r['l2_px'] for r in with_l2]
    print(f"L2>30: {len(bad_l2)}/{len(with_l2)} ({100*len(bad_l2)/len(with_l2):.1f}%)")
    print(f"Mean L2: {np.mean(l2_vals):.1f}px  Median: {np.median(l2_vals):.1f}px")
    dx = [r['dx_px'] for r in with_l2]
    dy = [r['dy_px'] for r in with_l2]
    print(f"Mean |dx|: {np.mean(dx):.1f}px  |dy|: {np.mean(dy):.1f}px  dy/dx: {np.mean(dy)/np.mean(dx):.2f}")

# FN by bbox size
print(f"\nFN by bbox area (pixels):")
for lo, hi, label in [(0, 20000, 'Tiny'), (20000, 80000, 'Small'), (80000, 200000, 'Medium'), (200000, 1e9, 'Large')]:
    fn_sub = [r for r in fns if lo <= r.get('bbox_area', 0) < hi]
    if fn_sub:
        near = sum(1 for r in fn_sub if r.get('hp_score', 0) > 0.4)
        print(f"  {label}: {len(fn_sub)}, near-miss={near}")

# L2 by size
if with_l2:
    print(f"\nL2 by bbox area:")
    for lo, hi, label in [(0, 20000, 'Tiny'), (20000, 80000, 'Small'), (80000, 200000, 'Medium'), (200000, 1e9, 'Large')]:
        sub = [r for r in with_l2 if lo <= r['bbox_area'] < hi]
        if sub:
            bad = sum(1 for r in sub if r.get('is_bad_l2'))
            print(f"  {label}: n={len(sub)} mean={np.mean([r['l2_px'] for r in sub]):.1f}px L2>30={bad}")

# FP profile
if fp_recs:
    hp = [r['hp_score'] for r in fp_recs]
    print(f"\nFP: {len(fp_recs)} total, mean HP={np.mean(hp):.3f}")
    print(f"  Borderline(0.5-0.7): {sum(1 for v in hp if 0.5<v<0.7)}")
    print(f"  Confident(>=0.7): {sum(1 for v in hp if v>=0.7)}")

# Worst L2
if with_l2:
    print(f"\nTop 10 worst L2:")
    for r in sorted(with_l2, key=lambda x: x['l2_px'], reverse=True)[:10]:
        print(f"  img={r['img_id']} L2={r['l2_px']:.1f} dx={r.get('dx_px',0):.1f} dy={r.get('dy_px',0):.1f} area={r['bbox_area']:.0f}")
print(f"{'='*50}")
