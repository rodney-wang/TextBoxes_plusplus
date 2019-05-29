import json
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from calculate_mean_ap_tbpp import get_avg_precision_at_iou, plot_pr_curve, COLORS

parser = argparse.ArgumentParser(description='Validate detection mAP')
parser.add_argument('--anno_json', default='/ssd/wfei/data/plate_detection/annotations/20181025_carplate_k11_det_gt.json',
                    type=str, help='Input image dir')
parser.add_argument('--det_json', default='/ssd/wfei/data/plate_detection/predictions/k11_bm_det_results_size500_caffe98000.json',
                    type=str, help='Output json file in COCO format for detection validation')
args = parser.parse_args()
anno_json = '/Users/fei/data/parking/plate_detection/20181025_carplate_k11_20181014.json'
det_json  = '/Users/fei/data/parking/plate_detection/k11_4k_bm_tbpp_vgg_res.json'

anno = json.load(open(anno_json))
res  = json.load(open(det_json))

gt_img_set = set()
gt_boxes = {}
for fname, det in anno.iteritems():
    if len(det)<1:
        continue
    box = det[0]["coordinates"]
    if len(box) >= 8:
        gt_img_set.add(fname)
        gt_boxes[fname] = [box]
print "Number of images in GT", len(gt_img_set)

pred_boxes = {}
for fname, det in res.iteritems():
    if fname not in gt_img_set:
        continue
    box = det["coordinates"]
    if 'score' not in det:
        score = np.random.random()
        score = score/10.+0.9
    else:
        score = det['score']
    pred_boxes[fname] = {'boxes': [box], 'scores':[score]}
print "Number of images in Detection",  len(pred_boxes)

ax = None
avg_precs = []
iou_thrs = []
start_time = time.time()
for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):#np.linspace(0.5, 0.95, 10)):
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    avg_precs.append(data['avg_prec'])
    iou_thrs.append(iou_thr)
    print "Average precision for ", iou_thr, "=", data['avg_prec']
    precisions = data['precisions']
    recalls = data['recalls']
# prettify for printing:
avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
print('map: {:.2f}'.format(100*np.mean(avg_precs)))
print('avg precs: ', avg_precs)
print('iou_thrs:  ', iou_thrs)
if False:
    ax = plot_pr_curve(
        precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx * 2], ax=ax)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    plt.show()
end_time = time.time()
print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
#results = get_avg_precision_at_iou(gt_boxes, pred_boxes, 0.5)

#print results
