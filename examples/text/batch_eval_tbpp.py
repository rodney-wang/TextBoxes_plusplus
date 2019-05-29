#coding=utf-8
import os
import glob
import time
import json
import argparse
import codecs
import numpy as np
from nms import polygon_iou
from plate_det_tbpp import PlateDet
import pdb
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
 Process all plates in benchmark
"""
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "不"]
prov = [i.decode('utf8') for i in provinces]

def tbpp_eval_end2end(label_json, img_dir, out_json, model_path):

    pdet = PlateDet(model_path)
    json_data = json.load(open(label_json, 'r'))
    img_folder = os.path.split(img_dir)[1]
    count = 0
    tbpp_res = {}
    start_time = time.time()
    for fname, label in json_data.iteritems():
        count += 1
        # if count >5:
        #    break
        print "### ", count, fname
        cor =[]
        for lab in label:
            ocr = lab['text'].strip().encode('utf8')
            #pdb.set_trace()
            if ocr[0:3] in provinces:
                 print ocr, len(ocr)
                 cor = lab["coordinates"]
        if len(cor) < 8:
            continue

        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue
        results = pdet(img_path)
        if len(results) == 0:
            continue
        corners = results[0][0:8]
        score   = results[0][8]
        json_data = {'text': ocr,
                     'coordinates': corners,
                     'score': str(score)}
        diff_avg = np.mean(np.abs(np.array(corners) - np.array(cor)))
        iou = polygon_iou(corners, cor)
        print "\tLabels: ", cor
        print "\tResults:", corners
        print "\tIOU:    ", iou
        print "\tAverage diff", diff_avg, " Score =", score
        tbpp_res[fname] = json_data
    print("--- %s seconds ---" % (time.time() - start_time))

    with codecs.open(out_json, 'w') as f:
         json.dump(tbpp_res, f, indent=4)
         print "\n #### Plate labels are written to:", out_json
    #eval(ocrtxt_file, out_dir, skip)

def parse_args():
    parser = argparse.ArgumentParser(description='Plate Segmentation')
    parser.add_argument('-l', '--label_json',
                        default='/ssd/wfei/data/plate_for_label/hk_double/20190505_HK_Double_Plates.json',
                        type=str, help='Plate label in Json format')
    parser.add_argument('-d', '--img_dir', default='/ssd/wfei/data/plate_for_label/hk_double/car_crop_20190505',
                        type=str, help='Input image dir')
    parser.add_argument('-o', '--out_json', default='/mnt/soulfs2/wfei/tmp/tbpp_res_hk0505.json',
                        type=str, help='Output tbpp results in json format')
    parser.add_argument('--model', default='/mnt/soulfs2/wfei/code/TextBoxes_plusplus/models/VGGNet/plate_0526/text_polygon_precise_fix_order_384x384/VGG_text_text_polygon_precise_fix_order_384x384_iter_100000.caffemodel',
                        type=str, help='Caffe model path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tbpp_eval_end2end(args.label_json, args.img_dir, args.out_json, args.model)

