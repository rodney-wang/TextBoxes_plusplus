#coding=utf-8
import numpy as np
import caffe
from nms import nms
import argparse
import time
#from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms, soft_nms

class PlateDet:

    #def __init__(self):
    #    self.net = self._init_det()

    def __init__(self, caffemodel, det_score_threshold=0.2, overlap_threshold=0.2):
        GPU_ID =3 
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)

        model_fold = '/mnt/soulfs2/wfei/code/TextBoxes_plusplus/models/'
        prototxt = model_fold + 'deploy_384x384.prototxt'
        prototxt = model_fold + 'deploy.prototxt'
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.det_score_threshold = det_score_threshold
        self.overlap_threshold = overlap_threshold

    def __call__(self, img, siz=384):
        """Detect car.

        Input
        -----
        net: network
        img: image

        Output
        ------
        results: num_box x 5
        """
        net = self.net
        #start_time = time.time()
        transformer = caffe.io.Transformer({'data': (1, 3, siz, siz)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        net.blobs['data'].reshape(1, 3, siz, siz)
        image = caffe.io.load_image(img)
        image_height, image_width, channels=image.shape
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']
        bboxes = self.extract_detections(detections, self.det_score_threshold, image_height, image_width )
        #print("--- %s seconds ---" % (time.time() - start_time))
        # apply non-maximum suppression
        #results = self.apply_quad_nms(bboxes, self.overlap_threshold)

        #return results
        return bboxes


    def extract_detections(self, detections, det_score_threshold, image_height, image_width):
        det_conf = detections[0, 0, :, 2]
        det_x1 = detections[0, 0, :, 7]
        det_y1 = detections[0, 0, :, 8]
        det_x2 = detections[0, 0, :, 9]
        det_y2 = detections[0, 0, :, 10]
        det_x3 = detections[0, 0, :, 11]
        det_y3 = detections[0, 0, :, 12]
        det_x4 = detections[0, 0, :, 13]
        det_y4 = detections[0, 0, :, 14]
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= det_score_threshold]
        top_conf = det_conf[top_indices]
        top_x1 = det_x1[top_indices]
        top_y1 = det_y1[top_indices]
        top_x2 = det_x2[top_indices]
        top_y2 = det_y2[top_indices]
        top_x3 = det_x3[top_indices]
        top_y3 = det_y3[top_indices]
        top_x4 = det_x4[top_indices]
        top_y4 = det_y4[top_indices]

        #print top_x1, top_y1, top_x2, top_y2
        bboxes = []
        for i in xrange(top_conf.shape[0]):
            x1 = int(round(top_x1[i] * image_width))
            y1 = int(round(top_y1[i] * image_height))
            x2 = int(round(top_x2[i] * image_width))
            y2 = int(round(top_y2[i] * image_height))
            x3 = int(round(top_x3[i] * image_width))
            y3 = int(round(top_y3[i] * image_height))
            x4 = int(round(top_x4[i] * image_width))
            y4 = int(round(top_y4[i] * image_height))
            #print x1, y1, x2, y2, x3, y3, x4, y4
            x1 = max(1, min(x1, image_width - 1))
            x2 = max(1, min(x2, image_width - 1))
            x3 = max(1, min(x3, image_width - 1))
            x4 = max(1, min(x4, image_width - 1))
            y1 = max(1, min(y1, image_height - 1))
            y2 = max(1, min(y2, image_height - 1))
            y3 = max(1, min(y3, image_height - 1))
            y4 = max(1, min(y4, image_height - 1))
            score = top_conf[i]
            bbox = [x1, y1, x2, y2, x3, y3, x4, y4, score]
            bboxes.append(bbox)
        #print bboxes
        bboxes_sorted = sorted(bboxes, key=lambda item: item[8])
        return bboxes_sorted

    def apply_quad_nms(self, bboxes, overlap_threshold):
        dt_lines = sorted(bboxes, key=lambda x: -float(x[8]))
        nms_flag = nms(dt_lines, overlap_threshold)
        results = []
        for k, dt in enumerate(dt_lines):
            if nms_flag[k]:
                if dt not in results:
                    results.append(dt)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='/ssd/wfei/data/plate_detection/images/images_sub/ch13021_20180921104417-00000007-00005383.jpg')
    opt = parser.parse_args()

    test_img = '/ssd/wfei/data/plate_for_label/wanda_10k/wanda_10k_filtered/1543391448985023026.jpg'
    model_path = '/mnt/soulfs2/wfei/code/TextBoxes_plusplus/models/VGGNet/plate_0526/text_polygon_precise_fix_order_384x384/VGG_text_text_polygon_precise_fix_order_384x384_iter_100000.caffemodel'
    #model_path ='./models/model_pre_train_syn.caffemodel'
    pdet = PlateDet(model_path)
    results = pdet(opt.image)
    print results
    results = pdet(test_img)
    print results

