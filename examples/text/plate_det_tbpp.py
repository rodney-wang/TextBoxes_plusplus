#coding=utf-8
import numpy as np
import caffe
import nms
#from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms, soft_nms

class PlateDet:

    #def __init__(self):
    #    self.net = self._init_det()

    def __init__(self, caffemodel, det_score_threshold=0.2, overlap_threshold=0.2):
        GPU_ID = 0
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)

        model_fold = '/mnt/soulfs2/wfei/code/TextBoxes_plusplus/models/'
        prototxt = model_fold + 'deploy_384x384.prototxt'
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
        transformer = caffe.io.Transformer({'data': (1, 3, siz, siz)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        net.blobs['data'].reshape(1, 3, siz, siz)

        image = caffe.io.load_image(img)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        detections = net.forward()['detection_out']
        bboxes = self.extract_detections(detections, self.det_score_threshold, siz, siz)
        # apply non-maximum suppression
        results = self.apply_quad_nms(bboxes, self.overlap_threshold)

        return results

        """
        dets = im_detect(net, img, siz)
        inds = np.where(dets[:, -1] == 1)[0]
        cls_dets2 = []
        if inds.shape[0] > 0:
            cls_dets = dets[inds, :-1].astype(np.float32)
            confidence_threshold = 0.00999999977648
            keep = soft_nms(cls_dets, sigma=0.5, Nt=0.30, threshold=confidence_threshold, method=1)
            cls_dets = cls_dets[keep, :]
            index = np.where(cls_dets[:, -1] > 0.3)[0]
            cls_dets2 = cls_dets[index, :]
        return cls_dets2
        """


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

    test_img = '/ssd/wfei/data/plate_for_label/wanda_10k/wanda_10k_filtered/1540722158754090215.jpg'
    model_path = '/mnt/soulfs2/wfei/code/TextBoxes_plusplus/models/VGGNet//text/text_polygon_precise_fix_order_384x384/VGG_text_text_polygon_precise_fix_order_384x384_iter_120000.caffemodel'
    pdet = PlateDet(model_path)
    results = pdet(test_img)
    print results
