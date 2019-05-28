#coding=utf-8
import os
import json
import cv2
import glob
import argparse


def make_xml_anno(json_file, xml_dir, img_dir):
    json_data = json.load(open(json_file, 'r'))
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)
    img_folder = os.path.split(img_dir)[1]
    count = 0
    for fname, label in json_data.iteritems():
        count += 1
        #if count >5:
        #    break
        print count, fname

        basename, file_extension = os.path.splitext(fname)
        xml_path = os.path.join(xml_dir, basename+'.xml')
        f = open(xml_path, 'w')
        line ="<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        f.write(line)
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t<folder>' + img_folder + '</folder>' + '\n'
        f.write(line)
        line = '\t<filename>' + fname + '</filename>\n'
        f.write(line)
        im_path = os.path.join(img_dir, fname)
        im = cv2.imread(im_path)
        if im is None: 
            os.remove(xml_path)
            continue
        (height, width, depth) = im.shape
        #(height, width, depth) = (70,200, 3)

        line = '\t<size>\n\t\t<width>' + str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
        line += '\t<depth>' + str(depth) +'</depth>\n\t</size>'
        f.write(line)

        valid_count = 0
        for lab in label:
            ocr = lab['text'].strip()
            print ocr, len(ocr)
            cor = lab["coordinates"]
            if len(cor) < 8 or not ocr:
                continue
            difficulty = 0
            if len(ocr)<7:
                difficulty = 1
                #continue   #uncomment if we want to exclude the 遮挡， 不清晰 cases

            xmin = min(cor[::2])
            xmax = max(cor[::2])
            ymin = min(cor[1::2])
            ymax = max(cor[1::2])
            line = '\n\t<object>'
            line += '\n\t\t<difficult>' + str(difficulty) + '</difficult>'
            line += '\n\t\t<name>text</name>'
            line += '\n\t\t<content>'+ocr.encode('utf8')+'</content>'
            line += '\n\t\t<bndbox>'
            line += '\n\t\t\t<x1>' + str(cor[0]) + '</x1>'
            line += '\n\t\t\t<y1>' + str(cor[1]) + '</y1>'
            line += '\n\t\t\t<x2>' + str(cor[2]) + '</x2>'
            line += '\n\t\t\t<y2>' + str(cor[3]) + '</y2>'
            line += '\n\t\t\t<x3>' + str(cor[4]) + '</x3>'
            line += '\n\t\t\t<y3>' + str(cor[5]) + '</y3>'
            line += '\n\t\t\t<x4>' + str(cor[6]) + '</x4>'
            line += '\n\t\t\t<y4>' + str(cor[7]) + '</y4>'
            line += '\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
            line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
            line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
            line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
            line += '\n\t\t</bndbox>'
            line += '\n\t</object>'
            f.write(line)
            valid_count += 1
        line='\n</annotation>' + '\n'
        f.write(line)
        f.close()
        if valid_count == 0:
            os.remove(xml_path)


def write_train_list_file(img_dir, xml_dir, list_file):

    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    with open(list_file, 'w') as f:
        for idx, xmlfile in enumerate(xml_files):
            bname = os.path.basename(xmlfile)
            if idx %1000 ==0:
                print idx, bname
            img_name = bname.replace('.xml', '.jpg')
            #xml_path = os.path.join('data/plate_detection/annotations/xml_wanda_0921', bname)
            #img_path = os.path.join('data/plate_detection/images/images_sub', img_name)
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path) and os.path.exists(xmlfile):
                f.write(' '.join([img_path, xmlfile])+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Plate end to end test')
    parser.add_argument('--img_dir', default='/ssd/wfei/data/plate_detection/images/images_sub',
                        type=str, help='Plate label file in txt format')
    parser.add_argument('--json', default='/ssd/wfei/data/plate_detection/annotations/20181119_carplate_wanda_0921.json',
                       type=str, help='Output plate label dir')
    parser.add_argument('--xml_dir', default='/ssd/wfei/data/plate_detection/annotations/xml_wanda_0921',
                        type=str, help='Output xml annotation dir')
    parser.add_argument('--list_file', default='/mnt/soulfs2/wfei/code/TextBoxes_plusplus/data/text/train_wanda_0921.txt',
                        type=str, help='List of image + xml (label) pair paths, which is used for conversion into lmdb')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    make_xml_anno(args.json, args.xml_dir, args.img_dir)
    write_train_list_file(args.img_dir, args.xml_dir, args.list_file)

