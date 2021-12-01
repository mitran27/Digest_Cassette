# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:32:07 2021

@author: mitran
"""

import numpy as np
import onnxruntime
import cv2
import pyclipper
from shapely.geometry import Polygon




class SegDetectorRepresenter():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=10000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.dilation_kernel = np.array([[1, 1], [1, 1]])
        # self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        
        for batch_index in range(pred.shape[0]):
            
            height, width = batch['shape'][batch_index]
            mask = segmentation[batch_index]
            boxes = self.boxes_from_bitmap(pred[batch_index], mask, width, height)
            
            boxes_batch.append(boxes)
        return boxes_batch

    def binarize(self, pred):
        return pred > self.thresh

    

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap
        pred = pred
        height, width = bitmap.shape
        
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates) # number of boxes
        
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)

        for i in range(num_contours):
            
            
            contour = contours[i].squeeze(1)
            
            points, sside = self.get_mini_boxes(contour)
            
            if sside < self.min_size:
                continue
            
            points = np.array(points)
            score = self.box_score_fast(pred, contour)

            if float(self.box_thresh) > float(score):
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            
            if sside < self.min_size + 2:
                continue
            
            box = np.array(box)
            
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[i, :, :] = box.astype(np.int16)
            
        return boxes

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

class Recognitionwotps:
   def __init__(self,path,shape=(100,32)):

       self.ort_session=onnxruntime.InferenceSession(path)
       self.S=shape
       list_special_token = ['[PAD]', '[UNK]', ' ']
       dict_character = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
       dict_character = list_special_token + dict_character

       self.character = ['[CTCblank]'] + dict_character
   def softmax(self,x,axis):
     return np.exp(x) / np.sum(np.exp(x),axis=axis,keepdims=True)
   def avg(self,x):
       return sum(x)/len(x)
   def decode(self,preds):
        text_index = np.argmax(preds, axis=1)
        score=str(format(np.max(self.softmax(preds,axis=1),axis=1).cumprod(axis=0)[-1],'.2f'))

        length = len(preds)

        texts = []
        characters = []
        for i in range(length):
            if text_index[i] != 0 and (not (i > 0 and text_index[i - 1] == text_index[i])):
                characters.append(self.character[text_index[i]])
            text = ''.join(characters)
        return (text,score)

   def __call__(self,batch):

       image_batch=np.array([np.transpose(cv2.resize(img, self.S, interpolation=cv2.INTER_CUBIC),(2, 0, 1)) for img in batch])
       image_batch = image_batch/127.5 - 1.0
       image_batch=image_batch.astype(np.float32)
       ort_inputs = {self.ort_session.get_inputs()[0].name: image_batch}
       predictions = self.ort_session.run(None, ort_inputs)[0]
       pred_text = [self.decode(p) for p in predictions]


       return pred_text

class Ourocr():
    def __init__(self,path):
        #sess_options = onnxruntime.SessionOptions()
        #sess_options.intra_op_num_threads = 6
        self.detection=onnxruntime.InferenceSession(path+'detection18.onnx')#,sess_options)
        self.recognition=Recognitionwotps(path+'CRNN-PR.onnx')
        self.post_process = SegDetectorRepresenter(thresh=0.5)

    def get_boxes(self,preds,h,w,padvalues):

        
        batch = {'shape': [(h, w)]}
        box_list = self.post_process(batch, preds, is_output_polygon=False)
        box_list = box_list[0]
        box_list = sorted(box_list, key=lambda x: (x[0][1], x[0][0]))

        if(len(box_list)==0):
            return []

        sorted_box_list = self.sort(box_list)

        padded_box_list = []
        for box in sorted_box_list:
            box = self.order_points(box)
            box=self.pad_boxes(box, padvalues)
            padded_box_list.append(box)

        return padded_box_list

    def order_points(self,pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = (rect[0], rect[1], rect[2], rect[3])

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # maxWidth = int(abs(max(rect[:, 0])-min(rect[:, 0])))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # maxHeight = int(abs(max(rect[:, 1]) - min(rect[:,1])))

        dst = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), borderValue=(0, 0, 0))

        return warped

    def sort(self,dt_boxes, run=5):
        dt_boxes=np.array(dt_boxes)
        clean_boxes = dt_boxes[abs(dt_boxes[:, 0][:, 1] - dt_boxes[:, 3][:, 1])<=
                               abs(dt_boxes[:, 0][:, 0] - dt_boxes[:, 1][:, 0])]
        heights = np.linalg.norm(clean_boxes[:, 3] - clean_boxes[:, 0], axis=1)
        h_thresh = np.mean(heights)/2
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = list(sorted(dt_boxes, key=lambda x: (np.mean(x[:,1]), x[0][0])))
        for j in range(run):
            for i in range(num_boxes - 1):
                h1 = abs(sorted_boxes[i+1][3][1]-sorted_boxes[i+1][0][1])
                h2 = abs(sorted_boxes[i][3][1]-sorted_boxes[i][0][1])
                if abs(np.mean(sorted_boxes[i + 1][:, 1]) - np.mean(sorted_boxes[i][:, 1])) < min(h_thresh, h1, h2)  and \
                        (sorted_boxes[i + 1][0][0] < sorted_boxes[i][0][0]):
                    tmp = sorted_boxes[i]
                    sorted_boxes[i] = sorted_boxes[i+1]
                    sorted_boxes[i+1] = tmp
        return sorted_boxes

    def pre_process_imagenet(self,image):

        sh=min(image.shape[:2])
        if(sh>1024):
            rat=1024/sh
            sha=[int(i*rat) for i in image.shape[:2]]
            image=cv2.resize(image,tuple(sha[::-1]))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        image = image.astype(np.float32)
        image /= 255.
        image -= mean
        image /= std
        return image

    def textdetection(self,image):
        h,w=image.shape[:2]

        # check dimesions
        if(len(image.shape)==3):
            image=np.expand_dims(image,0)
        if(image.shape[-1]==3):
            image=np.transpose(image,(0,3,1,2))

        inputs_onnx= {'input':image}

        #text detection

        pred = self.detection.run(None, inputs_onnx)
        return pred

    def pts4_box(self,box):
         x1,y1=np.min(box,axis=0)
         x2,y2=np.max(box,axis=0)
         return int(x1),int(y1),int(x2),int(y2)

    def pad_boxes(self, box, padvalues):

        heights = np.linalg.norm(box[3] - box[0])
        width = np.linalg.norm(box[1] - box[0])

        px = width*padvalues[0]/100
        py = heights*padvalues[1]/32
        # print(px, py)

        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]=box
        box=np.array([[x1-px,y1-py],[x2+px,y2-py],[x3+px,y3+py],[x4-px,y4+py]]).astype(np.int32)

        return box

    def recongnize(self,boundingBoxes,org_image):
        # detection
        text=[]
        data=[]
        image_points=[]
        image_array=[]

        for box in boundingBoxes:
            if box.all()==0:
                continue
            crop_img=self.four_point_transform(org_image, box)
            image_array.append(crop_img)
            image_points.append(box)

        if(len(image_array)>0):
          output=[]
          batch_size=128

          image_array_splited = [image_array[x:x+batch_size] for x in range(0, len(image_array), batch_size)]
          for batch in image_array_splited:
              ocr=self.recognition(batch)
              output.extend(ocr)

          for wrd, pts in zip(output, image_points):
            data.append([pts.tolist(), wrd[0], wrd[1]])

        return data

    def __call__(self,image,padvalues):
        org_image=image.copy()
        h,w=image.shape[:2]
        image=self.pre_process_imagenet(image)
        pred=self.textdetection(image)
        
        bboxes=self.get_boxes(pred[0], h, w, padvalues)
     
        data=self.recongnize(bboxes,org_image)
        return data
    
    



