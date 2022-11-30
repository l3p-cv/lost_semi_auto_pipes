# MIT License

# Copyright (c) 2022 pythonlessons

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import numpy as np
import pandas as pd
import json

import tritonclient.http as httpclient
import lost_ds as lds

class detection:
    
    def __init__(self, model_name, url, port):
        self.model_name = model_name
        self.inputs = []
        self.outputs = []
        self.input_size = 416
        self.url = url
        self.port = port
        self.model_version = 0
        
        
    def image_preprocess(self, image, target_size, gt_boxes=None):
        ih, iw    = target_size
        h,  w, _  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes
        
    def jsonKeys2int(self, x):
        # convert dict keys from string to int
        if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
        return x     
                  
    def read_class_names(self, class_file_name, fs_pipe):
        # loads class name from a json file
        
        with fs_pipe.open(class_file_name, 'r') as fp:
            names = json.load(fp)
        
        names = self.jsonKeys2int(names)
        return names
  
    def bboxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious    
    
    def nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0 
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes
    
    def postprocess_boxes(self, pred_bbox, original_image, input_size, score_threshold):
        valid_scale=[0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def get_image_data(self, filename, input_size):
            
        original_image = cv2.imread(filename)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = self.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        return (image_data, original_image)
            
    def load_model(self):
        '''
        load the model metadata and build the inputs and outputs
        '''
        
        self.triton_client = httpclient.InferenceServerClient(url=f'{self.url}:{self.port}',
                                                              verbose=False)
        
        try:    
            self.triton_client.is_server_live()
        except:
            raise Exception('no conection to server. Check IP and Port')
        
        self.metadata = self.triton_client.get_model_metadata(self.model_name)
        
        for input in self.metadata['inputs']:
            self.inputs.append(httpclient.InferInput(input['name'], 
                                                     [1, self.input_size, self.input_size, 3], 
                                                     "FP32"))

        for output in reversed(self.metadata['outputs']):
            self.outputs.append(httpclient.InferRequestedOutput(output['name']))
            
    def model_ready(self, version_path, fs_pipe):
        '''
        arg:
            version_path: path with the last used model version (init version is 0)
            fs_pipe: filesystem of the pipeline (local, cloud...)
        
        return: true, if a higher version of the model is available since the loop befor.
        '''
        model_stat = self.triton_client.get_inference_statistics(self.model_name)
        
        stat_list = model_stat['model_stats']
        act_version = stat_list[0]['version']
        
        with fs_pipe.open(version_path, 'r') as fp:
                last_version = json.load(fp)
                 
        if int(act_version) > int(last_version['version']):
            with fs_pipe.open(version_path, 'w') as fp:
                
                new_model_version_dict = {'version': act_version}
                json.dump(new_model_version_dict, fp)
            return True
        else:
            return False
    
    def predict(self, filename, lbl_names, fs_pipe):
        '''
        arg:
            filename: path of the image
            lbl_names: path of the class names json
            fs_pipe: filesystem of the pipeline (local, cloud...)
        
        return: df_bbox_lost
            :param df_bbox_lost: (anno_data,
                                anno_style, 
                                anno_format,
                                anno_class,
                                anno_dtype,
                                img_path)
        '''
        
        dict_class_names = self.read_class_names(lbl_names, 
                                                fs_pipe)
        
        image_data, original_image = self.get_image_data(filename, 
                                                        self.input_size)
            
        self.inputs[0].set_data_from_numpy(image_data)

        triton_prediction = self.triton_client.infer(model_name=self.model_name,
                                                    inputs=self.inputs,
                                                    outputs=self.outputs)
        results = []

        for output in reversed(self.metadata['outputs']):
            results.append(triton_prediction.as_numpy(output['name']))

        pred_bbox_tr = [np.reshape(x, (-1, np.shape(x)[-1])) for x in results]
        pred_bbox_tr = np.concatenate(pred_bbox_tr, axis=0)
        
        bboxes_tr = self.postprocess_boxes(pred_bbox_tr, 
                                            original_image, 
                                            self.input_size, 
                                            score_threshold=0.3)
        
        bboxes_tr = self.nms(bboxes_tr, 
                            iou_threshold=0.45, 
                            method='nms')

        
        """
        annos must be relativ for LOST
        annos transform from x1y1x2y2 to xcycwh
        annos transform from abs to rel
        """
                    
        dict_bbox = {'anno_data': [],
                    'anno_style': [],
                    'anno_format': [],
                    'anno_class': [],
                    'anno_dtype': [],
                    'img_path': []}
        
        for bboxes in bboxes_tr:
            dict_bbox['anno_data'].append(bboxes[:4]) 
            dict_bbox['anno_style'].append('x1y1x2y2')
            dict_bbox['anno_format'].append('abs') 
            dict_bbox['anno_class'].append(dict_class_names[bboxes[5]])
            dict_bbox['anno_dtype'].append('bbox')
            dict_bbox['img_path'].append(filename)     

        
        df_bbox = pd.DataFrame(dict_bbox)
        df_bbox_rel = lds.to_rel(df_bbox)
        df_bbox_lost = lds.transform_bbox_style('xcycwh', df_bbox_rel)
        
        return df_bbox_lost
  