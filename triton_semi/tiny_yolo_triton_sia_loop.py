from lost.pyapi import script
from lost.pyapi.utils.blacklist import ImgBlacklist
import os
import json

from detection_helper import detection

ENVS = ['lost']
EXTRA_PIP = ['tritonclient[all]']

ARGUMENTS = {'valid_imgtypes' : { 'value': "['.jpg', '.jpeg', '.png', '.bmp']",
                            'help': 'Img types where annotations will be requested for!'},
             'model_name' : { 'value': 'tiny_yolo_v4_marvel',
                            'help': 'name of the model that will be used'},
             'url' : { 'value': '192.168.1.23',
                            'help': 'url of the triton inference server (example: IP of device)'},
             'port' : { 'value': '8000',
                            'help': 'used port for request (example: 8000 for http'},
             'img_batch' : { 'value': '15',
                            'help': 'batch size of the images to annotate per loop'}
            }

class LostScript(script.Script):
    
    def manage_blacklist(self, filenames):
                
        anno_filenames = self.blacklist.get_whitelist(filenames, self.get_arg('img_batch'))
        self.blacklist.add(anno_filenames)
        self.blacklist.save()
        
        # to end the loop
        if len(filenames) == len(self.blacklist.blacklist):
            self.break_loop()
            
        return anno_filenames
    
        
    def main(self):
        
        fs_pipe = self.get_fs()
        
        # get images to annotate
        data_source = self.inp.datasources
        
        filenames = []
        for ds in data_source:
            fs = ds.get_fs()
            media_path = ds.path
            for path in fs.listdir(media_path, detail=False):
                if os.path.splitext(path)[1].lower() in self.get_arg('valid_imgtypes'):
                    filenames.append(path)
                        
        # blacklist with annotated images in the loops befor
        self.blacklist = ImgBlacklist(self, name='image_blacklist.json')
        
        # model version
        version_path = self.get_path(f'model_version.json', context='instance')
                  
        loop_itr = self.iteration
        
        # initial annotation
        if loop_itr == 0:
            for filename in self.manage_blacklist(filenames):
                self.outp.request_annos(filename, fs=fs)

            with fs_pipe.open(version_path, 'w') as fp:
                model_version_dict = {'version': 0}
                json.dump(model_version_dict, fp)
            
        # prediction with triton inference        
        else:
            
            try:
                # load triton model
                triton_client = detection(self.get_arg('model_name'),
                        self.get_arg('url'),
                        self.get_arg('port'))
                
                triton_client.load_model()
                
                
                if not triton_client.model_ready(version_path, fs_pipe):
                    self.reject_execution()
                    
                else:
                    lbl_path = self.get_path(f'labels_loop_{loop_itr - 1}.json', context='pipe')
                    
                    for filename in self.manage_blacklist(filenames):
                        
                        df_bbox_lost = triton_client.predict(filename, 
                                                            lbl_path, 
                                                            fs_pipe)
                        
                        self.outp.request_annos(filename,
                                            annos = df_bbox_lost.anno_data.values.tolist(), 
                                            anno_types = df_bbox_lost.anno_dtype.values.tolist(),
                                            anno_labels= df_bbox_lost.anno_class.values.tolist()
                                                )
            except:
                self.reject_execution()
            
            
            
if __name__ == "__main__":
    my_script = LostScript() 
