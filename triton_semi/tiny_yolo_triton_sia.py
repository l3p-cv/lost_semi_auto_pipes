from lost.pyapi import script
import os

from detection_helper import detection

ENVS = ['lost']
EXTRA_PIP = ['tritonclient[all]']

ARGUMENTS = {'valid_imgtypes' : { 'value': "['.jpg', '.jpeg', '.png', '.bmp']",
                            'help': 'These are the supported image types!'},
             'model_name' : { 'value': 'tiny_yolo_v4_marvel',
                            'help': 'name of the model that will be used'},
             'url' : { 'value': '192.168.1.23',
                            'help': 'url of the triton inference server (example: IP of device)'},
             'port' : { 'value': '8000',
                            'help': 'used port for request (example: 8000 for http'}
            }

class LostScript(script.Script):
    

    def main(self):
        
        fs_pipe = self.get_fs()
        
        #get images and labels
        data_source = self.inp.datasources
        
        filenames = []
        for ds in data_source:
            fs = ds.get_fs()
            media_path = ds.path
            for root, _ , files in fs.walk(media_path):
                
                for f in files:
                    if os.path.splitext(f)[1].lower() == '.json':
                        lbl_names = os.path.join(root, f)
                    else:
                        path = os.path.join(root, f)
                        if os.path.splitext(path)[1].lower() in self.get_arg('valid_imgtypes'):
                            filenames.append(path)
                        
        
        #load triton model
        triton_client = detection(self.get_arg('model_name'),
                                    self.get_arg('url'),
                                    self.get_arg('port'))
        
        triton_client.load_model()

        #prediction        
        for filename in filenames:
            
            df_bbox_lost = triton_client.predict(filename, 
                                                lbl_names, 
                                                fs_pipe)
            
            self.outp.request_annos(filename,
                                annos = df_bbox_lost.anno_data.values.tolist(), 
                                anno_types = df_bbox_lost.anno_dtype.values.tolist(),
                                anno_labels= df_bbox_lost.anno_class.values.tolist()
                                    )
            
if __name__ == "__main__":
    my_script = LostScript() 
