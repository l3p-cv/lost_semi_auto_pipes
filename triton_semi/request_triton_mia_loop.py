from lost.pyapi import script
from lost.pyapi.utils.blacklist import ImgBlacklist
import os
import json

from classification_helper import classification

ENVS = ['lost']
EXTRA_PIP = ['tritonclient[all]']
ARGUMENTS = {'valid_imgtypes' : { 'value': "['.jpg', '.jpeg', '.png', '.bmp']",
                            'help': 'Img types where annotations will be requested for!'},
             'model_name' : { 'value': 'marvel_2',
                            'help': 'name of the model that will be used'},
             'url' : { 'value': '192.168.1.23',
                            'help': 'url of the triton inference server (example: IP of device)'},
             'port' : { 'value': '8000',
                            'help': 'used port for request (example: 8000 for http'},
             'img_batch' : { 'value': '50',
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
        
        self.logger.info('Start Triton!')
        
        filenames = []
        for ds in self.inp.datasources:
            media_path = ds.path
            fs = ds.get_fs()
            for root, _ , files in fs.walk(media_path):
                for f in files:
                    path = os.path.join(root, f)
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
                triton_client = classification(self.get_arg('model_name'),
                                            self.get_arg('url'),
                                            self.get_arg('port'))
                
                try:
                    triton_client.load_model()    
                except Exception as e:
                    self.logger.warning(e)
                    raise
                
                if not triton_client.model_ready(version_path, fs_pipe):
                    self.reject_execution()
                    
                else:
                    try:
                        triton_client.triton_predict(self.manage_blacklist(filenames))
                    except Exception as e:
                        self.logger.warning(e)
                        raise
                        
                    for predict, img_sim_class in zip(triton_client.prediction, triton_client.img_sim_class):
                        self.outp.request_annos(predict, img_sim_class=img_sim_class)
                        
            except:
                self.reject_execution()

if __name__ == "__main__":
    my_script = LostScript() 
