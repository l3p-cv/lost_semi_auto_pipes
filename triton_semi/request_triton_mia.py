from lost.pyapi import script
import os

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
                            'help': 'used port for request (example: 8000 for http'}
            }
class LostScript(script.Script):
    
    
            
    def main(self):
        
        self.logger.info('Start Triton!')
        
        filenames = []
        for ds in self.inp.datasources:
            media_path = ds.path
            fs = ds.get_fs()
            for root, _ , files in fs.walk(media_path):
                for f in files:
                    path = os.path.join(root, f)
                    filenames.append(path)
                        
        triton_client = classification(self.get_arg('model_name'),
                                      self.get_arg('url'),
                                      self.get_arg('port'))
        
        try:
            triton_client.load_model()    
        except Exception as e:
            self.logger.warning(e)
            raise
            
        try:
            triton_client.triton_predict(filenames)
        except Exception as e:
            self.logger.warning(e)
            raise
            
        for predict, img_sim_class in zip(triton_client.prediction, triton_client.img_sim_class):
            self.outp.request_annos(predict, img_sim_class=img_sim_class)

if __name__ == "__main__":
    my_script = LostScript() 
