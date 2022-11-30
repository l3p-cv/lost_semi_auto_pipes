from lost.pyapi import script
import json
import lost_ds as lds

ENVS = ['lost']

class LostScript(script.Script):
    
    def merch_save_classes(self, class_dict, ds, path, fs):
        for label in list(ds.unique_labels(col='anno_lbl')):
            if not label in class_dict.values():
                new_id = len(class_dict)
                class_dict[new_id] = label
            
        with fs.open(path, 'w') as fp:
            json.dump(class_dict, fp)
    
    def main(self):

        fs_pipe = self.get_fs()
        
        df = self.inp.to_df()
        
        ds = lds.LOSTDataset(df)
        
        ds.remove_empty(inplace=True)
        
        loop_itr = self.iteration
        
        old_lbl_path = self.get_path(f'labels_loop_{loop_itr - 1}.json', context='pipe')
        new_lbl_path = self.get_path(f'labels_loop_{loop_itr}.json', context='pipe')
        
        # model class names to json
        if loop_itr == 0:
            class_dict = {}
            
            self.merch_save_classes(class_dict, ds, new_lbl_path, fs_pipe)
            
        else:
            with fs_pipe.open(old_lbl_path, 'r') as fp:
                class_dict = json.load(fp)
                
            self.merch_save_classes(class_dict, ds, new_lbl_path, fs_pipe)
        
        # export anno data as parquet
        anno_path = self.get_path(f'LOST_Annotation_{loop_itr}.parquet', context='pipe')
        ds.to_parquet(anno_path)
                       
        self.outp.add_data_export(new_lbl_path, fs_pipe)
        self.outp.add_data_export(anno_path, fs_pipe)
        
if __name__ == "__main__":
    my_script = LostScript() 
