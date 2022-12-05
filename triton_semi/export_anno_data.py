from lost.pyapi import script
import json
import lost_ds as lds

ENVS = ['lost']

class LostScript(script.Script):
        
    def main(self):
        loop_itr = self.iteration
        
        fs_pipe = self.get_fs()
        
        df = self.inp.to_df()
        df = df[df['img_iteration']==loop_itr]
        ds = lds.LOSTDataset(df)
        
        # export anno data as parquet
        anno_path = self.get_path(f'LOST_Annotation_{loop_itr}.parquet', context='pipe')
        ds.to_parquet(anno_path)
                       
        self.outp.add_data_export(anno_path, fs_pipe)
        
if __name__ == "__main__":
    my_script = LostScript() 
