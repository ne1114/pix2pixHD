import os.path
from pathlib import Path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import json

class StocktonDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = Path(opt.dataroot)
        
        with open(self.root.joinpath("dataset.json")) as f:
            dataset = json.load(f)
        
        self.ids = dataset[opt.phase]   # Get Training doc ids
        self.suffix = self.opt.dataset_suffix   # Get Dataset_suffix []
        self.dataset_size = len(self.ids) 
        
      
    def __getitem__(self, index):        
        B_tensor = inst_tensor = feat_tensor = 0 # Init 0 
        ### Get A and B 
        case_id = self.ids[index]
        AB_path = self.root.joinpath(case_id).joinpath(f"{case_id}{self.suffix}.png")
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
     
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': str(AB_path)}

        return input_dict

    def __len__(self):
        return len(self.ids) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'StocktonDataset'