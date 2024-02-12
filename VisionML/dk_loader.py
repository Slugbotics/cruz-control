import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import json

#TODO if manifest.json exists, delete everything above the dictionary with "paths" to make it valid json

class DataCC(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, "images")
        self.image_list = os.listdir(self.image_folder)
        self.current_catalog = 0
        self.curr_catalog_lines = self.open_file(self.current_catalog)
        self.manifest = json.load(open(os.path.join(self.root_dir, "manifest.json")))


    def __len__(self):
        return int(json.loads(open(os.path.join(self.root_dir, f"catalog_{((len(os.listdir(self.root_dir))-2)//2)-1}.catalog")).readlines()[-1])["_index"])
        

    def __getitem__(self, idx):
        while idx in self.manifest["deleted_indexes"]:
            idx += 1
            idx %= len(self.image_list)
            
        img_name = os.path.join(self.image_folder, f"{idx}_cam_image_array_.jpg")

        if idx > json.loads(self.curr_catalog_lines[-1])["_index"]: # TODO verify 
            self.curr_catalog_lines = self.open_file(self.current_catalog + 1)

        image = Image.open(img_name)
        labels = torch.FloatTensor(self.parse_labels(idx)) # throttle, steering      

        if self.transform:
            image = self.transform(image)

        return image, labels

    def open_file(self, file_idx): # loads desired .catalog file into memory as list 
        return open(os.path.join(self.root_dir, f"catalog_{file_idx}.catalog"),"r").readlines()
    

    def parse_labels(self, idx):
        label_data = json.loads(self.curr_catalog_lines[idx - idx // 1000 * 1000])
        return [label_data["user/angle"], label_data["user/throttle"]]




