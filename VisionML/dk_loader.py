import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
from PIL import Image
import json
import time

#TODO if manifest.json exists, delete everything above the dictionary with "paths" to make it valid json

class DataCC(Dataset):
    def __init__(self, root_dir, transform=None, test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, "images")
        self.image_list = os.listdir(self.image_folder)
        self.current_catalog = 0
        self.curr_catalog_lines = self.open_file(self.current_catalog)
        self.manifest = json.load(open(os.path.join(self.root_dir, "manifest.json")))
        self.test = test


    def __len__(self):
        return int(json.loads(open(os.path.join(self.root_dir, f"catalog_{((len(os.listdir(self.root_dir))-2)//2)-1}.catalog")).readlines()[-1])["_index"])
        

    def __getitem__(self, idx):
        while idx in self.manifest["deleted_indexes"]:
            idx += 1
            idx %= len(self.image_list)
            
        img_name = os.path.join(self.image_folder, f"{idx}_cam_image_array_.jpg")

        self.curr_catalog_lines = self.open_file(idx)

        image = Image.open(img_name)
        
        steering, throttle = self.parse_labels(idx) # steering, throttle      

        if self.transform:
            image = self.transform(image)

        return image, (torch.FloatTensor([steering]), torch.FloatTensor([throttle]))

    def open_file(self, idx): # loads desired .catalog file into memory as list 
        # get correct catalog indx from inpt indx
        return open(os.path.join(self.root_dir, f"catalog_{idx // 1000}.catalog"),"r").readlines()
    

    def parse_labels(self, idx):
        label_data = json.loads(self.curr_catalog_lines[idx - idx // 1000 * 1000])
        return label_data["user/angle"], label_data["user/throttle"]



def test_loader():
    import torchvision.transforms as transforms
    from torch.utils.data import random_split, DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Run on {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])

    dataset = DataCC("dc_dataset_2", transform=transform, test=True)
    train_size = int(0.3 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    valloader = DataLoader(val, batch_size=1, 
                        shuffle=True, num_workers=1)
    
    for inputs, (steering_targets, throttle_targets) in valloader:
        # inputs = inputs.to(device)
        steering_targets = steering_targets.to(device)
        throttle_targets = throttle_targets.to(device)

        # print(inputs, steering_targets, throttle_targets) # return img_name

if __name__ == '__main__':
    test_loader()
