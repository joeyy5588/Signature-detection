from PIL import Image
import os
import numpy as np
import torch

def transform(data_dir):
    fn_list = os.listdir(data_dir)
    image_list = []
    for img in fn_list:
        with open(data_dir+img, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = img.resize((170, 220), Image.ANTIALIAS)
            img = np.array(img).astype(np.float32)
            img = 255.0 - img
            img = (img - 127.5) / 127.5
            img = np.expand_dims(img, axis = 0)
            image_list.append(img)
    
    image_list = torch.from_numpy(np.array(image_list))
    save_fn = data_dir[:-1] + '.pt'
    torch.save(image_list, save_fn)
    return

transform('data/mixed/')
transform('data/handwritten/')
transform('data/form/')

        