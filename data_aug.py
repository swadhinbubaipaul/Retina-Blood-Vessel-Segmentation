import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate, GridDistortion, OpticalDistortion, ElasticTransform


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(train_path, test_path):
    train_x = sorted(glob(os.path.join(train_path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(train_path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(test_path, "test", "images", "*.ppm")))
    test_y = sorted(glob(os.path.join(test_path, "test", "1st_manual", "*.ppm")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        
        name = x.split("/")[-1].split(".")[0]

   
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]
            
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5, border_mode=cv2.BORDER_CONSTANT)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]
              
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]


            X = [x, x1, x2, x3, x4, x5, x6]  
            Y = [y, y1, y2, y3, y4, y5, y6]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":

    np.random.seed(42)


    train_data_path = "/content/drive/MyDrive/Datasets/DRIVE"
    test_data_path = "/content/drive/MyDrive/Datasets/STARE"
    (train_x, train_y), (test_x, test_y) = load_data(train_data_path, test_data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("/content/drive/MyDrive/Program/new_data/train/image/")
    create_dir("/content/drive/MyDrive/Program/new_data/train/mask/")
    create_dir("/content/drive/MyDrive/Program/new_data/test/image/")
    create_dir("/content/drive/MyDrive/Program/new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "/content/drive/MyDrive/Program/new_data/train/", augment=True)
    augment_data(test_x, test_y, "/content/drive/MyDrive/Program/new_data/test/", augment=False)
