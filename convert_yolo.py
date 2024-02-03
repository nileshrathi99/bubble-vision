import os
import sys
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def split_data(data_path):
    img_dir = os.path.join(data_path,'images')
    # annot_dir = os.path.join(data_path,'labels')
    
    train_img_dir = os.path.join(data_path,'train','images')
    train_annot_dir = os.path.join(data_path,'train','labels')
    val_img_dir = os.path.join(data_path,'val','images')
    val_annot_dir = os.path.join(data_path,'val','labels')

    os.makedirs(train_img_dir,exist_ok=True)
    os.makedirs(train_annot_dir,exist_ok=True)
    os.makedirs(val_img_dir,exist_ok=True)
    os.makedirs(val_annot_dir,exist_ok=True)

    img_list = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]
    train_list = img_list[:400]
    val_list = img_list[400:]

    for img in train_list:
        src = os.path.join(data_path,'images',img)
        dst = os.path.join(train_img_dir,img)
        os.rename(src,dst)

        src = os.path.join(data_path,'labels',img.replace('.jpg','.txt'))
        dst = os.path.join(train_annot_dir,img.replace('.jpg','.txt'))
        os.rename(src,dst)
    
    for img in val_list:
        src = os.path.join(data_path,'images',img)
        dst = os.path.join(val_img_dir,img)
        os.rename(src,dst)

        src = os.path.join(data_path,'labels',img.replace('.jpg','.txt'))
        dst = os.path.join(val_annot_dir,img.replace('.jpg','.txt'))
        os.rename(src,dst)

def main(data_path):
    image_dir = os.path.join(data_path,'Images')
    annot_dir = os.path.join(data_path,'Annotations')
    yolo_img_dir = os.path.join(data_path,'images')
    yolo_annot_dir = os.path.join(data_path,'labels')

    if os.path.exists(yolo_annot_dir):
        shutil.rmtree(yolo_annot_dir)

    if os.path.exists(yolo_img_dir):
        shutil.rmtree(yolo_img_dir)

    os.makedirs(yolo_img_dir)
    os.makedirs(yolo_annot_dir)

    min_width, min_height = 1e9, 1e9
    max_width, max_height = 0, 0

    for annot_name in tqdm(os.listdir(annot_dir)):
        image_path = os.path.join(image_dir,annot_name.replace('.txt','.jpg'))
        annot_path = os.path.join(annot_dir,annot_name)
        yolo_annot_path = os.path.join(yolo_annot_dir,annot_name)

        image = np.array(Image.open(image_path))
        height, width = image.shape[:2]
        
        min_width = min(min_width, width)
        min_height = min(min_height, height)
        max_width = max(max_width, width)
        max_height = max(max_height, height)

        Image.fromarray(image).save(os.path.join(yolo_img_dir,annot_name.replace('.txt','.jpg')))
        yolo_file = open(yolo_annot_path,"w")
        yolo_lines = []

        with open(annot_path) as file:
            for line in file:
                id,xmin,ymin,xmax,ymax = line.rstrip().split()
                id = int(id) - 1
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                
                xmid = ((xmin + xmax) / 2) / width
                ymid = ((ymin + ymax) / 2) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                yolo_lines.append(f'0 {xmid} {ymid} {w} {h}\n')

        yolo_file.writelines(yolo_lines)
        yolo_file.close()

    print('Splitting data now !!!')
    split_data(data_path)    
    print(min_width)
    print(min_height)
    print(max_width)
    print(max_height)


if __name__ == '__main__':
    data_path = str(sys.argv[1])
    main(data_path)