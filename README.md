## Bubble Vision

Automatic detection of formal features of comic book page 
</div>

## Info

This Repository contains the code to train and infer comic book page and get the statistics of its formal features

---
## Training

### Yolov5

- Install the requirements in your virtual environment:
    > pip install -r requirements.txt

- First convert the annotations into yolo format by running the convert_yolo file
    > python convert_yolo.py <path_to_comic_data_folder>

- Go to the yolov5 directory and run the following command:
    > python yolov5/train.py  \
    > --dataset <path_to_dataset.yaml> \
    > --img <img_size> \
    > --epochs <n_epochs> \
    > --weights yolov5m.pt \
    > --name <training_name>

- The checkpoints will be available under runs/train/{training_name}/weights/best.pt

- Use this checkpoint for inference

## Faster R-CNN
- All the code used for training the Faster R-CNN models can be found in Faster_RCNN/FasterRCNN.ipynb
- Can be run on a Google Colab notebook once the datasets are added and the respective data paths are updated.

---
## How to infer?

- Clone this repository
    > git clone https://github.iu.edu/cs-b657-sp2023/bubble-vision.git \
    > cd comicbook-infer.git

- Install the requirements in your virtual environment:
    > pip install -r requirements.txt

- Keep a directory that contains the images of comic book pages

- Run the infer.py file by specifying the command line arguments in the following format
    > python infer.py \
    > --panel_ckpt <path_to_panel_ckpt> \
    > --textbox_ckpt <path_to_panel_ckpt> \
    > --image_dir <path_to_image_directory> \
    > --save_dir <path_to_save_results_directory>

- The panel checkpoint and textbox checkpoint are available in the current directory

- To run inference on an image directory named "infer_images" and to save the results in "results_dir", the following command should be executed:
    > python infer.py \
    > --panel_ckpt panel_ckpt.pt \
    > --textbox_ckpt textbox_ckpt.pt \
    > --image_dir infer_images \
    > --save_dir result_dir

- The results_dir directory will now have JSONs corresponding to each image in the infer_images directory with the following key and values:

```
{
    "image_name" : image_name,
    "panel" : {
        "xmin" : [coordinates of xmin of panels],
        "ymin" : [coordinates of ymin of panels],
        "xmax" : [coordinates of xmax of panels],
        "ymax" : [coordinates of ymax of panels],
        "stats" : {
            "count": number of panels,
            "avg_size": average area of all panels,
            "max_min_ratio": ratio of biggest panel to smallest panel
        }
    },
    "textbox" : {
        "xmin" : [coordinates of xmin of textboxes],
        "ymin" : [coordinates of ymin of textboxes],
        "xmax" : [coordinates of xmax of textboxes],
        "ymax" : [coordinates of ymax of textboxes],
        "stats" : {
            "count": number of textboxes,
            "avg_size": average area of all textboxes,
            "max_min_ratio": ratio of biggest textbox to smallest textbox
        }
    }

}
```

## Results

<img width="851" alt="image" src="https://github.com/nileshrathi99/bubble-vision/assets/32071800/d6a1aee0-532c-4908-a922-9cfd219bc64e">

<img width="377" alt="image" src="https://github.com/nileshrathi99/bubble-vision/assets/32071800/e67cba2e-e08e-468b-b9a9-765eb34cfbf3">

## References

Yolov5 code: https://github.com/ultralytics/yolov5
Faster R-CNN: https://pytorch.org/vision/main/models/faster_rcnn.html
The Dataset used was from: https://obj.umiacs.umd.edu/comics/index.html 

