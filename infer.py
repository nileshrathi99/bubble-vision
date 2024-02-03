import os
import argparse
import json
import torch

def calculate_stats(comic_results):

    total = len(comic_results)
    all_areas = []

    for _, row in comic_results.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        curr_area = (xmax - xmin) * (ymax - ymin)
        all_areas.append(curr_area)

    avg_size = sum(all_areas) / len(all_areas)
    min_size = min(all_areas)
    max_size = max(all_areas)

    return {
        "count" : total,
        "avg_size" : float(f"{avg_size:.2f}"),
        "max_min_ratio" : float(f"{(max_size / min_size):.2f}")
    }


def main(panelbox, textbox, image_dir, save_dir, device):

    os.makedirs(save_dir, exist_ok=True)

    panelbox_model = torch.hub.load('yolov5',"custom",path=panelbox, source="local").to(device)
    textbox_model = torch.hub.load('yolov5',"custom",path=textbox, source="local").to(device)
    
    for image_name in os.listdir(image_dir):
        results = {}
        image_path = os.path.join(image_dir, image_name)
        no_ext = image_name.split('.')[0]
        save_file = os.path.join(save_dir, no_ext + ".json")
        
        panel_results = panelbox_model(image_path)
        panel_results = panel_results.pandas().xyxy[0][["xmin","ymin","xmax","ymax"]]
        panel_results = panel_results.round(2)

        textbox_results = textbox_model(image_path)
        textbox_results = textbox_results.pandas().xyxy[0][["xmin","ymin","xmax","ymax"]]
        textbox_results = textbox_results.round(2)

        panel_stats = calculate_stats(panel_results)
        textbox_stats = calculate_stats(textbox_results)

        panel_dict = panel_results.to_dict()
        textbox_dict = textbox_results.to_dict()

        panel_dict["stats"] = panel_stats
        textbox_dict["stats"] = textbox_stats

        results["image_name"] = image_name
        results["panel"] = panel_dict
        results["textbox"] = textbox_dict
        
        json.dump(results, open(save_file,"w"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to perform inference on image using panel and textbox detection models')
    parser.add_argument('--panel_ckpt', type=str, required=True, help='Path to the panel detection model checkpoint')
    parser.add_argument('--textbox_ckpt', type=str, required=True, help='Path to the textbox detection model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the input images for inference')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save the results')
    args = parser.parse_args()
    
    panelbox_ckpt_path = args.panel_ckpt
    textbox_ckpt_path = args.textbox_ckpt
    image_dir = args.image_dir
    save_dir = args.save_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(panelbox_ckpt_path, textbox_ckpt_path, image_dir, save_dir, device)