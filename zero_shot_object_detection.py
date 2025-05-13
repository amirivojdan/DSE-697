
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from transformers import (
    OwlViTProcessor, OwlViTForObjectDetection,
    Owlv2Processor, Owlv2ForObjectDetection,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)

# === CONFIGURATION ===
IMAGE_DIR = "./ChickenCoco/images/default"
ANNOTATION_PATH = "./ChickenCoco/annotations/instances_default.json"
CATEGORY_NAME = "Chicken"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "owlvit": {
        "id": "google/owlvit-base-patch32",
        "processor_class": OwlViTProcessor,
        "model_class": OwlViTForObjectDetection,
        "output_path": "owlvit_predictions.json",
        "post_process_fn": lambda processor, outputs, image: processor.post_process(outputs, target_sizes=torch.tensor([image.size[::-1]]).to(DEVICE))[0],
        "text_labels": ["a chicken"]
    },
    "owlvit2": {
        "id": "google/owlv2-base-patch16-ensemble",
        "processor_class": Owlv2Processor,
        "model_class": Owlv2ForObjectDetection,
        "output_path": "owlvit2_predictions.json",
        "post_process_fn": lambda processor, outputs, image: processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(image.height, image.width)]).to(DEVICE),
            threshold=0.2,
            text_labels=[["a chicken"]]
        )[0],
        "text_labels": [["a chicken"]]
    },
    "groundingdino": {
        "id": "IDEA-Research/grounding-dino-tiny",
        "processor_class": AutoProcessor,
        "model_class": AutoModelForZeroShotObjectDetection,
        "output_path": "groundingdino_predictions.json",
        "post_process_fn": lambda processor, outputs, image, inputs: processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.1, text_threshold=0.3, target_sizes=[image.size[::-1]]
        )[0],
        "text_labels": [["a Chicken"]]
    }
}


def load_coco_dataset(annotation_path):
    coco = COCO(annotation_path)
    return coco, coco.getImgIds()


def compute_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(predictions, iou_threshold=0.5):
    if not predictions:
        return []
    grouped = {}
    for pred in predictions:
        grouped.setdefault(pred["image_id"], []).append(pred)
    final_results = []
    for preds in grouped.values():
        preds.sort(key=lambda x: x["score"], reverse=True)
        keep = []
        while preds:
            best = preds.pop(0)
            keep.append(best)
            preds = [p for p in preds if compute_iou(best["bbox"], p["bbox"]) < iou_threshold]
        final_results.extend(keep)
    return final_results


def save_predictions(results, path):
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions to {path}")


def run_coco_evaluation(coco, path):
    coco_dt = coco.loadRes(path)
    with open(path) as f:
        preds = json.load(f)
    filtered = non_max_suppression(preds)
    with open(path, "w") as f:
        json.dump(filtered, f)
    for ann in coco.anns.values():
        ann["ignore"] = 0
        ann["iscrowd"] = 0
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def detect(model_id, config, coco, image_ids, category_id):
    model = config["model_class"].from_pretrained(config["id"]).to(DEVICE)
    processor = config["processor_class"].from_pretrained(config["id"])
    model.eval()
    results = []

    for img_id in tqdm(image_ids, desc=f"Running {model_id}"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        if model_id == "groundingdino":
            inputs = processor(images=image, text=config["text_labels"], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            result = config["post_process_fn"](processor, outputs, image, inputs)
        else:
            inputs = processor(images=image, text=config["text_labels"], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            result = config["post_process_fn"](processor, outputs, image)

        for box, score, *_ in zip(result["boxes"], result["scores"], result["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            if w > 1 and h > 1:
                results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 3)
                })

    return non_max_suppression(results)


def main():
    coco, image_ids = load_coco_dataset(ANNOTATION_PATH)
    category_id = coco.getCatIds(catNms=[CATEGORY_NAME])[0]

    for model_id, config in MODELS.items():
        print(f"Running model: {model_id}")
        results = detect(model_id, config, coco, image_ids, category_id)
        save_predictions(results, config["output_path"])
        run_coco_evaluation(coco, config["output_path"])


if __name__ == "__main__":
    main()
