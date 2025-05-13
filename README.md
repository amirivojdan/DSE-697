
# Zero-Shot Detection and Action Recognition of Broiler Chickens Using Vision-Language Models

This repository contains the final project for **DSE 697 - Large Language Modeling & Gen AI**, focused on applying **zero-shot object detection** using vision-language models (VLMs) to understand and detect chicken behaviors in farm settings.

## Project Presentation

You can view the project overview and key insights in the [ChickenVLM.pdf](ChickenVLM.pdf) presentation file.

## Datasets

Two datasets are used for evaluation:

- **ChickenCOCO**: Object detection dataset formatted in COCO-style annotations.
- **ChickenVerse4**: Visual classification dataset for behavior recognition.

Before running the code, please extract these datasets:

```bash
unzip ChickenCOCO.zip -d ./dataset
unzip ChickenVerse4.zip -d ./ChickenVerse4
```

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

> Note: This includes vision-language models from HuggingFace, LangChain components, and object detection tools.

## Running the Project

After setting up the environment and extracting the datasets, you can run the detection pipeline using:

```bash
python zero_shot_object_detection.py
```

This script will:
- Load all three VLMs (Grounding DINO, OWLViT, OWLViT v2)
- Run zero-shot object detection across the ChickenCOCO dataset
- Save predictions and evaluate them using COCO metrics

---

For more insights or reproduction, please refer to the detailed implementation in `zero_shot_object_detection.py`.

