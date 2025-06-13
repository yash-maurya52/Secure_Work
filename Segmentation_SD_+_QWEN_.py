# === Imports ===
import cv2
import pytesseract
import json
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import evaluate


import torch
# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
if HF_TOKEN:
    login(HF_TOKEN)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview").to(device)

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True).to(device)

wer_metric = evaluate.load("wer")

# === Heuristic Segmentation ===

def is_probable_heading(text):
    words = text.strip().split()
    if len(words) > 12 or not words:
        return False
    if any(text.strip().endswith(punct) for punct in ['.', ':', ';']):
        return False
    upper_ratio = sum(1 for w in words if w.isupper() or w.istitle()) / len(words)
    return upper_ratio > 0.5


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def classify_region(w, h, x, y, image_height):
    area = w * h
    aspect_ratio = w / h if h != 0 else 0
    if area < 1000 or (w < 25 and h < 25):
        return "symbol"
    elif area > 5000 and h < 80 and aspect_ratio > 3.5:
        return "heading"
    elif area > 3000 and aspect_ratio > 1.2:
        return "paragraph"
    else:
        return "symbol"


def merge_adjacent_paragraphs(boxes, max_x_gap=50, max_y_diff=20):
    merged = []
    used = [False] * len(boxes)
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        box1 = (x1, y1, x1 + w1, y1 + h1)
        merged_box = list(box1)
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = boxes[j]
            box2 = (x2, y2, x2 + w2, y2 + h2)
            vertical_aligned = abs(y1 - y2) <= max_y_diff
            horizontal_close = (0 < x2 - (x1 + w1) <= max_x_gap) or (0 < x1 - (x2 + w2) <= max_x_gap)
            if vertical_aligned and horizontal_close:
                merged_box[0] = min(merged_box[0], box2[0])
                merged_box[1] = min(merged_box[1], box2[1])
                merged_box[2] = max(merged_box[2], box2[2])
                merged_box[3] = max(merged_box[3], box2[3])
                used[j] = True
        used[i] = True
        mx, my, mx2, my2 = merged_box
        merged.append((mx, my, mx2 - mx, my2 - my))
    return merged


def segment_image_with_doctags(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]

    def sort_key(box):
        x, y, w, h = box
        return (y // 20, x)
    boxes = sorted(boxes, key=sort_key)
    boxes = merge_adjacent_paragraphs(boxes)

    doctags = []
    annotated = img.copy()
    image_height = img.shape[0]
    counts = {"heading": 0, "paragraph": 0, "symbol": 0, "sentence": 0}
    type_colors = {
        "heading": (0, 0, 255),
        "paragraph": (0, 255, 0),
        "symbol": (128, 128, 128),
        "sentence": (255, 0, 255)
    }

    for (x, y, w, h) in boxes:
        roi = img[y:y+h, x:x+w]
        ocr_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        if not ocr_text:
            continue
        region_type = classify_region(w, h, x, y, image_height)
        if is_probable_heading(ocr_text):
            region_type = "heading"
        counts[region_type] += 1
        tag = f"{region_type}_{counts[region_type]}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), type_colors[region_type], 2)
        cv2.putText(annotated, tag, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors[region_type], 1)
        doctags.append({
            "type": region_type,
            "tag": tag,
            "location": f"<loc_{x}><loc_{y}><loc_{x + w}><loc_{y + h}>",
            "text": ocr_text
        })
        if region_type in {"heading", "paragraph"}:
            sentences = split_into_sentences(ocr_text)
            for sentence in sentences:
                if len(sentence.strip()) == 0:
                    continue
                counts["sentence"] += 1
                sentence_tag = f"sentence_{counts['sentence']}"
                doctags.append({
                    "type": "sentence",
                    "tag": sentence_tag,
                    "location": f"<loc_{x}><loc_{y}><loc_{x + w}><loc_{y + h}>",
                    "text": sentence.strip()
                })

    base, ext = os.path.splitext(image_path)
    segmented_path = f"{base}_segmented.jpg"
    doctags_path = f"{base}_doctags.json"
    headings_path = f"{base}_headings.txt"

    cv2.imwrite(segmented_path, annotated)
    with open(doctags_path, "w") as f:
        json.dump(doctags, f, indent=4)
    headings = [tag["text"] for tag in doctags if tag["type"] == "heading"]
    with open(headings_path, "w") as f:
        for h in headings:
            f.write(h + "\n")

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Document")
    plt.axis("off")
    plt.show()

    print(f"Saved segmented image: {segmented_path}")
    print(f"Saved doctags: {doctags_path}")
    print(f"Saved extracted headings: {headings_path}")


# === Main pipeline ===

# Step 1: Segment image
image_path = "/content/samplepdf48.png"
segment_image_with_doctags(image_path)

# Step 2: Load segmented doctags
doctags_path = str(Path(image_path).with_name(Path(image_path).stem + "_doctags.json"))

with open(doctags_path, 'r') as f:
    doctags = json.load(f)

chunks = [tag['text'] for tag in doctags if tag['type'] in {'sentence', 'paragraph', 'heading'}]

# Step 3: Index into SmolDocling
docling = Docling()
docling.add_documents(chunks)

# Step 4: Load Qwen (can swap with any compatible LLM)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-1.8B-Chat",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
qwen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 5: Ask a question
query = "What is the main topic discussed in the document?"
top_chunks = docling.query(query, k=3)

# Build a prompt
context = "\n".join([c['text'] for c in top_chunks])
prompt = f"Given the following text:\n\n{context}\n\nAnswer this question: {query}"

# Step 6: Get answer from Qwen
response = qwen_pipeline(prompt, max_new_tokens=200)[0]['generated_text']
print("\n=== Qwen Answer ===")
print(response)
