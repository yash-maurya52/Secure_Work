import os
import time
import re
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import evaluate
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from bert_score import score as bertscore
from paddleocr import PaddleOCR
import numpy as np

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
if HF_TOKEN:
    login(HF_TOKEN)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview").to(device)

# Load WER metric
wer_metric = evaluate.load("wer")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_plain_text_from_docling_format(docling_output: str) -> str:
    docling_clean = re.sub(r'\b(doctag|text|loc_\d+)+\b', '', docling_output, flags=re.IGNORECASE)
    docling_clean = re.sub(r'<[^>]+>', '', docling_clean)
    docling_clean = re.sub(r'[\*\-_<>\[\]{}#@/\\|^%$]+', '', docling_clean)
    docling_clean = re.sub(r'[^\x20-\x7E]+', ' ', docling_clean)
    docling_clean = re.sub(r'\s+', ' ', docling_clean).strip()
    return docling_clean

def preprocess_with_paddleocr(image: Image.Image) -> Image.Image:
    """Use PaddleOCR to detect text regions and crop the image to only the relevant area."""
    print("🔍 Preprocessing image using PaddleOCR...")

    # Convert PIL image to numpy array for PaddleOCR compatibility
    image_np = np.array(image)

    results = ocr.ocr(image_np, cls=True)

    if not results or not results[0]:
        print("⚠️ No text regions found. Using original image.")
        return image

    # Get bounding box around all detected regions
    min_x = min([min(box[0][0][0], box[0][1][0], box[0][2][0], box[0][3][0]) for box in results[0]])
    min_y = min([min(box[0][0][1], box[0][1][1], box[0][2][1], box[0][3][1]) for box in results[0]])
    max_x = max([max(box[0][0][0], box[0][1][0], box[0][2][0], box[0][3][0]) for box in results[0]])
    max_y = max([max(box[0][0][1], box[0][1][1], box[0][2][1], box[0][3][1]) for box in results[0]])

    # Add padding
    padding = 10
    min_x = max(int(min_x) - padding, 0)
    min_y = max(int(min_y) - padding, 0)
    max_x = min(int(max_x) + padding, image.width)
    max_y = min(int(max_y) + padding, image.height)

    cropped = image.crop((min_x, min_y, max_x, max_y))
    return cropped

def run_combined_pipeline(image: Image.Image, ground_truth: str, task_prompt: str = "Convert this page to docling."):
    start = time.time()

    # Preprocess image using OCR
    preprocessed_image = preprocess_with_paddleocr(image)

    # Prepare SmolDocling input
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[preprocessed_image], return_tensors="pt").to(device)

    # Generate prediction
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_length:]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0]
    output_text = output_text.replace("<end_of_utterance>", "").strip()

    # Clean and save output
    clean_output = extract_plain_text_from_docling_format(output_text).lower()
    with open("recognized_output.txt", "w", encoding="utf-8") as f:
        f.write(clean_output)

    # Normalize ground truth
    ground_truth_clean = ' '.join(ground_truth.strip().lower().split())
    output_clean = ' '.join(clean_output.split())

    # WER
    try:
        error_rate = wer_metric.compute(predictions=[output_clean], references=[ground_truth_clean])
        wer_acc = max(0.0, (1 - error_rate) * 100)
    except Exception as e:
        print("WER error:", e)
        wer_acc = 0.0

    # BERTScore
    try:
        P, R, F1 = bertscore([output_clean], [ground_truth_clean], lang="en", rescale_with_baseline=True)
        bert_p = P[0].item() * 100
        bert_r = R[0].item() * 100
        bert_f1 = F1[0].item() * 100
    except Exception as e:
        print("BERTScore error:", e)
        precision = recall = f1 = 0.0

    # Logs
    print("\nRecognized Text:\n", clean_output)
    print("\nGround Truth:\n", ground_truth_clean)
    print(f"\nWER Accuracy:         {wer_acc:.2f}%")
    print(f"BERTScore Precision:  {bert_p:.2f}%")
    print(f"BERTScore Recall:     {bert_r:.2f}%")
    print(f"BERTScore F1:         {bert_f1:.2f}%")
    print(f"Total Time Taken:     {time.time() - start:.2f}s")

    return clean_output, wer_acc, (bert_p, bert_r, bert_f1)

# MAIN
if __name__ == "__main__":
    image_path = "/content/handwritting.jpg"  # Change this path
    image = Image.open(image_path).convert("RGB")

    ground_truth_text = """) the arm design philosophy: the arm processor has been specifically designed to be small to reduce power consumption & extend battery operation essential for applications such as mobile phones. high code density is another major requirement since embedded systems have limited memory due to cost or physical size restrictions. the smaller the area used by the embedded processor, the more available space for specialized peripherals.
    """

    result_text, acc, bert_scores = run_combined_pipeline(image, ground_truth_text)
    print("\nOutput saved to recognized_output.txt")
