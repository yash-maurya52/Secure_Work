import os
import time
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import evaluate
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from bert_score import score as bertscore

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

def extract_plain_text_from_docling_format(docling_output: str) -> str:
    """
    Cleans SmolDocling output by removing doctags, tags, locs, symbols, and junk.
    """
    docling_clean = re.sub(r'\b(doctag|text|loc_\d+)+\b', '', docling_output, flags=re.IGNORECASE)
    docling_clean = re.sub(r'<[^>]+>', '', docling_clean)
    docling_clean = re.sub(r'[\*\-_<>\[\]{}#@/\\|^%$]+', '', docling_clean)
    docling_clean = re.sub(r'[^\x20-\x7E]+', ' ', docling_clean)
    docling_clean = re.sub(r'\s+', ' ', docling_clean).strip()
    return docling_clean

def run_smol_docling_overlay(image: Image.Image, ground_truth: str, task_prompt: str = "Convert this page to docling."):
    start = time.time()

    # Prepare input
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": task_prompt}]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # Generate prediction
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_length:]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0]
    output_text = output_text.replace("<end_of_utterance>", "").strip()

    # Clean and lowercase output
    clean_output = extract_plain_text_from_docling_format(output_text).lower()

    # Save to .txt
    with open("recognized_output.txt", "w", encoding="utf-8") as f:
        f.write(clean_output)

    # Prepare lowercase ground truth for fair comparison
    ground_truth_clean = ' '.join(ground_truth.strip().lower().split())
    output_clean_for_eval = ' '.join(clean_output.split())

    # WER accuracy
    if ground_truth_clean:
        try:
            error_rate = wer_metric.compute(predictions=[output_clean_for_eval], references=[ground_truth_clean])
            accuracy = max(0.0, (1 - error_rate) * 100)
        except Exception as e:
            print("Error during WER calculation:", e)
            accuracy = 0.0
    else:
        print("No ground truth provided. Skipping WER.")
        accuracy = 0.0

    # BERTScore calculation
    if ground_truth_clean:
        try:
            P, R, F1 = bertscore([output_clean_for_eval], [ground_truth_clean], lang="en", rescale_with_baseline=True)
            precision = P[0].item() * 100
            recall = R[0].item() * 100
            f1 = F1[0].item() * 100
        except Exception as e:
            print("Error during BERTScore calculation:", e)
            precision = recall = f1 = 0.0
    else:
        print("No ground truth provided. Skipping BERTScore.")
        precision = recall = f1 = 0.0

    # Console logs
    print("\nRecognized Text:\n", clean_output)
    print("\nGround Truth:\n", ground_truth_clean)
    print(f"\nWER Accuracy: {accuracy:.2f}%")
    print(f"BERTScore Precision: {precision:.2f}%")
    print(f"BERTScore Recall: {recall:.2f}%")
    print(f"BERTScore F1: {f1:.2f}%")
    print(f"Time Taken: {time.time() - start:.2f}s")

    # Optional: WER Accuracy pie chart
    if accuracy > 0:
        labels = ['Correct', 'Error']
        sizes = [accuracy, 100 - accuracy]
        colors = ['green', 'red']
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title('WER Accuracy (Word-Level)')
        plt.axis('equal')
        plt.savefig("accuracy_pie_chart.png")
        plt.close()
        print("WER pie chart saved to accuracy_pie_chart.png")
    else:
        print("Skipping WER chart due to invalid accuracy.")

    # Optional: BERTScore bar chart
    if f1 > 0:
        metrics = ['Precision', 'Recall', 'F1']
        values = [precision, recall, f1]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(metrics, values, color=['blue', 'orange', 'green'])
        plt.ylim(0, 100)
        plt.ylabel('Score (%)')
        plt.title('BERTScore Metrics')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("bertscore_chart.png")
        plt.close()
        print("BERTScore chart saved to bertscore_chart.png")
    else:
        print("Skipping BERTScore chart due to invalid score.")

    return clean_output, accuracy, (precision, recall, f1)

# MAIN EXECUTION
if __name__ == "__main__":
    image_path = "/content/handwritting.jpg"  # Update this to your image path
    image = Image.open(image_path).convert("RGB")

    ground_truth_text = """) the arm design philosophy: the arm processor has been specifically designed to be small to reduce power consumption & extend battery operation essential for applications such as mobile phones. high code density is another major requirement since embedded systems have limited memory due to cost or physical size restrictions. the smaller the area used by the embedded processor, the more available space for specialized peripherals.
    """

    result_text, acc, bert_scores = run_smol_docling_overlay(image, ground_truth=ground_truth_text)
    print("\n📝 Cleaned output saved to recognized_output.txt")
    