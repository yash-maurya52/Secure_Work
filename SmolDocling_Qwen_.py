import os
import time
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import evaluate
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import bert_score

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
if HF_TOKEN:
    login(HF_TOKEN)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SmolDocling model and processor
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview").to(device)

# Load Qwen tokenizer and model for contextual evaluation
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True).to(device)

# Load WER metric
wer_metric = evaluate.load("wer")


def extract_plain_text_from_docling_format(docling_output: str) -> str:
    try:
        import json
        structured_data = json.loads(docling_output)
        text_lines = []
        for block in structured_data.get("blocks", []):
            line = " ".join(token.get("text", "") for token in block.get("tokens", []))
            text_lines.append(line.strip())
        raw_text = "\n".join(text_lines)
    except Exception:
        raw_text = docling_output.strip()

    clean_text = re.sub(r'\*\)\s*', '', raw_text)
    clean_text = re.sub(r'<[^>]*>', '', clean_text)
    clean_text = re.sub(r'[^\w\s.,;:!?\'"-]', '', clean_text)
    return clean_text.strip().lower()


def evaluate_text_quality(prediction: str, reference: str):
    try:
        wer_error = wer_metric.compute(predictions=[prediction], references=[reference])
        wer_accuracy = max(0.0, (1 - wer_error) * 100)
    except Exception as e:
        print("Error during WER calculation:", e)
        wer_accuracy = 0.0

    try:
        P, R, F1 = bert_score.score([prediction], [reference], lang="en", rescale_with_baseline=True)
        bert_precision = P[0].item() * 100
        bert_recall = R[0].item() * 100
        bert_f1 = F1[0].item() * 100
    except Exception as e:
        print("Error during BERTScore calculation:", e)
        bert_precision = bert_recall = bert_f1 = 0.0

    return wer_accuracy, bert_precision, bert_recall, bert_f1


def calculate_context_level_accuracy_qwen(recognized_text: str, ground_truth_text: str) -> float:
    qwen_prompt = (
        "You are a smart assistant that helps evaluate OCR output. "
        "Given two texts, determine how similar their meanings are. "
        "Even if words are different, give a high score if the overall meaning is the same.\n\n"
        f"Recognized Text:\n{recognized_text}\n\n"
        f"Ground Truth:\n{ground_truth_text}\n\n"
        "Rate the semantic similarity from 0 to 100. Only return the score as a number."
    )

    input_ids = qwen_tokenizer(qwen_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = qwen_model.generate(**input_ids, max_new_tokens=20)
    response = qwen_tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        score = float(re.findall(r"\d+(?:\.\d+)?", response)[-1])
    except:
        score = 0.0
    return max(0.0, min(score, 100.0))


def visualize_metrics(wer_acc: float, bert_p: float, bert_r: float, bert_f1: float, semantic_sim: float):
    if wer_acc > 0:
        plt.figure(figsize=(5, 5))
        plt.pie([wer_acc, 100 - wer_acc], labels=["Correct", "Error"], colors=["green", "red"], autopct='%1.1f%%')
        plt.title("WER Accuracy")
        plt.axis('equal')
        plt.savefig("wer_pie_chart.png")
        plt.close()

    if bert_f1 > 0 or semantic_sim > 0:
        metrics = ['BERT Precision', 'BERT Recall', 'BERT F1', 'Qwen Similarity']
        values = [bert_p, bert_r, bert_f1, semantic_sim]
        plt.figure(figsize=(7, 4))
        bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylim(0, 100)
        plt.ylabel('Score (%)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        plt.title("Semantic Evaluation Scores")
        plt.tight_layout()
        plt.savefig("semantic_scores_bar_chart.png")
        plt.close()


def run_smol_docling_overlay(image: Image.Image, ground_truth: str, task_prompt: str = "Convert this page to docling."):
    start = time.time()

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": task_prompt}]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[:, prompt_length:]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0]
    output_text = output_text.replace("<end_of_utterance>", "").strip()

    clean_output = extract_plain_text_from_docling_format(output_text)

    with open("recognized_output.txt", "w", encoding="utf-8") as f:
        f.write(clean_output)

    ground_truth_clean = ' '.join(ground_truth.split()).lower()
    output_clean = ' '.join(clean_output.split())

    wer_acc, bert_p, bert_r, bert_f1 = evaluate_text_quality(output_clean, ground_truth_clean)
    sem_sim = calculate_context_level_accuracy_qwen(clean_output, ground_truth_clean)

    print("\nRecognized Text:\n", clean_output)
    print("\nGround Truth:\n", ground_truth_clean)
    print(f"\nWER Accuracy:         {wer_acc:.2f}%")
    print(f"BERTScore Precision:  {bert_p:.2f}%")
    print(f"BERTScore Recall:     {bert_r:.2f}%")
    print(f"BERTScore F1:         {bert_f1:.2f}%")
    print(f"Contextual Similarity (Qwen): {sem_sim:.2f}%")
    print(f"Total Time Taken:     {time.time() - start:.2f}s")

    visualize_metrics(wer_acc, bert_p, bert_r, bert_f1, sem_sim)

    return clean_output, wer_acc, bert_p, bert_r, bert_f1, sem_sim


if __name__ == "__main__":
    image_path = "/content/handwritting.jpg"
    image = Image.open(image_path).convert("RGB")

    ground_truth_text = """
    The ARM Design Philosophy: The ARM Processor has been specifically designed to be small to reduce power consumption & extend battery operation essential for applications such as mobile phones. High code density is another major requirement since embedded systems have limited memory due to cost or physical size restrictions. The smaller the area used by the embedded processor, the more available space for specialized peripherals.
    """

    result_text, wer_acc, bert_p, bert_r, bert_f1, sem_sim = run_smol_docling_overlay(image, ground_truth_text)

    print("\n📝 Cleaned output saved to recognized_output.txt")