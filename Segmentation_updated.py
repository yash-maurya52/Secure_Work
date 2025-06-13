import cv2
import numpy as np
import pytesseract
import json
import matplotlib.pyplot as plt
import os
import re

def classify_region(w, h, x, y, image_height):
    area = w * h
    aspect_ratio = w / h if h != 0 else 0

    if area < 1000 or (w < 30 and h < 30):
        return "symbol"
    elif area > 4000 and h < 80 and aspect_ratio > 3 and y < image_height * 0.4:
        return "heading"
    else:
        return "paragraph"

def split_into_sentences(text):
    # Split based on punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def segment_image_with_doctags(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or invalid path!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]

    def sort_key(b):  # Sort vertically then left-right
        x, y, w, h = b
        return (y // 50, x)

    boxes = sorted(boxes, key=sort_key)

    annotated = img.copy()
    doctags = []
    image_height = img.shape[0]

    type_colors = {
        "heading": (255, 0, 0),
        "paragraph": (0, 255, 0),
        "symbol": (0, 0, 255),
        "sentence": (128, 0, 128)
    }

    counts = {
        "heading": 0,
        "paragraph": 0,
        "symbol": 0,
        "sentence": 0
    }

    for (x, y, w, h) in boxes:
        tag_type = classify_region(w, h, x, y, image_height)
        counts[tag_type] += 1
        tag_name = f"{tag_type}_{counts[tag_type]}"
        color = type_colors[tag_type]

        roi = img[y:y + h, x:x + w]
        ocr_text = pytesseract.image_to_string(roi)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, tag_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        doctags.append({
            "type": tag_type,
            "tag": tag_name,
            "location": f"<loc_{x}><loc_{y}><loc_{x + w}><loc_{y + h}>",
            "text": ocr_text.strip()
        })

        # === Sentence Segmentation ===
        if tag_type in {"paragraph", "heading"}:
            sentences = split_into_sentences(ocr_text)
            data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)

            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence_words = sentence.split()

                sent_x, sent_y, sent_w, sent_h = x + w, y + h, 0, 0

                for i, word in enumerate(data['text']):
                    if word.strip() in sentence:
                        wx, wy, ww, wh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        abs_x, abs_y = x + wx, y + wy
                        sent_x = min(sent_x, abs_x)
                        sent_y = min(sent_y, abs_y)
                        sent_w = max(sent_w, abs_x + ww - sent_x)
                        sent_h = max(sent_h, abs_y + wh - sent_y)

                if sent_w > 10 and sent_h > 10:
                    counts["sentence"] += 1
                    sentence_tag = f"sentence_{counts['sentence']}"
                    cv2.rectangle(annotated, (sent_x, sent_y), (sent_x + sent_w, sent_y + sent_h), type_colors["sentence"], 1)
                    doctags.append({
                        "type": "sentence",
                        "tag": sentence_tag,
                        "location": f"<loc_{sent_x}><loc_{sent_y}><loc_{sent_x + sent_w}><loc_{sent_y + sent_h}>",
                        "text": sentence.strip()
                    })

    # Save output
    base, ext = os.path.splitext(image_path)
    segmented_path = f"{base}_segmented.jpg"
    doctags_path = f"{base}_doctags.json"

    cv2.imwrite(segmented_path, annotated)
    with open(doctags_path, "w") as f:
        json.dump(doctags, f, indent=4)

    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Segmented with Headings, Paragraphs, Symbols, and Sentences")
    plt.axis('off')
    plt.show()

    print(f"Saved: {segmented_path}")
    print(f"DocTags JSON: {doctags_path}")

# Example usage
segment_image_with_doctags("/content/Why+do+we+need+to+use+paragraphs.jpg")
