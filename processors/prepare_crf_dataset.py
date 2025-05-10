import json
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def crf_dataset(input_path: str):
    X_data = []
    y_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            text = sample["data"]
            spans = sample.get("label", [])

            char_labels = ["O"] * len(text)
            for start, end, label in spans:
                char_labels[start] = "B-" + label
                for i in range(start + 1, end + 1):
                    if i < len(text):
                        char_labels[i] = "I-" + label

            # Tokenize bằng PhoBERT
            words = []
            offsets = []
            start = 0
            for token in text.split():
                idx = text.find(token, start)
                words.append(token)
                offsets.append(idx)
                start = idx + len(token)

            # Gán nhãn cho từng từ dựa vào ký tự đầu
            tokens = []
            bio_tags = []
            for word, start_idx in zip(words, offsets):
                if start_idx < len(char_labels):
                    label = char_labels[start_idx]
                else:
                    label = "O"
                tokens.append(word)
                bio_tags.append(label)

            X_data.append(tokens)
            y_data.append(bio_tags)

    return X_data, y_data

def main():
    print("Preparing CRF dataset...")

    input_path = "processed_datasets/hotel.jsonl"
    X_sentences, y_labels = crf_dataset(input_path)

    output_path = "processed_datasets/crf_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f_out:
        for tokens, tags in zip(X_sentences, y_labels):
            f_out.write(json.dumps({
                "tokens": tokens,
                "tags": tags
            }, ensure_ascii=False) + "\n")

    print(f"Đã ghi {len(X_sentences)} dòng vào {output_path}")


if __name__ == '__main__':
    main()