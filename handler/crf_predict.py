import joblib
from transformers import AutoTokenizer

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Load mô hình CRF đã huấn luyện
crf = joblib.load("model/crf_phobert_model.pkl")

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def extract_terms(text: str):
    words = text.split()
    features = sent2features(words)
    predicted = crf.predict_single(features)

    terms = []
    current = []
    for word, label in zip(words, predicted):
        if label.startswith("B-"):
            if current:
                terms.append(" ".join(current))
            current = [word]
        elif label.startswith("I-") and current:
            current.append(word)
        else:
            if current:
                terms.append(" ".join(current))
                current = []
    if current:
        terms.append(" ".join(current))
    return terms

# Ví dụ chạy
if __name__ == "__main__":
    review = "phòng rộng rãi, giường êm, nhân viên thân thiện, vị trí thuận lợi"
    result = extract_terms(review)
    print("Các cụm từ phản ánh khía cạnh:")
    for t in result:
        print("-", t)
