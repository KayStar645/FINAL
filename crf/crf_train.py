import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.prepare_crf_dataset import crf_dataset
import sklearn_crfsuite
import joblib
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

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

def main():
    input_path = "processed_datasets/hotel.jsonl"
    X_sentences, y_labels = crf_dataset(input_path)

    # Chia train/test
    X_train_sent, X_test_sent, y_train, y_test = train_test_split(
        X_sentences, y_labels, test_size=0.2, random_state=42
    )
    X_train = [sent2features(s) for s in X_train_sent]
    X_test = [sent2features(s) for s in X_test_sent]

    # Huấn luyện CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=2,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(X_train, y_train)

    # Đánh giá
    y_pred = crf.predict(X_test)
    print("\nBáo cáo phân loại trên tập test:")
    print(metrics.flat_classification_report(y_test, y_pred, digits=3))

    # In thông tin mô hình
    print("\nSố lượng nhãn đã học:", len(crf.classes_))
    print("Trọng số phân biệt mạnh nhất:")
    for (attr, label), weight in sorted(crf.state_features_.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"{attr} -> {label}: {weight:.3f}")

    # Lưu mô hình
    os.makedirs("model", exist_ok=True)
    joblib.dump(crf, "model/crf_phobert_model.pkl")
    print("\nĐã lưu mô hình tại: model/crf_phobert_model.pkl")

if __name__ == "__main__":
    main()
