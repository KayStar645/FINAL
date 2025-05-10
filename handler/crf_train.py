import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from processors.prepare_crf_dataset import crf_dataset
import sklearn_crfsuite
import joblib
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

from sklearn.metrics import make_scorer
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
import scipy.stats

def grid_search_crf(X_train, y_train):
    crf = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True,
        all_possible_states=True,
    )

    param_dist = {
        'c1': scipy.stats.expon(scale=0.5),   # tương đương [~0.01 - 1]
        'c2': scipy.stats.expon(scale=0.05),  # tương đương [~0.001 - 0.5]
    }

    rs = RandomizedSearchCV(
        crf,
        param_distributions=param_dist,
        n_iter=30,
        scoring=make_scorer(metrics.flat_f1_score, average='weighted'),
        verbose=1,
        n_jobs=-1,
        cv=3,
        random_state=42,
    )

    rs.fit(X_train, y_train)

    print("Best parameters:", rs.best_params_)
    print("Best CV score (f1_weighted):", rs.best_score_)

    return rs.best_estimator_

def train_crf_manual(X_train, y_train, c1=0.1, c2=0.1, max_iter=100):
    crf = CRF(
        algorithm='lbfgs',              # Thuật toán tối ưu hóa: L-BFGS
        c1=c1,                        # Hệ số regularization L1 (giảm overfitting)
        c2=c2,                        # Hệ số regularization L2 (giảm overfitting)
        max_iterations=max_iter,             # Số vòng lặp tối đa
        all_possible_transitions=True, # Cho phép các trạng thái chuyển không thấy trong train
        all_possible_states=True,      # Cho phép các trạng thái không xuất hiện trong train
        verbose=True                   # Hiển thị thông tin log từng vòng
    )

    crf.fit(X_train, y_train)

    return crf


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

    # Dùng grid_search_crf để tìm CRF tốt nhất
    # print("\nĐang tìm tham số tốt nhất cho mô hình CRF...")
    # crf = grid_search_crf(X_train, y_train)

    # Training CRF với tham số thủ công
    print("\nĐang huấn luyện mô hình CRF với tham số thủ công...")
    crf = train_crf_manual(
        X_train, y_train,
        c1=0.1, c2=0.1, max_iter=200
    )

    # Đánh giá
    y_pred = crf.predict(X_test)
    print("\nBáo cáo phân loại trên tập test:")
    print(metrics.flat_classification_report(y_test, y_pred, digits=3, zero_division=0))

    # Lưu báo cáo thành CSV
    report_dict = metrics.flat_classification_report(y_test, y_pred, digits=3, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    os.makedirs("results", exist_ok=True)
    df_report.to_csv("results/crf_evaluation_report.csv", encoding="utf-8-sig")
    print("Đã lưu báo cáo đánh giá vào: results/crf_evaluation_report.csv")

    # In thông tin mô hình
    print("\nSố lượng nhãn đã học:", len(crf.classes_))
    print("Trọng số phân biệt mạnh nhất:")
    for (attr, label), weight in sorted(crf.state_features_.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"{attr} -> {label}: {weight:.3f}")

    # Lưu mô hình
    os.makedirs("model", exist_ok=True)
    joblib.dump(crf, "model/crf_phobert_best.pkl")
    print("\nĐã lưu mô hình tại: model/crf_phobert_best.pkl")


if __name__ == "__main__":
    main()
