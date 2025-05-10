# FINAL

## Giai đoạn 1: Tiền xử lý văn bản (VietnameseTextPreprocessor)
vietnamese_processor.py 

## Giai đoạn 2: Huấn luyện mô hình CRF với PhoBERT
prepare_crf_dataset

Gán nhãn dữ liệu (BIO)

Tokenize với PhoBERT

Huấn luyện CRF

## Giai đoạn 3: Trích xuất các term từ review mới
Dự đoán nhãn BIO bằng PhoBERT + CRF

- loss: độ lỗi (càng giảm càng tốt)
- active: số lượng feature đang hoạt động (điều chỉnh qua c1, nên giữ mức trung bình)
- feature_norm: chuẩn hóa feature (tăng c2 để giảm feature_norm, nên giữ ở mức trung bình)

loss cao, active thấp, feature_norm thấp	            Underfitting → giảm c1, c2, tăng max_iterations
loss cao, active cao, feature_norm cao	                Overfitting → tăng c1, c2, giảm feature không cần thiết
loss giảm ổn định, active vừa, feature_norm ổn định	    Mô hình ổn

Trích xuất cụm từ phản ánh khía cạnh

## Giai đoạn 4: Ánh xạ term → aspect
Dựa trên KG (hotel.graphml) → ánh xạ term sang aspect

## Giai đoạn 5: Suy luận aspect & gán sentiment
Tìm quan hệ liên kết trong KG để mở rộng aspect

Gán sentiment theo aspect đã ánh xạ