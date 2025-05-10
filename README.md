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

Trích xuất cụm từ phản ánh khía cạnh

## Giai đoạn 4: Ánh xạ term → aspect
Dựa trên KG (hotel.graphml) → ánh xạ term sang aspect

## Giai đoạn 5: Suy luận aspect & gán sentiment
Tìm quan hệ liên kết trong KG để mở rộng aspect

Gán sentiment theo aspect đã ánh xạ