# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

```
| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |
```

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record nói về **cả bức ảnh**. Vì task là `image_classification`, model gán một nhãn duy nhất cho toàn bộ ảnh, không có tọa độ bbox hay polygon cho từng vật. Ảnh traffic chứa nhiều xe bus/car nhưng model vẫn chọn 1 lớp “cab” cho cả khung hình.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Đi kèm taxonomy/checkpoint. Model `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K, nên `taxonomy_name` = ImageNet-1K và `class_id`/`class_name` lấy từ danh sách 1000 lớp cố định của ImageNet, không phải model tự nghĩ ra lúc inference.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Vì tên lớp có thể trùng hoặc mơ hồ giữa các taxonomy khác nhau. `class_id` là số nguyên ổn định trong taxonomy, `taxonomy_name` chỉ rõ nguồn danh sách lớp, giúp tránh nhầm lẫn khi map nhãn hoặc so sánh giữa các model.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần quy định tiêu chí ưu tiên: chọn lớp mô tả **chủ đề chính / vật chiếm diện tích lớn nhất / vật nằm ở trung tâm**. Cũng cần nêu ví dụ ảnh đa chủ đề và cách xử lý khi hai lớp cạnh tranh.
6. Vì sao không được lấy `score` của model làm ground truth?
   - `score` chỉ là độ tin cậy nội bộ của model trên prediction hiện tại, không phản ánh nhãn đúng theo guideline. Ground truth phải do annotator gán theo quy tắc người viết, có thể khác hoàn toàn với lớp model đoán (như ảnh traffic bị đoán “cab” dù thực tế là đường đông xe bus/car).

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Góc trên-trái tại pixel (x=385.33, y=69.24), góc dưới-phải tại (x=498.92, y=348.92). Box rộng 113.58 pixel, cao 279.68 pixel. Đây là người đứng quay lưng (mặc tạp dề) ở bên phải ảnh bếp.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Theo file JSON (threshold 0.35): ảnh kitchen có **11** prediction.
   - Ngưỡng 0.60: còn **6** prediction (person 0.912625, bowl 0.719063, bowl 0.701282, oven 0.686791, oven 0.634300, person 0.610931).
   - Ngưỡng 0.20: nhiều hơn 11 (các box score thấp hơn 0.35 chỉ xuất hiện khi notebook chạy so sánh threshold; JSON hiện tại đã lọc ở 0.35).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Phải kiểm tra thêm nhiều box nhiễu (false positive), quyết định box nào giữ / xóa / chỉnh tọa độ, và xác nhận class có đúng không. Công việc QC tăng, dễ mệt và dễ bỏ sót lỗi.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - “Vẽ bbox chặt nhất có thể bao quanh toàn bộ vật thể nhìn thấy; nếu vật bị cắt mép ảnh thì box chạm mép ảnh; không mở rộng box vào vùng nền trống hoặc vật khác.”
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - Person góc trái (score 0.610931, bbox [0.12, 263.18, 61.55, 310.78]) bị cắt mép ảnh và một phần bị che. Đây là tình huống phổ biến → viết thành **quy tắc trong guideline** (“nếu vật bị cắt mép thì vẫn gán class và vẽ box đến mép ảnh; nếu bị che >50% thì vẫn gán nếu còn nhận diện được class”). Chỉ escalation khi guideline chưa có quy tắc rõ hoặc ca quá mơ hồ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Polygon mô tả **đường biên chính xác** theo silhouette của vật (vai, tay, chân, tạp dề…), không chỉ hình chữ nhật bao ngoài. Giúp tính diện tích chính xác, tách vật chồng lên nhau và huấn luyện model segmentation.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - `instance_id` là mã định danh duy nhất cho **một instance cụ thể** trong ảnh hiện tại (ví dụ “kitchen-001”). Nó giúp phân biệt nhiều vật cùng class (nhiều “bowl”, nhiều “person”) trong cùng một ảnh. Không phải class_id (vì class_id chỉ nói loại vật), cũng không phải tracking ID (vì không theo dõi qua nhiều frame video).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - “Vẽ biên mask bám sát mép vật thể nhìn thấy; bao gồm phần bị che một phần nếu còn nhận ra được; không bao gồm bóng đổ hay vùng nền; nếu hai vật chạm nhau thì tách biên rõ ràng theo từng instance.”
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - Các bowl trên bàn phía trước chồng lên nhau và một phần bị che; biên giữa dining table và các bowl cũng mờ. Dùng **guideline** trước (quy tắc “tách instance rõ khi hai vật chạm; vẽ theo phần nhìn thấy”). Chỉ escalation nếu guideline chưa quy định cách xử lý vùng mờ hoặc khi không chắc class.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

```
| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | --------------------------------------------- | -------------------------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         | Một class_id + class_name cho cả ảnh (theo taxonomy) | Ảnh traffic bị đoán “cab” (score 0.510915) trong khi thực tế là đường đông xe bus/car; top-5 còn minibus, police_van… cho thấy model không chắc | Chọn đúng 1 lớp theo guideline (ưu tiên chủ đề chính) | Đúng lớp? Có đúng taxonomy không? Ảnh đa chủ đề đã chọn đúng tiêu chí chưa? |
| Phát hiện vật thể     | Mỗi vật = 1 bbox_xyxy + class_name | Nhiều box chồng (nhiều bowl); person bị cắt mép (score 0.610931); cup score thấp (0.451108, 0.381542) dễ nhiễu | Vẽ bbox chặt, gán class đúng, quyết định giữ/bỏ vật bị che | Box có sát vật không? Class đúng? Có sót/thừa box không? |
| Instance segmentation | Mỗi instance = polygon_xy + class_name + instance_id | Biên các bowl chồng nhau; person mask dài (348 điểm); spoon score thấp (0.487579, 0.359147) | Vẽ polygon bám mép, tách instance rõ, gán instance_id | Mask có lọt nền/bóng không? Hai vật chạm đã tách đúng chưa? |
```

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Chỉ dùng ảnh/JSON trong phạm vi lab, không tải lên nơi công cộng, không ghi thông tin cá nhân vào báo cáo hay output, và không chia sẻ file ngoài nhóm làm việc được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay và báo Lab Coach / giảng viên phụ trách lab để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md` (notebook tạo khi chạy)
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
