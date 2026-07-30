===============================================================
 ĐO TRÊN JETSON AGX THOR  —  chạy trực tiếp trên máy, không cần mạng
===============================================================

MỤC ĐÍCH
  Lấy số latency + điện năng cho 4 nấc precision (FP32 / FP16 / INT8 / FP8)
  của 5 detector YOLO11, để so trực tiếp với Jetson Orin (không có FP8) và
  với RTX 5090. Đây là phần còn thiếu duy nhất của bài báo về phần cứng.

CẦN CHUẨN BỊ
  - Thor đã cài JetPack 7.x (có sẵn /usr/src/tensorrt/bin/trtexec).
    Nếu chưa có trtexec:   sudo apt-get install -y tensorrt
  - Khoảng 8 GB trống (engine build ra khá nặng).
  - Cắm nguồn đầy đủ, để máy chạy yên ~40–60 phút.

CÁC BƯỚC
  1. Chép cả thư mục này vào Thor (USB là được), ví dụ vào ~/thor_bundle
  2. Mở terminal, vào đúng thư mục đó:
         cd ~/thor_bundle
  3. Đặt máy về chế độ hiệu năng tối đa và khoá xung nhịp
     (bước này QUAN TRỌNG — không khoá thì số đo dao động, bài phải ghi chú):
         sudo nvpmodel -m 0
         sudo jetson_clocks
  4. Chạy:
         bash thor_bench.sh
  5. Chạy xong, gửi lại DUY NHẤT file này:
         ~/thor_bundle/thor_results.tar.gz

CHẠY THỬ NHANH TRƯỚC (khoảng 3 phút, nên làm để chắc chắn không lỗi)
         MODELS="yolo11n" DURATION=5 bash thor_bench.sh

GHI CHÚ
  - Bị ngắt giữa chừng cũng không sao: chạy lại lệnh cũ, phần đã xong được
    giữ nguyên, nó chỉ làm tiếp phần còn thiếu.
  - Muốn chạy ngắn hơn (3 model thay vì 5):
         MODELS="yolo11n yolo11m yolo11x" bash thor_bench.sh
  - Nếu FP8 báo build lỗi: ĐỪNG xoá gì cả. Đó cũng là một kết quả có ý nghĩa
    cho bài báo (nghĩa là FP8 chưa dùng được cho detector trên Thor), log lỗi
    đã được lưu tự động trong results/logs/.
  - Script không gửi gì lên mạng, không sửa gì ngoài thư mục này.

KẾT QUẢ SẼ IN RA MÀN HÌNH
  Một bảng: p50 / p99 (ms), img/s, công suất (W), năng lượng (mJ/ảnh) và tỉ số
  tốc độ so với FP16. Cần nhất là cột cuối: nếu INT8/FP8 < 1.00x nghĩa là
  lượng tử hoá KHÔNG nhanh hơn FP16 ở cỡ model đó — đúng hiện tượng bài báo
  đang mô tả trên Orin, và cần biết Thor có lặp lại không.
===============================================================
