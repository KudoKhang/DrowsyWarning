# Drowsy Warning
Đối với các tài xế chạy về đêm trên những tuyến đường dài thì không trách khỏi việc nhiều lúc mất ngủ dẫn đến ngủ gật. Điều đó thật sự nguy hiểm tới tính mạng của bản thân và cả những người tham gia giao thông xung quanh. Ứng dụng Drowsy Warning là một ứng dụng đơn giản nhưng có thể giúp giám sát và phát chuông cảnh báo nếu phát hiện tài xế ngủ gật trên xe.

<p align="center">
	<img src="https://github.com/KudoKhang/DrowsyWarning/blob/main/Sources/dowsywarning.gif?raw=true" />
</p>


# How it work
Đầu tiên chúng ta cần lấy các điểm landmarks trên mặt bằng Haarcascade để thực hiện việc tính toán

```python
face_detect = cv2.CascadeClassifier("Sources/haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("Sources/shape_predictor_68_face_landmarks.dat")
```

Cụ thể là ta sẽ lấy 6 điểm như sau:

```python
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

leftEye = landmark[left_eye_start:left_eye_end] # 6 landmark
rightEye = landmark[right_eye_start:right_eye_end]
```

<p align="center">
	<img src="https://github.com/KudoKhang/DrowsyWarning/blob/main/Sources/1.png?raw=true" />
</p>

Tiếp đó ta cần 2 hàm để tính khoảng cách giữa các landmark và tỉ lệ mí mắt trên dưới:

```python
def e_dist(A, B):
	return np.linalg.norm(A - B)

def eye_ratio(eye):
	d_V1 = e_dist(eye[1], eye[5])
	d_V2 = e_dist(eye[2], eye[4])
	d_H = e_dist(eye[0], eye[3])
	eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)
	return eye_ratio_val
```

<p align="center">
	<img src="https://github.com/KudoKhang/DrowsyWarning/blob/main/Sources/2.png?raw=true" />
</p>

Sau đó tính tỉ lệ trung bình của của 2 mắt:

```python
eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
```

Cuối cùng là so sánh `eye_avg_ratio` với ngưỡng, nếu thấp hơn ngưỡng và trong thời gian 16 frame thì tiến hành phát chuông cảnh báo:

```python
if eye_avg_ratio < EYE_RATIO_THRESHOLD:
	SLEEP_FRAMES += 1
	if SLEEP_FRAMES >= MAX_SLEEP_FRAMES:
		cv2.rectangle(frame, (500, 200), (800, 270), (0, 0, 255), -1)
		cv2.putText(frame, "WARNING!!!", (510, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		cv2.putText(frame, f"EYE RATIO: {round(eye_avg_ratio, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
		mixer.music.play()
else:
	SLEEP_FRAMES = 0
	cv2.putText(frame, f"EYE RATIO: {round(eye_avg_ratio, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
```

# Usage
Để khởi chạy project:
```bash
git clone https://github.com/KudoKhang/DrowsyWarning
cd DrowsyWarning
python main.py
```
Ta có thể triển khai project lên xe thực tế bằng cách tích hợp code vào raspberry Pi + 1 camera + 1 loa (loa có thể đấu vào hệ thống loa của xe) 
