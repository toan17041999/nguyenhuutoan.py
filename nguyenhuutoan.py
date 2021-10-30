import cv2
import numpy as np
#doc anh
anh = cv2.imread("anhchandung2.jpg", 0)
#Bo loc 5x5
kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])
kernel = kernel/sum(kernel)
#Bo loc phat hien canh
kernel2 = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])
kernel2 = kernel2/(np.sum(kernel2) if np.sum(kernel2)!=0 else 1)
#Loc nguon anh
img_loctt = cv2.filter2D(anh,-1,kernel)
img_loctc= cv2.filter2D(anh,-1,kernel2)
#lọc đồng hình
hh, ww = anh.shape[:2]
# lấy hình ảnh
img_log = np.log(np.float64(anh), dtype=np.float64)
# tiết kiệm dft dưới dạng đầu ra
dft = np.fft.fft2(img_log, axes=(0,1))
# áp dụng sự thay đổi điểm gốc đến trung tâm của hình ảnh
dft_shift = np.fft.fftshift(dft)
# tạo vòng tròn màu đen trên nền trắng cho bộ lọc thông cao
radius = 13
mask = np.zeros_like(anh, dtype=np.float64)
cy = mask.shape[0] // 2
cx = mask.shape[1] // 2
cv2.circle(mask, (cx,cy), radius, 1, -1)
mask = 1 - mask
# Mặt nạ ntialias thông qua làm mờ
mask = cv2.GaussianBlur(mask, (47,47), 0)
#  mặt nạ cho dft_shift
dft_shift_filtered = np.multiply(dft_shift,mask)
#chuyển điểm gốc
back_ishift = np.fft.ifftshift(dft_shift_filtered)
# tiết kiệm idft có phức tạp
img_back = np.fft.ifft2(back_ishift, axes=(0,1))
# kết hợp các thành phần thực và ảo phức tạp để tạo lại (độ lớn cho) hình ảnh ban đầu
img_back = np.abs(img_back)
# áp dụng exp để đảo ngược
img_homomorphic = np.exp(img_back, dtype=np.float64)
# quy mô kết quả
img_locdonghinh = cv2.normalize(img_homomorphic, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#ghi kết quả
cv2.imwrite("locthongthap.png",img_loctt)
cv2.imwrite("locthongcao.png",img_loctc)
cv2.imwrite("locdonghinh.png",img_locdonghinh)
#hiện thị ảnh
cv2.imshow("anh_xam", anh)
cv2.imshow("loc_thong_thap", img_loctt)
cv2.imshow("loc_thong_cao", img_loctc)
cv2.imshow("loc_dong_hinh", img_locdonghinh)
cv2.waitKey()
cv2.destroyAllWindows()