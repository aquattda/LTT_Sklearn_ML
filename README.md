# TÌM HIỂU TẤT CẢ CÁC THUẬT TOÁN MACHINE LEARING
## I.	Supervised Learning
### 1.	Linear Regression (Hồi quy tuyến tính)
-	Đây là thuật toán tìm ra mối quan hệ tuyến tính tồn tại $1$-$N$ input **X** và output **Y**.
-	Mối quan hệ tuyến tính là mối quan hệ bậc 1 ($y = ax + b$) giữa 2 biến **X**, **Y**.  </br>
⟶ Dựa vào đó để đưa ra kết quả **Y** dựa vào đầu vào **X**. </br>
⟶ Trong đó **X** là **Independent Variable** (**Feature**) và **Y** là **Dependent Variable** (**Target**). </br>
#### 1.1. Phân loại input X và bài toán thực tế:
-	Nếu tồn tại duy nhất một input **X** (**Simple Linear Regression**): Dự đoán mức lương của một nhân viên dựa vào số năm kinh nghiệm của nhân viên đó. </br> 
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/Simple_Linear.png) </br> 
**Trong đó:** </br>
-	A (**Intercept/Bias**): giá trị kỳ vọng của y hay gọi chung là giá trị trung bình khi **X** = 0.
-	B (**Slope/ Coefficient**): độ dốc, khi **X** thay đổi một $\Delta X$ thì giá trị kỳ vọng của **Y** sẽ là $\Delta Y = b \cdot \Delta X$ </br>
**Giả sử**: Dự đoán điểm thi **Y** = 40 + 5*(giờ học) </br>
$A$ = 40: Nếu giờ học = 0 thì điểm dự đoán TB là 40 nhưng sẽ có sai số $\varepsilon$.</br>
$B$ = 5: Nếu tăng giờ học lên 1 giờ thì số điểm sẽ tăng 5; thêm 2 giờ học thì điểm tăng 10; tăng $k$ giờ  học thì điểm tăng 5k. </br>

⇒ Nếu tồn tại $N$ input $X$ (**Multiple Linear Regression**): Dự đoán mức lương của một nhân viên dựa vào số năm kinh nghiệm của nhân viên và trình độ học vấn. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/Multiple_Linear.png) </br>
**Trong đó:** </br>
-	$β0$: hệ số chặn (intercept) giống giá trị kỳ vọng. </br>
⟶ Việc tất cả $Xi = 0$ (siêu hiếm) không có ý nghĩa. Khi nằm ngoài phạm vi dữ liệu thì $β0$ là ngoại suy (**extrapolation:** dự đoán bên ngoài khoảng đó), muốn $β0$ có ý nghĩa ta cần center các biến: </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/extrapolatrionpng.png) </br>
⇒ Nếu tồn tại $N$ input $X$ (**Multiple Linear Regression**): Dự đoán mức lương của một nhân viên dựa vào số năm kinh nghiệm của nhân viên và trình độ học vấn.
#### 1.2. Biểu đồ: </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/chart_linear_regression.png) </br>
#### 1.3. Ưu điểm: </br>
-	Dễ chơi dễ trúng thưởng đối với các bộ data đơn giản và có mối liên hệ $x, y$ là tuyến tính. </br>
#### 1.4.Nhược điểm:  </br>
- Tuy nhiên trong các trường hợp phức tạp hơn thì không ai ngu mà chọn linear cả </br>
**Giả sử:**  </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/chart_linear_regression_difficult.png) </br>
⟶ Ta có thể thấy bộ dự liệu rắc rối này không thể đồng bộ thành một đường thẳng có giá trị TB MIN ($X, Y$ quá lệch nhau)
