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
⟶ Ta có thể thấy bộ dự liệu rắc rối này không thể đồng bộ thành một đường thẳng có giá trị TB MIN ($X, Y$ quá lệch nhau) </br>
### 2.	Polynomial Regression (Đa thức)
- Đây được coi là bản cải tiến của linear do có sự xuất hiện của các phần tử bậc $1, 2, … N$. Sử dụng khi chúng ta muốn biết giá trị $N$ là bao nhiêu.
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/polynomial.jpeg)  </br>
So sánh Simple linear vs Polynomial Regresstion:
| Simple linear model | Polynomial model |
|---|---|
| • Chỉ số $X$ nằm ở dưới *(subscript)* giúp phân biệt các $x$ đầu vào khác nhau. | • Chỉ số $X$ ở phía trên *(superscript)* để nói về **bậc** của $x$ tương ứng. |
#### 2.2 Overfitting
-	Xảy ra khi độ phức tạp của mô hình > độ phức tạp của dữ liệu
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
Trong sơ đồ này:
- Có **6 data points** được huấn luyện bởi **2 mô hình Polynomial** ($N=2$, $N=10$).
  - Với **$N=2$** *(đường vàng)*: sai số ở mức **chấp nhận được**.
  - Với **$N=10$** *(đường đỏ)*: sai số **train** thấp hơn $N=2$, nhưng khi thêm dữ liệu mới/suy rộng ra **test**, sai số **tăng** (tức khả năng khái quát kém).
- **$N=10$** đại diện cho **hàm bậc cao** ⇒ mô hình **phức tạp**. </br>
⟶ Đây là ví dụ điển hình của **Overfitting**.  
⟶ **$N$ càng cao** ⇒ **train error** thường **thấp**, nhưng **test error** thường **cao**.
**Giả sử:** Ở kiểm tra giữa kỳ, sinh viên “học vẹt” đạt điểm rất cao vì cấu trúc bài không đổi.  
Đến **cuối kỳ**, cấu trúc/bậc phức tạp **thay đổi** ⇒ điểm **giảm**.

⟶ Đây là ví dụ điển hình của **overfitting**: học thuộc “đề cũ” nhưng **khái quát hoá** kém trên đề mới.

⟶ Khi triển khai, cần **tối thiểu hoá sai số** giữa dự đoán và thực tế **đồng thời** **kiểm soát độ phức tạp mô hình** ⇒ **Regularization (Điều chuẩn)**.

#### 2.3. Regularization cơ bản:
##### 2.3.1. Kỹ thuật 1 **(Lasso)**
- Thành phần **Regulaziation** của trọng số Wj được tính bằng tổng giá trị tuyệt đối.
$$
\text{Cost}_{\mathrm{L1}}
= \sum_{i=0}^{N}\!\left(y_i-\sum_{j=0}^{M} x_{ij} W_j\right)^2
+ \lambda \sum_{j=0}^{M} \lvert W_j\rvert
$$
**So sánh Normal & Lasso:**
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
- **Biểu đồ bên trái (chưa Regularization):**
  - Độ dài đoạn thẳng tỉ lệ với $W_i$. Đoạn càng dài ⇒ feature $X_i$ ảnh hưởng càng mạnh đến output.

- **Biểu đồ bên phải (đã Regularization):**
  - Trọng số bị **thu nhỏ** về gần 0; một số **bị ép đúng 0** (ví dụ $j=0$ và $j=5$) ⇒ các feature tương ứng **không còn ảnh hưởng** đến output.

- **Kết luận:** **LASSO (L1)** hay dùng cho **Feature Selection** vì có thể làm nhiều $W_i=0$;
##### 2.3.2. Kỹ thuật 2 **(Ridge)**
- Thành phần regura của trọng số Wj được tính bằng tổng bình phương.
$$
\text{Cost}_{\mathrm{L2}}
= \sum_{i=0}^{N}\!\left(y_i-\sum_{j=0}^{M} x_{ij} W_j\right)^2
+ \lambda \sum_{j=0}^{M} W_j^{2}
$$
**So sánh Normal & Ridge:**
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
-	Đối với ridge thì nó sẽ giảm độ ảnh hưởng của toàn bộ các feature chứ không “ép” như lasso </br>
⟶ Trong ML ta có thể áp dụng các kỹ thuật regularization trong nhiều bài toán khác nhau kể cả bài toán regression, classification

- **Đối với Ridge (L2)**: giảm (shrink) **độ lớn của tất cả hệ số** → hiếm khi bằng 0. </br>
⟶ Trong ML, **regularization** được áp dụng rộng rãi cho cả **regression** lẫn **classification** để kiểm soát độ phức tạp và giảm overfitting.
##### 2.3.3. Kỹ thuật 3 **(Elastic Net)**
- Sự kết hợp giữa L1 & L2
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
