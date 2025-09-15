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
So sánh Simple linear vs Polynomial Regresstion: </br>

| Simple linear model | Polynomial model |
|---|---|
| Chỉ số *X* nằm ở dưới *(subscript)* giúp phân biệt các \(x\) đầu vào khác nhau. | Chỉ số *X* ở phía trên *(superscript)* để nói về **bậc** của \(x\) tương ứng. |
| Ví dụ: ($x_i, x_j$) | Ví dụ: ($x^2, x^3$) |

#### 2.1. Overfitting
-	Xảy ra khi độ phức tạp của mô hình > độ phức tạp của dữ liệu </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
Trong sơ đồ này:
- Có **6 data points** được huấn luyện bởi **2 mô hình Polynomial** ($N=2$, $N=10$).
  - Với **$N=2$** *(đường vàng)*: sai số ở mức **chấp nhận được**.
  - Với **$N=10$** *(đường đỏ)*: sai số **train** thấp hơn $N=2$, nhưng khi thêm dữ liệu mới/suy rộng ra **test**, sai số **tăng** (tức khả năng khái quát kém).
- **$N=10$** đại diện cho **hàm bậc cao** ⇒ mô hình **phức tạp**. </br>
⟶ Đây là ví dụ điển hình của **Overfitting**.  
⟶ **$N$ càng cao** ⇒ **train error** thường **thấp**, nhưng **test error** thường **cao**. </br>
**Giả sử:** Ở kiểm tra giữa kỳ, sinh viên “học vẹt” đạt điểm rất cao vì cấu trúc bài không đổi.  
Đến **cuối kỳ**, cấu trúc/bậc phức tạp **thay đổi** ⇒ điểm **giảm**.

⟶ Đây là ví dụ điển hình của **overfitting**: học thuộc “đề cũ” nhưng **khái quát hoá** kém trên đề mới.

⟶ Khi triển khai, cần **tối thiểu hoá sai số** giữa dự đoán và thực tế **đồng thời** **kiểm soát độ phức tạp mô hình** ⇒ **Regularization (Điều chuẩn)**.

#### 2.2. Regularization cơ bản:
##### 2.2.1. Kỹ thuật 1 **(Lasso)**
- Thành phần **Regulaziation** của trọng số Wj được tính bằng tổng giá trị tuyệt đối. </br>

$$
\text{Cost}_{\mathrm{L1}}
= \sum_{i=0}^{N}\!\left(y_i-\sum_{j=0}^{M} x_{ij} W_j\right)^2 + \lambda \sum_{j=0}^{M} \lvert W_j\rvert
$$ 

**So sánh Normal & Lasso:** </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/normal%26lasso.png) </br>
- **Biểu đồ bên trái (chưa Regularization):**
  - Độ dài đoạn thẳng tỉ lệ với $W_i$. Đoạn càng dài ⇒ feature $X_i$ ảnh hưởng càng mạnh đến output.

- **Biểu đồ bên phải (đã Regularization):**
  - Trọng số bị **thu nhỏ** về gần 0; một số **bị ép đúng 0** (ví dụ $j=0$ và $j=5$) ⇒ các feature tương ứng **không còn ảnh hưởng** đến output.

- **Kết luận:** **LASSO (L1)** hay dùng cho **Feature Selection** vì có thể làm nhiều $W_i=0$;
##### 2.2.2. Kỹ thuật 2 **(Ridge)**
- Thành phần regura của trọng số Wj được tính bằng tổng bình phương. </br>

$$
\text{Cost}_{\mathrm{L2}}
= \sum_{i=0}^{N}\!\left(y_i-\sum_{j=0}^{M} x_{ij} W_j\right)^2 + \lambda \sum_{j=0}^{M} W_j^{2}
$$

**So sánh Normal & Ridge:** </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/normal_ridge.png) </br>
- **Đối với Ridge (L2)**: giảm (shrink) **độ lớn của tất cả hệ số** → hiếm khi bằng 0. </br>
⟶ Trong ML, **regularization** được áp dụng rộng rãi cho cả **regression** lẫn **classification** để kiểm soát độ phức tạp và giảm overfitting.
##### 2.2.3. Kỹ thuật 3 **(Elastic Net)**
- Sự kết hợp giữa L1 & L2 </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/Elastic_Net.png) </br>

**Tình huống:** Tập trainning $Y$ có giá trị $[-∞, +∞]$, và nó phục vụ cho bài toán Linear. Vậy nếu ta muốn áp dụng thuật toán này vô **Classification (Binary Classification)** và nó có giá trị output $[0, 1]$.

### 3.	Logistic Regression
-	Ý tưởng của bài toán này là thực hiện việc ánh xạ Y sao cho nó nằm trong khoảng $[0, 1]$. Sau đó đặt một giá trị ngưỡng ($p$). Nếu các giá trị output $> p$ thì sẽ ở class số 1 và output $< p$ sẽ ở class 0. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/logistic.webp) </br>
-	Thuật toán **Logistic Regression** dựa vào thuật toán **Linear Regression** tạo ra output và cho output đó đi qua **Sigmoid function**. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/sigmoid_func.png) </br>
- Trong đó $Z$ là input đầu vào có giá trị bất kỳ $[-∞, +∞]$ ⇒ output có giá trị ($0,1$)
- Do ý tưởng của thuật toán **Logistic regression** áp dụng giải thuật **Linear regression** mà thuật toán **Linear regression** là một hàm bậc 1 ($ax + b$) cho nên đối với **Logistic Regression* thì $Z$ sẽ thay đổi theo. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/logistic_sigmoid.png) </br>
- Thứ tự thực hiện:
    -	Đầu tiên sẽ chạy thuật toán **Linear regression** sau đó tìm ra được các output.
    -	Lấy các output đó sử dụng **Sigmoid function** tạo ra output mới là $0.85$.
    -	Sử dụng **Logistic regression** so sánh giá trị output mới đi so sánh với giá trị ngưỡng ($0.5$)
      
**LƯU Ý:** </br>
-	Tuy thuật toán **Logistic Regression** có chữ **Regression** nhưng đây là thuật toán để thực hiện bài toán **classification**
- **Trong Sklearn:**
    -	Nếu muốn kết hợp logistic regression với các kỹ thuật **Regularzation (Lasso, Ridge, Elastic Net)** => sử dụng tham số *(penalty{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’)* </br>

⇒ Giải quyết bài toán **Classification (Binary Classification)* khi chỉ có 2 class nhưng target $[0, 1]$ </br>

**Tình huống:** Vậy nếu bài toán **Classification** có nhiều hơn 2 class thì sao?

#### 3.1.	Multinomial Logistic Regression
- Giống như **Linear** thì Logistic cũng có **Multinomial** để giải quyết các bài tonas **Classification** có nhiều hơn 2 class. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/multinomial_logistic.png) </br>
-	Đối với **Multinomial Logistic Regression** nó sẽ sử dụng thuật toán **Softmax Activation Function**.
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/softmax_function.png) </br>
-	Thuật toán này sẽ ánh xạ các dữ liệu là một vector có giá trị $[-∞, +∞]$ ⇒ Các vector có giá trị $[0, 1]$, cuối cùng sẽ chọn **Max Value** làm phần tử cuối </br>

⇒ Dựa vào đó nên thuật toán này thường sử dụng giải thích xác xuất và dự đoán dành cho từng class

**Trong Sklearn:**
-	Khi sử dụng model **LogisticRegression** nếu có nhiều hơn 2 class thì sẽ tự chuyển đổi thành **Multinomial Logistic Regresstion** 
    (use **OneVsRestClassifier**)  

### 4. Binary Classification
#### 4.1. Bayes Theorem
- Trong lý thuyết xác suất định lý **Bayes Theorem** là một định lý áp dụng rộng rãi trong DS , ML. Giúp ta tính được xác suất xảy ra ở một sự kiện nào đó, mà biết được một sự kiện khác đã xảy ra. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/bayes.png) </br>

**Tình Huống:**
- Nếu có một text và cần kiểm tra xem xác suất của đoạn text đó thuộc loại nào. </br>

![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/clasifier_bayes.png) </br>
- Trong ảnh, cần kiểm tra xem đoạn text thuộc loại Positive hay Negative. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/clasifier_words.png) </br>
-	Về bản chất thì đoạn text sẽ là các từ riêng lẻ, cho nên cần kiểm tra tỷ lệ % của các từ đó. Rõ ràng chúng ta cần tính một xác suất có điều kiện cho nên dựa vào định lý **Bayes**. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/text_bayes.png) </br>
-	Khi áp dụng đính lý vào thì sẽ xảy ra vấn đề rằng 3 phép tính đằng sau sẽ phức tạp cho nên cần áp dụng thêm một giả thuyết.  </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/fitting_text.png) </br>
-	Giả thuyết cho rằng nếu các từ được phân chia ở các class riêng biệt thì xác suất của các từ xuất hiện là độc lập với nhau. Nhưng trong thực tế, việc chia các từ độc lập là không hợp lý vì các từ xuất hiện sẽ có mối liên hệ với nhau. Tuy nhiên việc áp dụng giả thuyết này chỉ mang tính chất giúp giải quyết bài toán trở nên dễ dàng hơn. </br>
#### 4.2. Naive Bayes
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/naive_bayes.png) </br>
##### 4.2.1. Multinomial Naive Bayes
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/multinomial_naive.png) </br>
-	Nếu input đầu vào là các biến rời rạc và giá trị được xác định bằng **số lần xuất hiện trong đoạn text.** </br>
##### 4.2.2. Bernoulli Naive Bayes
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/bernoulli_naive.png) </br>
-	Nếu input đầu vào là các biến rời rạc và giá trị được xác định bằng **chúng có xuất hiện hay không trong đoạn text.** </br>
##### 4.2.3. Gaussian Naive Bayes
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
-	Nếu input đầu vào là biến liên tục và có thêm giả thuyết về Bayes. </br>

**Tình huống:** Ta đã học các thuật toán về hồi quy và phân loại bây giờ ta sẽ học các thuật toán áp dụng cho cả 2 thuật toán này </br>

### 5. Decision Tree
-	Mở đầu là một thuật toán dễ giải thích và trực quan nhất trong ML </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
**Giả sử:** Áp dụng thuật toán giải quyết bài toán tuyển nhân viên. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
- Dựa vào độ lệch của các leaf node để xác định, nếu độ lệch tương đối cần phải phân nhánh tiếp. Ngược lại, ta dựa vào đó mà đưa ra quyết định. </br>
So sánh độ lệch:
 - Table </br>
**Tình Huống:** Trong thực tế, việc phân chia này phụ thuộc vào việc phân chia các feature phù hợp để tối ưu level của cây và độ lệch lớn nhất. Có rất nhiều cách để giúp ta xác định được việc lựa chọn các feature phù hợp. Một trong những cách phổ biến là dựa vào **Gini Impurity** và **Information Gain (Entropy)** </br>
#### 5.1. Gini Inpurity
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
Trong đó:
-	C: là tổng số lượng class trong Target
-	Pi là xác suất của một phần tử thuộc về class i

Trường hợp:
- Gini [$0, 0.5$] ⇒ Feature làm cho leaf node bị phân chia cao (độ lệch lớn)
- Gini [$> 0.5$] ⇒ Feature làm cho leaf node bị phân chia thấp (độ lệch nhỏ) </br>
**Giả sử:** Áp dụng thuật toán giải quyết bài toán cho vay ngân hàng </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
 - Đối với Age[Youth]: 
    - Yes(2) ~ Pi = 0.4
    - No(3) ~ Pi = 0.6 
</br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
- Sau khi có Gini ta cần tính Mean của các Gini đó.

⇒ Các Mean(Gini) thấp nhất sẽ chọn best feature. </br>
#### 5.2. Infomation Gain (Entropy)
- Đây là một chỉ số khác để xác định các best decision node.
- Thực hiện việc tìm các ngưỡng tách (trên feature) làm sao cho độ hỗn tạp của nhãn sau khi tách giảm nhiều nhất. </br>
- Công thức markdown </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
**Nhược điểm:** 
- Dễ bị **Overfitting** ( đặc biệt là cây quá nhiều bậc)	 
- Chỉ một thay đổi nhỏ của dữ liệu sẽ ảnh hưởng lớn đến cấu trúc của toàn bộ cây.
- Tuy **Desition Tree** có thể sử dụng cho **Regresstion or Classification** nhưng trong thực tế không nên sử dụng với **Regresstion** vì rất khó để tìm được khoảng cách TB đối với **Overfitting** và **Underfitting**. </br>
#### 5.3. Underfitting
-	Đây là hiện tượng đối lập với **Overfitting**. Tức là độ phức tạp của dữ liệu > mô hình </br>
=> Mô hình không đủ để tổng quát hóa xu hướng </br>

**Tình Huống:**
-	Việc sử dụng **Decision Tree** một mình khiến độ chính xác không cao, nên cần kết hợp nhiều **Decision Tree** lại với nhau để gia tăng độ chính xác </br>

### 6. Random Forest (Dừng ngẫu nhiên)
#### 6.1. Classification
- Đối với bài toán **Classification** thì cách thức kết hợp là sử dụng **Majority vote**. 
##### 6.1.1. Majority Vote
- Các vote nhiều nhất của từng **Decision** sẽ được chọn vào **Final Prediction** (Dự đoán cuối) </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
**Giả Sử:** Ứng dụng vào bài toán phân loại chó, mèo. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
#### 6.2. Regression
- Đối với bài toán **Regression** thì sẽ sử dụng **Averaging**. </br>
##### 6.2.1. Averaging
- Lấy giá trị trung bình của các Decision Tree sẽ được cho vào **Final Prediction** (Dự đoán cuối) </br>

**Giả Sử:** Ứng dụng vào bài toán mật độ ảnh hưởng ABC. </br>
![alt text](https://github.com/aquattda/LTT_Sklearn_ML/blob/main/images/overfitting.png) </br>
















