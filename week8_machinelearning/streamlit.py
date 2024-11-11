# Import các thư viện cần thiết
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Bước 1: Tải và chia dữ liệu
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 2: Xây dựng mô hình KNN với k = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Bước 3: Dự đoán và đánh giá mô hình
y_pred = knn.predict(X_test)

# Tính toán các chỉ số
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

# Tạo giao diện trực quan với Streamlit
st.title("Kết quả mô hình KNN trên tập dữ liệu Wine")
st.write("**Tỷ lệ chia dữ liệu:** 70% train, 30% test")
st.write("**Số lượng hàng xóm (k):** 5")

# Hiển thị các chỉ số đánh giá
st.subheader("Kết quả đánh giá")
st.write(f"Độ chính xác (Accuracy): {accuracy:.2f}")
st.write(f"Độ hồi đáp (Recall): {recall:.2f}")
st.write(f"Độ chính xác (Precision): {precision:.2f}")

# Trực quan hóa kết quả dự đoán bằng biểu đồ
import matplotlib.pyplot as plt
import numpy as np

# Vẽ biểu đồ phân loại các mẫu trong tập kiểm tra
fig, ax = plt.subplots()
ax.hist([y_test, y_pred], bins=np.arange(4) - 0.5, label=['Actual', 'Predicted'], color=['#4c72b0', '#55a868'], align='mid', edgecolor='black')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(data.target_names)
ax.set_xlabel("Wine Class")
ax.set_ylabel("Count")
ax.legend(loc="upper right")
st.pyplot(fig)
