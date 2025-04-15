import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Đọc dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
dataset_path = "bank/bank-full.csv"
df = pd.read_csv(dataset_path, sep=';')

# 2. Chia tập train và test theo định nghĩa của dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Chuyển đổi các đặc trưng thành dạng số
label_encoders = {}
for col in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train = scaler.fit_transform(train_df.drop(columns=['y']))
x_test = scaler.transform(test_df.drop(columns=['y']))
y_train = train_df['y']
y_test = test_df['y']

# 5. Phân loại bằng Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# 6. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
