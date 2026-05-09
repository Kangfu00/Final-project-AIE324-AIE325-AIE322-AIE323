import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# โหลดข้อมูลที่แปลงแล้ว
df = pd.read_csv("BU_Data_transformed.csv")

# กำหนด Target ที่ต้องการทำนาย
# 1 = อยากลองรสชาติใหม่
# 0 = ไม่อยากลอง
target = "Try_New_Flavor"

# ลบคอลัมน์ที่ไม่ควรเอาไปเป็น Feature
drop_cols = [
    target,
    "Like_Stronger_Ebisen_Flavor"
]

# เลือกเฉพาะคอลัมน์ตัวเลขเท่านั้น
X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=["int64", "float64"])

y = df[target]

# แบ่งข้อมูล Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# สร้างโมเดล Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train โมเดล
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินผลโมเดล
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ดู Feature ที่สำคัญที่สุด
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Save โมเดลไว้ใช้ใน Dashboard
with open("supervised_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nบันทึกโมเดลเรียบร้อย: supervised_model.pkl")