import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# โหลดข้อมูลที่แปลงแล้ว
df = pd.read_csv("BU_Data_transformed.csv")

# กำหนด Target ที่ต้องการทำนาย
target = "Try_New_Flavor"

# ลบคอลัมน์ Target และคอลัมน์ที่ใกล้เคียงกับคำตอบเกินไป
drop_cols = [
    target,
    "Like_Stronger_Ebisen_Flavor"
]

# เลือกเฉพาะข้อมูลตัวเลขเพื่อใช้เป็น Feature
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

# Predict ผลลัพธ์
y_pred = model.predict(X_test)

# แสดงผล Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# สร้าง Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Try", "Try"],
    yticklabels=["Not Try", "Try"]
)

plt.title("Confusion Matrix: Try New Flavor Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# สร้าง Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features for Try New Flavor Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()


print("\nสร้าง Visualization เสร็จแล้ว")
print("ได้ไฟล์:")
print("- confusion_matrix.png")
print("- feature_importance.png")