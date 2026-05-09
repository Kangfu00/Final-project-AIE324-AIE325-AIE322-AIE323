import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate

# โหลดข้อมูล
df = pd.read_csv("BU_Data_transformed.csv")

df["Sales_Opportunity"] = (
    (df["Try_New_Flavor"] == 1) |
    (df["Like_Stronger_Ebisen_Flavor"] == 1)
).astype(int)

# ==========================================
# 💡 เพิ่มโค้ด Feature Engineering ตรงนี้
# ==========================================
# 1. กลุ่มคนเน้นคุณภาพชีวิต (Quality Seeker)
df["Quality_Seeker_Score"] = df["Purchase_Factor_Quality_Ingredients"] + df["Purchase_Factor_Healthy"]

# 2. กลุ่มคนเชื่อมั่นในแบรนด์ (Brand Trust Score)
df["Brand_Trust_Score"] = df["Believe_Ebisen_Shrimp"] + df["Calvora_Tagline_Reflection"]
# ==========================================

# เลือกเฉพาะ Feature ที่หน้าเว็บใช้จริง (เพิ่ม 2 ตัวใหม่เข้าไปด้านล่างสุด)
features = [
    "Purchase_Factor_Quality_Ingredients",
    "Calvora_Tagline_Reflection",
    "Believe_Ebisen_Shrimp",
    "Purchase_Factor_Many_Flavors",
    "Purchase_Factor_Healthy",
    "Calvora_Association_General_Snack_Association",
    "Purchase_Factor_Crispy",
    "Age",
    "Know_Ebisen",
    "Strength_มีคุณภาพดี (Good quality)",
    "Calvora_Association_Calvora_Association",
    "Quality_Seeker_Score",  # <-- เพิ่มเข้ามา
    "Brand_Trust_Score"      # <-- เพิ่มเข้ามา
]

target = "Sales_Opportunity"

# กันกรณีบางคอลัมน์ไม่มี
available_features = [f for f in features if f in df.columns]

X = df[available_features]
y = df[target]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# สร้างโมเดลหลายแบบ
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=4,           # ป้องกัน overfit
        min_samples_leaf=5,    # ต้องการตัวอย่างอย่างน้อย 5 ในแต่ละ leaf
        class_weight="balanced"
)
}

best_model = None
best_score = 0
best_name = ""

print("\n========== MODEL BENCHMARK ==========\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print(classification_report(y_test, y_pred))
    print("-" * 50)

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

# Save Best Model
with open("supervised_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print(f"\nBest Model: {best_name}")
print("บันทึกโมเดลเรียบร้อย: supervised_model.pkl")

# Feature Importance
if best_name == "Random Forest":

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Important Features")
    print(importance)