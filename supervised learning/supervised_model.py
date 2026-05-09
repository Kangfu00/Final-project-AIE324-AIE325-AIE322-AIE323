import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# โหลดข้อมูล
df = pd.read_csv("BU_Data_transformed.csv")

# สร้าง Super Target
df["Sales_Opportunity"] = (
    (df["Try_New_Flavor"] == 1) |
    (df["Like_Stronger_Ebisen_Flavor"] == 1)
).astype(int)

# เลือกเฉพาะ Feature ที่หน้าเว็บใช้จริง
features = [
    "Know_Ebisen",
    "Age",
    "Purchase_Factor_Tasty",
    "Purchase_Factor_Many_Flavors",
    "Purchase_Factor_Crispy",
    "Purchase_Factor_Healthy",
    "Purchase_Factor_Quality_Ingredients",
    "Believe_Ebisen_Shrimp",
    "Ebisen_Flavor_Original",
    "Known_Snack_เอบินาริ",
    "Tasted_Snack_เอบินาริ"
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
        random_state=42
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
    f1 = f1_score(y_test, y_pred)

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