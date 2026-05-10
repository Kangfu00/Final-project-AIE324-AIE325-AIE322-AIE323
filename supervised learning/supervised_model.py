import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. โหลดข้อมูลที่มี Cluster_ID จาก unsupervised แล้ว
# ==========================================
df = pd.read_csv("BU_Data_3_Segments_Final_Complete.csv")
print(f"โหลดข้อมูล: {len(df)} คน")
print(f"Cluster distribution:\n{df['Cluster_ID'].value_counts().sort_index()}")

# ==========================================
# 2. สร้าง Target
# ==========================================
df["Sales_Opportunity"] = (
    (df["Try_New_Flavor"] == 1) |
    (df["Like_Stronger_Ebisen_Flavor"] == 1)
).astype(int)
print(f"\nSales_Opportunity distribution:\n{df['Sales_Opportunity'].value_counts()}")

# ==========================================
# 3. Feature Engineering
# ==========================================
df["Quality_Seeker_Score"] = df["Purchase_Factor_Quality_Ingredients"] + df["Purchase_Factor_Healthy"]
df["Brand_Trust_Score"]    = df["Believe_Ebisen_Shrimp"] + df["Calvora_Tagline_Reflection"]

# ==========================================
# 4. Features (เพิ่ม Cluster_ID เป็น feature)
# ==========================================
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
    "Cluster_ID",            # เพิ่มจาก unsupervised
    "Quality_Seeker_Score",
    "Brand_Trust_Score",
]

available_features = [f for f in features if f in df.columns]
missing = [f for f in features if f not in df.columns]
if missing:
    print(f"⚠️  ไม่พบคอลัมน์: {missing}")

X = df[available_features]
y = df["Sales_Opportunity"]

# ==========================================
# 5. Scale features
# ==========================================
sup_scaler = StandardScaler()
X_scaled = pd.DataFrame(
    sup_scaler.fit_transform(X),
    columns=available_features
)

# ==========================================
# 6. Train/Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 7. StratifiedKFold Cross-Validation (แก้ไขจาก train_test เพียว)
# ==========================================
# n=123 น้อย → ใช้ 5-fold เพื่อให้ประเมินผลน่าเชื่อถือขึ้น
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=4,
        min_samples_leaf=5, class_weight="balanced"
    ),
}

best_model_obj  = None
best_cv_f1      = 0
best_name       = ""

print("\n========== MODEL BENCHMARK (5-Fold CV) ==========\n")

for name, model in models.items():
    cv_results = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=['accuracy', 'f1_macro'],
        return_train_score=False
    )
    mean_acc = cv_results['test_accuracy'].mean()
    mean_f1  = cv_results['test_f1_macro'].mean()
    std_f1   = cv_results['test_f1_macro'].std()

    # fit บน train set ด้วยเพื่อดู classification report
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{name}")
    print(f"  CV Accuracy : {mean_acc:.4f}")
    print(f"  CV F1-Macro : {mean_f1:.4f} ± {std_f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-" * 50)

    if mean_f1 > best_cv_f1:
        best_cv_f1     = mean_f1
        best_model_obj = model
        best_name      = name

# fit best model บน full data ก่อน save
best_model_obj.fit(X_scaled, y)

# ==========================================
# 8. Feature Importance (ถ้าเป็น Random Forest)
# ==========================================
if best_name == "Random Forest":
    importance = pd.DataFrame({
        "Feature":    available_features,
        "Importance": best_model_obj.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"\nTop Features ({best_name}):")
    print(importance.to_string(index=False))

# ==========================================
# 9. บันทึก Pipeline ทั้งชุด (สำหรับ Streamlit)
# ==========================================
joblib.dump(best_model_obj,     'supervised_model.pkl')
joblib.dump(sup_scaler,         'supervised_scaler.pkl')
joblib.dump(available_features, 'supervised_features.pkl')

print(f"\n✅ Best Model: {best_name}  (CV F1-Macro: {best_cv_f1:.4f})")
print("✅ บันทึก: supervised_model.pkl, supervised_scaler.pkl, supervised_features.pkl")