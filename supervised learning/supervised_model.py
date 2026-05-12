import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma' 
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
# 8. Feature Importance / Coefficients
# ==========================================
if hasattr(best_model_obj, 'coef_'):
    # สำหรับ Logistic Regression
    importance_vals = best_model_obj.coef_[0]
    importance_type = "Coefficient"
else:
    # สำหรับ Random Forest หรือ Decision Tree
    importance_vals = best_model_obj.feature_importances_
    importance_type = "Feature Importance"

importance = pd.DataFrame({
    "Feature":    available_features,
    importance_type: importance_vals
}).sort_values(importance_type, ascending=False)

print(f"\nTop Features ({best_name} - {importance_type}):")
print(importance.to_string(index=False))

# ==========================================
# 9. บันทึก Pipeline ทั้งชุด (สำหรับ Streamlit)
# ==========================================
joblib.dump(best_model_obj,     'supervised_model.pkl')
joblib.dump(sup_scaler,         'supervised_scaler.pkl')
joblib.dump(available_features, 'supervised_features.pkl')

print(f"\n✅ Best Model: {best_name}  (CV F1-Macro: {best_cv_f1:.4f})")
print("✅ บันทึก: supervised_model.pkl, supervised_scaler.pkl, supervised_features.pkl")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Opportunity", "High Opportunity"],
    yticklabels=["Low Opportunity", "High Opportunity"]
)

plt.title("Confusion Matrix: Sales Opportunity Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_sales_opportunity.png", dpi=300)
plt.show()

# ==========================================
# สร้างกราฟ Pie Chart การกระจายตัวของ Sales Opportunity
# ==========================================
sales_counts = df["Sales_Opportunity"].value_counts()

# กำหนดสีให้คล้ายกับในสไลด์ (สีเขียวอมฟ้า และ สีเทาอ่อน)
colors = ['#16a085', '#e2e6e9'] 
# หากค่า 0 มีมากกว่า 1 ให้สลับสี เพื่อให้สีเขียวตรงกับ High Opportunity เสมอ
if sales_counts.index[0] == 0:
    colors = ['#e2e6e9', '#16a085']

plt.figure(figsize=(6, 6))
plt.pie(
    sales_counts, 
    labels=['High Opportunity (1)', 'Low Opportunity (0)'] if sales_counts.index[0] == 1 else ['Low Opportunity (0)', 'High Opportunity (1)'], 
    autopct='%1.0f%%', 
    colors=colors, 
    startangle=90,
    textprops={'fontsize': 12}
)

plt.title("การกระจาย Sales Opportunity", fontsize=14, fontweight='bold')
plt.tight_layout()

# บันทึกเป็นรูปภาพ
plt.savefig("pie_chart_sales_opportunity.png", dpi=300)
plt.show()

print("\nสร้างกราฟ Pie Chart สำเร็จ! บันทึกไฟล์ชื่อ pie_chart_sales_opportunity.png")

if hasattr(best_model_obj, 'coef_'):
    # สำหรับ Logistic Regression
    importance_vals = best_model_obj.coef_[0]
    importance_type = "Coefficient"
elif hasattr(best_model_obj, 'feature_importances_'):
    # สำหรับ Random Forest และ Decision Tree
    importance_vals = best_model_obj.feature_importances_
    importance_type = "Feature Importance"
else:
    importance_vals = np.zeros(len(available_features))
    importance_type = "Importance"

coef_df = pd.DataFrame({
    "Feature": available_features,
    importance_type: importance_vals
})

# ใช้ค่า Absolute เพื่อหาฟีเจอร์ที่มีผลกระทบแรงที่สุด (ไม่ว่าบวกหรือลบ)
coef_df["Abs_Importance"] = coef_df[importance_type].abs()

coef_df = coef_df.sort_values(
    by="Abs_Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=coef_df,
    x="Abs_Importance",
    y="Feature",
    palette="viridis"
)

plt.title(f"Top 10 Feature Effects: Sales Opportunity ({best_name})")
plt.xlabel("Effect Strength (Absolute Value)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_effect_sales_opportunity.png", dpi=300)
plt.show()

print("\nสร้าง Visualization ใหม่เสร็จแล้ว")
print("- confusion_matrix_sales_opportunity.png")
print("- feature_effect_sales_opportunity.png")