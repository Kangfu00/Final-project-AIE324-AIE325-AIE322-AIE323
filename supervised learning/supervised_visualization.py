import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


plt.rcParams['font.family'] = 'Tahoma' 

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

available_features = [col for col in features if col in df.columns]

X = df[available_features]
y = df["Sales_Opportunity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

with open("supervised_model.pkl", "rb") as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Best Model: Logistic Regression")
print("Accuracy:", accuracy)
print("F1-Score:", f1)

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

coef_df = pd.DataFrame({
    "Feature": available_features,
    "Coefficient": model.coef_[0]
})

coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()

coef_df = coef_df.sort_values(
    by="Abs_Coefficient",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=coef_df,
    x="Abs_Coefficient",
    y="Feature"
)

plt.title("Top 10 Feature Effects: Sales Opportunity")
plt.xlabel("Effect Strength")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_effect_sales_opportunity.png", dpi=300)
plt.show()

print("\nสร้าง Visualization ใหม่เสร็จแล้ว")
print("- confusion_matrix_sales_opportunity.png")
print("- feature_effect_sales_opportunity.png")