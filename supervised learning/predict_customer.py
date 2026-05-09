import pandas as pd
import pickle

# โหลดข้อมูลเดิม เพื่อเอาชื่อ Feature ให้ตรงกับตอน Train
df = pd.read_csv("BU_Data_transformed.csv")

target = "Try_New_Flavor"

drop_cols = [
    target,
    "Like_Stronger_Ebisen_Flavor"
]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=["int64", "float64"])

# โหลดโมเดลที่ Train ไว้แล้ว
with open("supervised_model.pkl", "rb") as file:
    model = pickle.load(file)

# สร้างข้อมูลลูกค้าจำลอง 1 คน
# เริ่มจากใส่ค่า 0 ให้ทุก Feature ก่อน
sample_customer = pd.DataFrame(
    [[0] * len(X.columns)],
    columns=X.columns
)

# กำหนดค่าตัวอย่างของลูกค้า
sample_customer["Know_Ebisen"] = 2
sample_customer["Age"] = 1
sample_customer["Purchase_Factor_Tasty"] = 5
sample_customer["Purchase_Factor_Many_Flavors"] = 5
sample_customer["Purchase_Factor_Crispy"] = 4
sample_customer["Purchase_Factor_Healthy"] = 3
sample_customer["Believe_Ebisen_Shrimp"] = 1

# ถ้ามีคอลัมน์นี้ในข้อมูล ให้ใส่ค่า 1
if "Ebisen_Flavor_Original" in sample_customer.columns:
    sample_customer["Ebisen_Flavor_Original"] = 1

if "Known_Snack_เอบินาริ" in sample_customer.columns:
    sample_customer["Known_Snack_เอบินาริ"] = 1

if "Tasted_Snack_เอบินาริ" in sample_customer.columns:
    sample_customer["Tasted_Snack_เอบินาริ"] = 1

# ทำนายผล
prediction = model.predict(sample_customer)[0]
probability = model.predict_proba(sample_customer)[0]

print("ผลการทำนาย:", prediction)

if prediction == 1:
    print("ลูกค้าคนนี้มีแนวโน้มจะลองรสชาติใหม่")
else:
    print("ลูกค้าคนนี้มีแนวโน้มจะไม่ลองรสชาติใหม่")

print("\nความน่าจะเป็น")
print("ไม่ลอง:", probability[0])
print("ลอง:", probability[1])