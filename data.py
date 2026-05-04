import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

df = pd.read_csv("BU_Data.csv", skiprows=1)

df.rename(columns={
    'Timestamp': 'Timestamp',
    'คุณเคยได้ยินชื่อแบรนด์ Calvora (คาลโวร่า) หรือไม่?': 'Heard_Calvora',
    'เมื่อพูดถึง Calvora (คาลโวร่า) คุณนึกถึงอะไร?': 'Calvora_Association',
    'จากภาพต่อไปนี้ ภาพใดทำให้คุณนึกถึงแบรนด์ Calvora (คาลโวร่า) บ้าง?': 'Calvora_Image_Association',
    'จากผลิตภัณฑ์ขนมขบเคี้ยวดังต่อไปนี้ คุณรู้จักผลิตภัณฑ์ใดบ้าง? (เลือกได้หลายข้อ)': 'Known_Snack_Products',
    'คุณเคยทานผลิตภัณฑ์ขนมขบเคี้ยวยี่ห้อใดบ้าง? (เลือกได้หลายข้อ)': 'Tasted_Snack_Brands',
    'คุณสมบัติของขนมขบเคี้ยวที่มีผลต่อการตัดสินใจซื้อ [วัตถุดิบคุณภาพ]': 'Purchase_Factor_Quality_Ingredients',
    'คุณสมบัติของขนมขบเคี้ยวที่มีผลต่อการตัดสินใจซื้อ [รสชาติอร่อย]': 'Purchase_Factor_Tasty',
    'คุณสมบัติของขนมขบเคี้ยวที่มีผลต่อการตัดสินใจซื้อ [รสชาติให้เลือกเยอะ]': 'Purchase_Factor_Many_Flavors',
    'คุณสมบัติของขนมขบเคี้ยวที่มีผลต่อการตัดสินใจซื้อ [สัมผัสกรุบกรอบ เคี้ยวเพลิน]': 'Purchase_Factor_Crispy',
    'คุณสมบัติของขนมขบเคี้ยวที่มีผลต่อการตัดสินใจซื้อ [เพื่อสุขภาพที่ดี (แคลอรี่น้อย, คั่วไม่ทอด)]': 'Purchase_Factor_Healthy',
    'คุณชอบทานขนมขบเคี้ยวตอนไหน พิมพ์สั้นๆ': 'Snack_Time',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [คาลโวร่า]': 'Is_Calvora_Calvora',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [เอบินาริ]': 'Is_Calvora_Ebinari',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [เอบินาริ X]': 'Is_Calvora_Ebinari_X',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [ฮานาโร]': 'Is_Calvora_Hanaro',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [แบ็กซ์]': 'Is_Calvora_Bax',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [พัฟโมริ]': 'Is_Calvora_Puffmori',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [Jomona]': 'Is_Calvora_Jomona',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [บิบิป๊อป]': 'Is_Calvora_Bibipop',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [สแน็คแบ๊ค]': 'Is_Calvora_SnackBack',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [Veggie Snap]': 'Is_Calvora_VeggieSnap',
    'คุณคิดว่าขนมยี่ห้อต่อไปนี้ เป็นของ Calvora หรือไม่? [ฟรูทร่า]': 'Is_Calvora_Frutra',
    'คุณคิดว่าผลิตภัณฑ์ของ Calvora มีจุดเด่นด้านใด? (เลือกได้หลายข้อ) อื่นๆ โปรดระบุ': 'Calvora_Product_Strengths_Other',
    'คุณรู้หรือไม่ว่า Calvora มี Tagline ว่า “Harvest the Power of Nature” "เก็บเกี่ยวพลังของธรรมชาติ" ?': 'Calvora_Tagline_Awareness',
    'จากข้อความ “Harvest the Power of Nature” คุณจะตีความว่าอย่างไร?': 'Calvora_Tagline_Interpretation',
    'คุณรู้สึกว่า tagline “Harvest the Power of Nature” ของ Calvora สะท้อนภาพลักษณ์ของแบรนด์มาก-น้อยเพียงใด?': 'Calvora_Tagline_Reflection',
    'อะไรจะทำให้คุณรู้สึกเชื่อมั่นในวัตถุดิบจากธรรมชาติที่ Calvora ใช้ (เลือกได้หลายข้อ) อื่นๆ โปรดระบุ': 'Calvora_Natural_Ingredient_Trust_Other',
    'เมื่อนึกถึงแบรนด์ที่มี "วัตถุดิบจากธรรมชาติ" คุณนึกถึงแบรนด์ใด (ไม่จำเป็นต้องเป็นผลิตภัณฑ์ขนมขบเคี้ยว) เพราะอะไร ขอเหตุผลสั้นๆ': 'Natural_Brand_Association',
    'คุณรู้จักหรือเคยทาน Calvora เอบินาริ (Ebisen) หรือไม่': 'Know_Ebisen',
    'คุณเคยทาน เอบินาริ รสชาติไหนบ้าง': 'Ebisen_Flavors',
    'เลือกทาน Calvora เอบินาริ เพราะอะไร': 'Why_Choose_Ebisen',
    'เชื่อหรือไม่ว่า Calvora เอบินาริ ทำมาจากกุ้งแท้ๆ ?': 'Believe_Ebisen_Shrimp',
    'จากข้อก่อนหน้า เชื่อหรือไม่เชื่อ เพราะอะไร?': 'Believe_Ebisen_Shrimp_Reason',
    'ถ้าออกรสชาติใหม่จะลองไหม': 'Try_New_Flavor',
    'รสชาติไหนที่อยากให้มีเพิ่ม': 'Desired_New_Flavor',
    'ทำไมถึงยังไม่เคยทาน': 'Reason_Never_Tried',
    'เชื่อหรือไม่ว่า Calvora เอบินาริ ทำมาจากกุ้งแท้ๆ ? 2': 'Believe_Ebisen_Shrimp_2',
    'จากข้อก่อนหน้า เชื่อหรือไม่เชื่อ เพราะอะไร? 2': 'Believe_Ebisen_Shrimp_Reason_2',
    'ถ้าเอบินาริมีรสชาติที่เข้มข้นขึ้น คุณจะชอบ/อยากลองกินมั้ย': 'Like_Stronger_Ebisen_Flavor',
    'ถ้าเอบินาริมี \'รสชาติที่เข้มข้นขึ้น\' คุณคิดว่าจะเป็นรสชาติอย่างไร': 'Expected_Stronger_Flavor',
    'ทำไมคุณถึงชอบ \'รสชาติที่เข้มข้น\'': 'Why_Like_Stronger_Flavor',
    'แล้วรสชาติที่ เข้มข้น แบบนี้คุณคิดว่าเหมาะกับการกินในช่วงเวลาไหน หรือ กินคู่กับอะไร?': 'Strong_Flavor_Occasion',
    'ขอเหตุผลหน่อย ทำไมถึงไม่อยากลอง?': 'Reason_Not_Willing',
    'อายุของคุณ': 'Age',
    'เพศของคุณ': 'Gender',
}, inplace=True)

print(df.info())

# =========================================================
# ขั้นตอนที่ 2: การทำความสะอาดข้อมูล (Data Cleaning)
# =========================================================
# ในขั้นตอนนี้ เราจะทำความสะอาดข้อมูลโดยเลือกเฉพาะผู้ตอบที่เคยได้ยินแบรนด์ Calvora
# และลบคอลัมน์ที่ว่างทั้งหมดออก เพื่อลดความซับซ้อนและป้องกันข้อมูลผิดพลาด
print("2. กำลังทำความสะอาดข้อมูล...")
# เลือกเฉพาะคนที่ตอบว่า "เคย" และลบคอลัมน์ที่ว่างทั้งหมด
df = df[df['Heard_Calvora'] == 'เคย'].copy()
df.dropna(how='all', axis=1, inplace=True)

# =========================================================
# ขั้นตอนที่ 3: การแปลงข้อมูลเป็นตัวเลข (Encoding & Transformation)
# =========================================================
# ในขั้นตอนนี้ เราจะแปลงข้อมูลที่เป็นข้อความหรือหมวดหมู่ให้เป็นตัวเลข
# เพื่อให้สามารถนำไปใช้ในการวิเคราะห์และสร้างโมเดลได้
# รวมถึงการจัดการกับค่าว่างและการเตรียมข้อมูลสำหรับการคำนวณ

# 3.1 แปลงคอลัมน์ Likert Scale (ระดับการวัดความเห็น 1-5)
# เช่น "มีผลมากที่สุด" = 5, "ไม่มีผล" = 1
likert_cols = ['Purchase_Factor_Quality_Ingredients', 'Purchase_Factor_Tasty', 'Purchase_Factor_Many_Flavors', 'Purchase_Factor_Crispy', 'Purchase_Factor_Healthy']
likert_mapping = {'มีผลมากที่สุด': 5, 'มีผลมาก': 4, 'ปานกลาง': 3, 'มีผลน้อย': 2, 'ไม่มีผล': 1, 'ไม่มีผลเลย': 1}
for col in likert_cols:
    df[col] = df[col].map(likert_mapping)

# 3.2 แปลงคำถาม "เป็นของ Calvora หรือไม่?" เป็น 1 (ใช่) หรือ 0 (ไม่ใช่)
# ใช้ lambda function เพื่อตรวจสอบและแปลงค่า
is_calvora_cols = [col for col in df.columns if col.startswith('Is_Calvora_')]
for col in is_calvora_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).strip() == 'ใช่ เป็นของ Calvora' else (0 if str(x).strip() == 'ไม่ใช่' else np.nan))

# 3.3 แปลงข้อมูลหมวดหมู่ทั่วไป เช่น เพศ อายุ ความรู้จักแบรนด์ เป็นตัวเลข
df['Heard_Calvora'] = df['Heard_Calvora'].map({'เคย': 1})
df['Calvora_Tagline_Awareness'] = df['Calvora_Tagline_Awareness'].map({'รู้': 1, 'ไม่รู้': 0})
df['Know_Ebisen'] = df['Know_Ebisen'].map({'รู้จัก และเคยทาน': 2, 'รู้จัก แต่ไม่เคยทาน': 1, 'ไม่รู้จักเลย': 0})
df['Age'] = df['Age'].map({'ต่ำกว่า 20ปี': 0, '20-29ปี': 1, '30-39ปี': 2, '40-49ปี': 3, '50ปี ขึ้นไป': 4})
df['Gender'] = df['Gender'].map({'หญิง': 0, 'ชาย': 1, 'LGBTQ+': 2})
df['Believe_Ebisen_Shrimp'] = df['Believe_Ebisen_Shrimp'].map({'เชื่อ': 1, 'ไม่เชื่อ': 0})
df['Try_New_Flavor'] = df['Try_New_Flavor'].map({'ลอง': 1, 'ไม่ลอง': 0})
df['Like_Stronger_Ebisen_Flavor'] = df['Like_Stronger_Ebisen_Flavor'].map({'ชอบรสเข้มข้น อยากลอง': 1, 'ไม่อยากลอง': 0})

# จัดการกับคอลัมน์ที่อาจมีหรือไม่มี
if 'Believe_Ebisen_Shrimp_2' in df.columns:
    df['Believe_Ebisen_Shrimp_2'] = df['Believe_Ebisen_Shrimp_2'].map({'เชื่อ': 1, 'ไม่เชื่อ': 0})

# เติมค่าว่าง (NaN) ในคอลัมน์ตัวเลขทั้งหมดให้เป็น 0 เพื่อพร้อมเข้า Model
# เพื่อป้องกันปัญหาในการคำนวณ
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# ใช้ get_dummies แยกด้วยเครื่องหมายลูกน้ำ สำหรับคอลัมน์จุดเด่นของผลิตภัณฑ์
# เพื่อสร้างฟีเจอร์ใหม่จากข้อความที่ผู้ใช้เลือกหลายข้อ
strengths_dummies = df['Calvora_Product_Strengths_Other'].str.get_dummies(sep=',')

# เติมคำว่า 'Strength_' นำหน้าชื่อคอลัมน์ใหม่ จะได้ไม่งงตอนดูกราฟ
strengths_dummies.columns = ['Strength_' + str(col).strip() for col in strengths_dummies.columns]

# นำคอลัมน์ใหม่ไปต่อท้าย DataFrame หลัก
df = pd.concat([df, strengths_dummies], axis=1)
'''
string_cols = df.select_dtypes(include=['object']).columns
print(df[string_cols].info())
for col in string_cols:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print("\n")
    
'''
df.drop(columns=['Calvora_Product_Strengths_Other'], inplace=True)
print(df.info())