import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from thefuzz import process
from sklearn.preprocessing import StandardScaler

# =========================================================
# ขั้นตอนที่ 1: โหลดข้อมูลและตั้งชื่อคอลัมน์ใหม่ให้เข้าใจง่าย
# =========================================================
# อ่านไฟล์ CSV โดยข้ามบรรทัดหัวเรื่องที่ไม่ใช่ header
# แล้วทำการเปลี่ยนชื่อคอลัมน์จากภาษาไทยเป็นภาษาอังกฤษเพื่อให้ใช้งานต่อได้สะดวก
# และลดความซับซ้อนของชื่อคอลัมน์เวลาเขียนโค้ด

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

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def categorize_snack_time(text):
    text = str(text).lower()
    if any(word in text for word in ['ดูทีวี', 'ดู TV', 'ดูหนัง', 'ซีรีส์', 'netflix', 'อนิเมะ', 'ยูทูป', 'ดูคลิป', 'ดูภาพยนตร์']): return 'Watching_Media'
    if any(word in text for word in ['ทำงาน', 'เรียน', 'อ่านหนังสือ', 'ทำการบ้าน']): return 'Working_Studying'
    if any(word in text for word in ['ว่าง', 'พัก', 'เบื่อ', 'เพลิน', 'เล่นๆ', 'หลังกินข้าว', 'หลังอาหาร', 'ปากว่าง', 'ตอนอยาก']): return 'Free_Time'
    if any(word in text for word in ['ดึก', 'ก่อนนอน']): return 'Late_Night'
    if any(word in text for word in ['หิว', 'ท้องว่า', 'อาหารว่าง', 'เวลาอาหาร']): return 'Hungry'
    if 'เกม' in text: return 'Playing_Games'
    if any(word in text for word in ['ปาร์ตี้', 'เหล้า', 'เบียร์', 'สังสรรค์', 'เมา']): return 'Party_Drinking'
    return 'Other'

df['Snack_Time_Category'] = df['Snack_Time'].apply(categorize_snack_time)
snack_time_dummies = pd.get_dummies(df['Snack_Time_Category'], prefix='Time', dtype=int)
df = pd.concat([df, snack_time_dummies], axis=1)
df.drop(columns=['Snack_Time' , 'Timestamp', 'Calvora_Product_Strengths_Other'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Calvora_Association(text):
    text = str(text).lower()
    if any(word in text for word in ['ข้าวเกรียบกุ้ง', 'ข้าวเกรียบ', 'กุ้ง', 'ข้างเกรียบกุ้ง', 'ขนมกุ้ง']): return 'Shrimp_Chip_Association'
    if any(word in text for word in ['ฮานาโร']): return 'Hanaro_Association'
    if any(word in text for word in ['แบ็กซ์']): return 'Bax_Association'
    if any(word in text for word in ['คาลโวร่า']): return 'Calvora_Association'
    if any(word in text for word in ['โจโมน่า']): return 'Jomona_Association'
    if any(word in text for word in ['ขนม', 'Snacks', 'ขบเคี้ยว']): return 'General_Snack_Association'
    if any(word in text for word in ['มันฝรั่ง']): return 'potato_chip_association'
    if any(word in text for word in ['ถั่ว']): return 'soybean_chip_association'
    if any(word in text for word in ['ญี่ปุ่น']): return 'Japanese_Association'
    if any(word in text for word in ['ตัวหนังสือสีแดง']): return 'Red_Text_Association'
    return 'Other'

df['Calvora_Association_Category'] = df['Calvora_Association'].apply(Calvora_Association)
Calvora_Association_dummies = pd.get_dummies(df['Calvora_Association_Category'], prefix='Calvora_Association', dtype=int)
df = pd.concat([df, Calvora_Association_dummies], axis=1)
df.drop(columns=['Calvora_Association' , 'Calvora_Association_Category'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Calvora_Tagline_Interpretation(text):
    text = str(text).lower()
    if any(word in text for word in ['วัตถุดิบจากธรรมชาติ', 'ธรรมชาติ', 'ออแกนนิก', 'ทำจากพืช' , 'แสงแดด' , 'วัตถุสังเคราะห์น้อย' , 'ไม่มีสารตกค้าง', 'Natural' , 'เก็บเกี่ยว']): return 'Natural'
    if any(word in text for word in ['-', 'ไม่แน่ใจ', ' ', 'ไม่สนใจ' , 'ไม่รู้' , 'อ่านภาษาอังกฤษไม่ออก']): return 'Negative'
    if any(word in text for word in ['ดี', 'สุจภาพดี' , 'คุภาพดี', 'สุขภ' , 'คุณภาพ' , 'มีประโยช']): return 'Positive'
    return 'Other'

df['Calvora_Tagline_Interpretation_Category'] = df['Calvora_Tagline_Interpretation'].apply(Calvora_Tagline_Interpretation)
Calvora_Tagline_Interpretation_dummies = pd.get_dummies(df['Calvora_Tagline_Interpretation_Category'], prefix='Calvora_Tagline_Interpretation', dtype=int)
df = pd.concat([df, Calvora_Tagline_Interpretation_dummies], axis=1)
df.drop(columns=['Calvora_Tagline_Interpretation'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Natural_Brand_Association(text):
    text = str(text).lower()
    if any(word in text for word in ['เลย์']): return 'Lay'
    if any(word in text for word in ['ทาโร่' ,'taro']): return 'Taro'
    if any(word in text for word in ['Blue diamond']): return 'Blue_diamond'
    if any(word in text for word in ['Yoguruto']): return 'Yoguruto'
    if any(word in text for word in ['Innisfree']): return 'Innisfree'
    if any(word in text for word in ['อายิโนโมโตะ', 'อะยิโนโมโตะ', 'อะยิโนโมโต๊ะ', 'อายิโนะโมะโต๊ะ', 'อาฮิโยโมโต๊ะ', 'อายิโนโมโต๊ะ' , 'อายิโนะโมะโต๊ะ']): return 'Ajinomoto'
    if any(word in text for word in ['Plantae']): return 'Plantae'
    if any(word in text for word in ['Whole food']): return 'Whole_food'
    if any(word in text for word in ['ซันไบร์ท']): return 'Sunbright'
    if any(word in text for word in ['สแน๊กแบ๊ก']): return 'SnackBack'
    if any(word in text for word in ['กาโนล่า']): return 'Canola'
    if any(word in text for word in ['sizzler']): return 'sizzler'
    if any(word in text for word in ['Golden place']): return 'Golden_place'
    if any(word in text for word in ['ข้าวแสนดี']): return 'Khao_Saendee'
    if any(word in text for word in ['โนริ' , 'สาหร่าย']): return 'Nori'
    if any(word in text for word in ['โก๋แก่']): return 'Koh_Kae'
    if any(word in text for word in ['innisfree']): return 'Innisfree'
    if any(word in text for word in ['dairy home']): return 'Dairy_Home'
    if any(word in text for word in ['mister potato']): return 'Mister_Potato'
    if any(word in text for word in ['moleculogy']): return 'Moleculogy'
    if any(word in text for word in ['shisuoka']): return 'Shisuoka'
    if any(word in text for word in ['diamond grain']): return 'Diamond_grain'
    return 'Other'
    
df['Natural_Brand_Association_Category'] = df['Natural_Brand_Association'].apply(Natural_Brand_Association)
df.drop(columns=['Natural_Brand_Association'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Why_Choose_Ebisen(text):
    text = str(text).lower()
    if any(word in text for word in ['อร่อย', 'รสชาติดี', 'ชอบ']): return 'Delicious'
    if any(word in text for word in ['ลอง', 'ทานของเพื่อน', 'อยากลอง' , 'ทดลอง', 'อยากกิน' , 'ลองชิม' , 'เพื่อนน่าจะชอบ ซื้อหลายๆรสไปกอนกับเพื่อน']): return 'Try'
    if any(word in text for word in ['เค็มกว่า' , 'เค็ม']): return 'Saltier'
    if any(word in text for word in ['เอกลักษณ์']): return 'Unique'
    if any(word in text for word in ['รสชาติ', 'ดูไลท์ ไม่เค็มมาก']): return 'Flavor'
    if any(word in text for word in ['แบรนด์']): return 'Brand'
    if any(word in text for word in ['ราคาถูก']): return 'Affordable'
    return 'Other'

df['Why_Choose_Ebisen_Category'] = df['Why_Choose_Ebisen'].apply(Why_Choose_Ebisen)
df.drop(columns=['Why_Choose_Ebisen'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Desired_New_Flavor(text):
    text = str(text).lower()
    if any(word in text for word in ['รสไปซี่' , 'เผ็ด', 'สไปซี่']): return 'Spicy'
    if any(word in text for word in ['ทรัฟเฟิล']): return 'Truffle'
    if any(word in text for word in ['ชีส']): return 'Cheese'
    if any(word in text for word in ['ลาบ' , 'แซ่บๆ']): return 'Laab'
    if any(word in text for word in ['กล้วย']): return 'Banana'
    if any(word in text for word in ['บาบีคิว']): return 'BBQ'
    if any(word in text for word in ['ข้าวโพด']): return 'Corn'
    if any(word in text for word in ['วาซาบิ', 'วาซาบิ']): return 'Wasabi'
    if any(word in text for word in ['โนริ', 'สาหร่าย']): return 'Nori'
    if any(word in text for word in ['มันม่วง']): return 'Purple_Potato'
    if any(word in text for word in ['หมาล่า']): return 'Mala'
    if any(word in text for word in ['คั่วพริกเกลือ']): return 'Spicy_Salt_Roasted'
    if any(word in text for word in ['ปู']): return 'Crab'
    if any(word in text for word in ['เห็ด']): return 'Mushroom'
    if any(word in text for word in ['ถั่วลันเตา']): return 'Pea'
    if any(word in text for word in ['พริกหวาน']): return 'Bell_Pepper'
    if any(word in text for word in ['กุ้งแช่น้ำปลา']): return 'Shrimp_in_Fish_Sauce'
    if any(word in text for word in ['ซีฟู๊ด']): return 'Seafood'
    if any(word in text for word in ['ต้มยำกุ้ง']): return 'Tom_Yum_Shrimp'
    if any(word in text for word in ['รสน้ำผึ้ง']): return 'Honey'
    return 'Other'
    
df['Desired_New_Flavor_Category'] = df['Desired_New_Flavor'].apply(Desired_New_Flavor)
df.drop(columns=['Desired_New_Flavor'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Reason_Never_Tried(text):
    text = str(text).lower()
    if any(word in text for word in ['ไม่ค่อยทานขนม']): return 'Not_Snack_Person'
    if any(word in text for word in ['ไม่ค่อยได้ลอง', 'ไม่มีโอกาสได้ลอง', 'ยังไม่เคยซื้อทาน', 'ไม่เคยซื้อลอง', 'ไม่เคยเห็น', 'อยากลองชิมก่อน', 'ยังไม่มีโอกาสซื้อ']): return 'Never_Had_Chance'
    if any(word in text for word in ['ชินกับแบบเดิม' ,'เคยชินกับ' ,'ซื้อตัวดั้งเดิม']): return 'Used_to_Old_Flavors'
    if any(word in text for word in ['ยังไม่ดึงดูด', 'คิดว่ารสเหมือนกัน' ,'ไม่ได้รู้สึกอยากกิน', 'น่าจะเหมือนฮานาโร' ,'เฉยๆ', 'มีอันอื่นให้กินเยอะ']): return 'Not_Attractive'
    return 'Other'

df['Reason_Never_Tried_Category'] = df['Reason_Never_Tried'].apply(Reason_Never_Tried)
df.drop(columns=['Reason_Never_Tried'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Expected_Stronger_Flavor(text):
    text = str(text).lower()
    if any(word in text for word in ['BBQ']): return 'BBQ'
    if any(word in text for word in ['ต้มยำ']): return 'Tom_Yum_Flavor'
    if any(word in text for word in ['รสกุ้ง', 'กุ้ง']): return 'Shrimp_Flavor'
    if any(word in text for word in ['เค็ม']): return 'Saltier'
    if any(word in text for word in ['โนริ', 'สาหร่าย']): return 'Nori_Flavor'
    if any(word in text for word in ['เข้มข้น', 'รสจัด', 'เด่นชัด', 'ชัดเจน', 'จัดจ้าน', 'ที่ชัดขึ้น']): return 'More_Intense_Flavor'
    if any(word in text for word in ['หมาล่า']): return 'Mala_Flavor'
    if any(word in text for word in ['เผ็ด', 'พริก']): return 'Spicier'
    if any(word in text for word in ['ลาบ' , 'แซบ']): return 'Laab_Flavor'
    if any(word in text for word in ['ผงเยอะๆ']): return 'More_Seasoning'
    if any(word in text for word in ['อร่อย']): return 'Delicious'
    return 'Other'

df['Expected_Stronger_Flavor_Category'] = df['Expected_Stronger_Flavor'].apply(Expected_Stronger_Flavor)
df.drop(columns=['Expected_Stronger_Flavor'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Why_Like_Stronger_Flavor(text):
    text = str(text).lower()
    if any(word in text for word in ['จัดจ้าน', 'แซ่บ' ,'ได้กลิ่นถึงวัดถุดิบมากขึ้น' ,'เข้มข้น' ,'ทานแล้วมีรสชาติในขณะที่ทาน' ,
                                     'ไม่ชอบกินจืด', 'ชอบ', 'รสจัด', 'เข้าถึงรสชาติ', 'ถึงใจ', 'ชอบรสชาติเยอะๆ', 'กินเพลิน']): return 'Like_Spicy_Flavor'
    if any(word in text for word in ['เค็ม']): return 'Like_Saltier_Flavor'
    if any(word in text for word in ['เผ็ด ']): return 'Like_Spicy_Flavor'
    if any(word in text for word in ['นัว']): return 'Like_Rich_Flavor'
    if any(word in text for word in ['ติดปาก']): return 'Like_Addictive_Flavor'
    if any(word in text for word in ['อร่อย', 'อร่อน']): return 'Like_Delicious_Flavor'
    return 'Other'

df['Why_Like_Stronger_Flavor_Category'] = df['Why_Like_Stronger_Flavor'].apply(Why_Like_Stronger_Flavor)
df.drop(columns=['Why_Like_Stronger_Flavor'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Strong_Flavor_Occasion(text):
    text = str(text).lower()
    if any(word in text for word in ['ดูทีวี', 'ดู TV', 'ดูหนัง', 'ซีรีส์', 'netflix', 'อนิเมะ', 'ยูทูป', 'ดูคลิป', 'ดูภาพยนตร์']): return 'Watching_Media'
    if any(word in text for word in ['ทำงาน', 'เรียน', 'อ่านหนังสือ', 'ทำการบ้าน']): return 'Working_Studying'
    if any(word in text for word in ['ว่าง', 'พัก', 'เบื่อ', 'เพลิน', 'เล่นๆ', 'หลังกินข้าว', 'หลังอาหาร', 'ปากว่าง', 'ตอนอยาก']): return 'Free_Time'
    if any(word in text for word in ['ดึก', 'ก่อนนอน']): return 'Late_Night'
    if any(word in text for word in ['หิว', 'ท้องว่า', 'อาหารว่าง', 'เวลาอาหาร']): return 'Hungry'
    if any(word in text for word in ['คู่กับเครื่องดื่ม', 'โค๊ก', 'น้ำอัดลม' ,'แป๊บซี่', 'ชา']): return 'With_Drinks'
    if 'เกม' in text: return 'Playing_Games'
    if any(word in text for word in ['ปาร์ตี้', 'เหล้า', 'เบียร์', 'สังสรรค์', 'เมา', 'ดื่ม']): return 'Party_Drinking'
    return 'Other'

df['Strong_Flavor_Occasion_Category'] = df['Strong_Flavor_Occasion'].apply(Strong_Flavor_Occasion)
df.drop(columns=['Strong_Flavor_Occasion'], inplace=True)

# สร้างฟังก์ชันเพื่อจัดหมวดหมู่เวลาที่ทานขนมจากข้อความ
def Reason_Not_Willing(text):
    text = str(text).lower()
    if any(word in text for word in ['ไม่อร่อย' ,'ชอบรสไม่จัดมาก' ,'ชอบรสชาตกลาง' ,'เข้มข้นมาก ผงปรุงรสก็มากตาม', 'ไม่ค่อยชอบ' ,'รสเข้มปกติจะเค็มไป', 'เคยลองกินขนมขบเคี้ยวที่รสชาติเข้มข้น ทำให้รู้สึกมีรสชาติที่เค็มกว่าปกติ']): return 'Not_Delicious'
    if any(word in text for word in ['ถ้ามีเพื่อนกิน อาจจะอยากลอง']): return 'Might_Try_With_Friends'
    if any(word in text for word in ['เค็มอยู่แล้ว' , 'ชอบรสดั้งเดิม' ,'ของเดิม', 'ชอบทานฮานาโรมากกว่า' ,'ปกติทานแค่รสชาติเดียว' ,'ค็มพอแล้ว', 'รสชาติตอนนี้ก็อร่อยอยู่แล้ว']): return 'Already_Salty'
    if any(word in text for word in ['เฉยๆ' , 'ไม่คาดหวัง' ,'งดทานขนมขบเคี้ยว']): return 'Indifferent'
    if any(word in text for word in ['ก็อร่อยอยู่แล้ว', 'รสชาตินี้ดีแล้ว']): return 'Already_Delicious'
    return 'Other'

df['Reason_Not_Willing_Category'] = df['Reason_Not_Willing'].apply(Reason_Not_Willing)
df.drop(columns=['Reason_Not_Willing'], inplace=True)

Gender = pd.get_dummies(df['Gender'], prefix='Gender', dtype=int)
df = pd.concat([df, Gender], axis=1)
df.drop(columns=['Gender'], inplace=True)

cols_to_scale = [
    'Know_Ebisen',
    'Age',
    'Calvora_Tagline_Reflection',
    'Purchase_Factor_Quality_Ingredients', 
    'Purchase_Factor_Tasty', 
    'Purchase_Factor_Many_Flavors', 
    'Purchase_Factor_Crispy', 
    'Purchase_Factor_Healthy'
]

# สั่ง Scale พร้อมกันทีเดียว
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


col_A = ['Known_Snack_Products', 'Calvora_Natural_Ingredient_Trust_Other', 'Tasted_Snack_Brands', 'Ebisen_Flavors' , 'Calvora_Image_Association']
col_N = ['Known_Snack_', 'Calvora_Natural_Ingredient_', 'Tasted_Snack_', 'Ebisen_Flavor_', 'Calvora_Image_']
for idx, col in enumerate(col_A):
    if col in df.columns:
        Add_c = df[col].str.get_dummies(sep=',')
        Add_c.columns = [col_N[idx] + str(c).strip() for c in Add_c.columns]
        df = pd.concat([df, Add_c], axis=1)
        df.drop(columns=[col], inplace=True)

df.drop(columns=['Believe_Ebisen_Shrimp_Reason', 'Believe_Ebisen_Shrimp_Reason_2' , 'Heard_Calvora'], inplace=True)
# ลบคอลัมน์ที่ชื่อซ้ำกันออก
df = df.loc[:, ~df.columns.duplicated()]

# เช็คคอลัมน์ตัวเลขที่มีค่า 2 หรือมากกว่า เพื่อเตรียมการ Scaling ต่อ
numeric_cols = df.select_dtypes(include=[np.number]).columns
columns_with_ge2 = []
for col in numeric_cols:
    if (df[col] >= 2).any():
        columns_with_ge2.append(col)

print("Columns with values >= 2 (for scaling):")
for col in columns_with_ge2:
    print(f"- {col}: max={df[col].max()}, count>=2={(df[col] >= 2).sum()}")

#'''
string_cols = df.select_dtypes(include=['object']).columns
print(df[string_cols].info())
for col in string_cols:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print("\n")
    
#'''



#print(df['Known_Snack_คาลโวร่า'])
# =========================================================
# ขั้นตอนสุดท้าย: จัดเรียงคอลัมน์ตามหัวข้อ และบันทึกผลเป็น CSV
# =========================================================
column_groups = [
    'Timestamp',
    'Know_Ebisen', 'Age', 'Gender',
    'Calvora_Tagline_Awareness',
    'Purchase_Factor_',
    'Is_Calvora_',
    'Calvora_Tagline_',
    'Strength_',
    'Calvora_Natural_Ingredient_',
    'Known_Snack_',
    'Tasted_Snack_',
    'Ebisen_Flavor_',
    'Calvora_Image_',
    'Time_',
    'Calvora_Association_',
    'Natural_Brand_Association_',
    'Why_Choose_Ebisen_',
    'Desired_New_Flavor_',
    'Reason_Never_Tried_',
    'Expected_Stronger_Flavor_',
    'Why_Like_Stronger_Flavor_',
    'Strong_Flavor_Occasion_',
    'Reason_Not_Willing_'
]
ordered_cols = []
for prefix in column_groups:
    if prefix in df.columns and prefix not in ordered_cols:
        ordered_cols.append(prefix)
    ordered_cols.extend([col for col in df.columns if col.startswith(prefix) and col not in ordered_cols])
ordered_cols.extend([col for col in df.columns if col not in ordered_cols])
if ordered_cols:
    df = df[ordered_cols]

output_file = 'BU_Data_transformed.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"เขียนไฟล์เรียบร้อย: {output_file}")
print(df.info(verbose=True, show_counts=True))