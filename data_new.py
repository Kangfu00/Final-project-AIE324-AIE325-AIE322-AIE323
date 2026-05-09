import pandas as pd
import numpy as np
from thefuzz import process
from sklearn.impute import KNNImputer

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


# 2. สร้างฟังก์ชันสำหรับจับคู่คำ
def clean_brand_name(text, choices, threshold=70):
    text = str(text).strip()
    if text == 'nan' or text == '':
        return 'Other'
        
    # extractOne จะคืนค่า (คำที่ใกล้เคียงที่สุด, คะแนนความเหมือน)
    best_match, score = process.extractOne(text, choices)
    
    # ถ้าคะแนนความคล้ายคลึงผ่านเกณฑ์ (เช่น 70%) ให้ใช้คำมาตรฐาน
    if score >= threshold:
        return best_match
    else:
        return 'Other'

standard_brands = [
    'เลย์', 'ทาโร่', 'Blue_diamond', 'Yoguruto', 'Innisfree', 
    'อายิโนโมโตะ', 'Plantae', 'Whole_food', 'ซันไบร์ท', 'สแน๊กแบ๊ก', 'กาโนล่า', 'sizzler', 'Golden_place', 'ข้าวแสนดี', 'โนริ', 'โก๋แก่', 'innisfree', 'dairy home', 'mister potato', 'moleculogy', 'shisuoka', 'diamond grain'
]

df['Natural_Brand_Association_Category'] = df['Natural_Brand_Association'].apply(
    lambda x: clean_brand_name(x, standard_brands)
)
fig = {'เลย์': 'Lay', 'ทาโร่': 'Taro', 'อายิโนโมโตะ': 'Ajinomoto', 'ซันไบร์ท': 'Sunbrite', 'สแน๊กแบ๊ก': 'SnackBack', 'กาโนล่า': 'Canola', 'ข้าวแสนดี': 'Khao_Saendee', 'โนริ': 'Nori', 'โก๋แก่': 'Koh_Kae', 'innisfree': 'Innisfree', 'dairy home': 'Dairy_Home', 'mister potato': 'Mister_Potato', 'moleculogy': 'Moleculogy', 'shisuoka': 'Shisuoka', 'diamond grain': 'Diamond_Grain'}
df['Natural_Brand_Association_Category'] = df['Natural_Brand_Association_Category'].replace(fig)
Natural_Brand_Association = pd.get_dummies(df['Natural_Brand_Association_Category'], prefix='Natural_Brand', dtype=int)
df = pd.concat([df, Natural_Brand_Association], axis=1)
df.drop(columns=['Natural_Brand_Association'], inplace=True)

RULES = {
    'Watching_Media': ['ดูทีวี', 'ดู tv', 'ดูหนัง', 'ซีรีส์', 'netflix', 'อนิเมะ', 'ยูทูป', 'ดูคลิป', 'ดูภาพยนตร์'],
    'Working_Studying': ['ทำงาน', 'เรียน', 'อ่านหนังสือ', 'ทำการบ้าน'],
    'Free_Time': ['ว่าง', 'พัก', 'เบื่อ', 'เพลิน', 'เล่นๆ', 'หลังกินข้าว', 'หลังอาหาร', 'ปากว่าง', 'ตอนอยาก'],
    'Late_Night': ['ดึก', 'ก่อนนอน'],
    'Hungry': ['หิว', 'ท้องว่าง', 'อาหารว่าง', 'เวลาอาหาร'],
    'Playing_Games': ['เกม'],
    'Party_Drinking': ['ปาร์ตี้', 'เหล้า', 'เบียร์', 'สังสรรค์', 'เมา']
}

# 2. ฟังก์ชันจะเหลือแค่นี้ สั้นและสะอาดมาก
def categorize_snack_time(text):
    text = str(text).lower()
    
    # วนลูปเช็คตามกฏใน Dictionary
    for category, keywords in RULES.items():
        if any(word in text for word in keywords):
            return category
            
    return 'Other'

# 3. นำไปใช้งาน (เหมือนเดิม)
df['Snack_Time_Category'] = df['Snack_Time'].apply(categorize_snack_time)
Snack_Time = pd.get_dummies(df['Snack_Time_Category'], prefix='Snack_Time', dtype=int)
df = pd.concat([df, Snack_Time], axis=1)
df.drop(columns=['Snack_Time'], inplace=True)

standard_brands = [
    'ข้าวเกรียบกุ้ง', 'ฮานาโร', 'แบ็กซ์', 'คาลโวร่า', 'โจโมน่า', 
    'ขนม', 'มันฝรั่ง', 'ถั่ว', 'ซันไบร์ท', 'ญี่ปุ่น', 'กาโนล่า', 'ตัวหนังสือสีแดง', 'Snacks', 'ข้าวแสนดี', 'ขบเคี้ยว'
]
df['Calvora_Association_Category'] = df['Calvora_Association'].apply(
    lambda x: clean_brand_name(x, standard_brands)
)
fig = {'ข้าวเกรียบกุ้ง': 'Shrimp_Chip_Association', 'ฮานาโร': 'Hanaro_Association', 'แบ็กซ์': 'Bax_Association', 'คาลโวร่า': 'Calvora_Association', 'โจโมน่า': 'Jomona_Association', 'ขนม': 'General_Snack_Association', 'มันฝรั่ง': 'Potato_Chip_Association', 'ถั่ว': 'Soybean_Chip_Association', 'ซันไบร์ท': 'Sunbrite', 'ญี่ปุ่น': 'Japanese_Association', 'กาโนล่า': 'Canola', 'ตัวหนังสือสีแดง': 'Red_Text_Association', 'Snacks': 'General_Snack_Association', 'ขบเคี้ยว': 'General_Snack_Association'}
df['Calvora_Association_Category'] = df['Calvora_Association_Category'].replace(fig)
Calvora_Association = pd.get_dummies(df['Calvora_Association_Category'], prefix='Calvora', dtype=int)
df = pd.concat([df, Calvora_Association], axis=1)
df.drop(columns=['Calvora_Association'], inplace=True)

RULES = {
    'Natural': ['จากธรรมชาติ', 'ออแกนนิก', 'ทำจากพืช', 'แสงแดด', 'วัตถุสังเคราะห์น้อย', 'ไม่มีสารตกค้าง', 'Natural', 'เก็บเกี่ยว'],
    'Negative': ['ไม่', '-'],
    'Positive': ['ดี', 'มีประโยช', 'คุณภาพ', 'สุจภาพดี'],
}

df['Calvora_Tagline_Interpretation_Category'] = df['Calvora_Tagline_Interpretation'].apply(categorize_snack_time)
Calvora_Tagline_Interpretation = pd.get_dummies(df['Calvora_Tagline_Interpretation_Category'], prefix='Calvora', dtype=int)
df = pd.concat([df, Calvora_Tagline_Interpretation], axis=1)
df.drop(columns=['Calvora_Tagline_Interpretation'], inplace=True)


RULES = {
    'Delicious': ['อร่อย', 'รสชาติดี', 'ชอบ'],
    'Try': ['ลอง', 'ของเพื่อน', 'อยาก'],
    'Saltier': ['เค็ม'],
    'Unique': ['เอกลักษณ์'],
    'Flavor': ['รสชาติ'],
    'Brand': ['แบรนด์'],
    'Affordable': ['ราคาถูก'],
}

df['Why_Choose_Ebisen_Category'] = df['Why_Choose_Ebisen'].apply(categorize_snack_time)
Why_Choose_Ebisen = pd.get_dummies(df['Why_Choose_Ebisen_Category'], prefix='Why_Choose_Ebisen', dtype=int)
df = pd.concat([df, Why_Choose_Ebisen], axis=1)
df.drop(columns=['Why_Choose_Ebisen'], inplace=True)

RULES = {
    'Spicy': ['รสไปซี่', 'เผ็ด', 'สไปซี่', 'หมาล่า', 'คั่วพริกเกลือ'], 
    'Truffle': ['ทรัฟเฟิล'],
    'Cheese': ['ชีส'],
    'Laab': ['ลาบ', 'แซ่บๆ'],
    'Banana': ['กล้วย'],
    'BBQ': ['บาบีคิว'],
    'Corn': ['ข้าวโพด'],
    'Wasabi': ['วาซาบิ'], 
    'Nori': ['โนริ', 'สาหร่าย'],
    'Purple_Potato': ['มันม่วง'],
    'Seafood': ['ปู', 'กุ้งแช่น้ำปลา', 'ซีฟู๊ด', 'ต้มยำกุ้ง'], 
    'Mushroom': ['เห็ด'],
    'Veggie': ['ถั่วลันเตา', 'พริกหวาน'], 
    'Honey': ['รสน้ำผึ้ง']
}

df['Desired_New_Flavor_Category'] = df['Desired_New_Flavor'].apply(categorize_snack_time)
Desired_New_Flavor = pd.get_dummies(df['Desired_New_Flavor_Category'], prefix='Desired_New_Flavor', dtype=int)
df = pd.concat([df, Desired_New_Flavor], axis=1)
df.drop(columns=['Desired_New_Flavor'], inplace=True)

RULES = {
    'Not_Snack_Person': ['ไม่ค่อยทานขนม'], 
    'Never_Had_Chance': ['ไม่ค่อยได้ลอง', 'ไม่เคยเห็น', 'ยังไม่มีโอกาสซื้อ'],
    'Used_to_Old_Flavors': ['ชิน', 'เคยชิน', 'ซื้อตัวดั้งเดิม'],
    'Not_Attractive': ['ไม่ดึงดูด', 'ไม่ได้รู้สึกอยาก'],
}

df['Reason_Never_Tried_Category'] = df['Reason_Never_Tried'].apply(categorize_snack_time)
Reason_Never_Tried = pd.get_dummies(df['Reason_Never_Tried_Category'], prefix='Reason_Never_Tried', dtype=int)
df = pd.concat([df, Reason_Never_Tried], axis=1)
df.drop(columns=['Reason_Never_Tried'], inplace=True)

standard_brands = [
    'BBQ', 'ต้มยำ', 'รสกุ้ง', 'เค็ม', 'โนริ', 'เข้มข้น', 'หมาล่า', 'เผ็ด', 'ลาบ', 'ผงเยอะๆ', 'อร่อย',
]
df['Expected_Stronger_Flavor_Category'] = df['Expected_Stronger_Flavor'].apply(
    lambda x: clean_brand_name(x, standard_brands)
)
fig = {'BBQ': 'BBQ_Association', 'ต้มยำ': 'Tom_Yum_Association', 'รสกุ้ง': 'Shrimp_Flavor_Association', 'เค็ม': 'Saltier_Association', 'โนริ': 'Nori_Association', 'เข้มข้น': 'More_Intense_Association', 'หมาล่า': 'Mala_Association', 'เผ็ด': 'Spicier_Association', 'ลาบ': 'Laab_Association', 'ผงเยอะๆ': 'More_Seasoning_Association', 'อร่อย': 'Delicious_Association'}
df['Expected_Stronger_Flavor_Category'] = df['Expected_Stronger_Flavor_Category'].replace(fig)
Expected_Stronger_Flavor_Category = pd.get_dummies(df['Expected_Stronger_Flavor_Category'], prefix='Expected_Stronger_Flavor', dtype=int)
df = pd.concat([df, Expected_Stronger_Flavor_Category], axis=1)
df.drop(columns=['Expected_Stronger_Flavor_Category'], inplace=True)

RULES = {
    'Like_Spicy_Flavor': ['จัดจ้าน', 'แซ่บ' ,'ได้กลิ่นถึงวัดถุดิบมากขึ้น' ,'เข้มข้น' ,'ทานแล้วมีรสชาติในขณะที่ทาน' ,
                                     'ไม่ชอบกินจืด', 'ชอบ', 'รสจัด', 'เข้าถึงรสชาติ', 'ถึงใจ', 'ชอบรสชาติเยอะๆ', 'กินเพลิน'], 
    'Like_Saltier_Flavor': ['เค็ม'],
    'Like_Spicy_Flavor': ['เผ็ด'],
    'Like_Rich_Flavor': ['นัว'],
    'Like_Addictive_Flavor': ['ติดปาก'],
    'Like_Delicious_Flavor': ['อร่อย'],
}

df['Why_Like_Stronger_Flavor_Category'] = df['Why_Like_Stronger_Flavor'].apply(categorize_snack_time)
Why_Like_Stronger_Flavor = pd.get_dummies(df['Why_Like_Stronger_Flavor'], prefix='Why_Like_Stronger_Flavor', dtype=int)
df = pd.concat([df, Why_Like_Stronger_Flavor], axis=1)
df.drop(columns=['Why_Like_Stronger_Flavor'], inplace=True)

RULES = {
    'Not_Delicious': ['ไม่อร่อย', 'ชอบรสไม่จัด' ,'จะเค็มไป'], 
    'Might_Try_With_Friends': ['อาจจะอยากลอง'],
    'Already_Salty': ['เค็มอยู่แล้ว', 'ของเดิม', 'อร่อยอยู่แล้ว'],
    'Indifferent': ['เฉยๆ', 'ไม่คาดหวัง', 'งดทานขนมขบเคี้ยว'],
    'Already_Delicious': ['อร่อยอยู่แล้ว', 'รสชาตินี้ดีแล้ว'],
}

df['Reason_Not_Willing_Category'] = df['Reason_Not_Willing'].apply(categorize_snack_time)
Reason_Not_Willing = pd.get_dummies(df['Reason_Not_Willing_Category'], prefix='Reason_Not_Willing', dtype=int)
df = pd.concat([df, Reason_Not_Willing], axis=1)
df.drop(columns=['Reason_Not_Willing'], inplace=True)


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

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# ใช้ KNN Imputer หาคนที่มีลักษณะคล้ายกัน 5 คน (n_neighbors=5) มาช่วยทายค่า
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

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

