import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import warnings
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma'
# 1. โหลดข้อมูล
df = pd.read_csv('BU_Data_transformed.csv')

# 2. เลือกคอลัมน์การรับรู้แบรนด์ (Known_Snack_...)
known_cols = [col for col in df.columns if col.startswith('Known_Snack_')]

# 3. คำนวณร้อยละการรู้จักแบรนด์
awareness_rates = df[known_cols].mean() * 100
awareness_rates.index = [idx.replace('Known_Snack_', '') for idx in awareness_rates.index]
awareness_rates = awareness_rates.sort_values(ascending=False)

# 4. สร้างกราฟแท่งเปรียบเทียบ
plt.figure(figsize=(12, 6))
# ไฮไลท์สีชมพูที่แบรนด์แม่ (คาลโวร่า)
colors = ['#ff9999' if 'คาลโวร่า' in name else '#66b3ff' for name in awareness_rates.index]
sns.barplot(x=awareness_rates.values, y=awareness_rates.index, palette=colors)

plt.title('Brand Awareness Percentage (Calvora vs Sub-brands/Others)', fontsize=15, fontweight='bold')
plt.xlabel('Awareness Rate (%)', fontsize=12)
plt.ylabel('Brand Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# ใส่ตัวเลขเปอร์เซ็นต์ที่ปลายแท่ง
for i, v in enumerate(awareness_rates.values):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

import pandas as pd

df = pd.read_csv('BU_Data_transformed.csv')

# ดึงคอลัมน์ Known และ Tasted ของสินค้าในเครือ
sub_brands = ['คาลโวร่า', 'ฮานาโร', 'สแน็คแบ๊ค', 'แบ็กซ์', 'บิบิป๊อป', 'เอบินาริ', 'Jomona']

summary_list = []
for brand in sub_brands:
    known_val = df[f'Known_Snack_{brand}'].mean() * 100
    tasted_val = df[f'Tasted_Snack_{brand}'].mean() * 100
    # คำนวณ Conversion Ratio: รู้จักแล้วยอมลองชิมกี่ %
    conversion = (tasted_val / known_val * 100) if known_val > 0 else 0
    
    summary_list.append({
        'Brand': brand,
        'Awareness (%)': round(known_val, 1),
        'Trial Rate (%)': round(tasted_val, 1),
        'Conversion (%)': round(conversion, 1)
    })

summary_df = pd.DataFrame(summary_list).sort_values('Awareness (%)', ascending=False)
print(summary_df)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. เตรียมข้อมูล
df = pd.read_csv('BU_Data_transformed.csv')
known_cols = [col for col in df.columns if col.startswith('Known_Snack_')]
awareness = df[known_cols].mean() * 100
awareness.index = [idx.replace('Known_Snack_', '') for idx in awareness.index]
awareness = awareness.sort_values(ascending=False)

# 2. คำนวณค่าเพื่อดู Gap
tagline_awareness = df['Calvora_Tagline_Awareness'].mean() * 100
parent_awareness = awareness['คาลโวร่า']
top_3_subbrands_avg = awareness.head(4).drop('คาลโวร่า').mean()

# 3. สร้างกราฟช่องว่างการรับรู้ (Consumer Perception Gap)
gap_data = pd.DataFrame({
    'Metric': ['รู้จักชื่อแบรนด์ Calvora', 'รู้จักแบรนด์ลูก (เฉลี่ย Top 3)', 'จำสโลแกนบริษัทแม่ได้ (Tagline)'],
    'Value': [parent_awareness, top_3_subbrands_avg, tagline_awareness]
})

plt.figure(figsize=(10, 6))
colors = ['#E63946', '#457B9D', '#A8DADC'] # แดง (แม่), น้ำเงิน (ลูก), ฟ้าอ่อน (Tagline)
sns.barplot(x='Value', y='Metric', data=gap_data, palette=colors)

plt.title('ช่องว่างการรับรู้ (Consumer Perception Gap)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('เปอร์เซ็นต์การรับรู้ (%)', fontsize=12)
plt.xlim(0, 105)

# ใส่ตัวเลขกำกับ
for i, v in enumerate(gap_data['Value']):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()