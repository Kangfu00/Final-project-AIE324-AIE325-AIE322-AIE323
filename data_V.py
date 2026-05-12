import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma'

# =====================================================================
# กำหนดรายชื่อแบรนด์ที่ถูกต้องของบริษัท (แบรนด์แม่ 1 + แบรนด์ลูก 7)
# =====================================================================
allowed_brands = ['คาลโวร่า', 'แบ็กซ์', 'บิบิป๊อป', 'Jomona', 'เอบินาริ', 'Veggie Snap', 'ฟรูทร่า', 'เอบินาริ X']
allowed_known_cols = [f'Known_Snack_{b}' for b in allowed_brands]
allowed_corp_cols = ['Is_Calvora_Bax', 'Is_Calvora_Bibipop', 'Is_Calvora_Jomona', 'Is_Calvora_Ebinari', 'Is_Calvora_VeggieSnap', 'Is_Calvora_Frutra', 'Is_Calvora_Ebinari_X']


# =====================================================================
# กราฟที่ 1: Brand Awareness Percentage (Calvora vs 7 Sub-brands)
# =====================================================================
df = pd.read_csv('BU_Data_transformed.csv')

# เลือกคอลัมน์การรับรู้แบรนด์ เฉพาะ 7 แบรนด์ + แบรนด์แม่
known_cols = [col for col in df.columns if col in allowed_known_cols]

awareness_rates = df[known_cols].mean() * 100
awareness_rates.index = [idx.replace('Known_Snack_', '') for idx in awareness_rates.index]
awareness_rates = awareness_rates.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
colors = ['#ff9999' if 'คาลโวร่า' in name else '#66b3ff' for name in awareness_rates.index]
sns.barplot(x=awareness_rates.values, y=awareness_rates.index, palette=colors)

plt.title('Brand Awareness Percentage (Calvora vs 7 Sub-brands)', fontsize=15, fontweight='bold')
plt.xlabel('Awareness Rate (%)', fontsize=12)
plt.ylabel('Brand Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(awareness_rates.values):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('1_brand_awareness_7brands.png', dpi=300)
plt.show()


# =====================================================================
# ตารางที่ 2: สรุป Conversion Ratio (Known vs Tasted)
# =====================================================================
summary_list = []
for brand in allowed_brands:
    known_val = df[f'Known_Snack_{brand}'].mean() * 100
    tasted_val = df[f'Tasted_Snack_{brand}'].mean() * 100
    conversion = (tasted_val / known_val * 100) if known_val > 0 else 0
    
    summary_list.append({
        'Brand': brand,
        'Awareness (%)': round(known_val, 1),
        'Trial Rate (%)': round(tasted_val, 1),
        'Conversion (%)': round(conversion, 1)
    })

summary_df = pd.DataFrame(summary_list).sort_values('Awareness (%)', ascending=False)
print("\n--- ตารางสรุปการรู้จักและทดลองชิม ---")
print(summary_df)


# =====================================================================
# กราฟที่ 3: ช่องว่างการรับรู้ (Consumer Perception Gap)
# =====================================================================
tagline_awareness = df['Calvora_Tagline_Awareness'].mean() * 100
parent_awareness = awareness_rates['คาลโวร่า']
# คำนวณค่าเฉลี่ยของแบรนด์ลูก Top 3 จาก 7 แบรนด์ที่มี
top_3_subbrands_avg = awareness_rates.drop('คาลโวร่า').head(3).mean()

gap_data = pd.DataFrame({
    'Metric': ['รู้จักชื่อแบรนด์ Calvora', 'รู้จักแบรนด์ลูก (เฉลี่ย Top 3 ในเครือ)', 'จำสโลแกนบริษัทแม่ได้ (Tagline)'],
    'Value': [parent_awareness, top_3_subbrands_avg, tagline_awareness]
})

plt.figure(figsize=(10, 6))
colors_gap = ['#E63946', '#457B9D', '#A8DADC'] 
sns.barplot(x='Value', y='Metric', data=gap_data, palette=colors_gap)

plt.title('ช่องว่างการรับรู้ (Consumer Perception Gap)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('เปอร์เซ็นต์การรับรู้ (%)', fontsize=12)
plt.xlim(0, 105)

for i, v in enumerate(gap_data['Value']):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('2_perception_gap_7brands.png', dpi=300)
plt.show()


# =====================================================================
# กราฟที่ 4: ความเหลื่อมล้ำในการรับรู้ (Corporate Linkage Gap) 
# =====================================================================
# จับคู่เฉพาะ 7 แบรนด์
association_pairs = {
    'แบ็กซ์': ('Known_Snack_แบ็กซ์', 'Is_Calvora_Bax'),
    'บิบิป๊อป': ('Known_Snack_บิบิป๊อป', 'Is_Calvora_Bibipop'),
    'Jomona': ('Known_Snack_Jomona', 'Is_Calvora_Jomona'),
    'เอบินาริ': ('Known_Snack_เอบินาริ', 'Is_Calvora_Ebinari'),
    'Veggie Snap': ('Known_Snack_Veggie Snap', 'Is_Calvora_VeggieSnap'),
    'ฟรูทร่า': ('Known_Snack_ฟรูทร่า', 'Is_Calvora_Frutra'),
    'เอบินาริ X': ('Known_Snack_เอบินาริ X', 'Is_Calvora_Ebinari_X')
}

data_list = []
for brand_th, (known_col, is_cal_col) in association_pairs.items():
    if known_col in df.columns and is_cal_col in df.columns:
        product_awareness = df[known_col].mean() * 100
        corporate_link = df[is_cal_col].mean() * 100 
        
        data_list.append({'Brand': brand_th, 'Metric': 'รู้จักชื่อขนม (Product Brand)', 'Percentage': product_awareness})
        data_list.append({'Brand': brand_th, 'Metric': 'รู้ว่าเป็นของบริษัทแม่ Calvora', 'Percentage': corporate_link})

df_assoc = pd.DataFrame(data_list)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_assoc, x='Brand', y='Percentage', hue='Metric', palette=['#457B9D', '#E63946'])

plt.title('ความเหลื่อมล้ำในการรับรู้: แบรนด์สินค้า vs บริษัทแม่ (เฉพาะ 7 แบรนด์ในเครือ)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('ชื่อแบรนด์สินค้า', fontsize=12)
plt.ylabel('เปอร์เซ็นต์ (%)', fontsize=12)
plt.ylim(0, 105)
plt.legend(title='', fontsize=11, loc='upper right')

for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().text(p.get_x() + p.get_width()/2., height + 2, f'{height:.1f}%', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('3_corporate_linkage_7brands.png', dpi=300)
plt.show()


# =====================================================================
# กราฟที่ 5: Cluster vs Corporate Linkage (คิดคะแนนเฉพาะ 7 แบรนด์)
# =====================================================================
df_cluster = pd.read_csv('BU_Data_3_Segments_Final_Complete.csv')

# คำนวณคะแนน Corporate Linkage Score โดยใช้เฉพาะ 7 คอลัมน์ที่ได้รับอนุญาตเท่านั้น!
df_cluster['Corporate_Linkage_Score'] = df_cluster[allowed_corp_cols].mean(axis=1) * 100

cluster_linkage = df_cluster.groupby('Cluster_ID')['Corporate_Linkage_Score'].mean().reset_index()

cluster_names = {
    1: 'Cluster 1: The Mass Market\n(กลุ่มเน้นซื้อตามความคุ้นเคย)', 
    2: 'Cluster 2: Quality & Trust Seekers\n(กลุ่มพรีเมียม สายคุณภาพ)', 
    3: 'Cluster 3: Young Flavor Explorers\n(กลุ่มสายลองรสชาติใหม่)'
}
cluster_linkage['Cluster_Name'] = cluster_linkage['Cluster_ID'].map(cluster_names)

plt.figure(figsize=(10, 6))
colors_cluster = ['#A8DADC', '#E63946', '#457B9D'] 
sns.barplot(data=cluster_linkage, x='Cluster_Name', y='Corporate_Linkage_Score', palette=colors_cluster)

plt.title('ความสามารถในการเชื่อมโยงแบรนด์แม่ แยกตามกลุ่มลูกค้า (เฉพาะ 7 แบรนด์ในเครือ)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('กลุ่มลูกค้า (Customer Segments)', fontsize=12)
plt.ylabel('อัตราการรู้ว่าสินค้าเป็นของ Calvora (%)', fontsize=12)
plt.ylim(0, 100)

for i, v in enumerate(cluster_linkage['Corporate_Linkage_Score']):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('4_cluster_linkage_7brands.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

# ==========================================
# สร้างกราฟเปรียบเทียบ CV F1-Macro (ข้อมูลจริง)
# ==========================================
# กำหนดข้อมูลจากผลรันจริงของคุณ
models = ['Logistic Reg.', 'Decision Tree', 'Random Forest']
# เรียงลำดับจากมากไปน้อยเพื่อความสวยงาม
f1_scores = [0.6526, 0.5326, 0.4880]
std_dev = [0.0811, 0.1005, 0.0901]

# กำหนดสี: ผู้ชนะ (Logistic Reg.) เป็นสีเขียวอมฟ้า ที่เหลือเป็นสีเทา
colors = ['#16a085', '#95a5a6', '#95a5a6']

fig, ax = plt.subplots(figsize=(7, 5))

# วาดกราฟแท่ง พร้อมใส่ yerr สำหรับค่า std
bars = ax.bar(models, f1_scores, color=colors, yerr=std_dev, capsize=5, width=0.5)

# ปรับขอบเขตแกน Y 
ax.set_ylim(0.3, 0.8)

# ใส่ตัวเลขค่า F1-Macro ไว้บนแท่งกราฟ
for bar in bars:
    yval = bar.get_height()
    # ปรับตำแหน่งตัวเลขให้ขยับขึ้นไปเหนือ error bar เล็กน้อย
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.11, f'{yval:.4f}', 
            ha='center', va='bottom', fontsize=11, color='#333333')

# ตกแต่งกราฟ
ax.set_title('CV F1-Macro (±std) แยกตามโมเดล\n(ผู้ชนะ: Logistic Regression)', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='-', alpha=0.3)

# ลบขอบกราฟด้านบนและด้านขวาเพื่อให้ดูคลีน
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

plt.tight_layout()

# บันทึกไฟล์ภาพ
plt.savefig('model_comparison_real_data.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nสร้างกราฟข้อมูลจริงเสร็จสมบูรณ์! บันทึกไฟล์ชื่อ model_comparison_real_data.png")