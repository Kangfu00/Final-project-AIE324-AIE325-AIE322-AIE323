import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from math import pi
import warnings

warnings.filterwarnings('ignore') # ปิดแจ้งเตือนจุกจิก

# --- ตั้งค่าฟอนต์ภาษาไทย ---
plt.rcParams['font.family'] = 'Tahoma' # หรือ 'Arial Unicode MS' สำหรับ Mac

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df_raw = pd.read_csv("BU_Data_transformed.csv")
print(f"ข้อมูลตั้งต้น: {len(df_raw)} คน")

# ==========================================
# 2. Feature Selection
# ==========================================
behavioral_cols = [
    col for col in df.columns 
    if 'Purchase_Factor_' in col 
    or 'Strength_' in col 
    or 'Time_' in col 
    or 'Calvora_Natural_Ingredient_' in col
]

print(f"ดึงคอลัมน์พฤติกรรมมาทั้งหมด {len(behavioral_cols)} คอลัมน์")

X = df[behavioral_cols].copy()

# ---------------------------------------------------------
# 🌟 จุดที่แก้ไข: จัดการข้อมูลที่เป็นตัวหนังสือ (Categorical Data)
# ---------------------------------------------------------
# ใช้ get_dummies แปลงคอลัมน์ที่เป็นข้อความ (เช่น 'Watching_Media') ให้กลายเป็นเลข 0, 1 อัตโนมัติ
X_encoded = pd.get_dummies(X)

# เติมค่าว่างด้วย 0 (ต้องทำหลังแปลงข้อความเสร็จแล้ว)
X_encoded.fillna(0, inplace=True)

print(f"หลังจากแปลงข้อความเป็นตัวเลข ได้คอลัมน์รวมที่จะใช้สอน AI ทั้งหมด {X_encoded.shape[1]} คอลัมน์")

# สเกลข้อมูลให้เป็นมาตรฐาน (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ==========================================
# 3. สร้างกราฟ Elbow Method หาจำนวนกลุ่ม (K) ที่ดีที่สุด
# ==========================================
print("\nกำลังสร้างกราฟ Elbow Method...")
inertia = []
K_range = range(1, 11) 

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(K_range, inertia, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
plt.title('Elbow Method: ค้นหาจำนวนกลุ่มลูกค้าที่เหมาะสมที่สุด (Optimal K)', fontsize=16, fontweight='bold')
plt.xlabel('จำนวนกลุ่ม (Number of Clusters - K)', fontsize=12)
plt.ylabel('ค่า Inertia (Sum of Squared Distances)', fontsize=12)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show() # <--- ดูจุดหักศอกที่กราฟนี้

# ==========================================
# 4. เลือกรันโมเดลด้วย K ที่เหมาะสม 
# ==========================================
# 💡 หากดูกราฟ Elbow แล้วพบว่าหักศอกที่ 4 สามารถแก้เลข 3 เป็น 4 ได้เลยครับ
optimal_k = 3 

print(f"\n💡 รัน K-Means จริง ด้วยจำนวนกลุ่ม K = {optimal_k}")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster_ID'] = kmeans_final.fit_predict(X_scaled) + 1 

# ==========================================
# 5. การทำ Profiling ร่วมกับข้อมูลประชากรศาสตร์ 
# ==========================================
print("\n=== จำนวนคนในแต่ละกลุ่ม ===")
print(df['Cluster_ID'].value_counts().sort_index())

print("\n=== ข้อมูลประชากรศาสตร์ (Demographics) ของแต่ละกลุ่ม ===")
demo_cols = ['Age', 'Gender_ชาย', 'Gender_หญิง', 'Gender_LGBTQ+']
available_demo_cols = [col for col in demo_cols if col in df.columns]

if available_demo_cols:
    demographic_profiling = df.groupby('Cluster_ID')[available_demo_cols].mean().round(2)
    print(demographic_profiling)
else:
    print("ไม่พบคอลัมน์ประชากรศาสตร์ใน Dataset นี้")

# บันทึกผล
output_filename = f"BU_Data_{optimal_k}_Segments_Unsupervised.csv"
df.to_csv(output_filename, index=False)
print(f"\n✅ บันทึกไฟล์ {output_filename} เรียบร้อยแล้ว")

# ==========================================
# 6. สร้างกราฟเพื่อวิเคราะห์ผล (Visualization)
# ==========================================
print("\nกำลังสร้างกราฟผลลัพธ์...")

# ---------------------------------------------------------
# กราฟที่ 1: Bar Chart (เปรียบเทียบสัดส่วนเพศในแต่ละ Cluster)
# ---------------------------------------------------------
gender_cols = ['Gender_ชาย', 'Gender_หญิง', 'Gender_LGBTQ+']
available_gender = [col for col in gender_cols if col in df.columns]

if available_gender:
    cluster_gender = df.groupby('Cluster_ID')[available_gender].mean()
    # ใช้ ax เพื่อวาดกราฟใน figure ที่เรากำหนดขนาดไว้
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_gender.plot(kind='bar', colormap='Set2', ax=ax)
    plt.title('สัดส่วนเพศในแต่ละกลุ่ม (Cluster)', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('สัดส่วน (Proportion)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='เพศ', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# กราฟที่ 2: Scatter Plot (ลดมิติข้อมูลด้วย PCA)
# ---------------------------------------------------------
# เนื่องจากเรามีฟีเจอร์เยอะมาก เราจะใช้ PCA ยุบให้เหลือแค่แกน X, Y (2 มิติ) เพื่อให้พล็อตจุดได้
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) 
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID', palette='tab10', s=100, alpha=0.7)
plt.title('Scatter Plot แสดงการกระจายตัวของแต่ละกลุ่ม (PCA 2D)', fontsize=14)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend(title='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# กราฟที่ 3: Radar Chart (ดูพฤติกรรมเด่นของแต่ละ Cluster)
# ---------------------------------------------------------
# เลือกคอลัมน์พฤติกรรมมาสัก 5-6 ตัว เพื่อไม่ให้กราฟรกเกินไป (สมมติเลือกปัจจัยการซื้อ Purchase_Factor)
radar_cols = [col for col in behavioral_cols if 'Purchase_Factor_' in col][:6] 

if radar_cols:
    cluster_radar = df.groupby('Cluster_ID')[radar_cols].mean().reset_index()
    
    # จัดชื่อตัวแปรให้สั้นลงสำหรับแสดงบนกราฟ
    categories = [col.replace('Purchase_Factor_', '') for col in radar_cols]
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=10)
    
    # วาดเส้นของแต่ละ Cluster
    for i in range(len(cluster_radar)):
        values = cluster_radar.loc[i, radar_cols].values.flatten().tolist()
        values += values[:1]
        cluster_id = cluster_radar.loc[i, 'Cluster_ID']
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.1)
        
    plt.title('Radar Chart: ปัจจัยการซื้อเฉลี่ยของแต่ละ Cluster', fontsize=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()