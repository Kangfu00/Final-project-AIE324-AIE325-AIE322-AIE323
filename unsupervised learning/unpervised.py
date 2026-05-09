import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ (สำคัญมาก ไม่งั้นตัวอักษรจะเป็นสี่เหลี่ยม) ---
# หากคุณใช้ Windows แนะนำ 'Tahoma' หรือ 'Cordia New'
# หากคุณใช้ Mac ให้เปลี่ยนเป็น 'Arial Unicode MS' หรือ 'Thonburi'
plt.rcParams['font.family'] = 'Tahoma' 

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df = pd.read_csv("BU_Data_transformed.csv")

# ==========================================
# 2. เลือกคอลัมน์ (Feature Selection) ให้ตรงกับสมมติฐาน 3 กลุ่ม
# ==========================================
target_cols = [
    # แกนที่ 1: สำหรับจับกลุ่ม "The Perfectionists" (สายพรีเมียม)
    'Purchase_Factor_Quality_Ingredients', 
    'Purchase_Factor_Tasty', 
    'Purchase_Factor_Crispy',
    'Strength_มีคุณภาพดี (Good quality)', 
    'Strength_เป็นแบรนด์ญี่ปุ่น (Japan brand)',
    
    # แกนที่ 2: สำหรับจับกลุ่ม "Binge-Watchers" (สายเคี้ยวหน้าจอ)
    'Time_Watching_Media', 
    'Time_Playing_Games', 
    'Time_Free_Time',
    
    # แกนที่ 3: สำหรับจับกลุ่ม "The Hungry Workers" (สายหิวตอนทำงาน)
    'Time_Working_Studying', 
    'Time_Hungry', 
    'Time_Late_Night'
]

# ดึงเฉพาะคอลัมน์ที่เลือกมาใช้งาน
X = df[target_cols].copy()

# เติมค่าว่างด้วย 0 (ถ้ามี)
X.fillna(0, inplace=True)

# ==========================================
# 3. ปรับสเกลข้อมูลให้อยู่ในมาตรฐานเดียวกัน
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. รัน K-Means จัดเป็น 3 กลุ่มตามที่เราวางแผนไว้
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42)

# ทำนายและบวก 1 เพื่อให้กลุ่มออกมาเป็น 1, 2, 3 (ปกติ AI จะเริ่มจาก 0)
df['Cluster_ID'] = kmeans.fit_predict(X_scaled) + 1 

# ==========================================
# 5. สรุปผลลัพธ์และตีความ (Cluster Profiling)
# ==========================================
print("=== จำนวนลูกค้าในแต่ละกลุ่ม ===")
print(df['Cluster_ID'].value_counts().sort_index())
print("\n=== ค่าเฉลี่ยพฤติกรรม (เพื่อใช้ตั้งชื่อกลุ่ม) ===")

# จัดกลุ่มและหาค่าเฉลี่ยของแต่ละคอลัมน์ เพื่อเช็คว่าพฤติกรรมตรงกับกลุ่มไหน
cluster_profiling = df.groupby('Cluster_ID')[target_cols].mean().round(2)

# .T (Transpose) เป็นการกลับตารางให้คอลัมน์มาอยู่ด้านซ้าย จะได้อ่านง่ายขึ้น
print(cluster_profiling.T)

# ==========================================
# 6. บันทึกผลลัพธ์ไปใช้งานต่อ
# ==========================================
output_filename = "BU_Data_3_Segments.csv"
df.to_csv(output_filename, index=False)
print(f"\n✅ เสร็จสิ้น! บันทึกไฟล์ '{output_filename}' ที่มีคอลัมน์ Cluster_ID เรียบร้อยแล้ว")

# ==========================================
# 7. สร้างกราฟ (Visualizations)
# ==========================================
print("\nกำลังสร้างกราฟ กรุณารอสักครู่...")

# --- กราฟที่ 1: Cluster Profile Bar Chart ---
cluster_profiling.T.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('พฤติกรรมและปัจจัยการเลือกซื้อของลูกค้าทั้ง 3 กลุ่ม', fontsize=16, fontweight='bold')
plt.xlabel('ตัวแปร (Features)', fontsize=12)
plt.ylabel('ค่าเฉลี่ย (Mean Value)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show() # หน้าต่างกราฟแรกจะเด้งขึ้นมา (ต้องปิดหน้าต่างนี้ก่อน กราฟที่สองถึงจะขึ้น)

# --- กราฟที่ 2: PCA Scatter Plot ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
df_pca['Cluster_ID'] = df['Cluster_ID']

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PCA_Component_1', 
    y='PCA_Component_2', 
    hue='Cluster_ID', 
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'], 
    data=df_pca, 
    s=100, 
    alpha=0.7
)

centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_pca[:, 0], 
    centroids_pca[:, 1], 
    marker='X', 
    s=250, 
    c='red', 
    label='Centroids (จุดศูนย์กลาง)',
    edgecolor='black'
)

plt.title('การกระจายตัวของกลุ่มลูกค้า 3 กลุ่ม (PCA Scatter Plot)', fontsize=16, fontweight='bold')
plt.xlabel('พฤติกรรมแกนที่ 1 (PCA 1)', fontsize=12)
plt.ylabel('พฤติกรรมแกนที่ 2 (PCA 2)', fontsize=12)
plt.legend(title='กลุ่ม (Cluster ID)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() # หน้าต่างกราฟที่สองจะเด้งขึ้นมา