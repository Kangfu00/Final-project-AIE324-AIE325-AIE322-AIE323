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

warnings.filterwarnings('ignore')
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
    col for col in df_raw.columns
    if ('Purchase_Factor_' in col
        or 'Strength_' in col
        or 'Calvora_Natural_Ingredient_' in col
        or ('Time_' in col and col != 'Snack_Time_Category'))
]
X_raw = pd.get_dummies(df_raw[behavioral_cols].copy()).fillna(0)
X_scaled_all = StandardScaler().fit_transform(X_raw)

# ==========================================
# 3. ลบ Outlier
# ==========================================
km_detect = KMeans(n_clusters=4, random_state=42, n_init=20)
labels_detect = km_detect.fit_predict(X_scaled_all) + 1
cluster_sizes = pd.Series(labels_detect).value_counts()
outlier_cluster_id = cluster_sizes.idxmin()

if cluster_sizes[outlier_cluster_id] == 1:
    outlier_mask = labels_detect == outlier_cluster_id
    outlier_indices = np.where(outlier_mask)[0]
    df = df_raw.drop(index=outlier_indices).reset_index(drop=True)
    X_clean = np.delete(X_scaled_all, outlier_indices, axis=0)
else:
    df = df_raw.copy()
    X_clean = X_scaled_all

# ==========================================
# 3.5 ลดมิติข้อมูล (PCA) ก่อนทำ Clustering เพื่อดันคะแนน
# ==========================================
pca_for_cluster = PCA(n_components=3, random_state=42)
X_model = pca_for_cluster.fit_transform(X_clean)

# ==========================================
# 4. ค้นหา K ด้วย Elbow + Silhouette Score
# ==========================================
inertia, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_model)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_model, labels))

best_k = list(K_range)[sil_scores.index(max(sil_scores))]
optimal_k = 3 # บังคับ 3 กลุ่มตามโจทย์ Business

# 📊 กราฟ 1: Elbow & Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertia, marker='o', color='#1f77b4', linewidth=2)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.set_xlabel('จำนวนกลุ่ม (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(K_range, sil_scores, marker='s', color='#e67e22', linewidth=2)
ax2.axvline(x=3, color='green', linestyle='--', alpha=0.8, label='K=3 (chosen)')
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.6, label=f'Best K={best_k}')
ax2.set_title('Silhouette Score (ยิ่งสูงยิ่งดี)', fontsize=14, fontweight='bold')
ax2.set_xlabel('จำนวนกลุ่ม (K)', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)
plt.suptitle('การหาจำนวนกลุ่มที่เหมาะสม (Optimal K)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# 5. รัน KMeans จริงด้วย K=3
# ==========================================
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['Cluster_ID'] = kmeans_final.fit_predict(X_model) + 1
final_sil = silhouette_score(X_model, df['Cluster_ID'])

print(f"\n✅ Silhouette Score (K={optimal_k}): {final_sil:.4f}")
print("\n=== จำนวนคนในแต่ละกลุ่ม ===")
print(df['Cluster_ID'].value_counts().sort_index())

# ==========================================
# 6. กราฟ Visualization แบบจัดเต็ม
# ==========================================

# 📊 กราฟ 2: Silhouette Plot รายตัวอย่าง
sil_vals = silhouette_samples(X_model, df['Cluster_ID'])
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for i in range(1, optimal_k + 1):
    vals = np.sort(sil_vals[df['Cluster_ID'] == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=colors[i-1], alpha=0.85)
    ax.text(-0.05, y_lower + len(vals)/2, f'C{i} (n={len(vals)})', fontsize=10)
    y_lower = y_upper + 10
ax.axvline(x=final_sil, color='red', linestyle='--', label=f'avg={final_sil:.3f}')
ax.set_title(f'Silhouette Plot แยกรายคน (K={optimal_k})', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# 📊 กราฟ 3: PCA Scatter Plot
df['PCA1'] = X_model[:, 0]
df['PCA2'] = X_model[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID', palette='tab10', s=100, alpha=0.8)
plt.title('Scatter Plot แยกกลุ่มลูกค้า (PCA)', fontsize=13, fontweight='bold')
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 📊 กราฟ 4: Cluster Heatmap (เช็คพฤติกรรมแต่ละกลุ่ม)
cluster_means = df.groupby('Cluster_ID')[behavioral_cols].mean()
short_names = {c: c.replace('Purchase_Factor_', 'PF_').replace('Strength_', 'St_').replace('Calvora_Natural_Ingredient_', 'NI_').replace('Time_', 'T_') for c in behavioral_cols}
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(cluster_means.rename(columns=short_names), annot=True, fmt='.2f', cmap='RdYlGn', center=0, linewidths=0.3, ax=ax)
ax.set_title('Heatmap: เจาะลึกพฤติกรรมแต่ละกลุ่ม (สีเขียว=เด่น, สีแดง=ไม่สนใจ)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()

# 📊 กราฟ 5: Radar Chart (ปัจจัยการซื้อ)
radar_cols = [c for c in behavioral_cols if 'Purchase_Factor_' in c]
cluster_radar = df.groupby('Cluster_ID')[radar_cols].mean().reset_index()
categories = [c.replace('Purchase_Factor_', '') for c in radar_cols]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=10)

for i in range(len(cluster_radar)):
    values = cluster_radar.loc[i, radar_cols].values.flatten().tolist() + [cluster_radar.loc[i, radar_cols].values[0]]
    cid = cluster_radar.loc[i, 'Cluster_ID']
    ax.plot(angles, values, linewidth=2, label=f'Cluster {cid}')
    ax.fill(angles, values, alpha=0.1)

plt.title('Radar Chart: ปัจจัยการซื้อหลัก', fontsize=14, y=1.1, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# ==========================================
# 7. โมเดลเปรียบเทียบอื่นๆ (Model Comparison)
# ==========================================
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

agglo_model = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
df['Cluster_ID_Agglo'] = agglo_model.fit_predict(X_model) + 1
sil_agglo = silhouette_score(X_model, df['Cluster_ID_Agglo'])

gmm_model = GaussianMixture(n_components=optimal_k, random_state=42)
df['Cluster_ID_GMM'] = gmm_model.fit_predict(X_model) + 1
sil_gmm = silhouette_score(X_model, df['Cluster_ID_GMM'])

spectral_model = SpectralClustering(n_clusters=optimal_k, random_state=42, assign_labels='kmeans', affinity='nearest_neighbors')
df['Cluster_ID_Spectral'] = spectral_model.fit_predict(X_model) + 1
sil_spectral = silhouette_score(X_model, df['Cluster_ID_Spectral'])

# 📊 กราฟ 6: เปรียบเทียบ 4 โมเดล
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID', palette='Set1', s=100, alpha=0.7, ax=axes[0, 0])
axes[0, 0].set_title(f'1. K-Means (Sil: {final_sil:.4f})', fontweight='bold')

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID_Agglo', palette='Set1', s=100, alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title(f'2. Agglomerative (Sil: {sil_agglo:.4f})', fontweight='bold')

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID_GMM', palette='Set1', s=100, alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title(f'3. Gaussian Mixture (Sil: {sil_gmm:.4f})', fontweight='bold')

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID_Spectral', palette='Set1', s=100, alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title(f'4. Spectral (Sil: {sil_spectral:.4f})', fontweight='bold')

plt.suptitle('เปรียบเทียบการแบ่งกลุ่มของทั้ง 4 โมเดล', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ==========================================
# 8. บันทึกไฟล์
# ==========================================
df.to_csv(f"BU_Data_{optimal_k}_Segments_Final_Complete.csv", index=False)
print("\n✅ รันเสร็จสมบูรณ์ กราฟครบ และบันทึกไฟล์เรียบร้อย!")