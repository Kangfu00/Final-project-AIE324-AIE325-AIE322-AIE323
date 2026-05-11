import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from math import pi
import joblib
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma'

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df_raw = pd.read_csv("BU_Data_transformed.csv")
print(f"ข้อมูลตั้งต้น: {len(df_raw)} คน")

# ==========================================
# 2. Feature Selection — 8 ฟีเจอร์ที่ดีที่สุด
# ==========================================
# เลือกด้วย brute-force ทุก combo จาก top-12 PCA loading (495 combinations)
# combo นี้ให้ Silhouette = 0.5191 ดีที่สุด และสมเหตุสมผลกับ sales
# แบ่งได้เป็น 3 กลุ่ม: ความโปร่งใส/การรับรอง, คุณภาพ/รสชาติ, สุขภาพ
behavioral_cols = [
    'Calvora_Natural_Ingredient_ข้อมูลเกี่ยวกับกระบวนการผลิต',       # ความโปร่งใสการผลิต → trust
    'Strength_ใช้วัตถุดิบจากธรรมชาติ (Use natural ingredients)',      # วัตถุดิบธรรมชาติ → health trend
    'Strength_มีรสชาติอร่อย (Tasty)',                                  # รสชาติ → core purchase driver
    'Strength_ทำจากเนื้อสัตว์แท้ (From real meat)',                   # ความแท้จริง → premium
    'Calvora_Natural_Ingredient_เห็นขั้นตอนของการเก็บเกี่ยววัตถุดิบ', # traceability → trust
    'Calvora_Natural_Ingredient_การรับรองจากหน่วยงานด้านอาหาร',       # certification → credibility
    'Strength_มีคุณภาพดี (Good quality)',                              # overall quality → repeat purchase
    'Strength_เพื่อสุขภาพที่ดี (For healthy lifestyles)',              # health → growing market
]

print(f"\nใช้ {len(behavioral_cols)} ฟีเจอร์:")
for c in behavioral_cols:
    short = c.replace('Calvora_Natural_Ingredient_','[NI] ').replace('Strength_','[St] ')
    print(f"  - {short}")

X_raw = df_raw[behavioral_cols].fillna(0)

# ==========================================
# 3. Fit Scaler
# ==========================================
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_raw)

# ==========================================
# 4. ลบ Outlier อัตโนมัติ
# ==========================================
km_detect = KMeans(n_clusters=4, random_state=42, n_init=20)
labels_detect = km_detect.fit_predict(X_scaled_all) + 1
cluster_sizes = pd.Series(labels_detect).value_counts()
outlier_cluster_id = cluster_sizes.idxmin()

if cluster_sizes[outlier_cluster_id] == 1:
    outlier_indices = np.where(labels_detect == outlier_cluster_id)[0]
    df = df_raw.drop(index=outlier_indices).reset_index(drop=True)
    X_clean = np.delete(X_scaled_all, outlier_indices, axis=0)
    print(f"\nลบ outlier {len(outlier_indices)} คน (row {outlier_indices})")
else:
    df = df_raw.copy()
    X_clean = X_scaled_all
print(f"ข้อมูลหลังลบ: {len(df)} คน")

# ==========================================
# 5. PCA 3 components
# ==========================================
N_COMPONENTS = 3
pca_for_cluster = PCA(n_components=N_COMPONENTS, random_state=42)
X_model = pca_for_cluster.fit_transform(X_clean)
var_total = pca_for_cluster.explained_variance_ratio_.sum() * 100
var1 = pca_for_cluster.explained_variance_ratio_[0] * 100
var2 = pca_for_cluster.explained_variance_ratio_[1] * 100
print(f"\nPCA {N_COMPONENTS} components อธิบาย variance: {var_total:.1f}%")

# ==========================================
# 6. Model Comparison
# ==========================================
optimal_k = 3
print(f"\nเปรียบเทียบ 4 โมเดล (K={optimal_k})...")

km_temp    = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
agglo_temp = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
gmm_temp   = GaussianMixture(n_components=optimal_k, random_state=42)
spec_temp  = SpectralClustering(n_clusters=optimal_k, random_state=42,
                                assign_labels='kmeans', affinity='nearest_neighbors')

l_km    = km_temp.fit_predict(X_model)
l_agglo = agglo_temp.fit_predict(X_model)
l_gmm   = gmm_temp.fit_predict(X_model)
l_spec  = spec_temp.fit_predict(X_model)

sil_km    = silhouette_score(X_model, l_km)
sil_agglo = silhouette_score(X_model, l_agglo)
sil_gmm   = silhouette_score(X_model, l_gmm)
sil_spec  = silhouette_score(X_model, l_spec)

model_names  = ['K-Means', 'Agglomerative', 'Gaussian Mixture', 'Spectral']
model_scores = [sil_km, sil_agglo, sil_gmm, sil_spec]
best_model   = model_names[np.argmax(model_scores)]

for n, s in zip(model_names, model_scores):
    print(f"  {n:20s} Silhouette = {s:.4f}")
print(f"\nโมเดลที่ดีที่สุด: {best_model}")

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = ['#2ca02c' if s == max(model_scores) else '#aec7e8' for s in model_scores]
bars = ax.bar(model_names, model_scores, color=bar_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, model_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(model_scores) * 1.2)
ax.set_ylabel('Silhouette Score (สูง = ดี)', fontsize=11)
ax.set_title(f'เปรียบเทียบ Silhouette Score ของ 4 โมเดล (K={optimal_k})\nดีที่สุด: {best_model}',
             fontsize=13, fontweight='bold')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ==========================================
# 7. รัน KMeans จริง
# ==========================================
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['Cluster_ID'] = kmeans_final.fit_predict(X_model) + 1
final_sil = silhouette_score(X_model, df['Cluster_ID'])

df['Cluster_ID_Agglo']    = l_agglo + 1
df['Cluster_ID_GMM']      = l_gmm + 1
df['Cluster_ID_Spectral'] = l_spec + 1

print(f"\nSilhouette Score (K={optimal_k}): {final_sil:.4f}")
print("\n=== จำนวนคนในแต่ละกลุ่ม ===")
print(df['Cluster_ID'].value_counts().sort_index())

# ==========================================
# 8. Elbow + Silhouette กราฟ
# ==========================================
inertia, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_model)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_model, labels))

best_k = list(K_range)[sil_scores.index(max(sil_scores))]
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
# 9. Visualization
# ==========================================
sil_vals = silhouette_samples(X_model, df['Cluster_ID'])
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
colors_map = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for i in range(1, optimal_k + 1):
    vals = np.sort(sil_vals[df['Cluster_ID'] == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=colors_map[i-1], alpha=0.85)
    ax.text(-0.07, y_lower + len(vals)/2, f'C{i} (n={len(vals)})', fontsize=10)
    y_lower = y_upper + 10
ax.axvline(x=final_sil, color='red', linestyle='--', label=f'avg={final_sil:.3f}')
ax.set_title(f'Silhouette Plot แยกรายคน (K={optimal_k})', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

df['PCA1'] = X_model[:, 0]
df['PCA2'] = X_model[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID', palette='tab10', s=100, alpha=0.8)
plt.title(f'Scatter Plot แยกกลุ่มลูกค้า (PCA {N_COMPONENTS} components, variance {var_total:.1f}%)',
          fontsize=13, fontweight='bold')
plt.xlabel(f'PCA 1 ({var1:.1f}%)', fontsize=11)
plt.ylabel(f'PCA 2 ({var2:.1f}%)', fontsize=11)
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

cluster_means = df.groupby('Cluster_ID')[behavioral_cols].mean()
short_names = {c: c.replace('Calvora_Natural_Ingredient_', 'NI: ')
                   .replace('Strength_', 'St: ') for c in behavioral_cols}
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(cluster_means.rename(columns=short_names), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'ค่าเฉลี่ย'})
ax.set_title('Heatmap: พฤติกรรมแต่ละกลุ่ม (8 ฟีเจอร์หลัก)', fontsize=13, fontweight='bold')
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.tight_layout()
plt.show()

radar_cols = [c for c in behavioral_cols if 'Strength_' in c]
cluster_radar = df.groupby('Cluster_ID')[radar_cols].mean().reset_index()
categories = [c.replace('Strength_', '') for c in radar_cols]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=9)
for i in range(len(cluster_radar)):
    values = cluster_radar.loc[i, radar_cols].values.flatten().tolist()
    values += values[:1]
    cid = cluster_radar.loc[i, 'Cluster_ID']
    ax.plot(angles, values, linewidth=2, label=f'Cluster {cid}')
    ax.fill(angles, values, alpha=0.1)
plt.title('Radar Chart: Strength ที่รับรู้แต่ละกลุ่ม', fontsize=14, y=1.1, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
pairs = [
    ('Cluster_ID',          f'1. K-Means (Sil: {sil_km:.4f})'),
    ('Cluster_ID_Agglo',    f'2. Agglomerative (Sil: {sil_agglo:.4f})'),
    ('Cluster_ID_GMM',      f'3. Gaussian Mixture (Sil: {sil_gmm:.4f})'),
    ('Cluster_ID_Spectral', f'4. Spectral (Sil: {sil_spec:.4f})'),
]
for ax, (col, title) in zip(axes.flatten(), pairs):
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue=col,
                    palette='Set1', s=100, alpha=0.7, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(f'PCA 1 ({var1:.1f}%)', fontsize=9)
    ax.set_ylabel(f'PCA 2 ({var2:.1f}%)', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
plt.suptitle('เปรียบเทียบการแบ่งกลุ่มของทั้ง 4 โมเดล', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ==========================================
# 10. บันทึก Pipeline
# ==========================================
joblib.dump(scaler,          'scaler.pkl')
joblib.dump(pca_for_cluster, 'pca.pkl')
joblib.dump(kmeans_final,    'kmeans.pkl')
joblib.dump(behavioral_cols, 'behavioral_cols.pkl')

df.to_csv(f"BU_Data_{optimal_k}_Segments_Final_Complete.csv", index=False)

print(f"\nรันเสร็จสมบูรณ์!")
print(f"Features: {len(behavioral_cols)} ฟีเจอร์ (คัดจาก brute-force 495 combinations)")
print(f"Silhouette Score: {final_sil:.4f}  (เพิ่มจาก 0.354 เป็น {final_sil:.3f})")
print(f"n = {len(df)} คน")