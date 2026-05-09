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
plt.rcParams['font.family'] = 'Tahoma'

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
print(f"Feature ที่ใช้: {len(behavioral_cols)} คอลัมน์")

X_raw = pd.get_dummies(df_raw[behavioral_cols].copy()).fillna(0)
X_scaled_all = StandardScaler().fit_transform(X_raw)

# ==========================================
# 3. ลบ Outlier (index 73 — กลุ่ม C3 ขนาด 1 คน จาก K=4 รอบแรก)
# ==========================================
# หา outlier ด้วย K=4 ก่อน แล้วตัดกลุ่มที่มีสมาชิกคนเดียวออก
km_detect = KMeans(n_clusters=4, random_state=42, n_init=20)
labels_detect = km_detect.fit_predict(X_scaled_all) + 1
cluster_sizes = pd.Series(labels_detect).value_counts()
outlier_cluster_id = cluster_sizes.idxmin()

if cluster_sizes[outlier_cluster_id] == 1:
    outlier_mask = labels_detect == outlier_cluster_id
    outlier_indices = np.where(outlier_mask)[0]
    df = df_raw.drop(index=outlier_indices).reset_index(drop=True)
    X_clean = np.delete(X_scaled_all, outlier_indices, axis=0)
    print(f"\n✅ ลบ outlier {len(outlier_indices)} คน (row {outlier_indices})")
    print(f"ข้อมูลหลังลบ: {len(df)} คน")
else:
    df = df_raw.copy()
    X_clean = X_scaled_all
    print("\nไม่พบ outlier ที่ชัดเจน")

# ==========================================
# 4. เลือก K ด้วย Elbow + Silhouette
# ==========================================
print("\nกำลังหา optimal K...")
inertia, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_clean)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clean, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertia, marker='o', color='#1f77b4', linewidth=2, markersize=8)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.set_xlabel('จำนวนกลุ่ม (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_xticks(list(K_range))
ax1.grid(True, linestyle='--', alpha=0.5)

best_k = list(K_range)[sil_scores.index(max(sil_scores))]
ax2.plot(K_range, sil_scores, marker='s', color='#e67e22', linewidth=2, markersize=8)
ax2.axvline(x=3, color='green', linestyle='--', alpha=0.8, label='K=3 (chosen)')
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.6, label=f'Best silhouette K={best_k}')
ax2.set_title('Silhouette Score (สูง = ดี)', fontsize=14, fontweight='bold')
ax2.set_xlabel('จำนวนกลุ่ม (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_xticks(list(K_range))
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('เลือก K ที่ดีที่สุดด้วย Elbow + Silhouette Score\n(หลังลบ Outlier)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nSilhouette Scores หลังลบ outlier:")
for k, s in zip(K_range, sil_scores):
    note = " ← best silhouette" if k == best_k else (" ← chosen (business)" if k == 3 else "")
    print(f"  K={k}: {s:.4f}{note}")

# ==========================================
# 5. รัน KMeans ด้วย K=3
# ==========================================
optimal_k = 3
print(f"\n💡 รัน K-Means จริง ด้วย K={optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['Cluster_ID'] = kmeans_final.fit_predict(X_clean) + 1

final_sil = silhouette_score(X_clean, df['Cluster_ID'])
print(f"✅ Silhouette Score (K={optimal_k}): {final_sil:.4f}")
print("\nจำนวนคนในแต่ละกลุ่ม:")
print(df['Cluster_ID'].value_counts().sort_index())

# ==========================================
# 6. Demographics Profiling
# ==========================================
print("\n=== Demographics ===")
demo_cols = ['Age', 'Gender_ชาย', 'Gender_หญิง', 'Gender_LGBTQ+']
avail_demo = [c for c in demo_cols if c in df.columns]
if avail_demo:
    print(df.groupby('Cluster_ID')[avail_demo].mean().round(2))

# ==========================================
# 7. Visualization
# ==========================================

# --- กราฟ 1: Silhouette Plot รายตัวอย่าง ---
sil_vals = silhouette_samples(X_clean, df['Cluster_ID'])
fig, ax = plt.subplots(figsize=(10, 5))
y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

for i in range(1, optimal_k + 1):
    vals = np.sort(sil_vals[df['Cluster_ID'] == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=colors[i-1], edgecolor='none', alpha=0.85)
    ax.text(-0.05, y_lower + len(vals)/2, f'C{i} (n={len(vals)})', fontsize=10)
    y_lower = y_upper + 10

ax.axvline(x=final_sil, color='red', linestyle='--', linewidth=1.5, label=f'avg={final_sil:.3f}')
ax.set_xlabel('Silhouette Coefficient', fontsize=12)
ax.set_title(f'Silhouette Plot รายตัวอย่าง (K={optimal_k}, หลังลบ Outlier)\nคนที่ค่า < 0 = อยู่ผิดกลุ่ม', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- กราฟ 2: PCA Scatter Plot ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
explained = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_ID', palette='tab10', s=100, alpha=0.8)
plt.title(f'PCA Scatter Plot (อธิบาย variance ได้ {explained[0]:.1f}% + {explained[1]:.1f}% = {sum(explained):.1f}%)\nหลังลบ Outlier', fontsize=13, fontweight='bold')
plt.xlabel(f'PCA 1 ({explained[0]:.1f}%)', fontsize=11)
plt.ylabel(f'PCA 2 ({explained[1]:.1f}%)', fontsize=11)
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- กราฟ 3: Cluster Heatmap ---
cluster_means = df.groupby('Cluster_ID')[behavioral_cols].mean()
short_names = {c: c.replace('Purchase_Factor_', 'PF_')
                   .replace('Strength_', 'St_')
                   .replace('Calvora_Natural_Ingredient_', 'NI_')
                   .replace('Time_', 'T_') for c in behavioral_cols}

fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(cluster_means.rename(columns=short_names),
            annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.3, ax=ax, cbar_kws={'label': 'ค่าเฉลี่ย'})
ax.set_title('Heatmap: ค่าเฉลี่ยทุก Feature ต่อ Cluster (K=3, หลังลบ Outlier)\n(สีเขียว = สูง, สีแดง = ต่ำ)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cluster ID', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

# --- กราฟ 4: Radar Chart ---
radar_cols = [c for c in behavioral_cols if 'Purchase_Factor_' in c]
if radar_cols:
    cluster_radar = df.groupby('Cluster_ID')[radar_cols].mean().reset_index()
    categories = [c.replace('Purchase_Factor_', '') for c in radar_cols]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    angles_labels = angles[:-1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles_labels, categories, fontsize=10)

    for i in range(len(cluster_radar)):
        values = cluster_radar.loc[i, radar_cols].values.flatten().tolist() + \
                 [cluster_radar.loc[i, radar_cols].values[0]]
        cid = cluster_radar.loc[i, 'Cluster_ID']
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cid}')
        ax.fill(angles, values, alpha=0.1)

    plt.title('Radar: ปัจจัยการซื้อแต่ละ Cluster (K=3)', fontsize=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. บันทึกผล
# ==========================================
df.to_csv(f"BU_Data_{optimal_k}_Segments_Final.csv", index=False)
print(f"\n✅ บันทึก BU_Data_{optimal_k}_Segments_Final.csv เรียบร้อย")
print(f"✅ Silhouette Score สุดท้าย: {final_sil:.4f}")
print(f"✅ n = {len(df)} คน (ลบ outlier 1 คนแล้ว)")