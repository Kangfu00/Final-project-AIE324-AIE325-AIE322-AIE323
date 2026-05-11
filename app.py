import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from math import pi

matplotlib.rcParams['font.family'] = 'Tahoma'

# ==========================================
# ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="Calvora Customer Insights",
    page_icon="🍤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e0e0e0 !important; }
[data-testid="stSidebar"] hr { border-color: #ffffff22; }
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.cluster-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1rem;
}
.result-box {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 1px solid #667eea44;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# โหลด Models
# ==========================================
@st.cache_resource
def load_models():
    try:
        return {
            'scaler':       joblib.load('scaler.pkl'),
            'pca':          joblib.load('pca.pkl'),
            'kmeans':       joblib.load('kmeans.pkl'),
            'behavioral_cols': joblib.load('behavioral_cols.pkl'),
            'sup_model':    joblib.load('supervised_model.pkl'),
            'sup_scaler':   joblib.load('supervised_scaler.pkl'),
            'sup_features': joblib.load('supervised_features.pkl'),
        }
    except FileNotFoundError as e:
        st.error(f"❌ ไม่พบไฟล์โมเดล: {e}\nกรุณารัน unsupervised.py และ supervised_model.py ก่อน")
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('BU_Data_3_Segments_Final_Complete.csv')
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ BU_Data_3_Segments_Final_Complete.csv")
        st.stop()

models = load_models()
df     = load_data()

behavioral_cols = models['behavioral_cols']

CLUSTER_NAMES = {
    1: "🔵 Cluster 1 — Casual Snacker",
    2: "🟠 Cluster 2 — Quality Seeker",
    3: "🔴 Cluster 3 — Health Conscious",
}
CLUSTER_COLORS = {1: "#4e79a7", 2: "#f28e2b", 3: "#e15759"}

# map ช่วงอายุที่เข้าใจง่าย → z-score ที่ตรงกันใน dataset (5 กลุ่ม ห่างเท่ากัน)
AGE_MAP = {
    "18 - 22 ปี":  -1.6003,
    "23 - 27 ปี":  -0.7523,
    "28 - 32 ปี":   0.0957,
    "33 - 37 ปี":   0.9438,
    "38 ปีขึ้นไป":  1.7918,
}

# ==========================================
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("## 🍤 Calvora Insights")
    st.markdown("---")
    page = st.radio(
        "เลือกหน้า",
        ["🏠 ภาพรวม", "🔮 พยากรณ์ลูกค้าใหม่", "📊 วิเคราะห์ Clustering"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**ข้อมูลโมเดล**")
    st.caption(f"n = {len(df)} คน (ลบ outlier แล้ว)")
    st.caption("Unsupervised: PCA(3) + KMeans K=3")
    st.caption("Supervised: Random Forest / LR / DT")

# ==========================================
# PAGE 1: ภาพรวม
# ==========================================
if page == "🏠 ภาพรวม":
    st.title("🍤 Calvora Customer Insights")
    st.markdown("ระบบวิเคราะห์และพยากรณ์กลุ่มลูกค้าของแบรนด์ **Calvora** ด้วย Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("จำนวนผู้ตอบ", f"{len(df)} คน")
    with col2:
        st.metric("จำนวน Cluster", "3 กลุ่ม")
    with col3:
        sales_opp = int(df['Sales_Opportunity'].sum()) if 'Sales_Opportunity' in df.columns else "-"
        st.metric("Sales Opportunity", f"{sales_opp} คน")
    with col4:
        pct = f"{sales_opp/len(df)*100:.1f}%" if isinstance(sales_opp, int) else "-"
        st.metric("คิดเป็น", pct)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📌 สัดส่วนแต่ละ Cluster")
        cluster_counts = df['Cluster_ID'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [CLUSTER_COLORS[i] for i in cluster_counts.index]
        wedges, texts, autotexts = ax.pie(
            cluster_counts.values, labels=[CLUSTER_NAMES[i] for i in cluster_counts.index],
            autopct='%1.1f%%', colors=colors, startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight('bold')
        ax.set_title('สัดส่วนลูกค้าแต่ละกลุ่ม', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("📌 Demographics แต่ละ Cluster")
        demo_cols = [c for c in ['Age', 'Gender_ชาย', 'Gender_หญิง', 'Gender_LGBTQ+'] if c in df.columns]
        demo_df = df.groupby('Cluster_ID')[demo_cols].mean().round(2)
        demo_df.index = [CLUSTER_NAMES[i] for i in demo_df.index]
        st.dataframe(demo_df, use_container_width=True)

        if 'Sales_Opportunity' in df.columns:
            st.subheader("📌 Sales Opportunity ต่อ Cluster")
            sales_by_cluster = df.groupby('Cluster_ID')['Sales_Opportunity'].mean() * 100
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            bars = ax2.bar(
                [CLUSTER_NAMES[i] for i in sales_by_cluster.index],
                sales_by_cluster.values,
                color=[CLUSTER_COLORS[i] for i in sales_by_cluster.index],
                edgecolor='white'
            )
            for bar, val in zip(bars, sales_by_cluster.values):
                ax2.text(bar.get_x() + bar.get_width()/2, val + 1,
                         f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
            ax2.set_ylim(0, 110)
            ax2.set_ylabel('% Sales Opportunity')
            ax2.set_title('โอกาสขายแต่ละกลุ่ม', fontsize=12, fontweight='bold')
            plt.xticks(rotation=15, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ==========================================
# PAGE 2: พยากรณ์ลูกค้าใหม่
# ==========================================
elif page == "🔮 พยากรณ์ลูกค้าใหม่":
    # CSS สำหรับ toggle button
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div {
        display: flex; flex-direction: row; gap: 8px; flex-wrap: wrap;
    }
    div[data-testid="stRadio"] > div > label {
        display: flex !important;
        align-items: center;
        padding: 6px 20px;
        border-radius: 20px;
        border: 1.5px solid #667eea;
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.15s;
        background: transparent;
        color: var(--text-color);
    }
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: #667eea;
        color: white !important;
        border-color: #667eea;
    }
    div[data-testid="stRadio"] > div > label > div:first-child { display: none; }
    .input-section {
        background: var(--background-color);
        border: 1px solid #e0e0e030;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .input-section h4 { margin: 0 0 0.75rem; font-size: 14px; color: #667eea; }
    .field-label {
        font-size: 13px;
        color: var(--text-color);
        margin-bottom: 2px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

    # helper: toggle ใช่/ไม่ใช่ (ต้องอยู่นอก form เพราะ radio ใน form มี limitation)
    def yn_toggle(label, key):
        st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
        val = st.radio("_", ["ไม่ใช่", "ใช่"], horizontal=True, key=key, label_visibility="collapsed")
        return 1 if val == "ใช่" else 0

    st.title("🔮 พยากรณ์ลูกค้าใหม่")
    st.markdown("---")

    tab_cluster, tab_sales = st.tabs([
        "🔵 ส่วนที่ 1 — พยากรณ์ Cluster (Unsupervised)",
        "💰 ส่วนที่ 2 — พยากรณ์ Sales Opportunity (Supervised)",
    ])

    # ==========================================
    # TAB 1: พยากรณ์ Cluster_ID
    # ==========================================
    with tab_cluster:
        st.subheader("🔵 พยากรณ์กลุ่มลูกค้า (Cluster_ID)")
        st.caption("ใช้ข้อมูลพฤติกรรมการซื้อและการบริโภค → StandardScaler + PCA(3) + KMeans K=3")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="input-section"><h4>🛒 ปัจจัยการซื้อ (Purchase Factor)</h4>', unsafe_allow_html=True)
            st.caption("ระดับความสำคัญ: -3 = ไม่สำคัญเลย, 0 = ปานกลาง, +3 = สำคัญมาก")
            pf_quality = st.slider("คุณภาพวัตถุดิบ", -3.0, 3.0, 0.0, 0.1, key="c_pf_q")
            pf_tasty   = st.slider("รสชาติอร่อย",    -3.0, 3.0, 0.0, 0.1, key="c_pf_t")
            pf_flavors = st.slider("ความหลากหลายรส", -3.0, 3.0, 0.0, 0.1, key="c_pf_f")
            pf_crispy  = st.slider("ความกรอบ",        -3.0, 3.0, 0.0, 0.1, key="c_pf_c")
            pf_healthy = st.slider("เพื่อสุขภาพ",     -3.0, 3.0, 0.0, 0.1, key="c_pf_h")
            strength_q = st.slider("Strength: คุณภาพดี", -3.0, 3.0, 0.0, 0.1, key="c_sq")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>🔗 Calvora Association</h4>', unsafe_allow_html=True)
            calvora_assoc = st.slider("Calvora Association",       -3.0, 3.0, 0.0, 0.1, key="c_ca")
            general_assoc = st.slider("General Snack Association", -3.0, 3.0, 0.0, 0.1, key="c_ga")
            tagline       = st.slider("Tagline Reflection",        -3.0, 3.0, 0.0, 0.1, key="c_tl")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-section"><h4>⏰ ช่วงเวลารับประทาน</h4>', unsafe_allow_html=True)
            st.caption("เลือกช่วงเวลาที่มักรับประทานขนม")
            t_free   = yn_toggle("ช่วงเวลาว่าง",  "c_t1")
            t_hungry = yn_toggle("ตอนหิว",         "c_t2")
            t_night  = yn_toggle("ดึก",            "c_t3")
            t_party  = yn_toggle("งานสังสรรค์",    "c_t4")
            t_game   = yn_toggle("เล่นเกม",        "c_t5")
            t_media  = yn_toggle("ดูสื่อ/ดูหนัง",  "c_t6")
            t_work   = yn_toggle("ทำงาน/เรียน",   "c_t7")
            t_other  = yn_toggle("อื่นๆ",          "c_t8")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>💪 Strength อื่นๆ</h4>', unsafe_allow_html=True)
            st_cols = [c for c in behavioral_cols if 'Strength_' in c and c != 'Strength_มีคุณภาพดี (Good quality)']
            st_vals = {}
            for c in st_cols:
                label = c.replace('Strength_', '')
                st_vals[c] = yn_toggle(label, f"c_{c}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="input-section"><h4>🌿 Natural Ingredient</h4>', unsafe_allow_html=True)
            st.caption("สิ่งที่ทำให้เชื่อมั่นว่าใช้วัตถุดิบธรรมชาติ")
            ni_cols = [c for c in behavioral_cols if 'Calvora_Natural_Ingredient_' in c]
            ni_vals = {}
            for c in ni_cols:
                label = c.replace('Calvora_Natural_Ingredient_', '')
                ni_vals[c] = yn_toggle(label, f"c_{c}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>👤 ข้อมูลทั่วไป</h4>', unsafe_allow_html=True)
            age_label = st.selectbox("ช่วงอายุ", list(AGE_MAP.keys()), key="c_age")
            age = AGE_MAP[age_label]
            know   = yn_toggle("รู้จัก Ebisen", "c_know")
            believe = yn_toggle("เชื่อว่า Ebisen ทำจากกุ้ง", "c_bel")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔵 พยากรณ์กลุ่มลูกค้า (Cluster)", use_container_width=True, type="primary"):
            input_dict = {c: 0.0 for c in behavioral_cols}
            input_dict.update({
                'Purchase_Factor_Quality_Ingredients': pf_quality,
                'Purchase_Factor_Tasty':               pf_tasty,
                'Purchase_Factor_Many_Flavors':        pf_flavors,
                'Purchase_Factor_Crispy':              pf_crispy,
                'Purchase_Factor_Healthy':             pf_healthy,
                'Time_Free_Time':        t_free,
                'Time_Hungry':           t_hungry,
                'Time_Late_Night':       t_night,
                'Time_Party_Drinking':   t_party,
                'Time_Playing_Games':    t_game,
                'Time_Watching_Media':   t_media,
                'Time_Working_Studying': t_work,
                'Time_Other':            t_other,
                'Strength_มีคุณภาพดี (Good quality)': strength_q,
                'Calvora_Association_Calvora_Association':       calvora_assoc,
                'Calvora_Association_General_Snack_Association': general_assoc,
                'Calvora_Tagline_Reflection': tagline,
            })
            input_dict.update(ni_vals)
            input_dict.update(st_vals)

            X_input = pd.DataFrame([input_dict])[behavioral_cols].fillna(0)
            X_s     = models['scaler'].transform(X_input)
            X_p     = models['pca'].transform(X_s)
            cluster = int(models['kmeans'].predict(X_p)[0]) + 1

            st.markdown(f"""
            <div class="result-box">
                <h3 style="margin:0 0 10px">📊 ผลการพยากรณ์ Cluster</h3>
                <span class="cluster-badge" style="background:{CLUSTER_COLORS[cluster]}22;
                      color:{CLUSTER_COLORS[cluster]};border:2px solid {CLUSTER_COLORS[cluster]};font-size:1.15rem;padding:8px 24px">
                    {CLUSTER_NAMES[cluster]}
                </span>
            </div>
            """, unsafe_allow_html=True)

            cluster_desc = {
                1: "มักบริโภคขนมในเวลาว่างหรือดูสื่อ ไม่ได้เน้นคุณภาพสูงมาก เน้นความสะดวกและความเคยชิน",
                2: "ให้ความสำคัญกับคุณภาพวัตถุดิบและความน่าเชื่อถือของแบรนด์เป็นหลัก",
                3: "สนใจเรื่องสุขภาพและความหลากหลายของรสชาติ มีแนวโน้มลองสินค้าใหม่สูง",
            }
            st.info(f"💡 **ลักษณะกลุ่ม:** {cluster_desc[cluster]}")
            st.caption("โมเดล: PCA(3 components) + KMeans K=3 — Silhouette Score = 0.354")

    # ==========================================
    # TAB 2: พยากรณ์ Sales Opportunity
    # ==========================================
    with tab_sales:
        st.subheader("💰 พยากรณ์ Sales Opportunity")
        st.caption("ใช้ข้อมูลทัศนคติและ Cluster_ID → StandardScaler + Best Model จาก RF / LR / DT")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="input-section"><h4>🛒 ปัจจัยการซื้อ</h4>', unsafe_allow_html=True)
            s_pf_quality  = st.slider("คุณภาพวัตถุดิบ",     -3.0, 3.0, 0.0, 0.1, key="s_pf_q")
            s_pf_flavors  = st.slider("ความหลากหลายรส",     -3.0, 3.0, 0.0, 0.1, key="s_pf_f")
            s_pf_crispy   = st.slider("ความกรอบ",            -3.0, 3.0, 0.0, 0.1, key="s_pf_c")
            s_pf_healthy  = st.slider("เพื่อสุขภาพ",         -3.0, 3.0, 0.0, 0.1, key="s_pf_h")
            s_strength_q  = st.slider("Strength: คุณภาพดี", -3.0, 3.0, 0.0, 0.1, key="s_sq")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>🔗 Association</h4>', unsafe_allow_html=True)
            s_calvora_assoc = st.slider("Calvora Association",       -3.0, 3.0, 0.0, 0.1, key="s_ca")
            s_general_assoc = st.slider("General Snack Association", -3.0, 3.0, 0.0, 0.1, key="s_ga")
            s_tagline       = st.slider("Tagline Reflection",        -3.0, 3.0, 0.0, 0.1, key="s_tl")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-section"><h4>🧠 ทัศนคติและความเชื่อ</h4>', unsafe_allow_html=True)
            s_believe = yn_toggle("เชื่อว่า Ebisen ทำจากกุ้ง", "s_bel")
            s_know    = yn_toggle("รู้จัก Ebisen",              "s_know")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>👤 ข้อมูลทั่วไป</h4>', unsafe_allow_html=True)
            s_age_label = st.selectbox("ช่วงอายุ", list(AGE_MAP.keys()), key="s_age")
            s_age = AGE_MAP[s_age_label]
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="input-section"><h4>🔵 Cluster_ID (จากส่วนที่ 1)</h4>', unsafe_allow_html=True)
            st.caption("ใส่ผลจากส่วนที่ 1 — ถ้ายังไม่ได้พยากรณ์ เลือก 'ไม่ทราบ'")
            s_cluster = st.radio(
                "_",
                options=[0, 1, 2, 3],
                format_func=lambda x: "ไม่ทราบ" if x == 0 else CLUSTER_NAMES[x].split("—")[1].strip(),
                horizontal=True,
                key="s_cluster",
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("💰 พยากรณ์ Sales Opportunity", use_container_width=True, type="primary"):
            quality_seeker = s_pf_quality + s_pf_healthy
            brand_trust    = s_believe + s_tagline
            cluster_val    = s_cluster if s_cluster != 0 else 2

            sup_input = {
                'Purchase_Factor_Quality_Ingredients':           s_pf_quality,
                'Calvora_Tagline_Reflection':                    s_tagline,
                'Believe_Ebisen_Shrimp':                         s_believe,
                'Purchase_Factor_Many_Flavors':                  s_pf_flavors,
                'Purchase_Factor_Healthy':                       s_pf_healthy,
                'Calvora_Association_General_Snack_Association': s_general_assoc,
                'Purchase_Factor_Crispy':                        s_pf_crispy,
                'Age':                                           s_age,
                'Know_Ebisen':                                   s_know,
                'Strength_มีคุณภาพดี (Good quality)':            s_strength_q,
                'Calvora_Association_Calvora_Association':       s_calvora_assoc,
                'Cluster_ID':                                    cluster_val,
                'Quality_Seeker_Score':                          quality_seeker,
                'Brand_Trust_Score':                             brand_trust,
            }
            sup_df     = pd.DataFrame([[sup_input.get(f, 0) for f in models['sup_features']]],
                                       columns=models['sup_features'])
            sup_scaled = models['sup_scaler'].transform(sup_df)
            sales_pred = int(models['sup_model'].predict(sup_scaled)[0])

            st.markdown("---")
            st.subheader("📊 ผลการพยากรณ์ Sales Opportunity")
            if sales_pred == 1:
                st.success("✅ **มีโอกาสซื้อ (Sales Opportunity = 1)**\n\nลูกค้ากลุ่มนี้มีแนวโน้มที่จะทดลองสินค้าใหม่หรือรสชาติที่เข้มข้นขึ้น")
            else:
                st.warning("⚠️ **โอกาสน้อย (Sales Opportunity = 0)**\n\nลูกค้ากลุ่มนี้ยังไม่พร้อมทดลองสินค้าใหม่ในตอนนี้")

            c1, c2, c3 = st.columns(3)
            c1.metric("Quality Seeker Score", f"{quality_seeker:.2f}")
            c2.metric("Brand Trust Score",    f"{brand_trust:.2f}")
            c3.metric("Cluster ที่ใช้",        f"C{cluster_val}" if s_cluster != 0 else "C2 (default)")
            st.caption("โมเดล: StratifiedKFold 5-fold CV — เลือก best จาก Random Forest / Logistic Regression / Decision Tree")

# ==========================================
# PAGE 3: วิเคราะห์ Clustering
# ==========================================
elif page == "📊 วิเคราะห์ Clustering":
    st.title("📊 วิเคราะห์ผล Clustering")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗺️ PCA Scatter", "🌡️ Heatmap", "🕸️ Radar Chart"])

    with tab1:
        st.subheader("PCA Scatter Plot — การกระจายตัวของแต่ละกลุ่ม")
        if 'PCA1' in df.columns and 'PCA2' in df.columns:
            fig, ax = plt.subplots(figsize=(9, 6))
            for cid, grp in df.groupby('Cluster_ID'):
                ax.scatter(grp['PCA1'], grp['PCA2'],
                           label=CLUSTER_NAMES[cid],
                           color=CLUSTER_COLORS[cid],
                           alpha=0.75, s=80, edgecolors='white', linewidth=0.5)
            ax.set_xlabel('PCA 1', fontsize=12)
            ax.set_ylabel('PCA 2', fontsize=12)
            ax.set_title('การกระจายตัวของลูกค้าแต่ละกลุ่ม (PCA 2D)', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("ไม่พบคอลัมน์ PCA1/PCA2 ในไฟล์ข้อมูล")

    with tab2:
        st.subheader("Heatmap — ค่าเฉลี่ยพฤติกรรมแต่ละกลุ่ม")
        avail_beh = [c for c in behavioral_cols if c in df.columns]
        cluster_means = df.groupby('Cluster_ID')[avail_beh].mean()
        short_names = {c: c.replace('Purchase_Factor_', 'PF_')
                           .replace('Strength_', 'St_')
                           .replace('Calvora_Natural_Ingredient_', 'NI_')
                           .replace('Time_', 'T_') for c in avail_beh}
        cluster_means.index = [CLUSTER_NAMES[i] for i in cluster_means.index]

        fig, ax = plt.subplots(figsize=(18, 4))
        sns.heatmap(cluster_means.rename(columns=short_names),
                    annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, linewidths=0.3, ax=ax,
                    cbar_kws={'label': 'ค่าเฉลี่ย'})
        ax.set_title('ค่าเฉลี่ยทุก Feature ต่อ Cluster\n(สีเขียว = สูง, สีแดง = ต่ำ)',
                     fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Radar Chart — ปัจจัยการซื้อแต่ละกลุ่ม")
        radar_cols = [c for c in behavioral_cols if 'Purchase_Factor_' in c and c in df.columns]
        cluster_radar = df.groupby('Cluster_ID')[radar_cols].mean().reset_index()
        categories = [c.replace('Purchase_Factor_', '') for c in radar_cols]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, fontsize=10)

        for i in range(len(cluster_radar)):
            cid = int(cluster_radar.loc[i, 'Cluster_ID'])
            values = cluster_radar.loc[i, radar_cols].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2,
                    label=CLUSTER_NAMES[cid], color=CLUSTER_COLORS[cid])
            ax.fill(angles, values, alpha=0.1, color=CLUSTER_COLORS[cid])

        plt.title('ปัจจัยการซื้อหลักแต่ละ Cluster', fontsize=14, y=1.1, fontweight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        st.pyplot(fig)
        plt.close()