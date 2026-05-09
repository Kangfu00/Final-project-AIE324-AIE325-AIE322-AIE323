import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv("BU_Data_transformed.csv")

target = "Try_New_Flavor"
drop_cols = [target, "Like_Stronger_Ebisen_Flavor"]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=["int64", "float64"])

with open("supervised_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Calvora AI Prediction",
    page_icon="🍤",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3, p, label, span, div {
    color: white !important;
}
.main-card {
    background: rgba(255,255,255,0.10);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.result-card {
    background: rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}
.metric-card {
    background: rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🍤 Calvora AI Customer Prediction")
st.markdown("### ระบบทำนายแนวโน้มการลองรสชาติใหม่ด้วย Supervised Learning")

col_intro1, col_intro2, col_intro3 = st.columns(3)

with col_intro1:
    st.markdown("""
    <div class="metric-card">
    <h3>Model</h3>
    <h2>Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)

with col_intro2:
    st.markdown("""
    <div class="metric-card">
    <h3>Task</h3>
    <h2>Classification</h2>
    </div>
    """, unsafe_allow_html=True)

with col_intro3:
    st.markdown("""
    <div class="metric-card">
    <h3>Target</h3>
    <h2>Try New Flavor</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🧾 Customer Information")

    know_ebisen = st.selectbox(
        "ระดับการรู้จัก Calvora Ebisen",
        ["ไม่รู้จักเลย", "รู้จัก แต่ไม่เคยทาน", "รู้จัก และเคยทาน"]
    )

    age = st.selectbox(
        "ช่วงอายุ",
        ["ต่ำกว่า 20 ปี", "20-29 ปี", "30-39 ปี", "40-49 ปี", "50 ปีขึ้นไป"]
    )

    st.markdown("### ปัจจัยที่มีผลต่อการซื้อ")

    purchase_tasty = st.slider("รสชาติอร่อย", 1, 5, 4)
    purchase_many_flavors = st.slider("มีรสชาติให้เลือกเยอะ", 1, 5, 4)
    purchase_crispy = st.slider("ความกรุบกรอบ", 1, 5, 4)
    purchase_healthy = st.slider("ความเป็นสินค้าสุขภาพ", 1, 5, 3)
    purchase_quality = st.slider("วัตถุดิบคุณภาพ", 1, 5, 4)

    believe_shrimp = st.radio(
        "เชื่อว่า Ebisen ทำมาจากกุ้งแท้หรือไม่",
        ["เชื่อ", "ไม่เชื่อ"],
        horizontal=True
    )

    ate_original = st.checkbox("เคยทาน Ebisen รส Original")
    know_ebisen_product = st.checkbox("รู้จักสินค้าเอบินาริ")
    tasted_ebisen_product = st.checkbox("เคยทานสินค้าเอบินาริ")

    predict_button = st.button("🔮 Predict Customer Behavior", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

know_map = {
    "ไม่รู้จักเลย": 0,
    "รู้จัก แต่ไม่เคยทาน": 1,
    "รู้จัก และเคยทาน": 2
}

age_map = {
    "ต่ำกว่า 20 ปี": 0,
    "20-29 ปี": 1,
    "30-39 ปี": 2,
    "40-49 ปี": 3,
    "50 ปีขึ้นไป": 4
}

sample_customer = pd.DataFrame(
    [[0] * len(X.columns)],
    columns=X.columns
)

if "Know_Ebisen" in sample_customer.columns:
    sample_customer["Know_Ebisen"] = know_map[know_ebisen]

if "Age" in sample_customer.columns:
    sample_customer["Age"] = age_map[age]

if "Purchase_Factor_Tasty" in sample_customer.columns:
    sample_customer["Purchase_Factor_Tasty"] = purchase_tasty

if "Purchase_Factor_Many_Flavors" in sample_customer.columns:
    sample_customer["Purchase_Factor_Many_Flavors"] = purchase_many_flavors

if "Purchase_Factor_Crispy" in sample_customer.columns:
    sample_customer["Purchase_Factor_Crispy"] = purchase_crispy

if "Purchase_Factor_Healthy" in sample_customer.columns:
    sample_customer["Purchase_Factor_Healthy"] = purchase_healthy

if "Purchase_Factor_Quality_Ingredients" in sample_customer.columns:
    sample_customer["Purchase_Factor_Quality_Ingredients"] = purchase_quality

if "Believe_Ebisen_Shrimp" in sample_customer.columns:
    sample_customer["Believe_Ebisen_Shrimp"] = 1 if believe_shrimp == "เชื่อ" else 0

if "Ebisen_Flavor_Original" in sample_customer.columns:
    sample_customer["Ebisen_Flavor_Original"] = 1 if ate_original else 0

if "Known_Snack_เอบินาริ" in sample_customer.columns:
    sample_customer["Known_Snack_เอบินาริ"] = 1 if know_ebisen_product else 0

if "Tasted_Snack_เอบินาริ" in sample_customer.columns:
    sample_customer["Tasted_Snack_เอบินาริ"] = 1 if tasted_ebisen_product else 0

with right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_button:
        prediction = model.predict(sample_customer)[0]
        probability = model.predict_proba(sample_customer)[0]

        not_try_prob = probability[0] * 100
        try_prob = probability[1] * 100

        if prediction == 1:
            st.success("ลูกค้าคนนี้มีแนวโน้มจะลองรสชาติใหม่")
            st.markdown("### ✅ Recommended Action")
            st.write("ควรนำเสนอรสชาติใหม่ โปรโมชั่นทดลองชิม หรือโฆษณาที่เน้นความแปลกใหม่")
        else:
            st.warning("ลูกค้าคนนี้มีแนวโน้มจะไม่ลองรสชาติใหม่")
            st.markdown("### ⚠️ Recommended Action")
            st.write("ควรเน้นความน่าเชื่อถือของแบรนด์ รสชาติดั้งเดิม หรือจุดเด่นด้านวัตถุดิบ")

        st.metric("โอกาสที่จะลองรสชาติใหม่", f"{try_prob:.2f}%")
        st.progress(int(try_prob))

        st.metric("โอกาสที่จะไม่ลอง", f"{not_try_prob:.2f}%")

        st.divider()

        if try_prob >= 70:
            st.info("กลุ่มนี้เหมาะกับแคมเปญเปิดตัวรสชาติใหม่มาก")
        elif try_prob >= 50:
            st.info("กลุ่มนี้มีโอกาสสนใจ ควรใช้โปรโมชันช่วยกระตุ้น")
        else:
            st.info("กลุ่มนี้ควรใช้การสร้างความเชื่อมั่นก่อนขายรสชาติใหม่")

    else:
        st.write("กรอกข้อมูลทางซ้าย แล้วกดปุ่ม Predict เพื่อดูผลลัพธ์")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

with st.expander("🔍 ดูข้อมูล Feature ที่ส่งเข้าโมเดล"):
    st.dataframe(sample_customer)

with st.expander("ℹ️ คำอธิบายระบบ"):
    st.write("""
    ระบบนี้ใช้โมเดล Supervised Learning ประเภท Random Forest Classifier 
    เพื่อทำนายว่าลูกค้ามีแนวโน้มจะลองรสชาติใหม่ของ Calvora หรือไม่ 
    โดยใช้ข้อมูลพฤติกรรม ความรู้จักสินค้า ความเชื่อมั่นต่อวัตถุดิบ 
    และปัจจัยที่มีผลต่อการตัดสินใจซื้อ
    """)