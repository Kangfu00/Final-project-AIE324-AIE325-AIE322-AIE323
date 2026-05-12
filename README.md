# 🍤 Calvora Customer Insights

โปรเจกต์นี้เป็นระบบวิเคราะห์และพยากรณ์โอกาสการขาย (Sales Opportunity) สำหรับแบรนด์ขนมขบเคี้ยว **Calvora** (กรณีศึกษา) ด้วยเทคนิค Machine Learning และ Data Science โดยมีการแบ่งกลุ่มลูกค้า (Customer Segmentation) ออกเป็น 3 กลุ่ม เพื่อหาอินไซต์และพยากรณ์แนวโน้มพฤติกรรมผ่าน Web Application ที่สร้างด้วย Streamlit

## 🚀 ฟีเจอร์หลัก (Key Features)
- **🏠 ภาพรวม (Overview):** สรุปสัดส่วน Demographics ของลูกค้าทั้ง 3 Clusters พร้อมแสดง % Sales Opportunity ในแต่ละกลุ่ม
- **💰 พยากรณ์ Sales Opportunity:** โมเดล Machine Learning (RF/LR/DT) สำหรับประเมินว่าลูกค้าที่มีทัศนคติและพฤติกรรมตามที่ระบุ จะมีโอกาสทดลองสินค้าใหม่หรือรสชาติที่เข้มข้นขึ้นหรือไม่
- **📊 วิเคราะห์ Clustering:** นำเสนอผลลัพธ์ผ่าน PCA Scatter Plot (2D), Heatmap เปรียบเทียบค่าเฉลี่ยพฤติกรรม, และ Radar Chart วิเคราะห์จุดเด่น (Strength) ที่แต่ละกลุ่มรับรู้

## 🛠️ โครงสร้างไฟล์ (File Structure)
- `app.py`: ไฟล์หลักสำหรับรัน Web Application ด้วย Streamlit
- `data.py`: สคริปต์สำหรับ Data Preprocessing, Data Cleaning, การทำ Encoding, สกัดฟีเจอร์, และ Imputation โดยแปลงไฟล์ `BU_Data.csv` เป็น `BU_Data_transformed.csv`
- โมเดลที่เทรนแล้ว (`.pkl`): `supervised_model.pkl`, `supervised_scaler.pkl`, `supervised_features.pkl`

---

## 💻 วิธีการรันโปรเจกต์ (How to Run)

1. ติดตั้งไลบรารีที่จำเป็น:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```
รันแอปพลิเคชัน:
```Bash
streamlit run app.py
```
🌿 กฎการทำงานกับ Git (Git Workflow)
เพื่อป้องกันไม่ให้โค้ดพังหรือตีกันเวลารวมงาน ห้าม Push โค้ดเข้าสาขา main โดยตรงเด็ดขาด! ให้ทุกคนสร้าง Branch ของตัวเองสำหรับทำงานแต่ละส่วน โดยทำตามขั้นตอนดังนี้:

Step 1: อัปเดตโค้ดล่าสุดเสมอ
ก่อนเริ่มงานใหม่ ให้แน่ใจว่าคุณอยู่ที่สาขา main และดึงโค้ดล่าสุดของเพื่อนๆ มาก่อน

```Bash
git checkout main
git pull origin main
```
Step 2: สร้างและสลับไปยัง Branch ของตัวเอง
สร้าง Branch ใหม่โดยตั้งชื่องานให้ชัดเจน (เช่น ชื่อคุณ/ชื่อฟีเจอร์)
ตัวอย่างการตั้งชื่อ: feature/kangfu-tools หรือ fix/memory-bug

```Bash
git checkout -b feature/ชื่อของคุณ-ชื่องาน
```
Step 3: ทำงานและ Save (Commit)

```Bash
git add .
git commit -m "อธิบายสั้นๆ ว่าอัปเดตอะไรไปบ้าง เช่น Add economy calculator tool"
```
Step 4: อัปโหลด Branch ของคุณขึ้น GitHub (Push)
อัปโหลดสาขาที่คุณเพิ่งทำเสร็จขึ้นไปยัง Repository ของกลุ่ม

```Bash
git push origin feature/ชื่อของคุณ-ชื่องาน
```
Step 5: สร้าง Pull Request (PR)

เข้าไปที่หน้าเว็บ GitHub ของโปรเจกต์เรา

กดปุ่ม Compare & pull request แถวๆ กล่องสีเหลืองที่เด้งขึ้นมา

แจ้งเพื่อนในกลุ่มให้ช่วยรีวิว (Review) โค้ด หากไม่มีปัญหา ค่อยกดปุ่ม Merge pull request เพื่อรวมโค้ดเข้าเส้น main

📄 ข้อมูลของ BU_Data_transformed.csv
ไฟล์ BU_Data_transformed.csv เป็นผลลัพธ์จากการรัน data.py โดยมีการแปลงข้อมูลและสกัดฟีเจอร์จากไฟล์ต้นฉบับ BU_Data.csv

หมวดหมู่หลักของคอลัมน์
Know_Ebisen, Age, Gender: ข้อมูลพื้นฐานของผู้ตอบ (แปลง Know_Ebisen เป็นตัวเลขตามระดับความรู้จัก)

Calvora_Tagline_Awareness: รู้หรือไม่ว่า Calvora มี tagline (1/0)

Purchase_Factor_*: ปัจจัยการซื้อ (คะแนน Likert 1-5 จากคำถามคุณสมบัติขนม)

Is_Calvora_*: คิดว่าเป็นของ Calvora หรือไม่ (1 = ใช่, 0 = ไม่ใช่)

Calvora_Tagline_Reflection: คะแนนความสอดคล้องของ tagline กับภาพลักษณ์แบรนด์

Calvora_Tagline_Interpretation_Category: จัดกลุ่มคำตอบการตีความ เช่น Natural, Positive, Negative, Other

Strength_*: ดัมมี่คุณสมบัติและจุดเด่นของผลิตภัณฑ์ที่ผู้ตอบเลือก (1 = เลือก, 0 = ไม่เลือก)

Calvora_Natural_Ingredient_*: ความเชื่อมั่นในวัตถุดิบธรรมชาติ (1 = ระบุ, 0 = ไม่ระบุ)

Known_Snack_* / Tasted_Snack_*: ผลิตภัณฑ์และแบรนด์ที่รู้จักหรือเคยทาน (ดัมมี่ 1/0)

Ebisen_Flavor_*: รสชาติ Ebisen ที่เคยทาน

Calvora_Image_*: ดัมมี่ภาพที่นึกถึงแบรนด์

Time_* และ Snack_Time_Category: โอกาสและเวลาที่ทานขนม เช่น Watching_Media, Free_Time

Calvora_Association_*: นึกถึง Calvora แล้วคิดถึงอะไร (เช่น Bax_Association)

Natural_Brand_Association_Category: แบรนด์ที่นึกถึงเมื่อพูดถึงวัตถุดิบธรรมชาติ

Why_Choose_Ebisen_Category: เหตุผลที่เลือกทาน (เช่น อร่อย, อยากลอง, ราคาถูก)

Desired_New_Flavor_Category / Expected_Stronger_Flavor_Category: รสชาติใหม่และระดับความเข้มข้นที่อยากให้มี

Why_Like_Stronger_Flavor_Category: เหตุผลที่ชอบรสเข้มข้น

Strong_Flavor_Occasion_Category: โอกาสที่เหมาะกับขนมรสเข้มข้น

Reason_Never_Tried_Category / Reason_Not_Willing_Category: เหตุผลที่ยังไม่เคยทานหรือไม่อยากลอง

Believe_Ebisen_Shrimp / Try_New_Flavor / Like_Stronger_Ebisen_Flavor: การตอบรับเชิงทัศนคติต่อการเชื่อว่าทำจากกุ้งแท้และความต้องการลองรสใหม่ (1/0)