### 🌿 กฎการทำงานกับ Git (Git Workflow)
เพื่อป้องกันไม่ให้โค้ดพังหรือตีกันเวลารวมงาน ห้าม Push โค้ดเข้าสาขา main โดยตรงเด็ดขาด! ให้ทุกคนสร้าง Branch ของตัวเองสำหรับทำงานแต่ละส่วน โดยทำตามขั้นตอนดังนี้:
- Step 1: อัปเดตโค้ดล่าสุดเสมอ
ก่อนเริ่มงานใหม่ ให้แน่ใจว่าคุณอยู่ที่สาขา main และดึงโค้ดล่าสุดของเพื่อนๆ มาก่อน

```Bash
git checkout main
git pull origin main
```
- Step 2: สร้างและสลับไปยัง Branch ของตัวเอง
สร้าง Branch ใหม่โดยตั้งชื่องานให้ชัดเจน (เช่น ชื่อคุณ/ชื่อฟีเจอร์)
ตัวอย่างการตั้งชื่อ: feature/kangfu-tools หรือ fix/memory-bug

```Bash
git checkout -b feature/ชื่อของคุณ-ชื่องาน
```
(คำสั่ง -b คือการสร้างสาขาใหม่และย้ายเข้าไปทำงานในสาขานั้นทันที)

- Step 3: ทำงานและ Save (Commit)
เมื่อเขียนโค้ดเสร็จแล้ว ให้ทำการ Save งานลงใน Branch ของคุณ

```Bash
git add .
git commit -m "อธิบายสั้นๆ ว่าอัปเดตอะไรไปบ้าง เช่น Add economy calculator tool"
```

- Step 4: อัปโหลด Branch ของคุณขึ้น GitHub (Push)
อัปโหลดสาขาที่คุณเพิ่งทำเสร็จขึ้นไปยัง Repository ของกลุ่ม

``` Bash
git push origin feature/ชื่อของคุณ-ชื่องาน
```

- Step 5: สร้าง Pull Request (PR)
1. เข้าไปที่หน้าเว็บ GitHub ของโปรเจกต์เรา
2. กดปุ่ม Compare & pull request แถวๆ กล่องสีเหลืองที่เด้งขึ้นมา
3. แจ้งเพื่อนในกลุ่มให้ช่วยรีวิว (Review) โค้ด หากไม่มีปัญหา ค่อยกดปุ่ม Merge pull request เพื่อรวมโค้ดเข้าเส้น main
---

### 📄 ข้อมูลของ `BU_Data_transformed.csv`
ไฟล์ `BU_Data_transformed.csv` เป็นผลลัพธ์จากการรัน `data.py` โดยมีการแปลงข้อมูลและสกัดฟีเจอร์จากไฟล์ต้นฉบับ `BU_Data.csv` แบบสรุปแล้ว

#### หมวดหมู่หลักของคอลัมน์
- `Know_Ebisen` (รู้จัก/เคยทาน Ebisen), `Age` (อายุ), `Gender` (เพศ)
  - ข้อมูลพื้นฐานของผู้ตอบ
  - `Know_Ebisen` แปลงเป็นตัวเลขตามระดับความรู้จัก

- `Calvora_Tagline_Awareness` (รู้หรือไม่ว่า Calvora มี tagline)
  - แปลงเป็น 1/0

- `Purchase_Factor_*` (ปัจจัยการซื้อ)
  - เช่น `Purchase_Factor_Quality_Ingredients` = วัตถุดิบคุณภาพ
  - เป็นคะแนน Likert 1-5 จากคำถามคุณสมบัติขนมที่มีผลต่อการตัดสินใจซื้อ

- `Is_Calvora_*` (คิดว่าเป็นของ Calvora หรือไม่)
  - เช่น `Is_Calvora_Calvora`, `Is_Calvora_Hanaro`
  - แปลงเป็น 1 = ใช่, 0 = ไม่ใช่

- `Calvora_Tagline_Reflection` (ความเห็นต่อ tagline)
  - คะแนนว่าตรงกับภาพลักษณ์แบรนด์มากน้อยแค่ไหน
- `Calvora_Tagline_Interpretation_Category` (ตีความ tagline)
  - จัดกลุ่มคำตอบเชิงข้อความ เช่น `Natural`, `Positive`, `Negative`, `Other`

- `Strength_*` (จุดเด่นของผลิตภัณฑ์)
  - ดัมมี่จากคำตอบว่าผู้ตอบเลือกคุณสมบัติใดบ้าง
  - 1 = เลือก, 0 = ไม่เลือก

- `Calvora_Natural_Ingredient_*` (ความเชื่อมั่นวัตถุดิบธรรมชาติ)
  - ตัวอย่างเช่น `Calvora_Natural_Ingredient_การรับรองจากหน่วยงานด้านอาหาร`
  - 1 = ระบุคุณสมบัตินั้น, 0 = ไม่ระบุ

- `Known_Snack_*` (รู้จักผลิตภัณฑ์)
  - ดัมมี่จากคำตอบว่า “รู้จักผลิตภัณฑ์ใดบ้าง”
  - 1 = รู้จัก, 0 = ไม่รู้จัก

- `Tasted_Snack_*` (เคยทานผลิตภัณฑ์)
  - ดัมมี่จากคำตอบว่า “เคยทานแบรนด์ใดบ้าง”
  - 1 = เคยทาน, 0 = ไม่เคยทาน

- `Ebisen_Flavor_*` (รสชาติ Ebisen ที่เคยทาน)
  - ดัมมี่จากรสชาติของ Ebisen เช่น `Ebisen_Flavor_Extra BBQ`

- `Calvora_Image_*` (ภาพที่นึกถึง Calvora)
  - ดัมมี่จากคำตอบภาพที่ผู้ตอบนึกถึง

- `Time_*` และ `Snack_Time_Category` (โอกาส/เวลาที่ทานขนม)
  - แปลงจากคำตอบข้อความเป็นหมวด เช่น `Watching_Media`, `Free_Time`, `Hungry`, `Party_Drinking`

- `Calvora_Association_*` (นึกถึง Calvora แล้วคิดถึงอะไร)
  - แยกเป็นกลุ่มเช่น `Bax_Association`, `General_Snack_Association`, `Shrimp_Chip_Association`

- `Natural_Brand_Association_Category` (แบรนด์ที่นึกถึงจากวัตถุดิบธรรมชาติ)

- `Why_Choose_Ebisen_Category` (เหตุผลเลือกทาน Ebisen)
  - เช่น `Delicious` (อร่อย), `Try` (อยากลอง), `Saltier` (เค็มกว่า), `Brand` (แบรนด์), `Affordable` (ราคาถูก)

- `Desired_New_Flavor_Category` (รสชาติใหม่ที่อยากให้มี)
  - เช่น `Spicy` (เผ็ด), `Cheese` (ชีส), `BBQ`, `Nori` (สาหร่าย), `Mala` (หมาล่า)

- `Reason_Never_Tried_Category` (เหตุผลที่ยังไม่เคยทาน)
  - เช่น `Never_Had_Chance` (ยังไม่มีโอกาส), `Not_Attractive` (ไม่ดึงดูด), `Not_Snack_Person` (ไม่ค่อยทานขนม)

- `Expected_Stronger_Flavor_Category` (ความคาดหวังรสเข้มข้น)
  - เช่น `BBQ`, `Shrimp_Flavor` (รสกุ้ง), `More_Intense_Flavor`

- `Why_Like_Stronger_Flavor_Category` (เหตุผลชอบรสเข้มข้น)
  - เช่น `Like_Spicy_Flavor`, `Like_Delicious_Flavor`

- `Strong_Flavor_Occasion_Category` (โอกาสที่เหมาะกับรสเข้มข้น)
  - เช่น `Watching_Media`, `Working_Studying`, `Party_Drinking`

- `Reason_Not_Willing_Category` (เหตุผลไม่อยากลอง)
  - เช่น `Not_Delicious`, `Already_Salty`, `Indifferent`

- `Believe_Ebisen_Shrimp` (เชื่อว่า Ebisen ทำมาจากกุ้งแท้)
- `Try_New_Flavor` (อยากลองรสใหม่)
- `Believe_Ebisen_Shrimp_2` (เชื่ออีกครั้งในคำถามชุดที่ 2)
- `Like_Stronger_Ebisen_Flavor` (ชอบรสเข้มข้น)
  - เป็นคอลัมน์ตัวชี้วัด 1/0 จากคำตอบตรงๆ
  - 1 = ใช่ / อยากลอง / เชื่อ, 0 = ไม่ใช่ / ไม่อยาก

#### การใช้งาน `BU_Data_transformed.csv`
- คอลัมน์ที่ขึ้นต้นด้วย `Known_Snack_`, `Tasted_Snack_`, `Ebisen_Flavor_`, `Strength_`, `Calvora_Natural_Ingredient_`, `Calvora_Image_`, `Time_`, `Calvora_Association_` เป็นฟีเจอร์ดัมมี่
- ดัมมี่ = 1 หมายถึง “มี/เลือก/รู้จัก/เคยทาน” และ 0 หมายถึง “ไม่มี/ไม่เลือก/ไม่รู้จัก/ไม่เคยทาน”
- `*Category` คือกลุ่มคำตอบจากข้อความที่แปลงเป็น label แล้ว
- เหมาะสำหรับนำไปวิเคราะห์เชิงสถิติ หรือใช้สร้างโมเดลต่อ

> สรุป: `BU_Data_transformed.csv` เป็นไฟล์ที่เตรียมข้อมูลให้พร้อมสำหรับการวิเคราะห์ต่อ ทั้งกลุ่มคำตอบเชิงตัวเลข ดัมมี่ และกลุ่มข้อความที่จัดหมวดแล้ว
