import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

df = pd.read_csv("BU_Data_transformed.csv")

print(df.info(verbose=True, show_counts=True))
print(df[['Try_New_Flavor', 'Like_Stronger_Ebisen_Flavor']])

# สร้างคอลัมน์ใหม่ 'Sales_Opportunity' (โอกาสในการขาย)
# ใช้ตรรกะ OR (|) คือ ถ้าลูกค้ายอมลองรสใหม่ (1) หรือ ชอบรสเข้มข้น (1) อย่างใดอย่างหนึ่ง ถือว่าเป็นโอกาสขาย (1)
df['Sales_Opportunity'] = (df['Try_New_Flavor'] | df['Like_Stronger_Ebisen_Flavor'])

# สิ่งสำคัญ: ต้องลบ 2 คอลัมน์ตั้งต้นทิ้งก่อนเทรนโมเดล เพื่อป้องกัน Data Leakage (โมเดลแอบดูคำตอบ)
# df.drop(columns=['Try_New_Flavor', 'Like_Stronger_Ebisen_Flavor'], inplace=True)

print(df['Sales_Opportunity'].value_counts())