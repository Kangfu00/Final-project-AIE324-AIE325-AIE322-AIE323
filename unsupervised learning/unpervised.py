import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

df = pd.read_csv("BU_Data_transformed.csv")