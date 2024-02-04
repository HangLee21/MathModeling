import statsmodels.api as sm
import pandas as pd

# 读取 Excel 文件
dataset = pd.read_excel('../data/standard_data.xlsx')
columns = dataset.columns[:-1]

X = dataset[columns]
y = dataset['label']

X = sm.add_constant(X)
model = sm.Logit(y, X)
result = model.fit()

print(result.summary())