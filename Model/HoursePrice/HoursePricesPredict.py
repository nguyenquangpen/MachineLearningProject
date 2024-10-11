import pandas as pd

train_df = pd.read_csv('D:/Academic/lap-trinh/Project/PycharmProject/MachineLearningProject/Model/HoursePrice/train.csv')
test_df = pd.read_csv('D:/Academic/lap-trinh/Project/PycharmProject/MachineLearningProject/Model/HoursePrice/test.csv')

print(train_df.columns)
print('....')
print(test_df.columns)
# missing = df.isnull().sum()
# missing = missing[missing > 0]
# print(missing)

# Electrical: 1, MasVnrArea:8, BsmtQual:37, BsmtCond:37, BsmtExposure:38, BsmtFinType1:37, BsmtFinType2:38,