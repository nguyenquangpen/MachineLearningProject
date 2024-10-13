import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Đọc dữ liệu
train_df = pd.read_csv('D:/Academic/lap-trinh/Project/PycharmProject/Data/housePrice/train.csv')

# Tính tỷ lệ giá trị thiếu
missing_ratio_train = train_df.isnull().sum() / train_df.shape[0]
train_df = train_df.drop(missing_ratio_train[missing_ratio_train > 0.5].index, axis=1)

# Tách cột SalePrice ra khỏi train_df
y = train_df['SalePrice']
x = train_df.drop(['SalePrice', 'Id'], axis=1)

# Xác định các cột numerical và categorical
numerical_cols = x.select_dtypes(include=['number']).columns.tolist()
categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
                    'GarageQual', 'PoolQC', 'GarageCond', 'MiscFeature',
                    'OverallQual', 'OverallCond']
ordinal_features = [col for col in ordinal_features if col in x.columns]

nominal_features = list(set(categorical_cols) - set(ordinal_features))

# Tạo pipelines cho các loại đặc trưng
num_feature = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
])

nom_feature = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ord_feature = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

# Tiền xử lý dữ liệu
preprocess = ColumnTransformer(transformers=[
    ('num', num_feature, numerical_cols),
    ('nom', nom_feature, nominal_features),
    ('ord', ord_feature, ordinal_features)
])

# Áp dụng preprocessing
x_processed = preprocess.fit_transform(x)
print(x_processed.shape)

# Chia dữ liệu thành tập train và test
x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=0)

# chọn mô hình tốt nhất
# rgs = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# model, predictions = rgs.fit(x_train, x_test, y_train, y_test)

params = {
    'loss': ['squared_error', 'absolute_error', 'quantile'],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

model = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=params,
    scoring='r2',
    cv=6,
    verbose=1,
    n_jobs=-1
)

model.fit(x_train, y_train)

best_model = model.best_estimator_

best_model.fit(x_train, y_train)
Y_pred = best_model.predict(x_test)
print("Best parameters found:", model.best_params_)
print('R2: ', r2_score(y_test, Y_pred))

test_df = pd.read_csv('D:/Academic/lap-trinh/Project/PycharmProject/Data/housePrice/test.csv')

test_ids = test_df['Id']

test_df = test_df.drop('Id', axis=1)

x_test_processed = preprocess.transform(test_df)

pred = best_model.predict(x_test_processed)

final_output = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
final_output.to_csv('prediction.csv', index=False)

# score in kaggle: 0.14275