import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2, SelectPercentile

train_df = pd.read_csv('C:/Users/Qann/OneDrive - Phenikaa Univesity/machine-learning/Data/LoanPredition/train.csv')
train_df = train_df.drop('id', axis=1)

train_df['Z_score'] = abs(stats.zscore(train_df['loan_status']))
train_df = train_df[train_df['Z_score'] < 3]
train_df = train_df.drop('Z_score', axis=1)

x = train_df.drop('loan_status', axis=1)
y = train_df['loan_status']

categorical_cols = x.select_dtypes(include=['object']).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

preprocessor = ColumnTransformer(transformers=[
    ('nom_feature', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# smote = SMOTE(random_state=42, sampling_strategy='auto')
# x_train, y_train = smote.fit_resample(preprocessor.fit_transform(x_train), y_train)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectPercentile(score_func = chi2, percentile=80)),
    ('classifier', RandomForestClassifier())
])

model.fit(x_train, y_train)

Y_pred = model.predict(x_test)

print("Classification report:\n", classification_report(y_test, Y_pred))
print("Accuracy: ", accuracy_score(y_test, Y_pred))

test_df = pd.read_csv('C:/Users/Qann/OneDrive - Phenikaa Univesity/machine-learning/Data/LoanPredition/test.csv')
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

test_df_transformed = preprocessor.transform(test_df)
pred_test = model.predict(test_df_transformed)

# final_output = pd.DataFrame({'id': test_ids, 'LoanStatus': pred_test})
# final_output.to_csv('predictionLoan2.csv', index=False)


# train:
#                precision    recall  f1-score   support
#
#            0       0.95      0.98      0.97     10051
#            1       0.98      0.95      0.97     10067
#
#     accuracy                           0.97     20118
#    macro avg       0.97      0.97      0.97     20118
# weighted avg       0.97      0.97      0.97     20118

