import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("/Data/StudentScore.xls")

# First I will create a file that contains all the information about the data.

# in this file i can see correlation coefficient between features is really hight
# Profile = ProfileReport(df, title = "Student_report")
# Profile.to_file("StudentReport.html")
# now i want to see correlation coefficient in the data between features
# print(df[["math score", "reading score", "writing score"]].corr())
# as i said before the correlation coefficient is quite high so i will lean towards using Regressor

# now i will divide data
target = "math score"
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# now i will transform data and filling missing data

# first is column has numeric data.
num_transform = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=3)),  # use KNNImputer for filling missing data
    ('scaler', StandardScaler())  # tranform data
])

#second is column has nominal data

# The length of the variables in the columns shown in the file is quite low which is one of reasons i chose onhotencoder
nom_transform = Pipeline(steps=
                         [('imputer', SimpleImputer(strategy='most_frequent')),  # Fill in the most common value type
                          ('Onhot', OneHotEncoder(sparse_output=False))
])

#third is column has ordinal data and i combine boolean data in here

education_level = ["associate's degree", "bachelor's degree", "high school", "master's degree",
                   "some college", "some high school"]

gender = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()

prep_courses = x_train["test preparation course"].unique()

ord_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_level, gender, lunchs, prep_courses]))])

# I just finished converting the data and will continue to apply those to the corresponding columns.
# ColumnTransformer is very suitable for it
preprocess = ColumnTransformer(transformers=[
    ('num_feature', num_transform, ['reading score', 'writing score']),
    ("ord_feature", ord_transform, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transform, ["race/ethnicity"])
])

# oke i will show the result of boolean data
result = preprocess.fit_transform(x_train)
#data have shape like:
    # reading,      writing,    educa   gender  lunch   cours  |        race/ethicity               |
    # -0.314662,    -0.100740,  4.0,    0.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0
    # -0.451732,    -0.429958,  4.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0
    # -0.451732,    -0.232427,  0.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0
    # 1.330181,     1.742875,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    1.0


# use LazyRegressor for finding best Regressor model
# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)

# Create a complete pipeline with preprocessing and regressor
model = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('regressor', LinearSVR(random_state=42))
])

# after running LazyRegressor it shows LinearSVR is the best model so I will choose it
# i want to choice best metric for this model so use GridSearchCV is the best
# Increase the max_iter in your parameters
params = {
    "regressor__epsilon": [0.1, 0.2],  # the range within which the predicted value can fall without being penalized
    "regressor__loss": ["epsilon_insensitive", "squared_epsilon_insensitive"], # Choose the type of loss function to optimize
    "regressor__C": [0.1, 1, 10],  # balance between optimizing the loss function and model complexity
    "regressor__max_iter": [1000, 2000, 5000]  # The maximum number of times the optimization algorithm will iterate
}

# Fitting the grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

grid_search.fit(x_train, y_train)

# Show best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# get best model
best_model = grid_search.best_estimator_

# Get predictions
y_pred =best_model.predict(x_test)

# user metric like r2_score to comment model
print("MAE : {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE : {}".format(mean_squared_error(y_test, y_pred)))
print("R2 : {}".format(r2_score(y_test, y_pred)))

# R2 : 0.8578482052595553

#because r2_score is higher than o.8 so the the model has good expected value and fits the data well.