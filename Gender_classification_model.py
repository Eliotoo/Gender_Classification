import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Display settings for PyCharm
desired_width=420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

# Import data set
df_titanic = pd.read_csv("train.csv")
# Locate missing values
print(df_titanic.isnull().sum())
print(df_titanic['Embarked'].value_counts(dropna=False))

# Replace 2 missing values with the most common category
df_titanic['Embarked'].fillna(value="S",inplace=True)
#df_titanic.dropna(subset=['Embarked'], how='any')

# Create new feature for passengers having cabins
df_titanic = df_titanic.assign(HasCabin=(df_titanic.Cabin.notnull()).astype(int) )
# Drop columns
df_titanic = df_titanic.drop(columns=['Name', 'Cabin', 'PassengerId', 'Ticket'])
# Make dummies for categorical variables
df_titanic = pd.get_dummies(df_titanic, columns=['Embarked', 'Pclass'], drop_first=True)

# Encode target variable. 1 = Male, 0 = Female
class_le = preprocessing.LabelEncoder()
y = pd.DataFrame(class_le.fit_transform(df_titanic['Sex'].values) )
# Assign input variable
X = df_titanic.loc[:, df_titanic.columns != 'Sex']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=1)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Imput missing values in age variable
Imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(Imputer.fit_transform((X_train)))
X_test = pd.DataFrame(Imputer.transform(X_test))
#X_val = pd.DataFrame(Imputer.transform(X_val))

#Random Forest model
RF_model = RandomForestClassifier(random_state=1)
RF_model.fit(X_train, y_train.values.ravel())
RF_pred = RF_model.predict(X_test)
print("Random Forest accuracy =",accuracy_score(y_test, RF_pred)*100,"%")

#Random Forest Grid Search tune

RF_grid = {
            'n_estimators': [100, 500],
            'max_depth': [10, 30],
            'min_samples_split': [2, 4],
            'max_features': ['sqrt', 'log2']
}
grid = GridSearchCV(RandomForestClassifier(random_state=1), param_grid=RF_grid, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
grid_model = grid.fit(X_train, y_train.values.ravel())
best_estimator = grid_model.best_estimator_
best_pred_y = best_estimator.predict(X_test)
print("Random forest grid search accuracy =",accuracy_score(y_test, best_pred_y)*100,"%")
print("Best results achieved using %s" % (grid_model.best_params_))



# Create and fit neutral network model
MLPC_model = MLPClassifier(random_state=1, max_iter=1000)
MLPC_model.fit(X_train, y_train.values.ravel())
MLPC_pred = MLPC_model.predict(X_test)
print("Neutral Network accuracy =",accuracy_score(y_test, MLPC_pred)*100,"%")

# Neutral Network GridSeach tune

NN_grid = {
            'hidden_layer_sizes':[50, 100],
            'solver': ['adam'],
            'max_iter': [800, 1000, 1500],
            'alpha': [1e-4, 1e-5, 2e-5]
}

grid = GridSearchCV(MLPClassifier(random_state=1), param_grid=NN_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_model = grid.fit(X_train, y_train.values.ravel())
best_estimator = grid_model.best_estimator_
MLPC_pred_grid = best_estimator.predict(X_test)
print("Neutral network grid search accuracy =",accuracy_score(y_test, MLPC_pred_grid)*100,"%")
print("Best results achieved using %s" % (grid_model.best_params_))

