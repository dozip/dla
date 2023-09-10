import utils as lib

import math
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate(metrics, Y_test, Y_pred):
    for m in metrics:

        try:
            if m == 'MAE':
                mae = mean_absolute_error(Y_test, Y_pred)
                print(f"MAE: {mae}")
            elif m == 'RMSE':
                rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
                print(f"RMSE: {rmse}")
            elif m == 'R_2':
                r_squared = r2_score(Y_test, Y_pred)
                print(f"R-squared: {r_squared}")
            if m == 'accuracy':
                accuracy = accuracy_score(Y_test, Y_pred)
                print("Accuracy:", accuracy)
            elif m == 'precision_score':
                precision = precision_score(Y_test, Y_pred, average='macro')
                print("Precision:", precision)              
            elif m == 'recall_score':
                recall = recall_score(Y_test, Y_pred, average='macro')
                print("Recall:", recall)    
            if m == 'f1_score':
                f1 = f1_score(Y_test, Y_pred, average='macro')
                print("F1-Score:", f1)
            elif m == 'confusion_matrix':
                cm = confusion_matrix(Y_test, Y_pred)
                print("Confusion Matrix:")
                print(cm)
            elif m == 'report':
                report = classification_report(Y_test, Y_pred)
                print("Classification Report:")
                print(report)
        
        except: print(m + ' is false metric for the chosen model')
"""
    The 'evaluate' method facilitates the computation and display of various performance metrics for evaluating a machine learning model. 
    The method performs the computations and prints the results for the selected metrics. It also includes error handling to ensure the program does not crash when invalid metrics are specified.

    Parameters:
    metrics (list of strings): A list of metrics to be calculated and displayed, such as 'MAE,' 'RMSE,' 'R^2,' 'accuracy,' 'precision_score,' 'recall_score,' 'f1_score,' 'confusion_matrix,' or 'report.'
    Y_test (array-like): The actual test data.
    Y_pred (array-like): The model-predicted data.

    Returns:
    None
"""

def show_residual_plot(residuals, Y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_pred, residuals, c='b', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
"""
    Generates a residual plot to visualize the model's prediction errors.

    Parameters:
    residuals (array-like): The residual values (actual - predicted).
    Y_pred (array-like): The predicted values from the machine learning model.

    Returns:
    None
"""

def train_the_model(model, X_train, Y_train):
    

    if model == 'neural_network_regression':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Eingabeschicht mit der Anzahl der Features
            tf.keras.layers.Dense(64, activation='relu'),      # Versteckte Schicht mit 64 Neuronen und ReLU-Aktivierung
            tf.keras.layers.Dense(32, activation='relu'),      # Versteckte Schicht mit 32 Neuronen und ReLU-Aktivierung
            tf.keras.layers.Dense(1)                           # Ausgabeschicht mit 1 Neuron (Regression)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        # Modelltraining
        model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=2)  # Anzahl der Epochen anpassen
        return 'Regression', model
        
    elif model == 'neural_network_classification': 
        num_classes = len(np.unique(Y_train))

        # Erstellen des neuronalen Netzwerkmodells
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Eingabeschicht mit der Anzahl der Features
            tf.keras.layers.Dense(64, activation='relu'),      # Versteckte Schicht mit 64 Neuronen und ReLU-Aktivierung
            tf.keras.layers.Dense(32, activation='relu'),      # Versteckte Schicht mit 32 Neuronen und ReLU-Aktivierung
            tf.keras.layers.Dense(num_classes, activation='softmax')  # Ausgabeschicht für Klassifikation
        ])

        # Kompilieren des Modells
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Modelltraining
        model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=2)
        return 'Classification', model

    else:
        model.fit(X_train, Y_train)

        str_model = ""

        if(isinstance(model, SVC) or isinstance(model, DecisionTreeClassifier) or isinstance(model, RandomForestClassifier)):
            str_model = 'Classification'
        else: str_model = 'Regression' 

        return str_model, model
"""
    Trains a machine learning model and returns the trained model.

    Parameters:
    model: str or machine learning model
        If 'neural_network', a neural network model is created and trained.
        Otherwise, the provided model is trained.
    X_train: array-like
        Training features.
    Y_train: array-like
        Training target variable.

    Returns:
    trained_model: The trained machine learning model.
    Regression or Classification - important for evaluation
"""

def show_coefficients(model):
    try:
        coefficients = model.coef_
        intercept = model.intercept_

        print("Coefficients:", coefficients)
        print("Intercept:", intercept)
    except:
        print("No coefficients to show!")
"""
    Prints the coefficients and intercept of a linear regression model.

    Parameters:
    model: machine learning model
    
    This function retrieves the coefficients and intercept from a machine learning model
    and prints them to the console. If an exception occurs during retrieval, it prints
    a message indicating that there are no coefficients to show.

    Returns:
    None
"""

def balance_data(dataframe, strategy, n_iterations, target):

    new_data = pd.DataFrame()
    sampler = RandomOverSampler(random_state=42) # Default Value
    
    if strategy == 'oversampling':
        print('Oversampling')
    elif strategy =='undersampling':
        print('Undersampling')
        sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        
    elif strategy == 'smote':
        print('Smote')
        sampler = SMOTE(random_state=42)
    
    elif strategy == 'bootstrapping':

        print('Bootstrapping')
        # Liste zum Speichern der Bootstrapped-Stichproben
        bootstrapped_samples = []

        # Durchführen des Bootstrappings
        for _ in range(n_iterations):
            # Zufällige Stichprobe mit Ersatz aus dem DataFrame ziehen
            bootstrap_sample = dataframe.sample(n=len(dataframe), replace=True, random_state=42)
            bootstrapped_samples.append(bootstrap_sample)

        # Den neuen DataFrame erstellen, der die Bootstrapped-Stichproben enthält
        return pd.concat(bootstrapped_samples, ignore_index=True)

    else: 
        print("This strategy is not known...") 
        return dataframe
    
    X_resampled, Y_resampled = sampler.fit_resample(dataframe.drop(target, axis=1), dataframe[target])
    combined_df = pd.concat([X_resampled, Y_resampled], axis=1)
    combined_df.rename(columns={0: target}, inplace=True)
    return combined_df
"""
    Balances a dataset using various strategies including Oversampling, Undersampling, SMOTE, or Bootstrapping.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing features and target variable.
    strategy (str): The strategy for data balancing ('oversampling', 'undersampling', 'smote', or 'bootstrapping').
    n_iterations (int): Number of iterations for bootstrapping (only applicable if strategy='bootstrapping').
    target (str): Name of the target feature

    Returns:
    pd.DataFrame: DataFrame with balanced data based on the chosen strategy, or the original DataFrame if strategy is not recognized.
"""    

def select_the_features(dataframe, strategy, alpha, n_features, target):
    
    new_data = pd.DataFrame()

    if strategy=='Lasso':
        
        sel_ = SelectFromModel(Lasso(alpha=alpha, random_state=10))
        sel_.fit(dataframe.drop(target, axis=1), dataframe[target])

        for feature in sel_.get_feature_names_out():
            new_data[feature] = dataframe[feature]
        new_data[target] = dataframe[target]

    elif strategy =='PCA':
          
        # Initialisieren Sie den PCA-Modell
        pca = PCA(n_components=n_features)

        # Führen Sie PCA auf Ihren Daten durch
        new_data = pd.DataFrame(pca.fit_transform(dataframe))

        #  erklärte Varianz durch jede Hauptkomponente
        explained_variance_ratio = pca.explained_variance_ratio_

    else: print("This strategy is not known...")
 
    return new_data
"""
    Selects features from a DataFrame using strategy 'Lasso regression' or 'PCA'.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing features and target variable.
    strategy (str): The strategy for feature selection ('Lasso' or 'PCA').
    alpha (float): Alpha value for Lasso regularization.
    n_features (int): Number of components for PCA.
    target (str): Name of the target feature

    Returns:
    pd.DataFrame: DataFrame with selected features based on the chosen strategy.
"""

def OneHotEncoder(dataframe, columns):
    # Check if there are any categorical columns with 'object' data type and select them
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    categorical_columns_with_values = [col for col in categorical_columns if dataframe[col].nunique() > 1]

    # Apply one-hot encoding to categorical columns with categorical values
    if categorical_columns_with_values:
        encoded_categorical = pd.get_dummies(dataframe[categorical_columns_with_values])
        dataframe = pd.concat([dataframe.drop(categorical_columns_with_values, axis=1), encoded_categorical], axis=1)

    # Remove the changed columns from list columns
    for col in categorical_columns_with_values:
        columns.remove(col)
    print(columns)

    return dataframe, columns
"""
    Apply one-hot encoding to categorical columns in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing categorical columns.
    columns (list): List of columns in the DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded categorical columns.
    list: Updated list of columns after one-hot encoding.
"""

def handle_missing_data(dataframe, columns, strategy, missing_values):
  
    # Fill missing values in each column based on the mode of groups defined by "mistriage" and "KTAS_expert"
    if strategy=='mean':
        for col in columns:
            # numeric values got the mean value, categoric the most common value
            try: dataframe[col] = dataframe[col].transform(lambda x: x.fillna(x.mean()))
            except: dataframe[col] = dataframe[col].transform(lambda x: x.fillna(x.mode()[0]))
    
    elif strategy=='mode':
        # grouping the data based on the unique combinations of values in the "mistriage" and "KTAS_expert" columns
        # create a separate group of rows for each unique combination of "mistriage" and "KTAS_expert" values
        grouped_data = dataframe.groupby(["mistriage", "KTAS_expert"])    
        for col in columns:
            dataframe[col] = grouped_data[col].transform(lambda x: x.fillna(x.mode()[0]))
    
    elif strategy =='delete': 
            for col in missing_values: 
                dataframe = dataframe.drop(columns=col) 
                columns.remove(col)
           
    elif strategy=='estimate': print('Sorry, but this feature is not ready yet...')

    else: print("Unknown strategy for handling missing values")
       
    return dataframe, columns
"""
    This function handles missing values in a Pandas DataFrame using various strategies.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    columns (list): List of columns to be processed.
    strategy (str): The strategy to handle missing values ('mean', 'mode', 'delete', 'estimate').
    missing_values (list): List of columns with missing values.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    list: List of columns with values, which are still there.
"""

def identify_missing_values(dataframe, columns):
   # Replace the '#NULL!' and '??' values in the columns with NaN (missing value)
    dataframe[columns] = dataframe[columns].replace(["#NULL!", "??"], np.NaN)

    # Find columns with missing values
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    
    # Calculate the count of missing values for each column
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    
    # Calculate the ratio of missing values for each column as a percentage
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    
    # Create a DataFrame to display the missing values count and ratio
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    
    # Print the DataFrame showing missing values information
    print(missing_df)
    
    # Return the list of column names with missing values
    return variables_with_na
"""
    This function calculates and displays information about missing values in a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    columns (list): List of columns in the DataFrame.

    Returns:
    list: List of column names with missing values.
"""

def transformObjectToFloat(dataframe):

    # Convert commas to periods in columns with decimal comma separators
    dataframe.replace({',': '.'}, regex=True, inplace=True)

    # List of columns with 'object' data type containing numeric strings
    object_columns_with_numbers = dataframe.select_dtypes(include=['object']).apply(pd.to_numeric, errors='coerce').notna().all()

    # List of columns with numeric strings that need to be converted to 'float'
    columns_to_convert = object_columns_with_numbers[object_columns_with_numbers].index

    # Convert selected columns to 'float'
    dataframe[columns_to_convert] = dataframe[columns_to_convert].astype(float)
"""
    Convert numeric strings with decimal comma separators to float values in a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    None
"""

def show_distribution(dataframe, feature_names: str):
    if len(feature_names) == 1:
        sns.histplot(data=dataframe, x=feature_names[0])
    else:
        fig, axes = plt.subplots(nrows=len(feature_names), ncols=1, figsize=(12,8), dpi=300)
        for i, name in enumerate(feature_names):
            sns.histplot(data=dataframe, x=name, ax=axes[i])
"""
    Show the distribution of one or more features in a Pandas DataFrame using histograms.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    feature_names (str or list): Name(s) of the feature(s) to visualize.

    Returns:
    None
"""

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
"""
    Calculate the lower and upper outlier detection thresholds for a variable in a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    variable (str): Name of the variable for which to calculate the thresholds.

    Returns:
    float: Lower threshold for outlier detection.
    float: Upper threshold for outlier detection.
"""

def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            low_limit, up_limit = outlier_thresholds(dataframe, col)
            if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
                number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
                print(col, ":", number_of_outliers)
                variable_names.append(col)
                if plot:
                    sns.boxplot(x=dataframe[col])
                    plt.show()
    return variable_names
"""
    Check for the presence of outliers in numeric columns of a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    num_col_names (list): List of numeric column names to check for outliers.
    plot (bool): Whether to plot box plots for columns with outliers.

    Returns:
    list: List of column names with outliers.
"""

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
"""
    Replace outlier values in a variable of a Pandas DataFrame with outlier detection thresholds.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    variable (str): Name of the variable for which to replace outlier values.

    Returns:
    None
"""

def plot_histogram(names, data):
    size = math.ceil(len(names)/2)
   
    if size < 2:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        for name in names:
            sns.distplot(data[name], bins = 20, ax=ax[x]) 
            x = x + 1
    else:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        y=0
        for name in names:
            sns.distplot(data[name], bins = 20, ax=ax[x,y]) 
            y = (y + 1) % 2  # Wechsel zwischen 0 und 1
            if y == 0:
                x = x + 1
def plot_countplot(names, data):
    size = math.ceil(len(names)/2)
    if size < 2:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        for name in names:
            sns.countplot(x=name, data=data, ax=ax[x])
            x = x + 1
    else:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        y=0
        for name in names:
            sns.countplot(x=name, data=data, ax=ax[x,y])
            y = (y + 1) % 2  # Wechsel zwischen 0 und 1
            if y == 0:
                x = x + 1
def plot_boxplot(names, data):
    size = math.ceil(len(names)/2)
    if size < 2:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        for name in names:
            sns.boxplot(x=name, data=data, ax=ax[x]) 
            x = x + 1
    else:
        fig, ax = plt.subplots(size, 2, figsize=(15,size*5))
        x=0
        y=0
        for name in names:
            sns.boxplot(x=name, data=data, ax=ax[x,y]) 
            y = (y + 1) % 2  # Wechsel zwischen 0 und 1
            if y == 0:
                x = x + 1
"""
    The following three functions plot diagramms for multiple features in a Pandas DataFrame.
    The visualization is displayed in a grid layout, with the number of rows determined by the number of features
    and the number of columns set to two.

    Parameters:
    names (list): List of variable names to plot histograms for.
    data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    None
"""