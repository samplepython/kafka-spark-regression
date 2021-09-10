import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import sys

def load_csv_dataset(path_to_file, file_name):
    ''' Loads the csv file from fiven path '''
    data = pd.read_csv(path_to_file + '/' + file_name)
    return data

def preprocess_data(data_frame):
    ''' handling preprocessing the data'''
    X = data_frame.copy()
    y = X.pop("SalePrice")
  

    #-----Feature Engineering-----
    X['TotalSquareFootage'] = X['TotalBsmtSF'] + X['GrLivArea']

    # Converting float64 and categorical to int64
    # float_columns = X.select_dtypes(np.float64)
    # LotFrontage MasVnrArea GarageYrBlt
    for float_column in X.select_dtypes(np.float64):
        X[float_column] = X[float_column].fillna(0).astype(np.int64)
        X[float_column].astype(np.int64)

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    #discrete_features = X.dtypes == int
    return X, y

def make_mi_scores(X, y):
    ''' caluculating the mutual information scores of features vs target and returning'''
    mi_scores = mutual_info_regression(X, y) #, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    ''' plotting mutual information scores'''
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.figure(dpi=100, figsize=(8, 5))
def select_high_mi_score_features(mi_scores, no_of_columns):
    ''' selecting only top features, those having high scores'''
    mi_scores_df = pd.DataFrame(mi_scores.head(no_of_columns))
    selected_features = list(mi_scores_df.index)
    return selected_features

def score_dataset_XGGradient(X, y, model=XGBRegressor()):
    # performing XGGradient to minimise the loss 
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Scoring Root Mean Squared Log Error
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def embedding_plot(X, y, title):
    ''' ploting the feature analysis'''
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10,10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=y/10.)

    plt.xticks([]), plt.yticks([])
    plt.legend(handles=sc.legend_elements()[0], labels=[i for i in range(10)])
    plt.title(title)
    plt.show()
def train_the_RandomForestRegressor_model(X, y,test_sample=0.2):
    ''' training the data with RandomForestRegressor model'''
    # Splitting the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample, random_state=1)

    # Applying the RandomForestRegressor on train dataset
    rf_model_object = RandomForestRegressor(n_estimators=50, random_state=1).fit(X_train,y_train)
    return rf_model_object, X_test, y_test

def begin():
    ''' Begining of the program in standalone application'''
    if(len(sys.argv) == 3 ):
        df = load_csv_dataset(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python <python-file> <absolute-path-to-csv-file> <csv-file-name>')

    X, y = preprocess_data((df))

    #mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores = make_mi_scores(X, y)
    mi_scores[::1].head(21)

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)

    X = X[select_high_mi_score_features(mi_scores, 20)]

    XGGradient_score = score_dataset_XGGradient(X,y)

    # PCA Analysis
    X_pca = PCA(n_components=2).fit_transform(X)
    embedding_plot(X_pca, y, "PCA")
    plt.show()

    # LDA Analysis
    X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    embedding_plot(X_lda, y, "LDA")
    plt.show()


    rf_model_object, X_test, y_test = train_the_RandomForestRegressor_model(X, y, 0.3)

    # predicting the SalePrice on test data
    predicted_sale_prices = rf_model_object.predict(X_test)

    Analyse = X_test.copy()
    Analyse['Actual_SalePrice'] = y_test
    Analyse['Pridicted_SalePrice'] = predicted_sale_prices
    Analyse['difference_in_SalePrice'] = Analyse.Actual_SalePrice - Analyse.Pridicted_SalePrice
    Analyse['Accuracy-in-Percentage'] = (Analyse.Pridicted_SalePrice / Analyse.Actual_SalePrice) *  100
    Analyse.to_csv('Analysis-test-data.csv')

if __name__ == '__main__':
    begin()
