import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import mpld3
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

magic_number = 93 #40005393

class ManageFile:
    def __init__(self, path:str) -> None:
        self.data_path = path

    def read_data(self):
        dataset = pd.read_csv(self.data_path)
        data = pd.DataFrame(dataset)
        return data

class PlotData:
    def __init__(self) -> None:
        pass

    def draw_heat_map(self, data, show:bool = False):
        for column in data.select_dtypes(include=['object']).columns.tolist():
            data[column] = LabelEncoder().fit_transform(data[column])

        fig = plt.subplots(figsize=(13, 13))
        sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
        if show:
            plt.show()

        self.save_fig(fig, "heat_map")

    def draw_pair_plot(self, data, show:bool = False):
        fig = plt.subplots(figsize=(13, 13))
        sns.pairplot(data.iloc[:, :5])
        if show:
            plt.show()
        self.save_fig(fig, "pair_plot")

    def save_fig(self, fig, name):
        html_content = mpld3.fig_to_html(plt.gcf())
        full_screen_html = f"""
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                #chart {{
                    max-width: 100%;
                    max-height: 100%;
                    width: auto;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div id="chart">{html_content}</div>
        </body>
        </html>
        """
        with open(name + ".html", "w", encoding="utf-8") as file:
            file.write(full_screen_html)

class PreProcessing:
    def __init__(self) -> None:
        self.select_flag = False

    def clean_data(self, data):
        df_cleaned = data[~data.isin(['Unknown']).any(axis=1)]
        return df_cleaned

    def calc_num_of_classes(self, data, column: str, plot:bool = False):
        print(data[column].value_counts())
        if plot:
            data[column].value_counts().plot(kind="pie")
            plt.show()

    def under_sampling_augmantation(self, data, column:str, majer:str, minor: str):
        df_major = data[data[column] == majer]
        df_minor = data[data[column] == minor]

        df_major_sample = resample(df_major, replace=True, n_samples=len(df_minor), random_state=magic_number)
        balanced_df = pd.concat([df_major_sample, df_minor])
        return balanced_df

    def split_data(self, data, column):
        data.sample(frac=1, random_state=magic_number)
        Y = data[column]
        del data[column]
        X = data

        X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, random_state=magic_number)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5, random_state=magic_number)
        return {"train" : (X_train, Y_train), "val" : (X_val, Y_val), "test" : (X_test, Y_test)}

    def normalize_encoding(self, data):
        scaler = RobustScaler()
        encoder = LabelEncoder()

        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        encoded_features = pd.DataFrame()
        for col in categorical_columns:
            encoded_col = encoder.fit_transform(data[col])
            encoded_features[col] = encoded_col

        numeric_scaled = scaler.fit_transform(data[numeric_columns])
        numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_columns)

        final_data = pd.concat([encoded_features, numeric_scaled_df], axis=1)

        return final_data

    def select_best_features(self, data):
        self.select_flag = True

        model = RandomForestClassifier(n_estimators=500, random_state=magic_number)
        X_train, y_train = data
        model.fit(X_train, y_train)

        feature_importances = model.feature_importances_

        important_features = [feature for feature, importance in zip(X_train.columns, feature_importances) if importance > 0.005]

        X_train = X_train[important_features]
        model = LogisticRegression(max_iter=1000)
        selector = SequentialFeatureSelector(model, n_features_to_select="auto", direction="forward")
        selector.fit(X_train, y_train)

        return X_train.columns[selector.get_support()]

    def delet_feature(self, data, features:list):
        for column in features:
            del data[column]
        return data

class Model:
    def __init__(self) -> None:
        self.select_flag = False

    def svm_train(self, data, column:list = []):
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        if self.select_flag:
            X_train = X_train[column]
            X_test = X_test[column]

        model_svm = SVC(kernel='linear')
        model_svm.fit(X_train, y_train)

        y_pred = model_svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
        confusion_matrix_df.columns = ["predict class 0", "predict class 1"]
        confusion_matrix_df.index = ["ground truth class 0", "ground truth class 1"]
        print(confusion_matrix_df)

    def grd_train(self, data, column:list = []):
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        if self.select_flag:
            X_train = X_train[column]
            X_test = X_test[column]

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
        confusion_matrix_df.columns = ["predict class 0", "predict class 1"]
        confusion_matrix_df.index = ["ground truth class 0", "ground truth class 1"]
        print(confusion_matrix_df)

    def xgb_train(self, data, column:list = []):
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]
        X_test,  y_test = data["test"]

        if self.select_flag:
            X_train = X_train[column]
            X_val = X_val[column]
            X_test = X_test[column]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'eta': 0.1,
            'eval_metric': 'logloss'
        }

        epoch = 1000
        watchlist = [(dtrain, 'train'), (dval, 'eval')]

        results = {}
        model = xgb.train(params, dtrain, epoch, watchlist, early_stopping_rounds=10, evals_result=results)

        plt.figure(figsize=(10, 6))
        plt.plot(results['train']['logloss'], label='Train')
        plt.plot(results['eval']['logloss'], label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('Log Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid()
        plt.show()

        y_pred = model.predict(dtest)
        y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]

        accuracy = accuracy_score(y_test, y_pred_binary)
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

        print("Classification Report:")
        print(classification_report(y_test, y_pred_binary))

        print("Confusion Matrix:")
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_binary))
        confusion_matrix_df.columns = ["predict class 0", "predict class 1"]
        confusion_matrix_df.index = ["ground truth class 0", "ground truth class 1"]
        print(confusion_matrix_df)

if __name__ == "__main__":
    file = ManageFile("BankChurners.csv")
    dataset = file.read_data()

    plotter = PlotData() 

    # plotter.draw_heat_map(dataset, True)
    # plotter.draw_pair_plot(dataset, True)

    preprocesser = PreProcessing()
    
    bad_feature = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
    
    dataset = preprocesser.delet_feature(dataset, bad_feature)

    clean_data = preprocesser.clean_data(dataset)

    preprocesser.calc_num_of_classes(clean_data, "Attrition_Flag", True)

    balanced_data =  preprocesser.under_sampling_augmantation(clean_data, "Attrition_Flag", "Existing Customer", "Attrited Customer")

    normlize_encoded_data = preprocesser.normalize_encoding(balanced_data)

    splited_data = preprocesser.split_data(normlize_encoded_data, "Attrition_Flag")

    # select_beast_feature = preprocesser.select_best_features(splited_data["train"])
    
    model = Model()
    # model.grd_train(splited_data)
    # model.xgb_train(splited_data)
    # model.train(splited_data)
