import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class Preprocess :
    def __init__(self, path) -> None:
        self.path = path
        
    def read_data(self):
        data = np.load(self.path)
        return data
    
    def split_data(self, data):
        scaler = RobustScaler()

        X = np.arange(len(data))[:, np.newaxis]
        Y = data[:, np.newaxis]

        X = scaler.fit_transform(X)
        Y = scaler.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        plt.figure()
        plt.plot(X, Y, color="orange", label="raw data")
        plt.scatter(X_train, Y_train, color="blue", label="train data")
        plt.scatter(X_test, Y_test, color="red", label="test data")
        plt.legend()
        plt.show()

        return {"train": (X_train.T, Y_train.T), "test": (X_test.T, Y_test.T)}

class Model:
    def __init__(self) -> None:
        pass
        
    def linear_regression(self, data: dict):
        x_train, y_train = data["train"]
        x_test, y_test = data["test"]

        w = 5
        b = 10

        mse_train_errors = []
        mse_test_errors = []
        r2_train_errors = []
        r2_test_errors = []
        for epoch in range(200):
            y_pred = w * x_train + b

            y_pred_test = w * x_test + b

            gradiant_weight = -2 * np.mean((y_train - y_pred) * x_train)
            gradiant_bise = -2 * np.mean(y_train - y_pred)

            w -= 0.1 * gradiant_weight
            b -= 0.1 * gradiant_bise

            mse_train = mean_squared_error(y_train, y_pred)
            mse_train_errors.append(mse_train)
            r2_train = r2_score(y_train[0], y_pred[0]) 
            r2_train_errors.append(r2_train)

            mse_test = mean_squared_error(y_test, y_pred_test)
            mse_test_errors.append(mse_test)
            r2_test = r2_score(y_test[0], y_pred_test[0]) 
            r2_test_errors.append(r2_test)

            plt.figure("mse")
            plt.clf()
            plt.plot(mse_train_errors, color="blue", label="mse train error")
            plt.plot(mse_test_errors, color="red", label="mse test error")
            plt.title(f"mse over epoch {epoch}")
            plt.xlabel("Epoch")
            plt.ylabel("error")
            plt.legend()
            plt.pause(0.025)

            plt.figure("r2")
            plt.clf()
            plt.plot(r2_train_errors, color="blue", label="r2 train error")
            plt.plot(r2_test_errors, color="red", label="r2 test error")
            plt.title(f"r2 over epoch {epoch}")
            plt.xlabel("Epoch")
            plt.ylabel("error")
            plt.legend()
            plt.pause(0.025)

            plt.figure("Regressor Line")
            plt.clf()
            plt.title(f"Pridect Line over epoch {epoch}")
            plt.scatter(x_train, y_train, color="green", label="Ground truth")
            plt.scatter(x_train, y_pred, color="red", label="Prediction")
            plt.legend()
            plt.pause(0.025)

        plt.show()

    def linear_regression_batch(self, data: set, num: int, process_show: bool = False):
        """_summary_

        Args:
            data (set): A dictionary containing training and testing data with the following keys:
                - "train": A tuple (x_train, y_train) where:
                    * x_train: numpy array of shape (features, samples) for training input.
                    * y_train: numpy array of shape (1, samples) for training labels.
                - "test": A tuple (x_test, y_test) where:
                    * x_test: numpy array of shape (features, samples) for testing input.
                    * y_test: numpy array of shape (1, samples) for testing labels.

            num (int): The number of features (dimensions) in the input data.
            process_show (bool, optional): If True, displays real-time plots for training loss 
                                           and regression performance. Defaults to False..
        """
        generated_data = self.add_term(data, num)
        x_train, y_train = generated_data["train"]
        x_test, y_test = generated_data["test"]

        w = np.full((1, num), -0.1) 
        b = np.array([0.4])  
        if num > 1:
            learning_rate = (1 / (10 ** 2))
        else: 
            learning_rate = 0.1

        buffer_train_X = []
        buffer_train_y = []

        buffer_test_X = []
        buffer_test_y = []

        error_train_list = []
        error_test_list = []
        
        X_test = x_test
        Y_test = y_test
        buffer_test_X.append(X_test)
        buffer_test_y.append(Y_test)

        for X_train, Y_train in zip(x_train.T, y_train.T):
            X_train =  X_train[:, np.newaxis]

            buffer_train_X.append(X_train)
            buffer_train_y.append(Y_train)
            X_train = np.hstack(buffer_train_X)  
            Y_train = np.array(buffer_train_y).reshape(1, -1) 
            
            X_test = np.hstack(buffer_test_X)  
            Y_test = np.array(buffer_test_y).reshape(1, -1)

            mse_train_list = []
            mse_test_list = []
            num_epochs = 200

            for i in range(num_epochs):
                y_pred = np.dot(w, X_train) + b 
                y_pred_test = np.dot(w, X_test) + b  

                error = y_pred - Y_train  
                gradient_w = (2 / X_train.shape[1]) * np.dot(error, X_train.T) 
                gradient_b = (2 / X_train.shape[1]) * np.sum(error)  

                loss_train = np.mean((y_pred - Y_train) ** 2)
                loss_test = np.mean((y_pred_test - Y_test) ** 2)

                mse_train_list.append(loss_train)
                mse_test_list.append(loss_test)

                w = w - (learning_rate * gradient_w)
                b -= learning_rate * gradient_b

                if process_show:
                    plt.figure("epoch")
                    plt.clf()
                    plt.plot(mse_train_list, color="blue")
                    plt.title(f"Loss over epochs {i}" )
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.pause(0.025)

                    y_pred_plot = np.squeeze(y_pred)
                    plt.figure("Regressor Line")
                    plt.clf()
                    plt.scatter(X_train[0, :], Y_train, color="green", label="Ground truth")
                    plt.scatter(X_train[0, :], y_pred_plot, color="red", label="Prediction")
                    plt.title(f"train in {X_train.shape[1]} number of points")
                    plt.legend()
                    plt.pause(0.025)

                if loss_train < 0.05:
                    mse_train = mean_squared_error(Y_train, y_pred)
                    if len(Y_train[0]) > 1:
                        r2_train = r2_score(Y_train[0], y_pred[0]) 
                    else:
                        r2_train = float('nan')
                    error_train_list.append({"mse": mse_train, "r2": r2_train})

                    mse_test = mean_squared_error(Y_test, y_pred_test)
                    if len(Y_train[0]) > 1:
                        r2_test = r2_score(Y_test[0], y_pred_test[0]) 
                    else:
                        r2_test = float('nan')
                    error_test_list.append({"mse": mse_test, "r2": r2_test})
                    break

                if i >= (num_epochs - 1):
                    mse_train = mean_squared_error(Y_train, y_pred)
                    if len(Y_train[0]) > 1:
                        r2_train = r2_score(Y_train[0], y_pred[0]) 
                    else:
                        r2_train = float('nan')
                    error_train_list.append({"mse": mse_train, "r2": r2_train})

                    mse_test = mean_squared_error(Y_test, y_pred_test)
                    if len(Y_train[0]) > 1:
                        r2_test = r2_score(Y_test[0], y_pred_test[0]) 
                    else:
                        r2_test = float('nan')
                    error_test_list.append({"mse": mse_test, "r2": r2_test})

        plt.figure("errors")
        plt.title(f"train vs test (mse error)")
        plt.xlabel("number of data")
        plt.ylabel("mse error")
        plt.plot([entry["mse"] for entry in error_train_list], color="blue", label="train error")
        plt.plot([entry["mse"] for entry in error_test_list], color="red", label="test error")
        plt.legend()
        plt.show()

        plt.figure("errors")
        plt.title(f"train vs test (r2 error)")
        plt.xlabel("number of data")
        plt.ylabel("r2 score")
        plt.plot([entry["r2"] for entry in error_train_list], color="blue", label="train error")
        plt.plot([entry["r2"] for entry in error_test_list], color="red", label="test error")
        plt.legend()
        plt.show()

    def add_term(self,data ,num):
        x_train, y_train = data["train"]
        X_test, y_test = data["test"]
        new_x_train = x_train
        new_X_test = X_test
        for power in range(2, num + 1):
            new_x_train = np.vstack((new_x_train, (x_train ** power)))
            new_X_test = np.vstack((new_X_test, (X_test ** power)))

        return {"train": (new_x_train, y_train), "test": (new_X_test, y_test)}

    def sk_regressor(self, data):
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        poly_features = PolynomialFeatures(degree=3)
        X_poly_train = poly_features.fit_transform(X_train)
        X_poly_test = poly_features.transform(X_test)
        poly_model = LinearRegression()
        poly_model.fit(X_poly_train, y_train)
        y_poly_pred = poly_model.predict(X_poly_test)

        svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
        svr_model.fit(X_train, y_train.ravel())
        y_svr_pred = svr_model.predict(X_test)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train.ravel())
        y_rf_pred = rf_model.predict(X_test)

        models = {
            "Polynomial Regression": y_poly_pred,
            "SVR": y_svr_pred,
            "Random Forest Regressor": y_rf_pred
        }

        model_names = []
        mse_values = []
        r2_values = []
        
        for name, y_pred in models.items():
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_names.append(name)
            mse_values.append(mse)
            r2_values.append(r2)
            print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

            plt.scatter(X_test, y_test, color="blue", label="Test Data")
            plt.scatter(X_test, y_pred, color="red", label="Predictions")
            plt.title(f"{name} (R2: {r2:.2f})")
            plt.xlabel("X")
            plt.ylabel("y")
            plt.legend()
            plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(model_names, mse_values, color='skyblue', label='MSE')
        plt.title('Mean Squared Error (MSE) Comparison')
        plt.xlabel('Models')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(model_names, r2_values, color='lightcoral', label='R2 Score')
        plt.title('$R^2$ Score Comparison')
        plt.xlabel('Models')
        plt.ylabel('$R^2$')
        plt.legend()
        plt.show()
if __name__ == "__main__":
    preprocesser = Preprocess("data.npy")
    data = preprocesser.read_data()
    data = preprocesser.split_data(data)

    model = Model()
    # model.linear_regression(data)
    # model.linear_regression_batch(data, 5, True)
    # model.sk_regressor(data)

