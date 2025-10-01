from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_linear_regression(X_train,y_train):

    model=LinearRegression()

    model.fit(X_train,y_train)

    return model

def eval_model(model,X_test,y_test):
    y_pred=model.predict(X_test)
    error_mse=mean_squared_error(y_test,y_pred)
    return error_mse

