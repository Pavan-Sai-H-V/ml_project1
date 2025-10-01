from datapreprocessing import prepare_data
from model_training import train_linear_regression,eval_model

def main():

    X_train, X_test, y_train, y_test = prepare_data()
    model = train_linear_regression(X_train, y_train)

    mse = eval_model(model, X_test, y_test)

    print(f"The Mean Squared Error of the model is: {mse:.4f}")


if __name__ == "__main__":
    main()