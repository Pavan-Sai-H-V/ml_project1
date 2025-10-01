from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(n_samples=100, test_size=0.2, random_state=42):
    X=2 * np.random.rand(n_samples, 1)
    y=1 + 2 * X + np.random.randn(n_samples, 1)
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test