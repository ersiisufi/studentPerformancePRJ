from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))
    
