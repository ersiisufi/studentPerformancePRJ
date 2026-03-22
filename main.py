from src.preprocess import load_data, preprocess
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)




if __name__ == "__main__":
    main()