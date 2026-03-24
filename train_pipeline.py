from src.preprocess import preprocess, load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model

def main ():
    print("Training pipeline started...")

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("Training pipeline finsihed.")
    

if __name__ == "__main__":
    main()
