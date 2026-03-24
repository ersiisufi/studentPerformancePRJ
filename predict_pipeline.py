from src.utils import load_model

def main():
    print("Prediction Pipeline started...")

    model = load_model()

    new_Data = [[0,1,3,4,6,7,8,]]

    predicition = model.predict(new_Data)

    print("Prediction: ", predicition)


    print("Prediction Pipeline finished")

if __name__ == "__main__":
    main()