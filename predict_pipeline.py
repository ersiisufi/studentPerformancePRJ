from src.utils import load_model

def main():
    print("Prediction Pipeline started...")

    model = load_model()

    print("Enter student Data: ")

    id = float(input(""))
    gender = float(input("Gender 1,0: "))
    lunch = float(input("Lunch 1,0: "))
    reading = float(input("Reading: "))
    writing = float(input("Writing: "))
    parental = float(input("Parental level of educ 1-3: "))
    prep = float(input("Test preparation 1,0; "))

    new_Data = [[id, gender, lunch, reading, writing, parental, prep]]

    predicition = model.predict(new_Data)
    print("Prediction: ", predicition)



if __name__ == "__main__":
    main()