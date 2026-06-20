import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)


def main():
    # obhet argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="prepare data")
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("--predict", action="store_true", help="predict outcomes")
    parser.add_argument("--save", action="store_true", help="save model")
    parser.add_argument("--load", action="store_true", help="load model")
    # analyse
    args = parser.parse_args()
    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("data prepared")
    if args.train:
        # preparer les donnees
        x_train, x_test, y_train, y_test = prepare_data()
        # train model
        model = train_model(x_train, y_train)
        # entregistrer model entrainé
        save_model(model)
        print("model trained and saved")
    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)
    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        predict(x_test)
    if args.save:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("model trained and saved")
    if args.load:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()


if __name__ == "__main__":
    main()
