def ask_user_for_metric(is_regression: bool) -> str:
    """Interactively ask the user which metric to optimize."""

    if is_regression:
        print("\nChoose a metric to optimize:")
        print("1) rmse")
        print("2) mae")
        print("3) mse")
        print("4) r2")
        print("5) mape")

        choice = input("Enter 1–5: ").strip()

        mapping = {
            "1": "rmse",
            "2": "mae",
            "3": "mse",
            "4": "r2",
            "5": "mape",
        }

        return mapping.get(choice, "rmse")

    else:
        print("\nChoose a metric to optimize:")
        print("1) accuracy")
        print("2) f1")
        print("3) precision")
        print("4) recall")

        choice = input("Enter 1–4: ").strip()

        mapping = {
            "1": "accuracy",
            "2": "f1",
            "3": "precision",
            "4": "recall",
        }

        return mapping.get(choice, "accuracy")
