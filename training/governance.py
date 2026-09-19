def validate_model(accuracy: float, threshold: float = 0.95) -> bool:
    """
    Validate whether model evaluation metric satisfies deployment threshold.
    """
    return accuracy >= threshold


if __name__ == "__main__":
    accuracy = 0.9721

    if validate_model(accuracy):
        print("MODEL VALIDATED")
    else:
        print("MODEL REJECTED")
