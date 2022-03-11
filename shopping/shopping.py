import csv
from ipaddress import collapse_addresses
import sys
from numpy import matrix

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.DictReader(f)

        evidence = []
        labels = []
        for row in reader:            
            # Month should be 0 for January, 1 for February, 2 for March, etc. up to 11 for December.
            # row['Month'] = row['Month'].replace({'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11})
            mths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for mth_num in range(len(mths)):
                if row['Month'] == mths[mth_num]:
                    row['Month'] = mth_num
                    break
                    
            # VisitorType should be 1 for returning visitors and 0 for non-returning visitors.
            row['VisitorType'] = 1 if row['VisitorType'] == 'Returning_Visitor' else 0

            # Weekend should be 1 if the user visited on a weekend and 0 otherwise.
            row['Weekend'] = 1 if row['Weekend'] == 'TRUE' else 0

            # Administrative, Informational, ProductRelated, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, and Weekend should all be of type int.
            for col in ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']:
                row[col] = int(row[col])

            # Administrative_Duration, Informational_Duration, ProductRelated_Duration, BounceRates, ExitRates, PageValues, and SpecialDay should all be of type float.
            for col in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']:
                row[col] = float(row[col])

            evidence.append(list(row.values())[:17])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    # Train model on training set
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # sensitivity, specificity = evaluate(y_test, predictions)
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            if labels[i] == 1:
                tp += 1
            if labels[i] == 0:
                tn += 1
        else:
            if labels[i] == 1:
                fn += 1
            if labels[i] == 0:
                fp += 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # using confusion matrix
    # tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    # sensitivity = tp / (tp + fn)
    # specificity = tn / (tn + fp)
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
