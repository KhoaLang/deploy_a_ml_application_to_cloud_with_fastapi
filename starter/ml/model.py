import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(max_iter=1000, random_state=8071)
    lr.fit(X_train, y_train.ravel())
    return lr


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Args:
        y (np.array): Known labels, binarized.
        preds (np.array): Predicted labels, binarized.
    Returns:
        precision : float
        recall : float
        fbeta : float
    """
    f_beta_score = fbeta_score(y, preds, beta=1, zero_division=1)
    rec = recall_score(y, preds, zero_division=1)
    prec = precision_score(y, preds, zero_division=1)
    return prec, rec, f_beta_score


def inference(model, X):
    """ Run model inferences and return the predictions.

    Args:
        model (LogisticRegression): Trained machine learning model.
        X (np.array): Data used for prediction.

    Returns:
        preds (np.array): Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_with_sliced_data(
        df, category_cols, label, encoder, label_binarizer, model, sliced_data_output_path):
    """
    Compute metrics of the model on slices of the data

    Args:
        df (pd.DataFrame): Input dataframe
        category_cols (list): list of categorical columns
        label (str): Class label string
        encoder (OneHotEncoder): fitted One Hot Encoder
        label_binarizer (LabelBinarizer): label binarizer
        model (starter.ml.model): Trained model binary file
        sliced_data_output_path (str): path to save the slice output

    Returns:
        metrics (pd.DataFrame): Dataframe containing the metrics
    """
    rows_list = list()
    for feature in category_cols:
        for category in df[feature].unique():
            row = {}
            tmp_df = df[df[feature] == category]

            x, y, _, _ = process_data(
                X=tmp_df,
                categorical_features=category_cols,
                label=label,
                training=False,
                encoder=encoder,
                lb=label_binarizer
            )

            preds = inference(model, x)
            precision, recall, fbeta = compute_model_metrics(y, preds)

            row['feature'] = feature
            row['precision'] = precision
            row['recall'] = recall
            row['f1'] = fbeta
            row['category'] = category
            rows_list.append(row)

    metrics = pd.DataFrame(
        rows_list,
        columns=[
            "feature",
            "precision",
            "recall",
            "f1",
            "category"])
    metrics.to_csv(sliced_data_output_path, index=False)
    return metrics
