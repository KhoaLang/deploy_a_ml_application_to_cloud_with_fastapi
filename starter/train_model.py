
import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

try:
    from starter.ml.data import process_data
    from starter.ml.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_sliced_data
    )
except ModuleNotFoundError:
    sys.path.append('./')
    from starter.ml.data import process_data
    from starter.ml.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_sliced_data
    )


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def training(config: DictConfig):
    """
    Trains a machine learning model and saves it.
    """
    logger.info(f"Config: {config}")

    SLICED_OUTPUT_PATH = config['main']['SLICED_OUTPUT_PATH']
    CATEGORY_FEATURES = config['main']['cat_features']
    MODEL_PATH = config['main']['model_path']
    TEST_SIZE = config['main']['test_size']
    DATA_PATH = config['main']['data_path']
    LABEL = config['main']['label']

    logger.info("Reading training data...")
    df = pd.read_csv(DATA_PATH)

    logger.info("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE)

    logger.info("Processing data...")
    X_train, y_train, encoder, label_binarizer = process_data(
        train_df, categorical_features=CATEGORY_FEATURES, label=LABEL, training=True)

    X_test, y_test, encoder, label_binarizer = process_data(
        X=test_df,
        categorical_features=CATEGORY_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=label_binarizer
    )

    logger.info("Training model...")
    model = train_model(X_train, y_train)
    logger.info(model)

    logger.info("Saving model...")
    if not os.path.exists("model/"):
        os.mkdir("model/")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump([encoder, label_binarizer, model], f)
    logger.info("Model saved.")

    logger.info("Inference model...")
    preds = inference(model, X_test)

    logger.info("Calculating model metrics...")
    prec, rec, f_beta_score = compute_model_metrics(y_test, preds)
    logger.info(f">>>Precision: {prec}")
    logger.info(f">>>Recall: {rec}")
    logger.info(f">>>Fbeta: {f_beta_score}")

    logger.info("Calculating model metrics on slices data...")
    computed_metrics = compute_metrics_with_sliced_data(
        df=test_df,
        category_cols=CATEGORY_FEATURES,
        label=LABEL,
        encoder=encoder,
        label_binarizer=label_binarizer,
        model=model,
        sliced_data_output_path=SLICED_OUTPUT_PATH
    )
    logger.info(f">>>Metrics with slices data: {computed_metrics}")
