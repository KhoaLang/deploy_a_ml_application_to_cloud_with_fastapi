"""
Project: Deploy a Machine Learning Model to Cloud with FastAPI
Author: khoalhd
Date: 2024-09-28
"""
from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import hydra

from .starter.ml.data import process_data
from .starter.ml.model import inference


app = FastAPI()


class Census(BaseModel):
    """
    A Pydantic model representing census data for an individual.

    Attributes:
        age (int): The individual's age.
        race (str): The individual's race.
        sex (str): The individual's sex (e.g., 'Male', 'Female').
        workclass (str): The work classification of the individual (e.g., 'State-gov').
        fnlgt (int): The final weight, a numerical value representing the number of people
                    the census entry represents.
        education (str): The individual's level of education (e.g., 'Bachelors').
        education_num (int): The numerical representation of the education level
                    (alias: 'education-num').
        marital_status (str): The marital status of the individual (alias: 'marital-status').
        occupation (str): The occupation of the individual (e.g., 'Exec-managerial').
        relationship (str): The relationship status of the individual within their household
                    (e.g., 'Husband').
        capital_gain (int): Capital gain recorded for the individual (alias: 'capital-gain').
        capital_loss (int): Capital loss recorded for the individual (alias: 'capital-loss').
        hours_per_week (int): The number of hours worked per week by the individual
                    (alias: 'hours-per-week').
        native_country (str): The country of origin for the individual (alias: 'native-country').

    Model Config:
        - Define an example of a typical census data record.
        - Usefull to show example in FastAPI docs.
    """
    age: int
    race: str
    sex: str
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 22,
                    "workclass": "State-gov",
                    "fnlgt": 215646,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Adm-clerical",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States"
                }
            ]
        }
    }


@app.get(path="/")
def index():
    """
    Return a simple welcome string
    """
    return {"message": "A machine learning deployment project!"}


@app.post(path="/predict")
async def predict(census_data: Census) -> Dict[str, str]:
    """
    Predict the salary for the input  from POST request.
    Args:
        census_data (BasicInputData) : Instance of a BasicInputData object. Collected data from
        web form submission.
    Returns:
        dict: Dictionary containing the model output.
    """
    with hydra.initialize(config_path=".", version_base="1.1"):
        conf = hydra.compose(config_name="config")

    [encoder, lab_bin, model] = pickle.load(
        open(conf["main"]["model_path"], "rb")
    )

    temp = {x: y for x, y in census_data.dict(by_alias=True).items()}
    input_dataframe = pd.DataFrame(
        temp, index=[0]
    )

    processed_data, _, _, _ = process_data(
        X=input_dataframe,
        categorical_features=conf['main']['category_features'],
        label=None,
        training=False,
        encoder=encoder,
        lb=lab_bin
    )

    prediction = inference(model, processed_data)
    return {"Output": ">50K"} if int(prediction[0]) == 1 else {
        "Output": "<=50K"}
