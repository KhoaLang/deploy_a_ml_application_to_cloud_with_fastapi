from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import hydra

# from starter.ml.data import process_data
# from starter.ml.model import inference

import uvicorn


app = FastAPI()


class CensusData(BaseModel):
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



@app.get(path="/")
def index():
    return {"message": "A machine learning deployment project!"}


# @app.post(path="/infer")
# # @hydra.main(config_path=".", config_name="config", version_base="1.2")
# async def predict(input_data: CensusData) -> Dict[str, str]:
#     """
#     Example function for returning model output from POST request.
#     Args:
#         input_data (BasicInputData) : Instance of a BasicInputData object. Collected data from
#         web form submission.
#     Returns:
#         dict: Dictionary containing the model output.
#     """
#     with hydra.initialize(config_path=".", version_base="1.2"):
#         conf = hydra.compose(config_name="config")

#     [encoder, lab_bin, model] = pickle.load(
#         open(conf["main"]["model_path"], "rb")
#     )

#     temp = {x: y for x, y in input_data.dict(by_alias=True).items()}
#     input_dataframe = pd.DataFrame(
#         temp, index=[0]
#     )

#     processed_data, _, _, _ = process_data(
#         X=input_dataframe,
#         categorical_features=conf['main']['cat_features'],
#         label=None,
#         training=False,
#         encoder=encoder,
#         lb=lab_bin
#     )

#     prediction = inference(model, processed_data)
#     return {"Output": ">50K" if int(prediction[0]) == 1 else "<=50K"}


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000)
