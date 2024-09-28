import requests
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

base_url = "https://deploy-a-ml-application-to-cloud-with.onrender.com"
endpoint = "/predict_predict_post"

response = requests.post(
    url=base_url+endpoint,
    data={
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
)

logging.info(response.status_code)
logging.info(response.json())