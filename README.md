# **Deploy a Machine Learning model to Cloud Application Platform with FastAPI**

Working on every OS because I have setup it in Docker container.

# Environment Set up

- Install Docker first.

- Run:

  - For prod:

  ```
  docker build -t ml_app .
  ```

  - For dev, use docker compose because it's easy to map the volume:

  ```
  docker build -t ml_app .

  docker compose up
  ```

## Repositories

- Create a directory for the project and initialize git.
  - As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
- Connect your local git repo to GitHub.
- Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
  - Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data

- Store in **/data** folder.

# Model

- Store in **/model** folder.

# API Creation

- Define in **main.py** file.

- There are 2 endpoints:

  - **/**: Default with a welcome string.

  - **/predict**: give the request's body a Census data as example show in FastAPI docs.

- Access the docs with:

  - Local URL: **http://127.0.0.1/docs**

  **OR**

  - After deployment: "https://{domain}/docs"
