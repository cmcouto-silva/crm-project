# CRM Project

## Context

### Business Problem

For efficiency purposes, Betsson is trying to predict which customers are going to call the Customer Service, based on their past behaviour.

### Analytical Problem

For a given day, predict whether a customer will call the Customer Service in the following 14 days.

## Repository Structure

The main notebooks are numbered in the folder "notebooks":
- 0-eda.ipynb
- 1-feature-selection.ipynb
- 2-predictive-modeling.ipynb
- pycaret.ipynb

The predictions for the test set can be found at `predictions/test_predictions.csv`.

## Additional Resources

Additionally, I've created a web app and an API to consume the model. I used Streamlit and FastAPI, respectively.

Links:
- [Webapp](https://crm-app.datargs.com/)
- [REST API](https://crm-api.datargs.com/docs)

GitHub Repositories:
- [Webapp](https://github.com/cmcouto-silva/crm-app)
- [REST API](https://github.com/cmcouto-silva/crm-api)

Docker images:
- [Webapp](https://hub.docker.com/repository/docker/cmcoutosilva/crm-app)
- [Rest API](https://hub.docker.com/repository/docker/cmcoutosilva/crm-api)

_**Note:**_ I'm using the free resources from [Render](https://render.com/). Downtime might occur (there's limited memory usage as well).
