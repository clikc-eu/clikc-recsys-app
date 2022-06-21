# CLIKC Recsys App
This is the CLIKC Recommender System Application repository for development.
It has been developed by using LightFM for the recommender part and FastAPI
for the microservice part.

It is composed of two microservices:
- Model microservice: related to the recommender system. It communicates with the Backend Interface microservice only (via Docker Compose ports configuration).
- Backend Interface microservice: it receives requests and forwards them to the Model microservice. When it gets the recommendations it will send them to the application that made the request.
Moreover a NGINX server is used as a reverse proxy. It represents the TLS termination point. Model and Backend Interface microservices are supposed to run inside the same host machine; so no TLS communication is provided here.
In order to communicate with the Backend Interface microservice from the outside an authentication is required. This is performed by sending a token via the HTTP header.
Communication between Model and Backend Interface microservice requires authentication; this is performed by sending a token via the HTTP header.

Each microservice is delivered via Docker containers by using Docker Compose.

### How to run it
- Install Docker and Docker Compose.
- Clone the repository and go to the main folder (docker-compose.yml level).
- Open the Terminal application here and run "docker-compose up -d".
- In the end open your internet browser and open one of links here below.

WARNING: In order to run the project you need some configuration files which have not been uploaded to this repository.

Note: for a fast usage go to https://localhost:8080/api/v1/recsys-interface/docs.

### Backend Interface microservice endpoints (MUST USE THESE)
- Go to https://localhost:8080/api/v1/recsys-interface/docs (GET) to get the documentation about.
- Go to https://localhost:8080/api/v1/recsys-interface/status (GET) to get service status.
- Go to https://localhost:8080/api/v1/recsys-interface/train (POST) to manually train the model of Model Service.
- Go to https://localhost:8080/api/v1/recsys-interface/recommendations/user/{user_id} (GET) to get recommendations for a given user via its id.
- Go to https://localhost:8080/api/v1/recsys-interface/recommendations/user/features (POST) to get recommendations for a new user given a list of features (features must belong to the dataset used during training).
- Go to https://localhost:8080/api/v1/recsys-interface/recommendations/item/{item_id} (GET) to get similar items via cosine similarity.

### Model microservice Endpoints (used INTERNALLY by Backend Interface microservice to communicate with Model microservice)
- http://localhost:8080/api/v1/model/docs (GET).
- http://localhost:8080/api/v1/model/status (GET).
- http://localhost:8080/api/v1/model/train (POST).
- http://localhost:8080/api/v1/model/recommendations/user/{user_id} (GET).
- http://localhost:8080/api/v1/model/recommendations/user/features (POST).
- http://localhost:8080/api/v1/model/recommendations/item/{item_id} (GET).

### Notes
- Automatic training is set to run at 1:30 AM (CET).
- One training request can be accepted at a time. Next calls will be ignored until current training is done.
- Background training is performed via APScheduler library.
- Communication between microservices is performed via AIOHTTP library.
- Keyword extraction is performed via spaCy.
- Json schema (local dataset) validation is performed via 
jsonschema.

### Credits & Documentations
- LightFM Documentation: https://making.lyst.com/lightfm/docs/home.html
- LightFM repository: https://github.com/lyst/lightfm
- FastAPI Documentation: https://fastapi.tiangolo.com/
- FastAPI repository: https://github.com/tiangolo/fastapi
- APScheduler Documentation: https://apscheduler.readthedocs.io/en/3.x/index.html
- APScheduler repository: https://github.com/agronholm/apscheduler
- AIOHTTP Documentation: https://docs.aiohttp.org/en/stable/
- AIOHTTP repository: https://github.com/aio-libs/aiohttp
- spaCy Documentation: https://spacy.io
- spaCy repository: https://github.com/explosion/spaCy
- jsonschema Documentation: https://python-jsonschema.readthedocs.io/en/stable/#
- jsonschema repository: https://github.com/python-jsonschema/jsonschema
- Docker Compose Documentation: https://docs.docker.com/compose/
- Docker Compose repository: https://github.com/docker/compose


