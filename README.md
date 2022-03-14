# CLIKC Recsys App
This is the CLIKC Recommender System Application repository for development.
It has been developed by using LightFM for the recommender part and FastAPI
for the microservice part.
It is composed by two microservices:
- Model Service: related to the recommender system. It communicates with the Recsys Interface Service only (via Docker Compose ports configuration).
- Recsys Interface Service: it receives requests and forwards them to the Model Service.
Moreover a NGINX server is used.
Each microservice is delivered via Docker containers by using Docker Compose.

### How to run it
- Install docker and docker-compose.
- Clone the repository and go to the main folder.
- Open the Terminal application and run "docker-compose up -d". In the end open your internet browser.

### Model Service Endpoints (used INTERNALLY by Recsys Interface Service only)
- Go to http://localhost:8080/api/v1/model/docs (GET) to get the documentation about Model Service.
- Go to http://localhost:8080/api/v1/model/status (GET) to get the status of the Model Service.
- Go to http://localhost:8080/api/v1/model/train (POST) to manually train the model of the Model Service.
- Go to http://localhost:8080/api/v1/model/recommendations/user/{user_id} (GET) to get recommendations for a given user.
- Go to http://localhost:8080/api/v1/model/recommendations/user/features (POST) to get recommendations for a new user given a list of features (features must belong to the dataset used during training).
- Go to http://localhost:8080/api/v1/model/recommendations/item/{item_id} (GET) to get similar items via cosine similarity.

### Notes
- Automatic training is set to run at 1:30 AM (CET).
- One training request can be accepted at a time. Next calls will be ignored until current training is done.
- Background training is performed via APScheduler library.
- Communication between microservices is performed via AIOHTTP library.

### References
- LightFM Documentation: https://making.lyst.com/lightfm/docs/home.html
- FastAPI Documentation: https://fastapi.tiangolo.com/
- APScheduler Documentation: https://apscheduler.readthedocs.io/en/3.x/index.html
- AIOHTTP Documentation: https://docs.aiohttp.org/en/stable/
- Docker Compose Documentation: https://docs.docker.com/compose/


