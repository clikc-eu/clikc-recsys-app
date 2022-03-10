# clikc-recsys-app
This is the CLIKC Recommender System Application repository for development.

### How to run it
- Install docker and docker-compose.
- Clone the repository and go to the main folder.
- Open the Terminal application and run "docker-compose up -d". In the end open your internet browser.
- Go to http://localhost:8080/api/v1/model/docs to get the documentation about Model Service.
- Go to http://localhost:8080/api/v1/model/status to get the status of the Model Service.
- Go to http://localhost:8080/api/v1/model/recommendations/user/{user_id} to get recommendations for a given user.
- Go to http://localhost:8080/api/v1/model/recommendations/item/{item_id} to get similar items via cosine similarity.

