# Deploy machine learning using Docker and Flask

---------------------------------------------------------------------------------------------------------------------------------------------------

## 1. What?
This is an academic poject. The goal is using Flask or horoku with Docker to deploy a simple model seen in others course.
- We choose to use Flask app and a saved model (in pickle format) to predict given dataset.
- One container run for ui and one other for model.


## 2. How to run it?
```
# Build du docker
docker-compose build

# lunch le docker
docker-compose up

# Output of results
# connect to http://localhost for client
# connect to http://localhost:5000 for server
```
