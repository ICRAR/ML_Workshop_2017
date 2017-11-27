# Building Docker

This docker image starts with the standard Ubuntu 

Build the container with Pytorch and Tensorflow

`docker build --tag ml_workshop_2017_environment_:latest .`  

As Docker and git don't play nice together we need to tell docker to not cache the git clone 

`docker build --no-cache --tag ml_workshop_2017:latest .`  


`docker tag <id> kevinvinsen/ml_workshop_2017:latest`

You only need to do this to login in when pushing
`docker login`   

`docker push kevinvinsen/ml_workshop_2017`
