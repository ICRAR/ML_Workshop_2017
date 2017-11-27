# Building Docker

This docker image starts with the standard Ubuntu 

`docker build --tag ml_workshop_2017:latest .`

`docker tag  f22ac18be217 kevinvinsen/ml_workshop_2017:latest`

`docker login`  You only need to do this 

`docker run -i -t -p 8888:8888 kevinvinsen/ml_workshop_2017 /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/ML_Workshop_2017/notebooks --ip='*' --port=8888 --no-browser --allow-root"`  
