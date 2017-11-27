# Notebooks

To run the notebooks from the command line

`docker run --rm -i -t -p 8888:8888 kevinvinsen/ml_workshop_2017 /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/ML_Workshop_2017/notebooks --ip='*' --port=8888 --no-browser --allow-root"`  


If you need to removed stopped containers

` docker rm $(docker ps -a -q)`
