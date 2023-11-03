# Build the docker file 
docker build -t digits:v1 -f ./Dockerfile .
# Mount our volume to models directory (where train data is stored)
docker run -v /home/pravin/mlops-23/mlops2/Models:/digits/models digits:v1