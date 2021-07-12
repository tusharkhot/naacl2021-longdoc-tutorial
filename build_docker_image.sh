set -e

IMAGE_NAME=longformer_tutorial
DOCKERFILE_NAME=Dockerfile

# Image name
GIT_HASH=`git log --format="%h" -n 1`
IMAGE=${IMAGE_NAME}_tushark-${GIT_HASH}
echo $IMAGE
docker build -f $DOCKERFILE_NAME -t $IMAGE .

echo -e "\033[0;32m Built image $IMAGE. Now run: \033[0m"
echo -e "\033[0;35m beaker image create --name=$IMAGE --description \"NumNet Repo; Git Hash: $GIT_HASH\" $IMAGE \033[0m"
