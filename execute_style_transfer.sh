# sudo docker build -t my_tensorflow_image .
sudo docker run --gpus all -v $(pwd):/app -v $(pwd)/main.py:/app/main.py my_tensorflow_image
