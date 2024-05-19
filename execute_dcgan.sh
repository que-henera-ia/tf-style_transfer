#sudo docker build -t my_tensorflow_image .
sudo docker run --gpus all -v $(pwd):/app -v $(pwd)/tf-style_transfer.py:/app/tf-style_transfer.py my_tensorflow_image
