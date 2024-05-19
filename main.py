from models.tf_style_transfer import StyleContentModel
from models.vgg import *
from tf_utils.manage_data import *
import time


# LOAD IMAGES AND SAVE THEM
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
# content_path = "test_sample_original.png"
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
# style_path = "cat.png"

content_image = load_img(content_path)
style_image = load_img(style_path)
plt.subplot(1, 2, 1)
save_img(content_image[0], 'ou')
plt.subplot(1, 2, 2)
save_img(style_image[0], 'Style Image')


'''
So why do these intermediate outputs within our pretrained image classification network allow us to define style and content representations?

At a high level, in order for a network to perform image classification (which this network has been trained to do), it must understand the image. 
This requires taking the raw image as input pixels and building an internal representation that converts the raw image pixels into a complex 
understanding of the features present within the image.

This is also a reason why convolutional neural networks are able to generalize well: they’re able to capture the invariances and defining features 
within classes (e.g. cats vs. dogs) that are agnostic to background noise and other nuisances. Thus, somewhere between where the raw image is fed 
into the model and the output classification label, the model serves as a complex feature extractor. By accessing intermediate layers of the model, 
you're able to describe the content and style of input images.
'''

# Choose intermediate layers from the network to represent the style and content of the image:
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight=1e-2
content_weight=1e4

# Create the model
extractor = StyleContentModel(style_layers, content_layers,style_weight=1e-2,content_weight=1e4)
results = extractor(tf.constant(content_image))

#Look at the statistics of each layer's output
print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())


# Define style and content targets
extractor.style_targets = extractor(style_image)['style']
extractor.content_targets = extractor(content_image)['content']

# Run gradient descent
image = tf.Variable(content_image)

extractor.train_step(image)
extractor.train_step(image)
extractor.train_step(image)
image_out=tensor_to_image(image)
save_img(image_out,"img")



start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    extractor.train_step(image)
    print(".", end='', flush=True)
  save_img(tensor_to_image(image),"img"+str(n))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))


save_gif('style_transfer')