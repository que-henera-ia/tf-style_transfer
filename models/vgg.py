import tensorflow as tf


# TODO: CREATE MY OWN VGG NET AND DOWNLOAD ITS WIEGHTS SO I CAN HAVE IT OFFLINE
def vgg19_layers(layer_names, weights="imagenet"):
  """ Creates a VGG model that returns a list of intermediate output values (layers)."""

  # Load our model. Load pretrained VGG, trained on ImageNet data.
  # Load a VGG19 without the classification head
  vgg = tf.keras.applications.VGG19(include_top=False, weights=weights)
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


def load_and_test_vgg(image, weights="imagenet"):
    # LOAD VGG19 AND TEST IT
    x = tf.keras.applications.vgg19.preprocess_input(image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights=weights)
    prediction_probabilities = vgg(x)
    print(prediction_probabilities.shape)
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    print(predicted_top_5)