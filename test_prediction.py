#from resnet50 import ResNet50
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

#model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_test = preprocess_input(x)

#preds = model.predict(img_test)
#print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
