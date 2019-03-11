import keras
import foolbox
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image

# load image 
## inception save image will lost one dimension
image_path = 'panda.jpg'
raw_image = image.load_img(image_path, target_size=(224,224))

# preprocessing the image
input_image = image.img_to_array(raw_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = preprocess_input(input_image)

#print(input_image)
inception_clfr = InceptionV3(include_top=True,
                              weights='imagenet',
			     classes=1000)

clfr_output = inception_clfr.predict(input_image)

# Decode the pred
print('Before attack:', decode_predictions(clfr_output, top=3))
clfr_prediction = np.argmax(clfr_output)
print(clfr_prediction)

# Then we introduce foolbox
import foolbox
from foolbox.models import KerasModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack

image_path = 'panda.jpg'
raw_image = image.load_img(image_path, target_size=(224,224))
input_image = image.img_to_array(raw_image)

# Define fmodel as Keras inception v3
fmodel = KerasModel(inception_clfr, bounds=(0,255))

# Apply the attack
attack = LBFGSAttack(model=fmodel, criterion=TargetClassProbability(282, p=.5)) 
adversarial = attack(input_image, label=282)
#print(adversarial[np.newaxis,:,:,:])

from scipy.misc import imsave
#imsave('origin_panda.jpg',input_image)
#imsave('adv_panda.jpg',adversarial)

clfr_output = inception_clfr.predict(adversarial[np.newaxis,:,:,:])
print('After attack:', decode_predictions(clfr_output))
print(np.argmax(clfr_output))



# plot the difference
input_image = image.array_to_img(input_image)
advexp_image = image.array_to_img(adversarial)
difference = adversarial - input_image

import matplotlib.pyplot as plt
plt.figure()

plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Adversarial Image')
plt.imshow(advexp_image)
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Difference')
plt.imshow(difference/abs(difference).max()*0.2+0.5)
plt.axis('off')

plt.show()

