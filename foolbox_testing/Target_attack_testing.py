import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import matplotlib.pyplot as plt
import cv2

# instantiate the model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

image, label = foolbox.utils.imagenet_example()

#img = cv2.imread('example.png',0)
#height, width = img.shape[:2]

#print(width,' ',height)
# run the attack
#attack = foolbox.attacks.DeepFoolAttack(fmodel)
attack = LBFGSAttack(model=fmodel, criterion=TargetClassProbability(789, p=0.5))
adversarial = attack(image[:, :, ::-1], label)

# save image(jpg & png problem)
#import scipy.misc
#scipy.misc.imsave('origin_cat.png', image)
#scipy.misc.imsave('adv_cat.png', adversarial[:,:,::-1])
#scipy.misc.toimage(image, cmin=0.0, cmax=255.0).save('cat.jpg')
#scipy.misc.toimage(adversarial[:,:,::-1], cmin=0.0, cmax=255.0).save('adv_cat.jpg')

cv2.imwrite("origin.png",image[:,:,::-1])
cv2.imwrite("adv.png",adversarial)

# show results
print(np.argmax(fmodel.predictions(adversarial)))
print(foolbox.utils.softmax(fmodel.predictions(adversarial))[789])
adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))

# plot the diff

plt.figure()

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(image/255)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('adv')
plt.imshow(adversarial[:,:,::-1]/255)
plt.axis('off')

plt.subplot(1,3,3)
plt.title('dif')
diff = adversarial[:,:,::-1] - image
plt.imshow(diff/abs(diff).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()

