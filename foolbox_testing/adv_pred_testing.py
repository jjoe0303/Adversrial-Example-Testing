import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt

# instantiate the model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104,116,123]),1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0,255), preprocessing=preprocessing)

# get source image and label
image, label = foolbox.utils.imagenet_example()

print('label', label)
print('predicted class', np.argmax(fmodel.predictions(image[:,:,::-1]))) 

# apply attack on source image

attack = foolbox.attacks.FGSM(fmodel)
#attack = foolbox.attacks.SaltAndPepperNoiseAttack(fmodel,criterion = foolbox.criteria.ConfidentMisclassification(0.8))
#adversarial = attack(image[:,:,::-1], label)
adversarial = attack(image[:,:,::-1], label,epsilons = 50)

print('adversarial class', np.argmax(fmodel.predictions(adversarial)))

# save the adv_example
import scipy.misc
scipy.misc.imsave("Adv_example.png",adversarial[:,:,::-1])


# plot the difference

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image / 255)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:,:,::-1] / 255)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()
