# Foolbox Tutorial
Foolbox is a Python toolbox to create adversarial examples that fool neural networks. It requires Python, NumPy and SciPy.

## Installation
```
pip3 install foolbox
```
## Documentation
 readthedocs: <br>http://foolbox.readthedocs.io/</br>
 paper: <br> https://arxiv.org/abs/1707.04131</br>

## Tutorial
### Create a model
```
# instantiate the model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104,116,123]),1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0,255), preprocessing=preprocessing)
```
### Get the Image
Use foolbox utility
```
# get source image and label
image, label = foolbox.utils.imagenet_example()

print('label', label)
print('predicted class', np.argmax(fmodel.predictions(image[:,:,::-1]))) 

```
or

Download your own Image 
& Use image.load_img or cv2.imread
```
# load image 
## inception save image will lost one dimension
image_path = 'panda.jpg'
raw_image = image.load_img(image_path, target_size=(224,224))

# preprocessing the image
input_image = image.img_to_array(raw_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = preprocess_input(input_image)
```

### Specify the criterion
Foolbox's attack can be untargeted or targeted based on the criterion
#### Untargeted
|Misclassification   |Defines adversarials as images for which the predicted class is not the original class.   |
|---|---|
|TopKMisclassification   |Defines adversarials as images for which the original class is not one of the top k predicted classes.   |
|OriginalClassProbability   |Defines adversarials as images for which the probability of the original class is below a given threshold.   | 
|ConfidentMisclassification	   |Defines adversarials as images for which the probability of any class other than the original is above a given threshold.   | 

Ex:
```
from foolbox.criteria import Misclassification
criterion = Misclassification()
```
p.s. All Attacks set Misclassification() as default 

#### Targeted
|TargetClass   |Defines adversarials as images for which the predicted class is the given target class.   |
|---|---|
|TargetClassProbability   |Defines adversarials as images for which the probability of a given target class is above a given threshold.   |

Ex:
```
from foolbox.criteria import TargetClassProbability
criterion = TargetClassProbability(282, p=.5)
```
p.s. LBFG-Attack is the only targeted attack I see in foolbox's readthedocs

### Applying the attack
```
# apply attack on source image

attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image[:,:,::-1], label)
```

### Take Prediction
Print the most likely class number
```
print('adversarial class', np.argmax(fmodel.predictions(adversarial)))
```

we can even decode the top k class to see what they really are
```
#decode the class

print(np.argmax(fmodel.predictions(adversarial)))
print(foolbox.utils.softmax(fmodel.predictions(adversarial))[789])
adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))
```

### Save the Adversarial Example
```
import cv2
cv2.imwrite("adv.png",adversarial)
```

or

```
import scipy.misc
scipy.misc.toimage(adversarial[:,:,::-1], cmin=0.0, cmax=255.0).save('adv_cat.jpg')
```
### Plot the adversarial image
![](https://i.imgur.com/w5F2StT.png)

```
import matplotlib.pyplot as plt
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
```





