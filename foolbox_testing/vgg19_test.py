from keras.applications.vgg19 import VGG19
from keras.preprocessing import  image
from keras.applications.vgg19 import preprocess_input,decode_predictions
from keras.models import Model
import numpy as np
import os
base_model = VGG19(weights = 'imagenet', include_top = True)
script_dir = os.path.dirname(__file__)
rel_path = "training_network/panda.jpg"
img_path = os.path.join(script_dir, rel_path)
img = image.load_img(img_path, target_size = (224,224))
raw_image = image.img_to_array(img)
input_image = np.expand_dims(raw_image, axis = 0)
input_image = preprocess_input(input_image)
result = base_model.predict(input_image)
print(np.argmax(result))

import foolbox 
from foolbox.models import KerasModel
from foolbox.criteria import TargetClassProbability
fmodel = KerasModel(base_model, bounds = (0,255))


img = image.load_img(img_path, target_size = (224,224))
raw_image = image.img_to_array(img)

# Apply the attack
attack = foolbox.attacks.LBFGSAttack(model=fmodel, criterion=TargetClassProbability(282, p=.5)) 
##attack = foolbox.attacks.GradientAttack(model=fmodel) 
##attack = foolbox.attacks.FGSM(model=fmodel) 
##attack = foolbox.attacks.BIM(model=fmodel) 
##attack = foolbox.attacks.L1BasicIterativeAttack(model=fmodel) 
##attack = foolbox.attacks.L2BasicIterativeAttack(model=fmodel) 
##attack = foolbox.attacks.PGD(model=fmodel) 
##attack = foolbox.attacks.RandomPGD(model=fmodel) 
##attack = foolbox.attacks.MomentumIterativeAttack(model=fmodel) 
##attack = foolbox.attacks.DeepFoolAttack(model=fmodel) 
##attack = foolbox.attacks.NewtonFoolAttack(model=fmodel) ### nice
##attack = foolbox.attacks.DeepFoolL2Attack(model=fmodel) 
##attack = foolbox.attacks.DeepFoolLinfinityAttack(model=fmodel) 
##attack = foolbox.attacks.ADefAttack(model=fmodel) 
##attack = foolbox.attacks.SLSQPAttack(model=fmodel) ### not work
##attack = foolbox.attacks.SaliencyMapAttack(model=fmodel) 
##attack = foolbox.attacks.IterativeGradientAttack(model=fmodel) ### Great
##attack = foolbox.attacks.IterativeGradientSignAttack(model=fmodel) ### Great
##attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel) 
adversarial = attack(raw_image, label=388)
fresult = fmodel.predictions(adversarial)

print(np.argmax(fresult))
print('Top 5 predictions: ', decode_predictions(base_model.predict(adversarial[np.newaxis,:,:,:]), top = 5))


