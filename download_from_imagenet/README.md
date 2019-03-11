# ImageNet Download using python

## ImageNet
* image database organized according to the WordNet hierarchy
* useful resource for researchers, educators, students
* has 14,197,122 images <b>with 21841 synsets indexed</b> 

## Download Tutorial
![](https://i.imgur.com/yKhL0LU.png)

![](https://i.imgur.com/wFESI4u.png)

![](https://i.imgur.com/hGCcxZp.png)

![](https://i.imgur.com/LJsUWRm.png)
```
import numpy as np
from bs4 import BeautifulSoup
import requests
import cv2
import PIL.Image
import urllib


# get source image and label
# download the image

page = requests.get("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02123394")

##print(page.content)

# BeautifulSoup is an HTML parsing library

soup = BeautifulSoup(page.content, 'html.parser') #puts the content of the website into the soup variable, each url on a different line

str_soup = str(soup) # convert soup to string to be split
type(str_soup)
split_urls = str_soup.split('\r\n') # split so each url is a different position on a list
##print(len(split_urls)) # to know how many image we have

```

![](https://i.imgur.com/krdqfV7.png)
```
!mkdir /content/train #create the Train folder
!mkdir /content/train/pcat #create the persian cat folder
!mkdir /content/validation
!mkdir /content/validation/pcat #create the persian cat folder
```

![](https://i.imgur.com/Fie8E5A.png)
```
img_rows, img_cols = 32, 32 #number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)#format to store the images (rows, columns,channels) called channels last

def url_to_image(url):
 # download the image, convert it to a NumPy array, and then read
 # it into OpenCV format
 resp = urllib.request.urlopen(url)
 image = np.asarray(bytearray(resp.read()), dtype="uint8")
 image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 # return the image
 return image

n_of_training_images=100 #the number of training images to use

for progress in range(n_of_training_images):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print(progress)
    if not split_urls[progress] == None:
      try:
        I = url_to_image(split_urls[progress])
        if (len(I.shape))==3: #check if the image has width, length and channels
          save_path = '/content/train/pcat/img'+str(progress)+'.jpg'#create a name of each image
          cv2.imwrite(save_path,I)
      except:
        None
```

Result:
![](https://i.imgur.com/VuLqf5G.png)
![](https://i.imgur.com/HxQo96m.png)









