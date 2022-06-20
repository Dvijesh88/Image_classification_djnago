from django.shortcuts import render
import numpy as np
from django.core.files.storage import FileSystemStorage
from keras.utils.image_utils import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.applications import vgg16

image_height = 224
image_width = 224

vgg_model = vgg16.VGG16(weights = 'imagenet')

def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)

def predictclass(request):
    fileobject = request.FILES['filepath']
    fs = FileSystemStorage()
    filepathname = fs.save(fileobject.name, fileobject)
    filepathname = fs.url(filepathname)

    testimage = '.'+filepathname
    img = load_img(testimage, target_size=(image_height, image_width))
    x = img_to_array(img)
    final_image = np.expand_dims(x,axis=0)
    
    final_image1 = vgg16.preprocess_input(final_image.copy())
    vgg_predications = vgg_model.predict(final_image1)
    label_vgg = tf.keras.applications.imagenet_utils.decode_predictions(vgg_predications)[0]
        
    #print(label_vgg[0][1])
    confidence_score = label_vgg[0][2]*100
    context = {'filepathname':filepathname, 'predicationlable':label_vgg[0][1],'confidence_score':confidence_score}
    return render(request, 'index.html', context)
