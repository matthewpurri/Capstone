# -*- coding: utf-8 -*-



from keras import backend as K
K.set_image_dim_ordering('th')
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn import svm
from sklearn import cross_validation
import numpy as np 
import time

start = time.time()

img_width, img_height = 224, 224
data_dir = 'capstone/data/demux'
samples = 536                       # change it to 356 for gradient images, 364 for difference images



# loading pretrained VGG16 model
# include_top: whether to include the 3 fully-connected layers at the top of the network
base_model = VGG16(weights='imagenet', include_top=True)
base_model.summary()                # summary of the architecture of the model
# defining model for feature extraction from fc2 layer
# the layer for feature extraction can be changed by passing the name of layer needed for feature extraction
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

# extracting features from the images
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)           
features = model.predict_generator(generator, samples)

# features extracted from block 1 to 5 are reshaped from 4d to 2d
#features = np.reshape(features,(364,25088))        # block 5
#features = np.reshape(features,(364,100352))       # block 4
#features = np.reshape(features,(364,200704))       # block 3
#features = np.reshape(features,(364,401408))       # block 2 
#features = np.reshape(features,(364,802816))       # block 1 
print "features vector dimension: ",features.shape


# generating data labels for the wet and dry images, 0 for dry and 1 for wet
data_labels = np.array([0] * (samples / 2) + [1] * (samples / 2))
#trainig the SVM classifier and generating 5 fold cross validation scores
clf = svm.SVC(kernel='linear')
scores = cross_validation.cross_val_score(clf, features, data_labels, cv=5)

# printing the results
print 'scores arrray: ',scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
print (time.time() - start)


