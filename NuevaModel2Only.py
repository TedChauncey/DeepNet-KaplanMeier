### This model is to predict 2 year survival in Stage I NSCLC using pretrained VGG-16###
### Tafadzwa L. Chaunzwa 1-28-18 ###
### Make sure to run in a new window ###
########################################


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from sklearn.metrics import roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc

#from PIL import Image
from numpy import *
from keras import backend as K

## Define functions ##
def AUC(test_labels,test_prediction,nb):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 
    
def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = plt.plot(fpr, tpr)
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(['ROC','FPR'],loc=4)
    plt.show()
    #myAuc = auc(fpr, tpr)
    return myAuc
   
#### DEFINE MODEL########

model_base = VGG16(weights='imagenet', include_top=False, input_shape = (50,50,3) )
model_base.layers.pop()
model_base.outputs = [model_base.layers[-1].output]
model_base.layers[-1].outbound_nodes =[]

model_top = model_base.output
model_top = Flatten()(model_top)
model_top =Dense(4096, activation = 'relu')(model_top)
model_top = Dropout(0.5)(model_top)
model_top =Dense(4096, activation = 'relu')(model_top)
model_top = Dropout(0.5)(model_top)
model_top =Dense(2, activation = 'softmax')(model_top)

model = Model(input = model_base.input, output=model_top)
for layer in model.layers[:15]:  # layer 11 has best performance? 15 or 11 or 6 or 3  
    layer.trainable = False

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])


### DEFINE DATA ###

#input image dimensions
OG_size = 150 #original image size 
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 3

## DATA SOURCES ##

Outcomes_file = '/home/chintan/Desktop/tsne_python/Stage1/Stage2Train.csv' #define the outcomes file, sorted according to PID
path1 = '/home/chintan/Desktop/tsne_python/Stage1/Stage2Train'    #path of folder of images    
path2 = '/home/chintan/Desktop/tsne_python/Stage1/TrainImageCrops' #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples


for file in listing:
    im = Image.open(path1 + '/' + file) 
    img = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    #img = im.resize((img_rows,img_cols))
    gray = img.convert('RGB')  #gray is a misnormer since we are converting to RGB        
    gray.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path2)
imlist = sorted(imlist, key = len)
#imlist.sort() #make sure to SORT the images to match 

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

#create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes#

Outcomes = pd.read_csv(Outcomes_file) #outcomes file is sorted so as to match the image index
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'SURV2']) #pick the column with the labels of interest
PID = pd.Series.as_matrix(Outcomes.loc[:,'PID'])

label = Outcome_of_interest

data,Label = shuffle(immatrix,label, random_state = 2) #see how shuffle works
train_data = [data,Label]

#pick image index for demo. Not the same as im_index for layer visualization
im_index1 = 1
img=immatrix[im_index1].reshape(img_rows,img_cols,3)
plt.title('Image Preprocessing example')
plt.figure(1)
plt.imshow(img)
plt.imshow(img,cmap='gray')
plt.show()
print (train_data[0].shape)
print (train_data[1].shape)

#MODEL Parameters ##

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 100

from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])


# split X and y into training and testing sets
test_size = .1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)


X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#X_train = preprocess_input(X_train)
#X_test = preprocess_input(X_test) 


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


### TRAIN THE MODEL ####

#hist = model.fit(X_train, Y_train,
#                    batch_size=batch_size,
#                    nb_epoch=nb_epoch,
#                    verbose=1,
#                    validation_data=(X_test, Y_test))     

hist = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)     
### MODEL ASSESSMENT ###


# visualizing losses and accuracy

#train_loss=hist.history['loss']
#val_loss=hist.history['val_loss']
#xc=range(nb_epoch)
#plt.figure(2)
#plt.plot(xc,train_loss)
#plt.plot(xc,val_loss)
#plt.xlabel('num of Epochs')
#plt.ylabel('loss')
#plt.title('train_loss vs val_loss')
#plt.grid(True)
#plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])
#plt.show()



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


## VISUALIZE INTERMEDIATE LAYERS ##
im_index = 21 # 67 not bad. 69 ok for demostrating its ignoring stuff . 12 is the best
#get the index of the 2 layer you want to visualize
idx1 = 1 
idx2 =7

input_image=X_train[im_index,:,:,:].reshape(1,50,50,3)
plt.figure(3)
plt.title('Input Image')
plt.imshow(input_image[0,:,:,:],cmap ='gray')
plt.axis('off')
#plt.imshow(input_image[0,0,:,:])
plt.show()
input_image_aslist= [input_image] #convert input image to list

##PLOT ACTIVATIONS FOR SHALLOW vs DEEP LAYERS##

# build functions
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[idx1].output ] )
func2 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[idx2].output ] )

# then use it to plot output of filters
fig4 = plt.figure(4)
plt.title('First Conv2D Layer Activations')
plt.axis('off')

#iter = sort(np.append(range(31),-1, axis = None))
for i in range(nb_filters-1):
	func1out = func1(input_image_aslist)
	convnet_image1 = func1out[0][:,:,:,i]
	
	a=fig4.add_subplot(4,8,i+1)
	imgplot = plt.imshow(squeeze(convnet_image1), cmap = 'gray')
	plt.axis('off')
	plt.show()

fig5 = plt.figure(5)
plt.title('Deep Conv2D Layer Activations')
plt.axis('off')	

for i in range(nb_filters-1):
	func2out = func2(input_image_aslist)
	convnet_image2 = func2out[0][:,:,:,i]
	
	b=fig5.add_subplot(4,8,i+1)
	imgplot = plt.imshow(squeeze(convnet_image2), cmap = 'gray')
	plt.axis('off')
	plt.show()


## PLOT ACTIVATIONS HEAT MAP ##
idx3 = 5 #pick layer to plot heat map

from fractions import Fraction
import scipy.ndimage

#call the functions again
func3 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[idx3].output ] )
input_image_list= [input_image]
func3out = func3(input_image_list)

convnet_image3 = np.squeeze(func3out[0][:,:,:,1])
img_squeeze = np.squeeze(input_image)

#resize conv_layer output to match input size
dim_input = img_squeeze.shape[1]
dim_convlayer = convnet_image3.shape[1]
scale =Fraction(dim_input,dim_convlayer)
convnet_image3_scaled = scipy.ndimage.zoom(convnet_image3,scale,order =0)


#plot heat map and input image side by side
fig6 = plt.figure(6)
c=fig6.add_subplot(1,2,1)
input_plot = plt.imshow(img_squeeze, cmap = 'gray')
plt.title('Input Image')
plt.axis('off')

c=fig6.add_subplot(1,2,2)
heat_plot = plt.imshow(convnet_image3_scaled)
plt.title('Activations Heat Map (Conv Layer 9)')
plt.axis('off')
plt.show()

##Predictions - from test set ####
pred_results = []
for i in xrange(X_test.shape[0]):
    predictions = model.predict(np.expand_dims(X_test[i], axis = 0))
    pred_results = np.append(pred_results,[predictions])

Y_pred = pred_results.reshape(X_test.shape[0],2)
print(Y_pred)

nb_classes = 2

plt.figure(7)    
ROC = AUCalt(Y_test, Y_pred)
 
ROC2 = AUC(Y_test, Y_pred, nb_classes)

print ROC2[1]


#save model 
fname = "NuevaModel_Stage2Only.h5"
model.save(fname)
 
#determine pvalue of AUC
import scipy.stats as stat
a = Y_pred[:, 0]
b = Y_test[:, 0]
groups = [a[b == i] for i in xrange(2)]
pvalue = stat.ranksums(groups[0], groups[1])[1]
