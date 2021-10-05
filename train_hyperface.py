import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dense, Dropout, Activation, Flatten, concatenate, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
import requests
import sqlite3
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
data_path = 'data/face_data.csv'
data = pd.read_csv(data_path)
data.iloc[:,47:68] = data.iloc[:,47:68].astype('float32')
data['sex'] = data['sex'].astype('float32')
data_val = data.sample(1000)
data = data.drop(data_val.index).reset_index().drop(columns=['index'])
data_val = data_val.reset_index().drop(columns=['index'])
fnames = list(data['img_path'])
filelist_ds = tf.data.Dataset.from_tensor_slices(fnames)
def get_label(file_path):
  file_name= file_path.numpy().decode("utf-8")
  data_row = data[data['img_path']==file_name]
  pose = [tf.constant(data_row['roll'].values[0],dtype=tf.float32),tf.constant(data_row['pitch'].values[0],dtype=tf.float32),tf.constant(data_row['yaw'].values[0],dtype=tf.float32)]
  sex = [tf.constant(data_row['sex'].values[0],dtype=tf.float32)]
  landmarks = [tf.constant(data_row.iloc[:,i].values[0],dtype=tf.float32) for i in range(5,47)]
  visibility = [tf.constant(data_row.iloc[:,i].values[0],dtype=tf.float32) for i in range(47,68)]

  return tuple(landmarks+visibility+pose+sex)

def process_img(img):
  #color images
  img = tf.image.decode_jpeg(img, channels=3) 
  #convert unit8 tensor to floats in the [0,1]range
  img = tf.image.convert_image_dtype(img, tf.float32) 
  return img

def combine_images_labels(file_path: tf.Tensor):
  img = tf.io.read_file(file_path)
  img = process_img(img)
  outputs = get_label(file_path)

  img.set_shape((227,227,3))
  for out in outputs:
    out.set_shape(())

  return tuple([img]+list(outputs))
ds_train=filelist_ds.map(lambda x: tf.py_function(func=combine_images_labels,
          inp=[x], Tout=tuple([tf.float32]+[tf.float32 for _ in range(42)]+[tf.float32 for _ in range(21)]+[tf.float32 for _ in range(3)]+[tf.float32])),num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
def map_func(img,lm0,lm1,lm2,lm3,lm4,lm5,lm6,lm7,lm8,lm9,lm10,lm11,lm12,lm13,lm14,lm15,lm16,lm17,lm18,lm19,lm20,lm21,lm22,lm23,lm24,lm25,lm26,lm27,lm28,lm29,lm30,lm31,lm32,lm33,lm34,lm35,lm36,lm37,lm38,lm39,lm40,lm41,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,p0,p1,p2,s):
  lm = [lm0,lm1,lm2,lm3,lm4,lm5,lm6,lm7,lm8,lm9,lm10,lm11,lm12,lm13,lm14,lm15,lm16,lm17,lm18,lm19,lm20,lm21,lm22,lm23,lm24,lm25,lm26,lm27,lm28,lm29,lm30,lm31,lm32,lm33,lm34,lm35,lm36,lm37,lm38,lm39,lm40,lm41]
  v = [v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20]
  p = [p0,p1,p2]
  img.set_shape((227,227,3))
  for li in lm:
    li.set_shape(())
  for vi in v:
    vi.set_shape(())
  for pi in p:
    pi.set_shape(())
  s.set_shape(())
  indict = {'input_img':img}
  outdict = {}
  for i,li in enumerate(lm):
      outdict['landmark_'+str(i)] = li
  for i,vi in enumerate(v):
      outdict['visibility_'+str(i)] = vi
  for i,pi in enumerate(p):
      outdict['pose_'+str(i)] = pi
  outdict['sex'] = s
  
  return indict, outdict
  
ds_train = ds_train.map(map_func)
BATCH_SIZE = 128
ds_train_batched = ds_train.batch(BATCH_SIZE).cache()
# ds_val_batched = ds_val.batch(BATCH_SIZE).cache()
def create_model():
  Inputs = tf.keras.Input(shape=(227,227,3),name='input_img')
  conv1 = Conv2D(64, (11,11), strides = 4, activation = 'relu', padding = 'valid', name='features.0')(Inputs)
  max1 = MaxPooling2D((3,3), strides = 2, padding = 'valid')(conv1)
  max1 = BatchNormalization()(max1)

  conv1a = Conv2D(256, (4,4), strides = 4, activation = 'relu', padding = 'valid')(max1)
  conv1a = BatchNormalization()(conv1a)

  conv2 = Conv2D(192, (5,5), strides = 1, activation = 'relu', padding = 'same', name='features.3')(max1)
  max2 = MaxPooling2D((3,3), strides = 2, padding = 'valid')(conv2)
  max2 = BatchNormalization()(max2)

  conv3 = Conv2D(384, (3,3), strides = 1, activation = 'relu', padding = 'same', name='features.6')(max2)
  conv3 = BatchNormalization()(conv3)

  conv3a = Conv2D(256, (2,2), strides = 2, activation = 'relu', padding = 'valid')(conv3)
  conv3a = BatchNormalization()(conv3a)

  conv4 = Conv2D(256, (3,3), strides = 1, activation = 'relu', padding = 'same', name='features.8')(conv3)
  conv4 = BatchNormalization()(conv4)

  conv5 = Conv2D(256, (3,3), strides = 1, activation = 'relu', padding = 'same', name='features.10')(conv4)
  pool5 = MaxPooling2D((3,3), strides = 2, padding = 'valid')(conv5)
  pool5 = BatchNormalization()(pool5)

  concat = concatenate([conv1a, conv3a, pool5])
  concat = BatchNormalization()(concat)
  conv_all = Conv2D(192, (1,1), strides = 1, activation = 'relu', padding = 'valid', name='convall')(concat)
  flat = Flatten()(conv_all)
  fc_full = Dense(3072, activation = 'relu', name='fcfull')(flat)

  #fc_full = Dropout(0.5)(fc_full)

  #fc_detection = Dense(512, activation = 'relu')(fc_full)
  fc_landmarks = Dense(512, activation = 'relu')(fc_full)
  fc_visibility = Dense(512, activation = 'relu')(fc_full)
  fc_pose = Dense(512, activation = 'relu')(fc_full)
  fc_sex = Dense(512, activation = 'relu')(fc_full)

  #out_detection = Dense(1, activation = 'softmax')(fc_detection)
  out_landmarks = []
  out_visibility = []
  out_pose = []
  for i in range(42):
    out_landmarks.append(Dense(1, activation = None, name='landmarks'+str(i))(fc_landmarks))
  for i in range(21):
    out_visibility.append(Dense(1, activation = 'sigmoid', name='visibility'+str(i))(fc_visibility))
  for i in range(3):
    out_pose.append(Dense(1, activation = 'my_activation_pose', name='pose'+str(i))(fc_pose))
  out_sex = Dense(1, activation = 'sigmoid', name='sex')(fc_sex)
  outdict = {}
  for i,li in enumerate(out_landmarks):
    outdict['landmark_'+str(i)] = li
  for i,vi in enumerate(out_visibility):
    outdict['visibility_'+str(i)] = vi
  for i,pi in enumerate(out_pose):
    outdict['pose_'+str(i)] = pi
  outdict['sex'] = out_sex
  model = tf.keras.Model(inputs = {'input_img':Inputs}, outputs = outdict)

  return model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
# @tf.function
def MyActivation(x):
    return K.clip(x,-1.0,1.0)
# @tf.function
def MyActivationPose(x):
    return K.clip(x,-10.0,10.0)

get_custom_objects().update({'my_activation': Activation(MyActivation)})
get_custom_objects().update({'my_activation_pose': Activation(MyActivationPose)})
model = create_model()
# model.summary()
import torchvision.models as models
alexnet_model = models.alexnet(pretrained=True)
alexnet_wts = {}
for name,param in alexnet_model.named_parameters():
  param = param.detach().numpy()
  if (name.startswith('features')):
    shp = list(param.shape)
    shp.reverse()
    alexnet_wts[name] = np.reshape(param, tuple(shp))
model_layers = ['features.0','features.3','features.6','features.8','features.10']
for layer_name in model_layers:
  layer_weights = [alexnet_wts[layer_name+'.weight'],alexnet_wts[layer_name+'.bias']]
  model.get_layer(layer_name).set_weights(layer_weights)
  model.get_layer(layer_name).trainable = True
def loss_landmarks(landmarks_true,landmarks_pred,visibility):
    loss = K.constant([0.0])
    for i in range(len(landmarks_true)):
        if(K.eval(visibility[i])==1):
            loss = tf.math.add(loss, tf.math.squared_difference(landmarks_true[i],landmarks_pred[i]))
    loss = tf.multiply(loss,1.0/42)
    loss = K.squeeze(loss,0)
    return loss
def loss_visibility(visibility_true,visibility_pred):
    loss = K.constant([0.0])
    for i in range(len(visibility_true)):
        loss = tf.math.add(loss, tf.math.squared_difference(visibility_true[i],visibility_pred[i]))
    loss = tf.multiply(loss,1.0/21)
    loss = K.squeeze(loss,0)
    return loss
def loss_pose(pose_true,pose_pred):
    loss = K.constant([0.0])
    for i in range(len(pose_true)):
        loss = tf.math.add(loss, tf.math.squared_difference(pose_true[i],pose_pred[i]))
    loss = tf.multiply(loss,1.0/3)
    loss = K.squeeze(loss,0)
    return loss
def loss_sex(sex_true,sex_pred):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(sex_true,sex_pred)
    return loss
def loss_batched(y_true,y_pred):
    loss_lm = K.constant(0.0)
    loss_v = K.constant(0.0)
    loss_p = K.constant(0.0)
    loss_s = tf.math.multiply(loss_sex(y_true['sex'],y_pred['sex']),2.0)
    for j in range(42):
        loss_lm = tf.math.add(loss_lm,loss_landmarks(y_true['landmark_'+str(j)],y_pred['landmark_'+str(j)],y_true['visibility_'+str(j//2)]))
    loss_lm = tf.math.multiply(loss_lm,5.0)
    for j in range(21):
        loss_v = tf.math.add(loss_v,loss_visibility(y_true['visibility_'+str(j)],y_pred['visibility_'+str(j)]))
    loss_v = tf.math.multiply(loss_v,0.5)
    for j in range(3):
        loss_p = tf.math.add(loss_p,loss_pose(y_true['pose_'+str(j)],y_pred['pose_'+str(j)]))
    loss_p = tf.math.multiply(loss_p,5.0)
    return loss_lm,loss_v,loss_p,loss_s

optimizer = tf.keras.optimizers.Adam()

from PIL import Image, ImageDraw, ImageFont
def inference(path):
    img = cv2.imread(path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detects faces of different sizes in the input image
    face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    faces = list(face_cascade1.detectMultiScale(gray, 1.3, 5)) + list(face_cascade2.detectMultiScale(gray, 1.3, 5))
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')).copy()
    draw = ImageDraw.Draw(image)
    for (x,y,w,h) in faces:
        x,y,w,h = x-15,y-15,w,h
        # To draw a rectangle in a face
        draw.rectangle((x,y,x+w,y+h),None,5)
        face = np.copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y:y+h, x:x+w])
        to_size = 227
        face_rescaled = cv2.resize(face, (to_size,to_size), interpolation = cv2.INTER_AREA)
        face_rescaled = face_rescaled.astype('float32')/255.0
        outputs = model.predict({'input_img':np.expand_dims(face_rescaled,0)})
        landmarks_pred = np.array([np.squeeze(outputs['landmark_'+str(i)]) for i in range(42)])
        visibility_pred = np.array([np.squeeze(outputs['visibility_'+str(i)]) for i in range(21)])
        pose_pred = 180*(np.array([np.squeeze(outputs['pose_'+str(i)]) for i in range(3)])/np.pi)
        sex_pred = np.squeeze(outputs['sex'])
        r=2
        fcx,fcy = x+w/2.0,y+h/2.0
        for i in range(0,41,2):
            if (visibility_pred[i//2]>0.5):
                xl_pred = float(landmarks_pred[i]*w+fcx)
                yl_pred = float(landmarks_pred[i+1]*h+fcy)
                pt_pred=(xl_pred-r,yl_pred-r,xl_pred+r,yl_pred+r)
                draw.ellipse(pt_pred,fill=(0,255,0))
        gender = 'MALE' if sex_pred>0.5 else 'FEMALE'
        draw.text((x,y), str(round(pose_pred[0],2))+'°,'+str(round(pose_pred[1],2))+'°,'+str(round(pose_pred[2],2))+'°', fill=(255,0,0,0))
        draw.text((x,y-10),gender, fill=(0,255,0,0))
    display(image)
checkpoint_path = 'checkpoints/cp.ckpt'
model.load_weights(checkpoint_path)
epochs = 20
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(ds_train_batched):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            y_batch_pred = model(x_batch_train, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_lm,loss_v,loss_p,loss_s = loss_batched(y_batch_train, y_batch_pred)
            loss_total = K.sum([loss_lm,loss_v,loss_p,loss_s])

        grads = tape.gradient(loss_total, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Log every 50 batches.
        if step % 50 == 0:
            print("Training loss (for one batch) at step ",step," loss_total: ",float(loss_total)," loss_landmarks: ",float(loss_lm)," loss_visibility: ",float(loss_v)," loss_pose: ",float(loss_p)," loss_gender: ",float(loss_s))
            print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
    model.save_weights(checkpoint_path)
    idx = np.random.randint(len(data_val))
    plot_landmarks(idx,data_val)
    ind = np.random.randint(1,4)
    inference('data/sample_images/test'+str(ind)+'.jpg')