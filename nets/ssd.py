import keras.backend as K
import numpy as np
from keras.engine.topology import InputSpec, Layer
from keras.layers import (Activation, Concatenate, Conv2D, Flatten, Input,
                          Reshape)
from keras.models import Model

from nets.mobilenet import mobilenet

def SSD300(input_shape, num_classes=21):
    #---------------------------------#
    #   典型的输入大小为[300,300,3]
    #---------------------------------#
    input_tensor = Input(shape=input_shape)
    
    #------------------------------------------------------------------------#
    #   net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    #------------------------------------------------------------------------#
    net = mobilenet(input_tensor)
    
    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对conv4_3的通道进行l2标准化处理 
    # 38,38,512
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv4_3_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='conv4_3_loc')(net['conv4_3'])
    net['conv4_3_loc_flat'] = Flatten(name='conv4_3_loc_flat')(net['conv4_3_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv4_3_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv4_3_conf')(net['conv4_3'])
    net['conv4_3_conf_flat'] = Flatten(name='conv4_3_conf_flat')(net['conv4_3_conf'])

    # 对fc7层进行处理 
    # 19,19,1024
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['fc7_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat']    = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['fc7_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat']   = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])

    # 对conv6_2进行处理
    # 10,10,512
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv6_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat']    = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv6_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf_flat']   = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    # 对conv7_2进行处理
    # 5,5,256
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv7_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat']    = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv7_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf_flat']   = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])

    # 对conv8_2进行处理
    # 3,3,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv8_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat']    = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv8_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf_flat']   = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])

    # 对conv9_2进行处理
    # 1,1,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv9_2_mbox_loc']         = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc_flat']    = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv9_2_mbox_conf']        = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf_flat']   = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    
    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['conv4_3_loc_flat'],
                                                            net['fc7_mbox_loc_flat'],
                                                            net['conv6_2_mbox_loc_flat'],
                                                            net['conv7_2_mbox_loc_flat'],
                                                            net['conv8_2_mbox_loc_flat'],
                                                            net['conv9_2_mbox_loc_flat']])
                                    
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['conv4_3_conf_flat'],
                                                            net['fc7_mbox_conf_flat'],
                                                            net['conv6_2_mbox_conf_flat'],
                                                            net['conv7_2_mbox_conf_flat'],
                                                            net['conv8_2_mbox_conf_flat'],
                                                            net['conv9_2_mbox_conf_flat']])
    # 8732,4
    net['mbox_loc']     = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf']    = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf']    = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # 8732,25
    net['predictions']  = Concatenate(axis =-1, name='predictions')([net['mbox_loc'], net['mbox_conf']])

    model = Model(input_tensor, net['predictions'])
    return model
