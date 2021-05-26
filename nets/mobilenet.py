import keras.backend as K
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          DepthwiseConv2D, ZeroPadding2D)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)


def mobilenet(input_tensor):
    #----------------------------主干特征提取网络开始---------------------------#
    # SSD结构,net字典
    net = {} 
    # Block 1
    x = input_tensor
    # 300,300,3 -> 150,150,64
    x = Conv2D(32, (3,3),
            padding='same',
            use_bias=False,
            strides=(2, 2),
            name='conv1')(input_tensor)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    x = _depthwise_conv_block(x, 64, 1, block_id=1)
    
    # 150,150,64 -> 75,75,128
    x = _depthwise_conv_block(x, 128, 1,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, 1, block_id=3)

    
    # Block 3
    # 75,75,128 -> 38,38,256
    x = _depthwise_conv_block(x, 256, 1,
                              strides=(2, 2), block_id=4)
    
    x = _depthwise_conv_block(x, 256, 1, block_id=5)
    net['conv4_3'] = x

    # Block 4
    # 38,38,256 -> 19,19,512
    x = _depthwise_conv_block(x, 512, 1,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, 1, block_id=7)
    x = _depthwise_conv_block(x, 512, 1, block_id=8)
    x = _depthwise_conv_block(x, 512, 1, block_id=9)
    x = _depthwise_conv_block(x, 512, 1, block_id=10)
    x = _depthwise_conv_block(x, 512, 1, block_id=11)

    # Block 5
    # 19,19,512 -> 19,19,1024
    x = _depthwise_conv_block(x, 1024, 1,
                              strides=(1, 1), block_id=12)
    x = _depthwise_conv_block(x, 1024, 1, block_id=13)
    net['fc7'] = x

    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # 19,19,512 -> 10,10,512
    net['conv6_1'] = Conv2D(256, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu',
                                   name='conv6_2')(net['conv6_2'])

    # Block 7
    # 10,10,512 -> 5,5,256
    net['conv7_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same', 
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_2'])
    # Block 8
    # 5,5,256 -> 3,3,256
    net['conv8_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv8_2')(net['conv8_1'])

    # Block 9
    # 3,3,256 -> 1,1,256
    net['conv9_1'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='conv9_2')(net['conv9_1'])
    #----------------------------主干特征提取网络结束---------------------------#
    return net
