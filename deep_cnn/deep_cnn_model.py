import tensorflow as tf

def setup_model(X,y):
	    conv_block1 = conv_block(X, 48, (3,3))
	    conv_block2 = conv_block(conv_block1, 48, (3,3))
	    conv_block3 = conv_block(conv_block2, 96, (3,3))
	    conv_block4 = conv_block(conv_block3, 96, (3,3))
	    conv_block5 = conv_block(conv_block4, 96, (3,3))
	    dconv_block5 = dconv_block(conv_block5, 96, (4,4), (2,2))
	    dconv_block4 = dconv_block(dconv_block5, 96, (4,4), (2,2), conv_block4)
	    dconv_block3 = dconv_block(dconv_block4, 96, (4,4), (2,2), conv_block3)
	    dconv_block2 = dconv_block(dconv_block3, 48, (4,4), (2,2), conv_block2)
	    dconv_block1 = dconv_block(dconv_block2, 3, (4,4), (2,2))
	    y_out = dconv_block1
	    return y_out

def conv_block(inputs, conv_filter, conv_kernel):
    conv1 = tf.layers.conv2d(inputs, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    conv2 = tf.layers.conv2d(conv1, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    conv3 = tf.layers.conv2d(conv2, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    pooled = tf.layers.max_pooling2d(conv3, (2,2), (2,2))
    return pooled

def dconv_block(inputs, dconv_filter, dconv_kernel, dconv_strides, conv_input = None):
    if conv_input is not None:
        inputs = tf.concat([inputs, conv_input], 3)
    dconv = tf.layers.conv2d_transpose(inputs,
                                       dconv_filter,
                                       dconv_kernel,
                                       dconv_strides,
                                       padding='same',
                                       activation=parametric_relu)
    conv1 = tf.layers.conv2d(dconv, 
                             dconv_filter, 
                             (3,3),
                             padding='same', 
                             activation=parametric_relu)
    conv2 = tf.layers.conv2d(conv1, 
                             dconv_filter, 
                             (3,3),
                             padding='same', 
                             activation=parametric_relu)
    return conv2

def parametric_relu(x):
    with tf.variable_scope("PReLu") as scope:
        alpha = tf.get_variable('alpha', [1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        h = tf.maximum(alpha * tf.ones_like(x), x)
    return h


class deep_CNN_model(object):
	def __init__(self):
		epsilon = 0.1
		tf.reset_default_graph()
		self.X = tf.placeholder(tf.float32, [None, 128, 384, 6])
		self.y = tf.placeholder(tf.float32, [None, 128, 384, 3])
		self.y_out = setup_model(self.X, self.y)
		# Charbonnier Loss
		epsilon = 0.1
		self.loss = tf.reduce_sum(tf.sqrt((self.y_out - self.y) ** 2 + epsilon ** 2))
		self.saver = tf.train.Saver()