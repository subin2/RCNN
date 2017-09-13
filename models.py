import os
import csv
import time
import random
import cPickle
import pandas as pd

import numpy as np
import tensorflow as tf

class conv2d(object):
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool=None, pool_size=[1,4], nonlinearity=None, 
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
        with tf.variable_scope(name):
            self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
            network = tf.nn.bias_add( tf.nn.conv2d(input = input, filter = self.weight, strides=strides, padding=padding), 
                                     self.bias, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            if pool=='p':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1], 
                                         strides=[1,pool_size[0],pool_size[1],1], 
                                         padding='SAME')
            self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class res_conv2d(object):
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool=None, pool_size=[1,4], nonlinearity=None, 
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
        with tf.variable_scope(name):
            self.weight1 = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias1 = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
            self.weight2 = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias2 = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
            network = tf.nn.bias_add( tf.nn.conv2d(input = input, filter = self.weight1, strides=strides, padding=padding), 
                                     self.bias1, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            #
            network = tf.nn.bias_add( tf.nn.conv2d(input = network, filter = self.weight2, strides=strides, padding=padding), 
                                     self.bias2, name=name)
            network = tf.add(input, network)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            #
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            if pool=='p':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1], 
                                         strides=[1,pool_size[0],pool_size[1],1], 
                                         padding='SAME')
            self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class shared_depthwise_conv2d(object):
    """
    input: tensor of shape [batch, in_height, in_width, in_channels]
    weight_size: an array of the form [filter_height, filter_width, in_channels, channel_multiplier].
        Let in_channels be 1.
    returns:
        A 4D Tensor of shape [batch, out_height, out_width, in_channels * channel_multiplier].
    """
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], 
					nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, 
					offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_conv2d_default'):
		self.pool = pool
		self.weight_size = [weight_size[0],weight_size[1],1,weight_size[3]]
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.tile(tf.reduce_mean(tf.random_normal( weight_size, stddev=std, dtype=tf.float32), axis=2, keep_dims=True),
                                               [1,1,weight_size[2],1]))
			self.bias = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
			network = tf.add( tf.nn.depthwise_conv2d(input = input, filter = self.weight, strides=strides, padding=padding), 
							self.bias, name=name)
			if use_batchnorm:
				batch_mean, batch_var = tf.nn.moments(network, axes=[0])
				network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
			if nonlinearity != None:
				network = nonlinearity(network, name=name)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1], 
                                         strides=[1,pool_size[0],pool_size[1],1], 
                                         padding='SAME')                
			self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class depthwise_conv2d(object):
    """
    input: tensor of shape [batch, in_height, in_width, in_channels]
    weight_size: an array of the form [filter_height, filter_width, in_channels, channel_multiplier].
        Let in_channels be 1.
    returns:
        A 4D Tensor of shape [batch, out_height, out_width, in_channels * channel_multiplier].
    """
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], 
					nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, 
					offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_conv2d_default'):
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32))
			self.bias = tf.Variable( tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32) )
			network = tf.nn.bias_add( tf.nn.depthwise_conv2d(input = input, filter = self.weight, strides=strides, padding=padding), 
							self.bias, name=name)
			if use_batchnorm:
				batch_mean, batch_var = tf.nn.moments(network, axes=[0])
				network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
			if nonlinearity != None:
				network = nonlinearity(network, name=name)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1], 
                                         strides=[1,pool_size[0],pool_size[1],1], 
                                         padding='SAME')                
			self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class RCL(object):
	def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], num_iter=3, 
                nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
		"""
			when num_iter==1, same as conv2d
		"""
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal(weight_size, stddev=std, dtype=tf.float32) )
			self.biases = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32))
			"""
			rcl = tf.nn.bias_add(tf.nn.conv2d(input=input, filter=self.weight, strides=strides, padding=padding), 
                                 self.biases)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(rcl, [0])#[0,1,2]
                rcl = tf.nn.batch_normalization(rcl, batch_mean, batch_var, offset, scale, epsilon)
            if nonlinearity != None:
                rcl = nonlinearity(rcl)
            network = rcl
			"""
			network = input
 		 	if num_iter == 0:
 		 		network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding), 
                                         self.biases
                                        )
				if use_batchnorm:
					batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
					network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
				if nonlinearity != None:
					network = nonlinearity(network, name=name)
 		 	else:
				for i in range(num_iter):
					#network = tf.add( rcl, 
					#                 tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding), 
					#                               self.biases
					#                               )
					#                )
					network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding), 
                                         self.biases
                                        )
					if use_batchnorm:
						batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
						network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
					if nonlinearity != None:
						network = nonlinearity(network, name=name)
					network = tf.add(input, network)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='c':
				#input: [batch, height, width, channel]
				#kernel: [height, width, in_channels, out_channels]
				network = conv2d(input=network, 
								weight_size=[pool_size[0],pool_size[1],weight_size[-1],weight_size[-1]], 
                                 padding='VALID',
                                 nonlinearity=nonlinearity,
                                 use_dropout=use_dropout,
                                 keep_prob=keep_prob,
                                 name=name+'_convpool')
			elif pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1, pool_size[0], pool_size[1], 1], 
                                         strides=[1, pool_size[0], pool_size[1], 1], 
                                         padding='SAME')
			self.result = network
	def get_layer(self):
		if self.pool == 'c':
			return self.result.get_layer()
		return self.result
	def get_conv_layer(self):	
		if self.pool != 'c':
			raise ValueError('No conv layer is used for pooling.')
		return self.pool
	def get_weight(self):
		return self.weight
	def get_biases(self):
		return self.biases


class depthwise_RCL(object):
	def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], num_iter=3, 
                nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_RCL_default'):
		"""
		when num_iter==1, same as conv2d
		"""
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal(weight_size, stddev=std, dtype=tf.float32) )
			self.bias = tf.Variable(tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32))
			#self.biases = [tf.Variable( tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32)) for i \
			#			   in range(num_iter)]
			"""
			rcl = tf.nn.bias_add(tf.nn.depthwise_conv2d(input=input, filter=self.weight, strides=strides, padding=padding), 
                                 self.biases[0])
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(rcl, [0])#[0,1,2]
                rcl = tf.nn.batch_normalization(rcl, batch_mean, batch_var, offset, scale, epsilon)
            if nonlinearity != None:
                rcl = nonlinearity(rcl)
            network = rcl
			"""
			network = input
			for i in range(num_iter):
				#network = tf.add( rcl, 
                #                 tf.nn.bias_add(tf.nn.depthwise_conv2d(input=network, filter=self.weight, strides=strides, padding=padding), 
                #                               self.biases[i+1]
                #                               )
                #                )
				network = tf.nn.bias_add(tf.nn.depthwise_conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
								self.biases[i+1])
				if use_batchnorm:
					batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
					network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
				if nonlinearity != None:
					network = nonlinearity(network, name=name)
				network = tf.add(input, network)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='c':
				network = conv2d(input=network, 
								weight_size=[1,pool_size, weight_size[-1]*weight_size[-2], weight_size[-1]*weight_size[-2]],
								padding='VALID',
								nonlinearity=nonlinearity,
								use_dropout=use_dropout,
								keep_prob=keep_prob,
								name=name+'_convpool')
			elif pool=='p':
				network = tf.nn.max_pool(value=network,
										ksize=[1,pool_size[0],pool_size[1],1], 
										strides=[1,pool_size[0],pool_size[1],1], 
										padding='SAME')
			self.result = network
	def get_layer(self):
		return self.result
	def get_weight(self):
		return self.weight
	def get_biases(self):
		return self.biases


class feedforward(object):
    def __init__(self, input, weight_size, nonlinearity=None, use_dropout=False, keep_prob=1.0, use_batchnorm=False, 
                std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default'):
        with tf.variable_scope(name):
            self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias = tf.Variable( tf.random_normal( [weight_size[-1]], stddev=std, dtype=tf.float32) )
            network = tf.nn.bias_add( tf.matmul(input, self.weight), self.bias, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            self.result = network
    def get_layer(self):
        return self.result
    def get_bias(self):
        return self.bias
    def get_weight(self):
        return self.weight


class RCNN(object):
    def __init__(self, batch_size=128, width=1, height=1024, channel = 126, filters=[256,256,256,256], rrcl_iter=[2,2,2,2], 
                conv_num=4, forward_layers=[200, 3], pool=['p', 'p', 'p', 'p'], use_batchnorm=True, scale=1, offset=0.01, 
                epsilon=0.01, nonlinearity=None, keep_probs=None, std=0.01, w_filter_size=[1,9], p_filter_size=[1,4], 
                l_rate=0.01, l_decay=0.95, l_step=1000, decay=0.9, momentum=0.9, optimizer='RMSProp', opt_epsilon=0.1):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.channels = channels
        self.filters = filters
        self.rrcl_iter = rrcl_iter
        self.conv_num = conv_num
        self.use_batchnorm = use_batchnorm
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.nonlinearity = nonlinearity
        self.keep_probs = keep_probs
        self.use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])
        if keep_probs == None:
            self.keep_probs = [1.0 for i in range(1+rrcl_num+len(forward_layers)-1)]
        if self.use_dropout and len(keep_probs) != (conv_num + len(forward_layers)-1):
            raise ValueError('Parameter \'keep_probs\' length is wrong.')
        self.std = std
        self.w_filter_size = w_filter_size
        self.p_filter_size = p_filter_size
        #self.forward_layers = [out_channels] + forward_layers 
        self.pool = pool
        if len(self.pool) != conv_num :
            raise ValueError('Parameter \'pool\' length does not match with the model shape.')
        if len(self.filters) != conv_num:
            raise ValueError('Parameter \'filters\' length does not match with the model shape.')
        global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, global_step, l_step, l_decay, staircase=True)
        self.decay = decay
        self.momentum = momentum
        
        self.y = tf.placeholder(tf.float32, [batch_size, self.forward_layers[-1]], name='y')
        self.x = tf.placeholder(tf.float32, [batch_size, width, height, channel], name='x')
        
        self.build_model( )
        
        # Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.pow(self.y - self.output_layer, 2))
        if optimizer=='Adam':
            self.optimizer = tf.train.AdamOptimizer(self.l_rate, epsilon=opt_epsilon).minimize(self.cost, global_step=global_step)
        else :#optimizer=='RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer(self.l_rate, 
                                                   decay=self.decay, 
                                                   momentum=self.momentum).minimize(self.cost, global_step = global_step)
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()
        
        # Launch the session
        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=self.session_conf)
        #self.sess = tf.InteractiveSession()
        
        self.sess.run(init)
        
        self.saver = tf.train.Saver(max_to_keep=10000)
        
    def build_model(self):
        #self.weights, self.biases = self.init_weights()
        """
        RCL(input, filter, strides=[1,1,1,1], padding='SAME', num_iter=3, nonlinearity=None, use_dropout=True, keep_prob=1.0, 
            use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
        """
        #networks = self.conv1.get_layer()
        self.rrcls = []
        for r in range(self.conv_num):
            if r==0:
                network = RCL(input = self.x, 
                      weight_size = [self.w_filter_size[0], self.w_filter_size[1], self.channel, self.filters[r]], 
                      num_iter = self.rrcl_iter[r], 
                      nonlinearity = self.nonlinearity, 
                      use_dropout = self.use_dropout,
                      keep_prob = self.keep_probs[r], 
                      use_batchnorm = self.use_batchnorm,
                      std=self.std,
                      offset=self.offset,
                      scale=self.scale,
                      epsilon=self.epsilon, 
                      pool=self.pool[r],
                      pool_size=self.p_filter_size,
                      name='RCL'+str(r))
            else:
                network = RCL(input=network.get_layer(),
                                weight_size=[self.w_filter_size[0],self.w_filter_size[1], self.filters[r], self.filters[r+1]],
                                num_iter=self.rrcl_iter[r],
                                nonlinearity = self.nonlinearity,
                                use_dropout = self.use_dropout,
                                keep_prob = self.keep_probs[r],
                                use_batchnorm = self.use_batchnorm,
                                std = self.std,
                                offset=self.offset,
                                scale=self.scale,
                                epsilon=self.epsilon,
                                pool=self.pool[r],
                                pool_size=self.p_filter_size,
                                name='RCL'+str(r))
            self.rrcls.append(network)
            length = length/self.p_filter_size
            print'rrcl{} done'.format(r),
            print'    {}'.format(network.get_layer())
        #        
        network = tf.reshape(network.get_layer(), shape=[self.batch_size, -1])# * self.keep_probs[1]]) ###
        self.flatten = network
        print 'flatten to {}'.format(self.flatten)
        """
        (input, weight, nonlinearity=None, use_dropout=False, keep_prob=1.0, 
        use_batchnorm=False, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default')
        """
        if len(self.forward_layers) == 2:
            network = feedforward(input = network,
                                  weight_size=[self.forward_layers[0], self.forward_layers[1]],
                                  nonlinearity=None,
                                  use_dropout = False, 
                                  use_batchnorm = False,
                                  std=self.std,
                                  offset=self.offset,
                                  scale=self.scale,
                                  epsilon=self.epsilon, 
                                  name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print'feedforward {} done, {}'.format(i+1, self.output_layer)
            print'model built'
        else:
            self.forwards=[]
            for i in range(len(self.forward_layers)-1 -1):
                network  = feedforward(input = network, 
                                              weight_size=[self.forward_layers[i], self.forward_layers[i+1]],
                                              nonlinearity=self.nonlinearity, 
                                              use_dropout = self.use_dropout, 
                                              keep_prob = self.keep_probs[self.conv_num+i], 
                                              use_batchnorm = self.use_batchnorm,
                                              std=self.std,
                                              offset=self.offset,
                                              scale=self.scale,
                                              epsilon=self.epsilon, 
                                              name='forward'+str(i))
                self.forwards.append(network)
                network = network.get_layer()
                print'feedforward {} done, {}'.format(i, network)
            network =  feedforward(input = network,
                                         weight_size=[self.forward_layers[-2], self.forward_layers[-1]],
                                         nonlinearity=None,
                                         use_dropout = False, 
                                         use_batchnorm = False,
                                         std=self.std,
                                         offset=self.offset,
                                         scale=self.scale,
                                         epsilon=self.epsilon, 
                                         name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print'feedforward {} done, {}'.format(i+1, self.output_layer)
            print'model built'
            
    def train(self, data, target):
        ## data: [batch, time_idx]
        ## x: [batch, in_height, in_width, in_channels]
        train_feed_dict = {self.x:data}
        train_feed_dict.update({self.y:target})
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict=train_feed_dict
                                 )
        return cost
    
    def test(self, data, target):
        test_feed_dict = {self.x:data}
        test_feed_dict.update({self.y:target})
        cost = self.sess.run(self.cost, 
                             feed_dict=test_feed_dict
                            )
        return cost
    
    def reconstruct(self, data):
        recon_feed_dict = {self.x:data}
        return self.sess.run(self.output_layer, 
                             feed_dict=recon_feed_dict
                            )
    
    def save(self, save_path='./model.ckpt'):
        saved_path = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s"%saved_path)
        
    def load(self, load_path = './model.ckpt'):
        self.saver.restore(self.sess, load_path)
        print("Model restored")
    
    def terminate(self):
        self.sess.close()
        tf.reset_default_graph()


