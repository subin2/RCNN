# RCNN
Recurrent Convolutional Neural Network (RCNN) implementation using python-tensorflow.

## Library versions
python 2.7
numpy 1.12
tensorflow 1.12

## Usage
1. Import models.py

    import models

2. Create an RCNN object

    rcnn = models.RCNN(batch_size=128, 
                      width=1, # input shape \[batch, width, height, channel\]
                      height=1024, 
                      channel = 126, 
                      conv_num=4, #number of convolution layer
                      rrcl_iter=\[2,2,2,2\], #number of recurrents at each convolution layer. basic convolution if '0'.
                      filters=\[256,256,256,256\], # number of filters at each convolution
                      pool=['p', 'p', 'p', 'p'], #'p' for maxpooling, 'c' for convolution, 'n' for skipping pooling.
                      w_filter_size=[1,9], #convolution filter size
                      p_filter_size=[1,4], #pooling filter size
                      forward_layers=[200, 3], #feedforward layer after conv_num number of convolution layers.
                      keep_probs=[0.5,0.5,0.5,0.5,0.5],  #keep probability of dropout. Use None if you don't want to use dropout.
                      use_batchnorm=True, # batch normalization
                      nonlinearity=tf.nn.relu, 
                      std=0.01, 
                      l_rate=0.01, 
                      l_decay=0.95, #learning rate decay after l_step training.
                      l_step=100000, 
                      scale=1, offset=0.01, epsilon=0.01, 
                      decay=0.9, 
                      momentum=0.9)
           
3. Training

    model.train(data=train_input, target=input_target) 
train_input should be in shape [batch_size, width, height, channel] and input_target in [batch_size, forward_layers[-1]].
train function returns loss value.

4. Test

  model.test(data=test_input, target=test_target)
train_input should be in shape [batch_size, width, height, channel] and input_target in [batch_size, forward_layers[-1]].
test function returns test input's loss value without updating parameters.

5. Reconstruct
If you want to get the output value of your model, use reconstruct().
  model.reconstrut(data=test_input)
This returns the output values.

6. Save and load the model.

  model.save(save_path)
  model.load(load_path)

7. Terminate the model.
  
  model.terminate()
terminate() function closes the session and calls tf.reset_default_graph()
