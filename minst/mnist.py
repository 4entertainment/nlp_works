import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10]) # real values. "etiketler"

# y = x*w + b
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(x, w) + b # y = x*w + b. x and w are matrixes. so we use tf.matmul func.
# x: input (nmist images, contains nmist images' colors)
# w, b: model's optimization parameters. first form of w and b is matrix with full zeros. the matrix of w and b will change by model from training infos/steps 


y = tf.nn.softmax(logits) # softmax activision func will zip the values to [0-1].
# if the 3.neural shows "0.9": it means models says the input nmist image is "2" with 0.9(%90) probability
# if (y = x*w + b) calculation output is logits = [1.2 , 0.6 , -0.4]

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true) # we will get 500 loss value for 500 input image.
loss = tf.reduce_mean(xent) # mean aritmetic calculation for loss 

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

def training_step (iterations):
    for i in range (iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        sess.run(optimize, feed_dict=feed_dict_train)

def test_accuracy ():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(2000)
test_accuracy()