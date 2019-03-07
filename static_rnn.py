import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

true_file = "/home/raj/Documents/AI/fake_news/datasets/true_reduced5.txt"
fake_file = "/home/raj/Documents/AI/fake_news/datasets/fake_reduced5.txt"

lemmatizer = WordNetLemmatizer()
def create_lexicon(true, fake):
    lexicon = []
    print("started reading true file")
    with open(true, "r") as f:
        contents = f.readlines()
        for line in contents:
            words = word_tokenize(line.lower())
            lexicon += list(words)
    print("completed reading true file")
    print("started reading fake file")
    with open(fake, "r") as f:
        contents = f.readlines()
        for line in contents:
            words = word_tokenize(line.lower())
            lexicon += list(words)
    print("completed reading fake file")
    print("lemmatizing")
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    print("lexicon length", len(lexicon))
    l2 = []
    w_counts = Counter(lexicon)
    for word in w_counts:
        if 500 > w_counts[word] > 5:
            l2.append(word)
    print("length of lexicon is :", len(l2))
    return l2
def sample_handling(sample,lexicon,classification):

	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features,classification])
	return featureset


def create_feature_sets_and_labels(true,fake,test_size = 0.1):
	lexicon = create_lexicon(true,fake)
	features = []
	features += sample_handling(true,lexicon,[1,0])
	features += sample_handling(fake,lexicon,[0,1])
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = create_feature_sets_and_labels(true_file, fake_file)
print(len(train_x))
print(len(train_y))
print(len(test_x))
print(len(test_y))



n_classes = 2
batch_size = 28
seq_length = len(create_lexicon(true_file, fake_file))
num_features = 1
rnn_size = 200
hm_epochs = 20


input_data = tf.placeholder(tf.float32, [None, seq_length, num_features])
labels = tf.placeholder(tf.float32, [None, n_classes])


def test_data_modification(test_x, test_y):
    test_x = test_x.reshape((-1, seq_length, num_features))
    test_y = test_y.reshape((-1, n_classes))
    return test_x, test_y



def recurrent_neural_network(data):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
         'biases':tf.Variable(tf.random_normal([n_classes]))}
    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, num_features])
    data = tf.split(data, seq_length, 0)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = tf.nn.static_rnn(lstm_cell, data, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    print(output.shape)
    return output


def train_neural_network(data):
    prediction = recurrent_neural_network(data)
#    print(prediction)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            for _ in range(int(len(train_x)/batch_size)):
                start = i
                end = start + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                i += batch_size
                batch_x = batch_x.reshape((batch_size, seq_length, num_features))
#                batch_y = batch_y.reshape((batch_size, 2, num_features))
                batch_y = batch_y.reshape((batch_size, n_classes))
                _, c = sess.run([optimizer, cost], feed_dict = {input_data:batch_x, labels:batch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
    
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        t_x, t_y= test_data_modification(np.array(test_x), np.array(test_y))
        print('Accuracy:',accuracy.eval({input_data:t_x, labels:t_y}))
            
train_neural_network(input_data)

