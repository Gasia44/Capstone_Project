import argparse
import os
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--word_space_path', type=str, default='dns_space_ver_1.tsv')

	args = parser.parse_args()

	return args

LOG_DIR = 'logs/word_space'
STAR_SPACE_SIZE = 50

def create_metadata(index_word_map_dic):
	with open(LOG_DIR + '/metadata.tsv', 'w') as metadata:
		metadata.write('Word\tIndex\n')
		for index, word in index_word_map_dic.items():
			metadata.write('%s\t%d\n' % (word, index))

if __name__ == '__main__':

	args = parse_args()

	print('Reading word space data...')
	word_space_df = pd.read_csv(args.word_space_path, sep='\t',  header=None)
	print('Reading done.')

	embedding_matrix = word_space_df[list(range(1, STAR_SPACE_SIZE + 1))].values

	print('Creating metadata...')
	create_metadata(word_space_df[0].to_dict())
	print('Creating done.')

	print('Creating summary for projector...')
	tf.reset_default_graph()
	sess = tf.InteractiveSession()
	embeddings = tf.Variable([0.0], name='embeddings')
	_inputs = tf.placeholder(tf.float32, shape=[len(word_space_df), STAR_SPACE_SIZE])
	set_x = tf.assign(embeddings, _inputs, validate_shape=False)

	sess.run(tf.global_variables_initializer())
	sess.run(set_x, feed_dict={_inputs: embedding_matrix})

	# create a TensorFlow summary writer
	summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
	config = projector.ProjectorConfig()
	embedding_conf = config.embeddings.add()
	embedding_conf.tensor_name = 'embeddings'
	embedding_conf.metadata_path = 'metadata.tsv'
	projector.visualize_embeddings(summary_writer, config)
	print('Creating done.')

	print('Saving logs for tensorboard projector...')
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
	print('Saving done.')

	sess.close()
