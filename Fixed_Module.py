import tensorflow as tf

class LRRLLSTMS:
    def __init__(self):
        pass
    def apply(self,Input_txt_indices,batchSize,memoryLen, vocabularySize, global_len, embeddingSize, outputSize,training):


        word_embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1), dtype=tf.float32)
        sentence_embeddings = tf.nn.embedding_lookup(word_embeddings, Input_txt_indices, max_norm=None)



        #sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings, training = training, axis = -1)
        sentence_embeddings_left = sentence_embeddings

        left_vector = tf.get_variable(shape=[1, 2, embeddingSize], dtype=tf.float32, name='LeftVector')
        left_vector = tf.tile(left_vector, [batchSize, 1, 1])
        sentence_embeddings_left = tf.concat([left_vector,sentence_embeddings_left], axis = 1)
        sentence_embeddings_left = tf.layers.batch_normalization(sentence_embeddings_left,training=training)


        with tf.variable_scope('LRLSTM'):
            LeftSentenceEncoderCell = tf.nn.rnn_cell.LSTMCell(memoryLen)
            #LeftSentenceEncoderCell = bnLSTM.BNLSTMCell(memoryLen,training)
            LeftHypothesises, leftHypothesisesState = tf.nn.dynamic_rnn(LeftSentenceEncoderCell, sentence_embeddings_left,
                                                                        dtype=tf.float32)



        left_scores = LeftHypothesises


        #left_vector = [left_vector for i in range(0, batchSize)]
        #left_vector = tf.stack(left_vector, axis=0)
        #left_scores = tf.layers.dense(LeftHypothesises, embeddingSize)
        #left_scores = left_scores[:, 0:-1, :]
        #left_scores = tf.concat([left_vector, left_scores], axis=1)



        sentence_embeddings_right = tf.reverse(sentence_embeddings,[1])

        right_vector = tf.get_variable(shape=[1, 2, embeddingSize], dtype=tf.float32, name='RightVector')
        right_vector = tf.tile(right_vector, [batchSize, 1, 1])

        sentence_embeddings_right = tf.concat([ right_vector, sentence_embeddings_right], axis=1)
        sentence_embeddings_right = tf.layers.batch_normalization(sentence_embeddings_right, training=training)
        with tf.variable_scope('RLLSTM'):
            RightSentenceEncoderCell = tf.nn.rnn_cell.LSTMCell(memoryLen)
            #RightSentenceEncoderCell = bnLSTM.BNLSTMCell(memoryLen,training)
            RightHypothesises, rightHypothesisesState = tf.nn.dynamic_rnn(RightSentenceEncoderCell, sentence_embeddings_right,
                                                                          dtype=tf.float32)

        right_scores = tf.reverse(RightHypothesises,[1])
        #right_vector = tf.get_variable(shape=[1, embeddingSize], dtype=tf.float32, name='RightVector')
        #right_vector = [right_vector for i in range(0, batchSize)]
        #right_vector = tf.stack(right_vector, axis=0)
        #right_scores = tf.layers.dense(RightHypothesises, embeddingSize)
        #right_scores = right_scores[:, 0:-1, :]
        #right_scores = tf.concat([right_vector, right_scores], axis=1)
        left_vectors = left_scores
        right_vectors = right_scores
        # left_vectors = tf.contrib.layers.batch_normalization(left_scores,training=training, axis = -1)
        # right_vectors = tf.contrib.layers.batch_norm(right_scores,updates_collections=None,
        #                                                           is_training=training,center=True,scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"])

        left_vectors = left_vectors[:,1:-1,:]
        right_vectors = right_vectors[:,1:-1,:]

        #out_vectors = tf.concat([left_vectors,right_vectors],axis=-1)

        out_vectors = tf.concat([left_vectors ,  right_vectors],axis = -1)

        #out_vectors = tf.contrib.layers.batch_norm(out_vectors,decay = 0.9,updates_collections=None,
        #                                                          is_training=training,center=True,scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"])
        #out_vectors = tf.nn.tanh(out_vectors)

        out_vectors = tf.layers.dense(out_vectors, global_len)
        out_vectors = tf.layers.batch_normalization(out_vectors,training=training, axis= -1)
        out_vectors = tf.nn.tanh(out_vectors)

        return out_vectors, sentence_embeddings



class vanillaLSTM:
    def __init__(self):
        pass
    def apply(self,Input_txt_indices,batchSize,memoryLen, vocabularySize, global_len, embeddingSize, outputSize,training):

        word_embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1), dtype=tf.float32)
        sentence_embeddings = tf.nn.embedding_lookup(word_embeddings, Input_txt_indices, max_norm=None)



        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings, training = training, axis = -1)
        sentence_embeddings_left = sentence_embeddings

        left_vector = tf.get_variable(shape=[1, 1, embeddingSize], dtype=tf.float32, name='LeftVector')
        left_vector = tf.tile(left_vector, [batchSize, 1, 1])
        sentence_embeddings_left = tf.concat([left_vector,sentence_embeddings_left], axis = 1)

        with tf.variable_scope('LRLSTM'):
            LeftSentenceEncoderCell = tf.nn.rnn_cell.LSTMCell(memoryLen)
            #LeftSentenceEncoderCell = bnLSTM.BNLSTMCell(memoryLen,training)
            LeftHypothesises, leftHypothesisesState = tf.nn.dynamic_rnn(LeftSentenceEncoderCell, sentence_embeddings_left,
                                                                        dtype=tf.float32)



        return  tf.nn.tanh(LeftHypothesises[:,1:,:]), sentence_embeddings






class SpatialAttention:
    def __init__(self, name):
        pass
    def apply(self,vggFeatures, TxtSignal, globalLen, middleSize = 200,training=1):
        # Visual = tf.contrib.layers.batch_norm(vggFeatures,is_training=training,decay = 0.9,updates_collections=None,variables_collections=["batch_norm_non_trainable_variables_collection"])
        Visual = tf.layers.dense(vggFeatures,globalLen)
        Visual = tf.layers.batch_normalization(Visual,training=training)
        Visual = tf.nn.tanh(Visual)




        #TxtSignal = tf.contrib.layers.batch_norm(TxtSignal,decay = 0.9,updates_collections=None,is_training=training,center=True,scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"])
        TxtSignal = tf.layers.dense(TxtSignal,middleSize)
        Visual_middle = tf.layers.dense(Visual, middleSize)

        TxtSignal = tf.expand_dims(TxtSignal, axis=1)
        TxtSignal = tf.tile(TxtSignal, [1,Visual_middle.get_shape().as_list()[1],1])


        middle = Visual_middle + TxtSignal
        middle = tf.layers.batch_normalization(middle,  training=training)
        middle = tf.nn.tanh(middle)


        middle = tf.layers.dense(middle,1)
        middle = tf.squeeze(middle)
        middle = tf.layers.batch_normalization(middle, -1,training=training)
        Attention = tf.nn.softmax(middle, name='VisualAttention')


        output = tf.einsum('ijk,ij->ik', Visual, Attention)
        #output = weightedAverage(Visual, Attention)
        # output = tf.layers.dense(output,globalLen)
        # output = tf.contrib.layers.batch_norm(output,decay = 0.9,
        #                                                       updates_collections=None,is_training=training,center=True,scale=True,
        #                                       variables_collections=["batch_norm_non_trainable_variables_collection"])
        # output = tf.nn.tanh(output)
        return output, Attention


def weightedAverage(input, weights):
    weights = tf.expand_dims(weights,2)
    weights = tf.tile(weights,[1,1,input.get_shape().as_list()[-1]])

    return tf.reduce_sum(tf.multiply(input,weights), axis = 1)

class ConvolutionalDep:
    def apply_FB2(self,Input_txt_indices,positions,batchSize,memoryLen, vocabularySize, global_len, embeddingSize, outputSize,training):
        # word_embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1), dtype=tf.float32)
        word_embeddings = tf.get_variable('ConvWordEmbeddings', [vocabularySize, embeddingSize], tf.float32)
        sentence_embeddings_original = tf.nn.embedding_lookup(word_embeddings, Input_txt_indices, max_norm=1)

        position_embeddings = tf.get_variable('ConvPositionEmbeddings', [32, embeddingSize], tf.float32)
        position_sentence_embeddings = tf.nn.embedding_lookup(position_embeddings, positions, max_norm=1)
        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings_original, training = training)
        position_sentence_embeddings = tf.layers.batch_normalization(position_sentence_embeddings, training =training, axis = -1)

        posEMB = tf.nn.sigmoid(position_sentence_embeddings,name='posembMatrix')

        sentence_embeddings = sentence_embeddings * posEMB

        embedded = tf.layers.conv1d(sentence_embeddings, outputSize, 3, 1, 'SAME')

        position_embeddings2 = tf.Variable(tf.random_uniform([32, outputSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings2 = tf.nn.embedding_lookup(position_embeddings2, positions, max_norm=1)

        # embedded = embedded * tf.nn.sigmoid(position_sentence_embeddings2)
        embedded = tf.concat([embedded , position_sentence_embeddings2], axis=-1)

        embedded = tf.layers.conv1d(embedded, int(global_len/2), 3, 1, 'SAME')
        embedded = GLU(embedded, training)
        embedded = tf.layers.conv1d(embedded, global_len, 3, 1, 'SAME')
        embedded = GLU(embedded, training)


        return embedded, sentence_embeddings_original
    def apply(self, Input_txt_indices,positions,batchSize,memoryLen, vocabularySize, embeddingSize, outputSize,training):
        word_embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1), dtype=tf.float32)
        sentence_embeddings_original = tf.nn.embedding_lookup(word_embeddings, Input_txt_indices, max_norm=1)



        position_embeddings = tf.Variable(tf.random_uniform([32, embeddingSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings = tf.nn.embedding_lookup(position_embeddings, positions, max_norm=1)
        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings_original,
                                                                  training=training)
        position_sentence_embeddings = tf.layers.batch_normalization(position_sentence_embeddings, training= training)
        sentence_embeddings = sentence_embeddings * tf.nn.sigmoid(position_sentence_embeddings)


        embedded = tf.layers.conv1d(sentence_embeddings, outputSize,3,1,'SAME')

        position_embeddings2 = tf.Variable(tf.random_uniform([32, outputSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings2 = tf.nn.embedding_lookup(position_embeddings2, positions, max_norm=1)

        embedded = embedded * tf.nn.sigmoid(position_sentence_embeddings2)

        embedded = tf.layers.conv1d(embedded, outputSize, 3, 1, 'SAME')
        embedded = tf.nn.tanh(embedded)
        return embedded, sentence_embeddings_original

    def apply_FB(self, Input_txt_indices,positions,batchSize,memoryLen, vocabularySize, embeddingSize, outputSize,training):
        word_embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1), dtype=tf.float32)
        sentence_embeddings_original = tf.nn.embedding_lookup(word_embeddings, Input_txt_indices, max_norm=1)



        position_embeddings = tf.Variable(tf.random_uniform([32, embeddingSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings = tf.nn.embedding_lookup(position_embeddings, positions, max_norm=1)
        sentence_embeddings = tf.contrib.layers.batch_norm(sentence_embeddings_original,trainable=1,decay = 0.9,updates_collections=None,
                                                                  is_training=training,center=True,scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"])
        position_sentence_embeddings = tf.contrib.layers.batch_norm(position_sentence_embeddings,trainable=1,decay = 0.9,updates_collections=None,
                                                                  is_training=training,center=True,scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"])
        sentence_embeddings = sentence_embeddings * tf.nn.sigmoid(position_sentence_embeddings)


        embedded = tf.layers.conv1d(sentence_embeddings, outputSize,3,1,'SAME')

        position_embeddings2 = tf.Variable(tf.random_uniform([32, outputSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings2 = tf.nn.embedding_lookup(position_embeddings2, positions, max_norm=1)

        #embedded = embedded * tf.nn.sigmoid(position_sentence_embeddings2)
        embedded = embedded + position_sentence_embeddings2
        embedded = GLU(embedded, training)
        embedded = tf.layers.conv1d(embedded, outputSize, 3, 1, 'SAME')
        embedded = tf.nn.tanh(embedded)

        return embedded, sentence_embeddings_original


    def apply_roles(self, Input_roles_features,positions,batchSize,memoryLen, embeddingSize, outputSize,training):

        sentence_embeddings_original = Input_roles_features



        position_embeddings = tf.Variable(tf.random_uniform([32, embeddingSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings = tf.nn.embedding_lookup(position_embeddings, positions, max_norm=1)
        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings_original,training=training)
        position_sentence_embeddings = tf.layers.batch_normalization(position_sentence_embeddings, training=training)
        sentence_embeddings = sentence_embeddings * tf.nn.sigmoid(position_sentence_embeddings)


        embedded = tf.layers.conv1d(sentence_embeddings, outputSize,3,1,'SAME')

        position_embeddings2 = tf.Variable(tf.random_uniform([32, outputSize], -1, 1), dtype=tf.float32)
        position_sentence_embeddings2 = tf.nn.embedding_lookup(position_embeddings2, positions, max_norm=1)

        #embedded = embedded * tf.nn.sigmoid(position_sentence_embeddings2)
        embedded = embedded + position_sentence_embeddings2
        embedded = GLU(embedded, training)
        embedded = tf.layers.conv1d(embedded, outputSize, 3, 1, 'SAME')
        embedded = tf.nn.tanh(embedded)

        return embedded, sentence_embeddings_original



def GLU(Y, training):
    dims = Y.get_shape().as_list()[-1]
    if(dims%2 != 0):
        print('For GLU an even dims is required...')
        return -1
    A, B = tf.split(Y, 2, -1)
    B = tf.layers.batch_normalization(B, training= training ,axis = -1)
    return A*tf.nn.sigmoid(B)