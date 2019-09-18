import numpy as np
import tensorflow as tf
import os,shutil
import DataGenerator ### User must writes a costum Generator
import Fixed_Module as Modules
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(tf.__version__)
ty = 'ECCV18'


vocabularySize = 25000
embeddingSize = 500
batchSize = 32
globalLen = 3000

graph = tf.Graph()

with graph.as_default():



    Input_Txt = tf.placeholder(dtype=tf.int32, shape = [batchSize, 30], name = 'inputSentence')### Word indices of the sentence
    DetectionGTIndex = tf.placeholder(dtype=tf.int64, shape = [batchSize], name = 'DetectionGT')### The groundTruth of the inacurate word position in the sentence
    Input_positions = tf.placeholder(dtype=tf.int32, shape=[batchSize, 30], name='inputSentencePositions') ### The position of each word in the sentence. (To deal with different senteces len)
    Input_vgg19_196 = tf.placeholder(dtype=tf.float32, shape=[batchSize, 14, 14, 512], name='inputVgg19_196')#### vgg19 features extracted from video and max-pooled over time
    Input_C3D = tf.placeholder(dtype=tf.float32, shape=[batchSize, 4096], name='inputC3D') ### C3D features extracted from all 16 frames shots, max-pooled over time
    TrueWords = tf.placeholder(dtype=tf.int64, shape = [batchSize], name = 'TrueWords') ### Ground-Truth of the accurate word to be predicted
    training = tf.placeholder(tf.bool, name = 'trainingMode') # train or test mode, 1 or 0
    keep_prob = tf.placeholder(tf.float32, name='kepp_prob_input')

    DetectionGT = tf.one_hot(DetectionGTIndex, 30)

    with tf.variable_scope('DETECTION',initializer=tf.contrib.layers.xavier_initializer()):

        with tf.variable_scope('REST'):
            ##### Text Encodings
            TextEncoder = Modules.LRRLLSTMS()
            EncodedSentence_LSTM, sentence_embeddings = TextEncoder.apply(Input_Txt, batchSize, 400, vocabularySize, globalLen,
                                                                     embeddingSize,
                                                                     500, training)


            sentence_embeddings = tf.contrib.layers.batch_norm(sentence_embeddings, is_training = training)

            embeddings_LSMT = tf.nn.tanh(sentence_embeddings)




            TextEncoder = Modules.ConvolutionalDep()

            EncodedSentence_Conv, sentence_embeddings = TextEncoder.apply_FB2(Input_Txt,Input_positions, batchSize, 400, vocabularySize, globalLen,
                                                                     embeddingSize,
                                                                     500, training)
            sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings,training=training,axis = -1)

            embeddings_Conv = tf.nn.tanh(sentence_embeddings)


            embeddings1 = tf.concat([embeddings_LSMT, embeddings_Conv], axis=-1)
            senteces_encoded1 = tf.concat([EncodedSentence_LSTM, EncodedSentence_Conv], axis=-1)

            embeddings1 = tf.nn.tanh(embeddings1)
            senteces_encoded1 = tf.nn.tanh(senteces_encoded1)
            embeddings1 = tf.layers.dense(embeddings1, 250)
            senteces_encoded1 = tf.layers.dense(senteces_encoded1, 250)



            #### Visual bias Computation
            Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D, training=training, axis=-1)
            Input_C3D_normalized = tf.layers.dense(Input_C3D_normalized, 1000)
            Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D_normalized, training=training, axis=-1)
            Input_C3D_normalized = tf.nn.relu(Input_C3D_normalized)
            Input_C3D_normalized = tf.layers.dense(Input_C3D_normalized, globalLen)
            Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D_normalized, training=training, axis=-1)
            Input_C3D_normalized = tf.nn.tanh(Input_C3D_normalized)
            Input_C3D_normalized = tf.expand_dims(Input_C3D_normalized, axis=1)
            Input_C3D_normalized = tf.tile(Input_C3D_normalized, [1, 30, 1])

            word_embeddings_bias = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1, 1),
                                               dtype=tf.float32)
            sentence_embeddings_bias = tf.nn.embedding_lookup(word_embeddings_bias, Input_Txt, max_norm=None)
            word_embeddings_bias = tf.layers.batch_normalization(sentence_embeddings_bias, axis=-1)
            word_embeddings_bias = tf.nn.tanh(word_embeddings_bias)
            word_embeddings_bias = tf.layers.dense(word_embeddings_bias, senteces_encoded1.get_shape()[-1])
            word_embeddings_bias = tf.nn.sigmoid(word_embeddings_bias, name='biasgates')
            Input_C3D_normalized = tf.layers.dense(Input_C3D_normalized, word_embeddings_bias.get_shape()[-1])
            Input_C3D_normalized = tf.nn.tanh(Input_C3D_normalized)

            Input_C3D_normalized = tf.multiply(Input_C3D_normalized, word_embeddings_bias)




            ### Detection Scores
            cosVector1 = embeddings1 + senteces_encoded1 + Input_C3D_normalized
            cosVector1 = tf.layers.batch_normalization(cosVector1,axis = -1, training=training)
            cosVector1 = tf.nn.tanh(cosVector1)

            left_scores1 = tf.layers.dense(cosVector1, 1)

            left_scores1 = tf.squeeze(left_scores1, axis=-1)

            loss_detection = tf.losses.softmax_cross_entropy(DetectionGT, left_scores1)

            Detection_Scores = tf.nn.softmax(left_scores1,name='detectionScoresOut')
            prediction = tf.argmax(Detection_Scores, 1,name='detectionWord')
            equality_detection = tf.equal(DetectionGTIndex, prediction)
            accuracy_detection = tf.reduce_mean(tf.cast(equality_detection, tf.float32))

    with tf.variable_scope('FIB',initializer=tf.contrib.layers.xavier_initializer()):
        sentence_embeddings = tf.concat([EncodedSentence_LSTM, EncodedSentence_Conv], axis=-1)
        sentence_embeddings = tf.layers.dense(sentence_embeddings,2000)
        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings, training=training)
        sentence_embeddings = tf.nn.relu(sentence_embeddings)


        sentence_embeddings = tf.layers.dense(sentence_embeddings,globalLen)
        sentence_embeddings = tf.layers.batch_normalization(sentence_embeddings, training=training)
        sentence_embeddings = tf.nn.tanh(sentence_embeddings)


        EncodedSentence_FIB = sentence_embeddings
        l = EncodedSentence_FIB[:, 0, :]
        r = EncodedSentence_FIB[:, -1, :]
        FIBTxt = tf.reduce_max(EncodedSentence_FIB,1)
        FIBTxt = tf.layers.batch_normalization(FIBTxt, axis = -1, training=training)
        with tf.variable_scope('VISUAL196',initializer=tf.contrib.layers.xavier_initializer()):
            Input_vgg19_normalized = Input_vgg19_196
            Input_vgg19_normalized = tf.layers.batch_normalization(Input_vgg19_196,axis =-1,training=training)
            Input_vgg19_normalized = tf.reshape(Input_vgg19_normalized, [batchSize,196, 512])
            AttentionModel = Modules.SpatialAttention(name='FIBSpatialAttention')
            VisualFeatures, SpatialAttenton = AttentionModel.apply(Input_vgg19_normalized, FIBTxt, globalLen,training=training)
            VisualFeatures = tf.expand_dims(VisualFeatures, axis=1)
            VisualFeatures = tf.tile(VisualFeatures, [1, 30, 1])

            Input_C3D_normalized = Input_C3D
            Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D, training=training, axis = -1)
            #Input_C3D_normalized = tf.layers.dense(Input_C3D_normalized,1000)
            #Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D_normalized, training=training, axis = -1)
            #Input_C3D_normalized = tf.nn.relu(Input_C3D_normalized)
            Input_C3D_normalized = tf.layers.dense(Input_C3D_normalized,globalLen)
            Input_C3D_normalized = tf.layers.batch_normalization(Input_C3D_normalized, training=training, axis = -1)
            Input_C3D_normalized = tf.nn.tanh(Input_C3D_normalized)
            Input_C3D_normalized = tf.expand_dims(Input_C3D_normalized, axis=1)
            Input_C3D_normalized = tf.tile(Input_C3D_normalized, [1, 30, 1])


    with tf.variable_scope('INFERENCE',initializer=tf.contrib.layers.xavier_initializer()):
        #joint_representation_FIB = tf.concat([EncodedSentence], axis = -1)
        joint_representation_FIB = EncodedSentence_FIB + Input_C3D_normalized

        # joint_representation_FIB = tf.layers.batch_normalization(joint_representation_FIB,-1,training=training)
        # joint_representation_FIB = tf.nn.tanh(joint_representation_FIB)



        joint_representation_FIB = tf.nn.dropout(joint_representation_FIB, keep_prob)
        MissingWord = tf.layers.dense(joint_representation_FIB, 890)
        MissingWord_Smax = tf.nn.softmax(MissingWord, dim=-1,name='AllMissingWords')
        TrueWords_GT = tf.one_hot(TrueWords, 890)



        MissingWord_FGT = tf.einsum('ijk,ij->ik', MissingWord, DetectionGT, name='MissingWordScoresFGT')
        MissingWord_DT = tf.einsum('ijk,ij->ik', MissingWord, Detection_Scores, name='MissingWordScoresDT')
        #
        loss_fill = tf.losses.softmax_cross_entropy(TrueWords_GT, MissingWord_FGT)
        loss_fill_DT = tf.losses.softmax_cross_entropy(TrueWords_GT, MissingWord_DT)
        loss_fill = loss_fill
        prediction_words = tf.argmax(MissingWord_FGT, 1)
        equality_fill = tf.equal(TrueWords, prediction_words)

        accuracy_fill = tf.reduce_mean(tf.cast(equality_fill, tf.float32))

        equality = tf.logical_and(equality_fill, equality_detection)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


    loss =  loss_fill


    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                       if 'bias' not in v.name]) * 0.00005

    vars1 = [v for v in vars if v.name.startswith('DETECTION/')]
    print(len(vars1))
    vars2 = [v for v in vars if v.name.startswith('FIB/')]
    print(len(vars2))
    vars3 = [v for v in vars if v.name.startswith('INFERENCE/')]
    print(len(vars3))


    loss= loss + lossL2




    optimizer1 = tf.train.AdagradOptimizer(learning_rate=0.01)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
            train_step1 = optimizer1.minimize(loss = 10*loss_fill +lossL2+ loss_detection,var_list=vars)
    #train_step2 = optimizer2.minimize(loss=loss_fill+lossL2 + loss_detection, var_list=vars2)
    #train_step3 = optimizer3.minimize(loss=loss_fill + lossL2 + loss_detection, var_list=vars3)


    train_step = [train_step1]

    initialize = tf.initialize_all_variables()

print('builts')

main_path = '/media/amir/DATA/'
experiment_name = 'bad_rebutt_FIB_justC3D_'+ty
savedModelPath = main_path+'CorrectionModels/models/' + experiment_name+'/'+experiment_name
print(savedModelPath)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
resume  = 0
best_acc = 0
patient = 0
max_patient = 8
visual = 1
visual2 = 1
use_jpeg = 0
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= graph) as sess:
    counter = 0
    saver = tf.train.Saver()
    if resume==1:
        try:
            NewSaver = tf.train.Saver()
            NewSaver.restore(sess, main_path+'CorrectionModels/models/' + experiment_name+'/'+experiment_name)
            print('*****previous model loaded, training will be resumed*****')
        except:
            try:
                shutil.rmtree(main_path+'CorrectionModels/models/' + experiment_name+'/')
            except:
                pass
            sess.run(initialize)
    else:
        try:
            shutil.rmtree(main_path + 'CorrectionModels/models/' + experiment_name + '/')
        except:
            pass
        sess.run(initialize)



    train_writer = tf.summary.FileWriter(main_path+"CorrectionModels/Logs" + "/" + experiment_name + "_tr", sess.graph)
    tf.train.write_graph(sess.graph_def,main_path+"CorrectionModels/Logs" + "/" + experiment_name + "_tr","graph.pbtxt")

    train_writer.add_graph(graph=graph)

    os.makedirs( main_path+'CorrectionModels/models/' + experiment_name + '/',exist_ok=True)

    saver.save(sess, savedModelPath)

    #summary_writer = tf.summary.FileWriter(main_path+'CorrectionModels/models/' + experiment_name + '/'+"output", sess.graph)


    print('asljfsljflsd')


    for x, y, z, k, m,r, p, k2, filenames,jpegs in DataGenerator.gen_visual(ty, batchSize=batchSize,visual = visual,testmode=0,roles_features=1,visual2=visual2):
        feed_dict = feed_dict = {
            Input_Txt: x,
            Input_positions: p,
            Input_C3D: k,
            Input_vgg19_196: k2,
            DetectionGTIndex: y,
            TrueWords: z,
            keep_prob: 0.3,
            training: True
        }

        acc, acc_fill, l, _,Attn196 = sess.run([accuracy,accuracy_fill, loss_fill, train_step,SpatialAttenton], feed_dict=feed_dict)

        counter = counter + 1
        if(counter%3000==0):
            #saver.save(sess, './models/' + experiment_name)
            print(counter)
            print('accuracy is :' + str(acc))
            print('accuracy fill is :' + str(acc_fill))
            #print('accuracy detection is :' + str(acc_detection))
            print('loss is: ' + str(l))
            print(experiment_name)
            np.save('Attn196',Attn196)
            #np.save('mydetection', mydetection)
            #np.save('gtDetection', gtDetection)
            print('stoped')
            print('started')

        if counter==1:
            saver.save(sess, savedModelPath)
            NewSaver = tf.train.Saver()
            NewSaver.restore(sess, main_path + 'CorrectionModels/models/' + experiment_name + '/' + experiment_name)
            print('*****counter 1 saved model and loaded successfully*****')
        if(counter%int(296960/batchSize)==0):
            val_counter = 0

            acc_val_total = 0
            acc_val_detection_total = 0
            acc_val_fib_total = 0
            for x,y, z, k,m,r, p, k2, filenames, jpegs in DataGenerator.gen_visual(ty, batchSize =batchSize, setName = 'testing',visual = visual,testmode=1,roles_features=1,visual2=visual2):
                val_counter = val_counter + x.shape[0]
                feed_dict = feed_dict = {
                    Input_Txt: x,
                    Input_positions: p,
                    Input_C3D: k,
                    Input_vgg19_196: k2,
                    DetectionGTIndex: y,
                    TrueWords: z,
                    keep_prob: 1,
                    training: True
                }
                acc_val,    acc_val_fill,val_loss = sess.run([accuracy, accuracy_fill,loss], feed_dict=feed_dict)
                #acc_val_detection_total = acc_val_detection_total + x.shape[0]*acc_val_detection
                acc_val_total = acc_val_total + x.shape[0] * acc_val
                acc_val_fib_total = acc_val_fib_total + x.shape[0] * acc_val_fill


                if(val_counter>30300):
                    acc_val = acc_val_total/val_counter
                    #acc_val_detection = acc_val_detection_total / val_counter
                    acc_val_fill = acc_val_fib_total / val_counter
                    val_counter = 0
                    print('*********************validation accuracy is ')
                    print(acc_val)
                    #print(acc_val_detection)
                    print(acc_val_fill)

                    print(experiment_name)
                    print(patient)
                    #saver.save(sess, savedModelPath)
                    # print('saved!')
                    print(savedModelPath)
                    print('*********************')
                    break
            if(acc_val>best_acc):
                patient = 0
                best_acc = acc_val
                saver.save(sess, savedModelPath)
                print('saved!')
            else:
                patient = patient + 1
            if(patient>=max_patient):
                break