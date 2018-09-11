import argparse
import json
import datetime
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, RepeatVector
from tensorflow.keras.layers import Masking, Layer, Lambda, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from sclstm import SCLSTM
        

def get_args():
    parser = argparse.ArgumentParser(description="toy dataset to test SCLSTM")

    # data
    parser.add_argument("--n_class", type=int, default=5,
                        help="number of class")

    # training
    parser.add_argument("--eta", type=float, default=1e-4,
                        help="hyperparameter of loss function")
    parser.add_argument("--xi", type=float, default=100,
                        help="hyperparameter of loss function")

    # lstm
    parser.add_argument("--dim_hidden", type=int, default=32,
                        help="state size of hidden state in LSTM")
    parser.add_argument("--max_length", type=int, default=8,
                        help="max length in caption sequence")
    parser.add_argument("--dim_wordvec", type=int, default=32,
                        help="dimension of word vector")

    return parser.parse_args()


def gen_data(n_data, n_class, max_length):

    word_to_index = {}
    index_to_word = []
    for w in ["<NULL>", "<START>", "<END>", "<UNK>"]:
        word_to_index[w] = len(word_to_index)
        index_to_word.append(w)
    for w in ["this", "these", "is", "are", "and"]:
        word_to_index[w] = len(word_to_index)
        index_to_word.append(w)
    for i in range(n_class):
        word_to_index[str(i)] = len(word_to_index)
        index_to_word.append(str(i))

    captions = np.zeros((n_data, max_length), dtype=np.int32)
    controls = np.zeros((n_data, n_class), dtype=np.int32)

    for i in range(n_data):
        n = random.randint(1, min(3, n_class))
        objects = random.sample(range(n_class), n)
        for j in objects:
            controls[i,j] = 1 
        caption = ["<START>"]
        if n == 1:
            for w in ["this", "is", str(objects[0]), "<END>"]:
                caption.append(w)
        else:
            caption.append("these")
            caption.append("are")
            for o in objects:
                caption.append(str(o))
            caption.append(caption[-1])
            caption[-2] = "and"
            caption.append("<END>")
        for j, w in enumerate(caption):
            captions[i,j] = word_to_index[w]
        """
        print(objects)
        print(controls[i])
        print(caption)
        print(captions[i])
        """

    targets = np.zeros((n_data * max_length, len(word_to_index)))
    targets[np.arange(n_data * max_length), captions.reshape(-1)] = 1
    targets = targets.reshape((n_data, max_length, len(word_to_index)))
    targets[:,:-1,:] = targets[:,1:,:]

    return word_to_index, index_to_word, captions, targets, controls


def build_model(dim_control, dim_hidden, max_length, n_word, dim_wordvec):
    caption_input = Input(shape=(max_length,), name="caption_input")
    init_h = Input(shape=(dim_hidden,), name="init_h")
    init_c = Input(shape=(dim_hidden,), name="init_c")
    init_d = Input(shape=(dim_control,), name="init_d")

    embedding_layer = Embedding(n_word, dim_wordvec, input_length=max_length)
    caption_embedding = embedding_layer(caption_input)

    decoder = SCLSTM(dim_hidden, dim_control, return_sequences=True,
            return_state=True)
    #decoder = LSTM(dim_hidden, return_sequences=True, return_state=True)

    #decoder_output, decoder_h, decoder_c = decoder(caption_embedding,
    #        initial_state=[init_h, init_c])
    decoder_output, decoder_h, decoder_c, decoder_d = decoder(caption_embedding,
            initial_state=[init_h, init_c, init_d])
    print(decoder_output.shape)
    print(decoder_h.shape)
    print(decoder_c.shape)
    print(decoder_d.shape)

    def get_h(x):
        return x[:,:,:dim_hidden]

    def get_d(x):
        return x[:,:,dim_hidden:]

    decoder_h_all = Lambda(get_h)(decoder_output)
    decoder_d_all = Lambda(get_d)(decoder_output)

    print(decoder_h_all.shape)
    print(decoder_d_all.shape)

    decoder_dense = Dense(n_word, activation="softmax")
    pred_y = decoder_dense(decoder_h_all)

    print("pred_y.shape", pred_y.shape)

    out = Concatenate(name="concatename_out")([pred_y, decoder_d_all])
    #out = pred_y

    print("out.shape", out.shape)

    model_train = Model(inputs=[caption_input, init_h, init_c, init_d],
            outputs=[out])

    caption_input_inf = Input(shape=(1,), name="input_inf")
    caption_embedding_inf = embedding_layer(caption_input_inf)
    dec_output_inf, dec_h_inf, dec_c_inf, dec_d_inf = decoder(
            caption_embedding_inf,
            initial_state=[init_h, init_c, init_d])
    dec_h_all_inf = Lambda(get_h)(dec_output_inf)
    pred_y_inf = decoder_dense(dec_h_all_inf)

    model_inf = Model(inputs=[caption_input_inf, init_h, init_c, init_d],
            outputs=[pred_y_inf, dec_h_inf, dec_c_inf, dec_d_inf])

    return model_train, model_inf


def main():

    # limit GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    sess.as_default()

    args = get_args()

    n_data = 320

    word_to_index, index_to_word, captions, targets, controls = gen_data(n_data,
            args.n_class, args.max_length)

    print(captions.shape)
    print(targets.shape)
    print(controls.shape)
    # concate controls to targets just to make the shape match with model output
    true_y = np.zeros((targets.shape[0], targets.shape[1], targets.shape[2] + controls.shape[-1]))
    true_y[:,:,:targets.shape[2]] = targets
    # mask "<NULL>"
    true_y[:,:,0] = 0
    print("true_y.shape", true_y.shape)
    print(captions[0,:10])
    print(controls[0])
    print(targets[0,:10])
    print(true_y[0,:10])

    model_train, model_inf = build_model(args.n_class, args.dim_hidden,
            args.max_length, len(word_to_index), args.dim_wordvec)

    def sclstm_loss(y_true, y_pred):
        """
        y_pred: numpy array, shape=(batch_size, time_step, len(word) + len(control))
        y_true: similar as y_pred
        """
        #assert y_pred.shape == y_true.shape
        n_word = len(word_to_index)
        preds = y_pred[:,:,:n_word]
        controls = y_pred[:,:,n_word:]
        print(type(preds))
        xent = -K.sum(y_true[:,:,:n_word] * K.log(preds + 1e-9))
        print(xent)
        reg1 = K.sum(K.abs(controls[:,-1,:]))
        print(reg1)
        t = K.sum(K.abs(controls[:,:,1:] - controls[:,:,:-1]), axis=-1)
        reg2 = K.sum(args.eta * (args.xi ** t))
        print(reg2)
        #return xent + reg1 + reg2
        return xent

    model_train.summary()

    init_h = np.zeros((n_data, args.dim_hidden))
    init_c = np.zeros((n_data, args.dim_hidden))
    sclstm_loss(targets, targets)
    model_train.compile(optimizer='adam', loss=sclstm_loss)
    model_train.fit(x=[captions, init_h, init_c, controls],
            y=[true_y], epochs=200)

    while True:
        tmp_control = np.zeros((1,args.n_class,), dtype=np.int32)
        tmp_caption = np.zeros((1,1,), dtype=np.int32)
        s = input("enter string contain number from 0~{}: ".format(args.n_class))
        for w in s.split():
            tmp_control[0,int(w)] = 1
        tmp_caption[0,0] = word_to_index["<START>"]
        prev_h = np.zeros((1, args.dim_hidden,))
        prev_c = np.zeros((1, args.dim_hidden,))
        prev_control = tmp_control
        res = []
        for i in range(args.max_length):
            dec_output_inf, dec_h_inf, dec_c_inf, dec_d_inf = model_inf.predict(
                    x=[tmp_caption, prev_h, prev_c, prev_control])
            print(tmp_caption)
            print(prev_h)
            print(prev_c)
            print(prev_control)
            prev_h = dec_h_inf
            prev_c = dec_c_inf
            prev_control = dec_d_inf
            print("dec_output_inf.shape", dec_output_inf.shape)
            print(dec_output_inf[0,0])
            idx = np.argmax(dec_output_inf[0,0])
            tmp_caption[0,0] = idx
            print("idx", idx)
            res.append(index_to_word[idx])
            if idx == word_to_index["<END>"]:
                break
        print(res)

if __name__ == "__main__":
    main()
