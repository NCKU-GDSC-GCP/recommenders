from base.base_model import BaseModel
import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Activation
from keras.models import Model
from keras import regularizers


class recommender(BaseModel):
    def __init__(self, config, K, M, N):
        super(recommender, self).__init__(config)
        self.build_model(K, M, N)
        self.init_saver()

    # build model
    def build_model(self, K, M, N):
        customer = Input(shape=(1, ))
        vendor = Input(shape=(1, ))
        P_embedding = Embedding(M, K, embeddings_regularizer=regularizers.l2())(customer)
        Q_embedding = Embedding(N, K, embeddings_regularizer=regularizers.l2())(vendor)
        customer_bias = Embedding(M, 1, embeddings_regularizer=regularizers.l2())(customer)
        vendor_bias = Embedding(N, 1, embeddings_regularizer=regularizers.l2())(vendor)


        # Layers
        P_embedding = Flatten()(P_embedding)
        Q_embedding = Flatten()(Q_embedding)
        customer_bias = Flatten()(customer_bias)
        vendor_bias = Flatten()(vendor_bias)
        R = Concatenate()([P_embedding, Q_embedding, customer_bias, vendor_bias])

        # Neural network for Deep Learning
        R = Dense(2048)(R)
        R = Activation('linear')(R)
        R = Dense(256)(R)
        R = Activation('linear')(R)
        R = Dense(1)(R)

        # Model compile
        model = Model(inputs=[customer, vendor], outputs=R)
        self.model = model

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
