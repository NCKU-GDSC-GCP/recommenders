import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Embedding, Dense
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model



def get_retrieval(user_num, item_num):
    user_id = Input(shape=(1, ), name="user_id")
    gender = Input(shape=(1, ), name="gender")
    like_tags = Input(shape=(1, ), name="like_tags")

    restaurant_id = Input(shape=(1, ), name="restaurant_id")
    restaurant_tag = Input(shape=(1, ), name="restaurant_tag")
    rating = Input(shape=(1, ), name="rating")

    # User tower
    user_vector = concatenate([
        Embedding(user_num, 100)(user_id),
        Embedding(2, 2)(gender),
        Embedding(17, 2)(like_tags)
    ])
    user_vector = Dense(32, activation="relu")(user_vector)
    user_vector = Dense(8, activation="relu", name="user_embedding", kernel_regularizer="l2")(user_vector)

    # Item tower
    item_vector = concatenate([
        Embedding(item_num, 100)(restaurant_id),
        Embedding(17, 2)(restaurant_tag),
        Embedding(1, 1)(rating)
    ])
    item_vector = Dense(32, activation="relu")(item_vector)
    item_vector = Dense(8, activation="relu", name="item_embedding", kernel_regularizer="l2")(item_vector)

    sim = cosine_similarity(user_vector, item_vector)
    sim = tf.expand_dims(sim, 1)

    output = Dense(1, activation="sigmoid")(sim)

    return Model(inputs=[user_id, gender, like_tags, restaurant_id, restaurant_tag, rating], outputs=[output])
