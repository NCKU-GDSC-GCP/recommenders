import tensorflow as tf
from models import retrieval

# test model
r = retrieval.get_retrieval(100, 100)
r.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
r.summary()

