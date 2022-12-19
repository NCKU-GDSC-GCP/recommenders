from base.base_model import BaseModel
import tensorflow as tf


class recommender(BaseModel):
    def __init__(self, config):
        super(recommender, self).__init__(config)
        self.build_model()
        self.init_saver()

    # build model
    def build_model(self):
        pass

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
