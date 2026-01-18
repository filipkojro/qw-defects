import tensorflow as tf

class DistributionOverlap(tf.keras.Metric):
    def __init__(self, name='dist_overlap', **kwargs):
        super().__init__(name=name, **kwargs)
        self.overlap_sum = self.add_weight(name='overlap_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        intersection = tf.reduce_sum(tf.minimum(y_true, y_pred), axis=-1)

        self.overlap_sum.assign_add(tf.reduce_sum(intersection))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.overlap_sum, self.count)
    
    def reset_state(self):
        self.overlap_sum.assign(0.0)
        self.count.assign(0.0)

def DistributionOverlapSingle(y_true, y_pred):
    do = DistributionOverlap()
    do.reset_state()
    do.update_state(tf.expand_dims(y_true, axis=0), tf.expand_dims(y_pred, axis=0))

    return do.result().numpy()