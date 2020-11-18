class AverageIntervalWidth(tf.keras.metrics.Metric):

    def __init__(self, name='AIW', **kwargs):
        super(AverageIntervalWidth, self).__init__(name=name, **kwargs)
        self.average_interval_width = self.add_weight(name='AIW', initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # the state would be updated everytime we have a new calculation
        self.average_interval_width.assign(tf.math.reduce_mean(y_pred))

    def result(self):
        return self.average_interval_width

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.average_interval_width.assign(0.0)