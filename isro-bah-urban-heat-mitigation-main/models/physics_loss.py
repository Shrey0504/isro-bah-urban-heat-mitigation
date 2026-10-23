import tensorflow as tf

class PhysicsInformedLoss(tf.keras.losses.Loss):
    def __init__(self, scaler_mean, scaler_scale, lambda_physics=0.1, name="physics_informed_loss", **kwargs):
        super(PhysicsInformedLoss, self).__init__(name=name, **kwargs)
        self.lambda_physics = lambda_physics
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.scaler_mean = tf.constant(scaler_mean, dtype=tf.float32)
        self.scaler_scale = tf.constant(scaler_scale, dtype=tf.float32)

    def call(self, y_true, y_pred, inputs):
        data_loss = self.mse_loss(y_true, y_pred)

        raw_inputs = (inputs * self.scaler_scale) + self.scaler_mean

        ndvi = inputs[:, 0:1]
        ndbi = inputs[:, 1:2]
        air_temp = inputs[:, 2:3]

        predicted_le = ndvi * 250.0
        predicted_h = ndbi * 300.0 * (y_pred - air_temp)

        physics_residual = predicted_h + predicted_le - 400.0
        physics_loss = tf.reduce_mean(tf.square(physics_residual))

        return data_loss + (self.lambda_physics * physics_loss)