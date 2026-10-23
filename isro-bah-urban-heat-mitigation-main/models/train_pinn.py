import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class PhysicsInformedLoss(tf.keras.losses.Loss):
    def __init__(self, scaler_mean, scaler_scale, lambda_physics=0.05, name='physics_informed_loss', **kwargs):
        super(PhysicsInformedLoss, self).__init__(name=name, **kwargs)
        self.lambda_physics = lambda_physics
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # Store scaler statistics as TensorFlow constants
        self.scaler_mean = tf.constant(scaler_mean, dtype=tf.float32)
        self.scaler_scale = tf.constant(scaler_scale, dtype=tf.float32)

    def call(self, y_true, y_pred, inputs):
        data_loss = self.mse_loss(y_true, y_pred)

        ndvi = inputs[:, 0:1]
        ndbi = inputs[:, 1:2]
        air_temp = inputs[:, 2:3]

        predicted_le = ndvi * 250.0

        predicted_h = ndbi * 300.0 * (y_pred - air_temp)

        physics_residual = (predicted_h + predicted_le - 400.0) / 400.0
        physical_loss = tf.reduce_mean(tf.square(physics_residual))

        return data_loss + (self.lambda_physics * physical_loss)

def build_pinn_model(input_shape=(4,)):
    inputs = tf.keras.Input(shape=input_shape, name='environmental_input')
    x = tf.keras.layers.Dense(64, activation='swish', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(32, activation='swish', name='dense_2')(x)

    x = tf.keras.layers.Dense(1, activation='sigmoid', name='sigmoid_out')(x)

    outputs = tf.keras.layers.Lambda(lambda val:(val * 60.0) + 20.0, name='predicted_list')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Gujarat_UBH_PINN')

    return model


class PINNTrainer:
    def __init__(self, model, scaler_mean, scaler_scale, lambda_physics=0.1):
        self.model = model
        self.loss_fn = PhysicsInformedLoss(
            scaler_mean = scaler_mean,
            scaler_scale = scaler_scale,
            lambda_physics=lambda_physics
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss_value = self.loss_fn.call(y_true=y_batch, y_pred=predictions, inputs=x_batch)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.total_loss_metric.update_state(loss_value)
        self.mse_metric.update_state(y_batch, predictions)
        return loss_value


def run_training_pipeline(csv_path, epochs=100, batch_size=32):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing source file: {csv_path}. Run data extraction step first.")

    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    feature_cols = ['NDVI', 'NDBI', 'Air_Temp', 'Wind_Speed']
    X = df[feature_cols].values
    y = df[['LST']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled.astype(np.float32), y.astype(np.float32), test_size=0.2, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    pinn_net = build_pinn_model(input_shape=(4,))
    trainer = PINNTrainer(
        model=pinn_net,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        lambda_physics=0.0001
    )

    print("\n--- Starting Physics-Informed Training Loop ---")
    for epoch in range(1, epochs + 1):
        trainer.total_loss_metric.reset_state()
        trainer.mse_metric.reset_state()

        for x_batch, y_batch in train_dataset:
            trainer.train_step(x_batch, y_batch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} -> "
                  f"Total Combined Loss: {trainer.total_loss_metric.result().numpy():.4f} | "
                  f"Data MSE: {trainer.mse_metric.result().numpy():.4f}")

    print("\n--- Running Evaluation Validation ---")
    val_preds = pinn_net(tf.convert_to_tensor(X_val, dtype=tf.float32), training=False)
    val_mse = tf.keras.losses.MeanSquaredError()(y_val, val_preds)
    final_val_mse = tf.reduce_mean(val_mse).numpy()
    print(f"Final Out-Of-Sample Validation MSE: {final_val_mse:.4f}")

    os.makedirs("./saved_models", exist_ok=True)
    pinn_net.save("./saved_models/gujarat_uhi_pinn_model.keras")
    print("Model structure saved cleanly to './saved_models/gujarat_uhi_pinn_model.keras'")

    print("\n--- Running Evaluation Validation ---")
    val_preds = pinn_net(tf.convert_to_tensor(X_val, dtype=tf.float32), training=False)

    y_true_np = y_val
    y_pred_np = val_preds.numpy()

    # Calculate Regression Metrics
    val_mse = tf.reduce_mean(tf.square(y_val - val_preds)).numpy()
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_true_np, y_pred_np)
    val_r2 = r2_score(y_true_np, y_pred_np)

    print(f"Validation MSE:  {val_mse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f} °C")
    print(f"Validation MAE:  {val_mae:.4f} °C")
    print(f"Validation R²:   {val_r2:.4f}")

    os.makedirs("./saved_models", exist_ok=True)
    pinn_net.save("./saved_models/gujarat_uhi_pinn_model.keras")
    print("\nModel structure saved cleanly to './saved_models/gujarat_uhi_pinn_model.keras'")

    return pinn_net, scaler


if __name__ == "__main__":
    pass