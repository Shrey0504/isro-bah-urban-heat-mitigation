import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from sklearn.preprocessing import StandardScaler
from models.train_pinn import build_pinn_model


class UrbanHeatOptimizer:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def evaluate_intervention(self, base_feature, delta_ndvi=0.0, delta_ndbi=0.0):
        features = np.array(base_feature, dtype=np.float32).copy()
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)

        features[:, 0] = np.clip(features[:, 0] + delta_ndvi, -1.0, 1.0)
        features[:, 1] = np.clip(features[:, 1] + delta_ndbi, -1.0, 1.0)

        features_scaled = self.scaler.transform(features)

        predicted_lst = self.model(tf.convert_to_tensor(features_scaled, dtype=np.float32), training=False)
        return float(predicted_lst.numpy()[0][0])

    def first_optimal_intervention(self, base_feature, max_ndvi_increase=0.2, max_ndbi_decrease=0.2, steps=20, lr=0.01):
        raw_features = np.array(base_feature, dtype=np.float32).copy()
        if len(raw_features.shape) == 1:
            raw_features = np.expand_dims(base_feature, axis=0)

        base_scaled = self.scaler.transform(raw_features)
        baseline_lst = float(self.model(tf.convert_to_tensor(base_scaled, dtype=np.float32), training=False).numpy()[0][0])

        delta_ndvi = tf.Variable(0.0, dtype=np.float32)
        delta_ndbi = tf.Variable(0.0, dtype=np.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(steps):
            with tf.GradientTape() as tape:
                modified_features = tf.concat([
                    raw_features[:, 0:1] + tf.clip_by_value(delta_ndvi, 0.0, max_ndvi_increase),
                    raw_features[:, 1:2] - tf.clip_by_value(delta_ndbi, 0.0, max_ndbi_decrease),
                    raw_features[:, 2:4]
                ], axis=1)

                mean = tf.convert_to_tensor(self.scaler.mean_, dtype=np.float32)
                var = tf.convert_to_tensor(self.scaler.var_, dtype=np.float32)
                scaled_inputs = (modified_features - mean) / tf.math.sqrt(var + 1e-7)

                predicted_lst = self.model(scaled_inputs, training=False)

                loss = tf. reduce_mean(predicted_lst)

            gradient = tape.gradient(loss, [delta_ndvi, delta_ndbi])
            optimizer.apply_gradients(zip(gradient, [delta_ndvi, delta_ndbi]))

        opt_ndvi_shift = float(tf.clip_by_value(delta_ndvi, 0.0, max_ndvi_increase).numpy())
        opt_ndbi_shift = float(tf.clip_by_value(delta_ndbi, 0.0, max_ndbi_decrease).numpy())

        optimized_lst = self.evaluate_intervention(base_feature, delta_ndvi=opt_ndvi_shift, delta_ndbi=-opt_ndbi_shift)
        temp_reduction = baseline_lst - optimized_lst

        return {
            "baseline_lst_celsius": round(baseline_lst, 2),
            "optimized_lst_celsius": round(optimized_lst, 2),
            "estimated_reduction": round(temp_reduction, 2),
            "recommended_ndvi_increase": round(opt_ndvi_shift, 3),
            "recommended_ndbi_decrease": round(opt_ndbi_shift, 3)
        }

if __name__ == "__main__":
    print("Simulating Optimized Pipeline Test")

    mock_model = build_pinn_model()
    mock_scaler = StandardScaler()

    dummy_data = np.array([
        [0.15, 0.65, 38.0, 2.5],
        [0.12, 0.70, 40.0, 1.8],
        [0.30, 0.40, 35.0, 3.2]
    ])
    mock_scaler.fit(dummy_data)

    optimizer_engine = UrbanHeatOptimizer(model=mock_model, scaler=mock_scaler)

    hot_zone_features = [0.1, 75.0, 41.0, 1.5]

    manual_cooling = optimizer_engine.first_optimal_intervention(
        hot_zone_features, delta_ndvi=0.15, delta_ndbi=-0.10
    )

    print("Manual Intervention Prediction Completed")

    optimization_results = optimizer_engine.first_optimal_intervention(
        base_feature=hot_zone_features,
        max_ndvi_increase=0.25,
        max_ndbi_decrease=0.20
    )

    print("\n----Optimization Output Metrics----")
    for key, val in optimization_results.items():
        print(f"{key.replace('_',' ').title()}: {val}")