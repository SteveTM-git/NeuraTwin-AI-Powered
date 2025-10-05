import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class TurbineAnomalyDetector:
    """
    AI-powered anomaly detection system for turbine health monitoring.
    Uses Isolation Forest for real-time anomaly detection and LSTM for failure prediction.
    """
    
    def __init__(self, data_file='turbine_sensor_data.csv'):
        """Initialize the anomaly detector with sensor data."""
        print("🔄 Loading sensor data...")
        self.df = pd.read_csv(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Feature columns (sensor readings)
        self.feature_cols = ['temperature', 'vibration', 'rpm', 'pressure']
        
        # Prepare data
        self.X = self.df[self.feature_cols].values
        self.y = self.df['status'].map({'Normal': 0, 'Warning': 1, 'Critical': 2}).values
        
        # Initialize models
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.lstm_model = None
        
        print(f"✅ Loaded {len(self.df)} samples")
        print(f"📊 Status distribution:\n{self.df['status'].value_counts()}\n")
    
    def train_isolation_forest(self, contamination=0.15):
        """
        Train Isolation Forest for anomaly detection.
        
        Args:
            contamination: Expected proportion of anomalies (0.15 = 15%)
        """
        print("🌲 Training Isolation Forest...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            verbose=0
        )
        
        self.isolation_forest.fit(X_scaled)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)
        
        # Add predictions to dataframe
        self.df['predicted_anomaly'] = predictions
        self.df['anomaly_score'] = anomaly_scores
        
        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        pred_binary = (predictions == -1).astype(int)
        true_binary = (self.y > 0).astype(int)  # Warning or Critical = 1
        
        # Calculate accuracy
        accuracy = np.mean(pred_binary == true_binary) * 100
        
        print(f"✅ Isolation Forest trained!")
        print(f"🎯 Detection Accuracy: {accuracy:.2f}%")
        print(f"🚨 Detected {sum(predictions == -1)} anomalies")
        
        # Save model
        joblib.dump(self.isolation_forest, 'isolation_forest_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print("💾 Model saved: isolation_forest_model.pkl\n")
        
        return predictions, anomaly_scores
    
    def create_sequences(self, data, sequence_length=50):
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data array
            sequence_length: Number of time steps to look back
        
        Returns:
            X: Sequences of sensor readings
            y: Target labels (next state)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(data) - sequence_length):
            X_seq.append(data[i:i+sequence_length])
            y_seq.append(self.y[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, sequence_length=50, num_features=4):
        """Build LSTM neural network for time-series prediction."""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')  # 3 classes: Normal, Warning, Critical
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lstm(self, sequence_length=50, epochs=20, batch_size=32):
        """
        Train LSTM for failure prediction.
        
        Args:
            sequence_length: Number of time steps to consider
            epochs: Training epochs
            batch_size: Batch size for training
        """
        print("🧠 Training LSTM for failure prediction...")
        
        # Scale data
        X_scaled = self.scaler.transform(self.X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, sequence_length)
        
        # Split train/test (80/20)
        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Testing samples: {len(X_test)}")
        
        # Build and train model
        self.lstm_model = self.build_lstm_model(sequence_length, len(self.feature_cols))
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ LSTM Training Complete!")
        print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
        print(f"📉 Test Loss: {loss:.4f}")
        
        # Save model
        self.lstm_model.save('lstm_model.keras')
        print("💾 Model saved: lstm_model.keras\n")
        
        # Make predictions
        y_pred = np.argmax(self.lstm_model.predict(X_test, verbose=0), axis=1)
        
        return history, y_test, y_pred
    
    def visualize_results(self):
        """Visualize anomaly detection results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        # Color mapping
        status_colors = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
        colors = self.df['status'].map(status_colors)
        
        # 1. Temperature with anomaly detection
        ax1 = axes[0, 0]
        ax1.scatter(range(len(self.df)), self.df['temperature'], 
                   c=colors, alpha=0.5, s=10, label='Actual Status')
        anomalies = self.df[self.df['predicted_anomaly'] == -1]
        ax1.scatter(anomalies.index, anomalies['temperature'], 
                   c='purple', marker='x', s=50, label='Detected Anomaly', linewidths=2)
        ax1.set_ylabel('Temperature (°C)', fontweight='bold')
        ax1.set_title('Temperature Anomaly Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Vibration with anomaly detection
        ax2 = axes[0, 1]
        ax2.scatter(range(len(self.df)), self.df['vibration'], 
                   c=colors, alpha=0.5, s=10, label='Actual Status')
        ax2.scatter(anomalies.index, anomalies['vibration'], 
                   c='purple', marker='x', s=50, label='Detected Anomaly', linewidths=2)
        ax2.set_ylabel('Vibration (Hz)', fontweight='bold')
        ax2.set_title('Vibration Anomaly Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Anomaly Score Distribution
        ax3 = axes[1, 0]
        normal_scores = self.df[self.df['status'] == 'Normal']['anomaly_score']
        fault_scores = self.df[self.df['status'].isin(['Warning', 'Critical'])]['anomaly_score']
        
        ax3.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green')
        ax3.hist(fault_scores, bins=50, alpha=0.6, label='Fault', color='red')
        ax3.set_xlabel('Anomaly Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Anomaly Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Status Timeline
        ax4 = axes[1, 1]
        status_numeric = self.df['status'].map({'Normal': 0, 'Warning': 1, 'Critical': 2})
        ax4.plot(status_numeric.values, linewidth=2, color='blue', label='True Status')
        ax4.fill_between(range(len(self.df)), 0, status_numeric.values, 
                        alpha=0.3, color='blue')
        ax4.set_ylabel('Status Level', fontweight='bold')
        ax4.set_xlabel('Sample Index', fontweight='bold')
        ax4.set_title('Health Status Timeline')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Normal', 'Warning', 'Critical'])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        print("📈 Results saved: anomaly_detection_results.png")
        plt.show()
    
    def predict_realtime(self, sensor_reading):
        """
        Predict health status for a single sensor reading.
        
        Args:
            sensor_reading: Dict with keys: temperature, vibration, rpm, pressure
        
        Returns:
            Prediction dict with anomaly status and probability
        """
        # Convert to array
        reading_array = np.array([[
            sensor_reading['temperature'],
            sensor_reading['vibration'],
            sensor_reading['rpm'],
            sensor_reading['pressure']
        ]])
        
        # Scale
        reading_scaled = self.scaler.transform(reading_array)
        
        # Predict with Isolation Forest
        is_anomaly = self.isolation_forest.predict(reading_scaled)[0]
        anomaly_score = self.isolation_forest.score_samples(reading_scaled)[0]
        
        status = "🚨 ANOMALY DETECTED" if is_anomaly == -1 else "✅ Normal"
        
        return {
            'status': status,
            'is_anomaly': is_anomaly == -1,
            'anomaly_score': float(anomaly_score),
            'reading': sensor_reading
        }


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 TURBINE ANOMALY DETECTION - AI MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Initialize detector
    detector = TurbineAnomalyDetector('turbine_sensor_data.csv')
    
    # Train Isolation Forest
    print("\n" + "=" * 60)
    predictions, scores = detector.train_isolation_forest(contamination=0.15)
    
    # Train LSTM
    print("=" * 60)
    history, y_test, y_pred = detector.train_lstm(sequence_length=50, epochs=15, batch_size=32)
    
    # Visualize results
    print("=" * 60)
    print("📊 Generating visualizations...")
    detector.visualize_results()
    
    # Test real-time prediction
    print("\n" + "=" * 60)
    print("🧪 Testing Real-Time Prediction")
    print("=" * 60)
    
    # Test with normal reading
    normal_reading = {
        'temperature': 72,
        'vibration': 1.3,
        'rpm': 2200,
        'pressure': 98
    }
    result = detector.predict_realtime(normal_reading)
    print(f"\n📊 Normal Reading Test:")
    print(f"   Status: {result['status']}")
    print(f"   Anomaly Score: {result['anomaly_score']:.4f}")
    
    # Test with fault reading
    fault_reading = {
        'temperature': 105,
        'vibration': 5.8,
        'rpm': 1200,
        'pressure': 65
    }
    result = detector.predict_realtime(fault_reading)
    print(f"\n📊 Fault Reading Test:")
    print(f"   Status: {result['status']}")
    print(f"   Anomaly Score: {result['anomaly_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📦 Generated Files:")
    print("   • isolation_forest_model.pkl")
    print("   • lstm_model.keras")
    print("   • scaler.pkl")
    print("   • anomaly_detection_results.png")
    print("=" * 60)