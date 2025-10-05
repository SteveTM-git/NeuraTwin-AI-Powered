import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

class TurbineSensorSimulator:
    """
    Simulates IoT sensor data for a turbine/engine with configurable fault scenarios.
    Generates: Temperature, Vibration, RPM, and Pressure readings.
    """
    
    def __init__(self, duration_hours=24, sample_rate_seconds=10):
        """
        Initialize the simulator.
        
        Args:
            duration_hours: Total simulation duration in hours
            sample_rate_seconds: Time between sensor readings in seconds
        """
        self.duration_hours = duration_hours
        self.sample_rate_seconds = sample_rate_seconds
        self.num_samples = int((duration_hours * 3600) / sample_rate_seconds)
        
        # Normal operating ranges
        self.normal_ranges = {
            'temperature': (60, 80),      # °C
            'vibration': (0.5, 2.0),      # Hz
            'rpm': (1500, 3000),          # RPM
            'pressure': (80, 120)         # psi
        }
        
        # Fault ranges
        self.fault_ranges = {
            'temperature': (90, 120),
            'vibration': (3.0, 8.0),
            'rpm': (800, 1400),           # Low RPM indicates issues
            'pressure': (50, 70)          # Low pressure indicates leakage
        }
    
    def generate_normal_data(self):
        """Generate normal operating sensor data with slight noise."""
        timestamps = [datetime.now() + timedelta(seconds=i*self.sample_rate_seconds) 
                     for i in range(self.num_samples)]
        
        # Generate normal data with realistic noise
        data = {
            'timestamp': timestamps,
            'temperature': np.random.normal(70, 3, self.num_samples),  # Mean 70°C, std 3
            'vibration': np.random.normal(1.2, 0.3, self.num_samples), # Mean 1.2 Hz
            'rpm': np.random.normal(2250, 150, self.num_samples),      # Mean 2250 RPM
            'pressure': np.random.normal(100, 5, self.num_samples),    # Mean 100 psi
            'status': ['Normal'] * self.num_samples
        }
        
        return pd.DataFrame(data)
    
    def inject_gradual_temperature_fault(self, df, start_idx, duration_samples):
        """
        Simulate bearing failure - gradual temperature increase.
        """
        end_idx = min(start_idx + duration_samples, len(df))
        
        # Create gradual temperature rise
        rise_curve = np.linspace(0, 40, end_idx - start_idx)
        df.loc[start_idx:end_idx-1, 'temperature'] += rise_curve
        df.loc[start_idx:end_idx-1, 'status'] = 'Warning'
        
        # Mark critical when temp > 100°C
        critical_idx = df[df['temperature'] > 100].index
        df.loc[critical_idx, 'status'] = 'Critical'
        
        return df
    
    def inject_vibration_spike(self, df, start_idx, duration_samples):
        """
        Simulate mechanical imbalance - sudden vibration spikes.
        """
        end_idx = min(start_idx + duration_samples, len(df))
        
        # Add random spikes
        spikes = np.random.uniform(4, 8, end_idx - start_idx)
        df.loc[start_idx:end_idx-1, 'vibration'] = spikes
        df.loc[start_idx:end_idx-1, 'status'] = 'Critical'
        
        return df
    
    def inject_pressure_drop(self, df, start_idx, duration_samples):
        """
        Simulate seal leakage - gradual pressure drop.
        """
        end_idx = min(start_idx + duration_samples, len(df))
        
        # Gradual pressure decrease
        drop_curve = np.linspace(0, -40, end_idx - start_idx)
        df.loc[start_idx:end_idx-1, 'pressure'] += drop_curve
        df.loc[start_idx:end_idx-1, 'status'] = 'Warning'
        
        # Mark critical when pressure < 70 psi
        critical_idx = df[df['pressure'] < 70].index
        df.loc[critical_idx, 'status'] = 'Critical'
        
        return df
    
    def inject_rpm_fluctuation(self, df, start_idx, duration_samples):
        """
        Simulate fuel supply issues - RPM fluctuations.
        """
        end_idx = min(start_idx + duration_samples, len(df))
        
        # Create erratic RPM pattern
        fluctuation = np.random.uniform(-800, -500, end_idx - start_idx)
        df.loc[start_idx:end_idx-1, 'rpm'] += fluctuation
        df.loc[start_idx:end_idx-1, 'status'] = 'Warning'
        
        return df
    
    def generate_fault_scenarios(self, num_faults=3):
        """
        Generate data with randomly injected fault scenarios.
        
        Args:
            num_faults: Number of fault events to inject
        """
        df = self.generate_normal_data()
        
        fault_types = [
            self.inject_gradual_temperature_fault,
            self.inject_vibration_spike,
            self.inject_pressure_drop,
            self.inject_rpm_fluctuation
        ]
        
        # Inject faults at random intervals
        fault_positions = sorted(random.sample(range(100, self.num_samples - 500), num_faults))
        
        for i, fault_pos in enumerate(fault_positions):
            fault_func = random.choice(fault_types)
            duration = random.randint(50, 200)  # Random fault duration
            df = fault_func(df, fault_pos, duration)
            print(f"✅ Injected {fault_func.__name__} at sample {fault_pos}")
        
        return df
    
    def save_to_csv(self, df, filename='turbine_sensor_data.csv'):
        """Save generated data to CSV."""
        df.to_csv(filename, index=False)
        print(f"\n💾 Data saved to {filename}")
        print(f"📊 Total samples: {len(df)}")
        print(f"⚠️  Warning samples: {len(df[df['status'] == 'Warning'])}")
        print(f"🚨 Critical samples: {len(df[df['status'] == 'Critical'])}")
    
    def visualize_data(self, df):
        """Visualize the generated sensor data."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle('Turbine Sensor Data with Fault Injection', fontsize=16, fontweight='bold')
        
        # Color mapping for status
        colors = df['status'].map({'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'})
        
        # Temperature
        axes[0].scatter(range(len(df)), df['temperature'], c=colors, alpha=0.6, s=1)
        axes[0].set_ylabel('Temperature (°C)', fontweight='bold')
        axes[0].axhline(y=90, color='orange', linestyle='--', label='Warning Threshold')
        axes[0].axhline(y=100, color='red', linestyle='--', label='Critical Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Vibration
        axes[1].scatter(range(len(df)), df['vibration'], c=colors, alpha=0.6, s=1)
        axes[1].set_ylabel('Vibration (Hz)', fontweight='bold')
        axes[1].axhline(y=3, color='red', linestyle='--', label='Critical Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # RPM
        axes[2].scatter(range(len(df)), df['rpm'], c=colors, alpha=0.6, s=1)
        axes[2].set_ylabel('RPM', fontweight='bold')
        axes[2].axhline(y=1500, color='orange', linestyle='--', label='Warning Threshold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Pressure
        axes[3].scatter(range(len(df)), df['pressure'], c=colors, alpha=0.6, s=1)
        axes[3].set_ylabel('Pressure (psi)', fontweight='bold')
        axes[3].set_xlabel('Sample Index', fontweight='bold')
        axes[3].axhline(y=70, color='red', linestyle='--', label='Critical Threshold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sensor_data_visualization.png', dpi=300, bbox_inches='tight')
        print("\n📈 Visualization saved as 'sensor_data_visualization.png'")
        plt.show()


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("🚀 Turbine Sensor Data Simulator")
    print("=" * 50)
    
    # Initialize simulator
    # 24 hours of data, one reading every 10 seconds
    simulator = TurbineSensorSimulator(duration_hours=24, sample_rate_seconds=10)
    
    # Generate data with fault scenarios
    print("\n🔧 Generating sensor data with fault injection...")
    sensor_data = simulator.generate_fault_scenarios(num_faults=4)
    
    # Save to CSV
    simulator.save_to_csv(sensor_data, 'turbine_sensor_data.csv')
    
    # Visualize
    print("\n📊 Creating visualizations...")
    simulator.visualize_data(sensor_data)
    
    # Show sample data
    print("\n📋 Sample Data Preview:")
    print(sensor_data.head(10))
    print("\n" + "=" * 50)
    print("✅ Data generation complete!")
    print("\nNext Steps:")
    print("1. Check 'turbine_sensor_data.csv' for the generated data")
    print("2. Review 'sensor_data_visualization.png' for visual analysis")
    print("3. Ready to move to Phase 2: AI/ML Model Training!")