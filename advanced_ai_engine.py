import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from scipy import stats
import json

class AdvancedAIEngine:
    """
    Advanced AI capabilities for turbine health monitoring:
    - Root Cause Analysis
    - Remaining Useful Life (RUL) Prediction
    - Maintenance Scheduling
    """
    
    def __init__(self):
        """Initialize the advanced AI engine."""
        # Load trained models
        try:
            self.isolation_forest = joblib.load('isolation_forest_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("✅ AI Models loaded successfully")
        except FileNotFoundError:
            print("⚠️  Warning: Model files not found")
            self.isolation_forest = None
            self.scaler = None
        
        # Thresholds for different fault types
        self.thresholds = {
            'temperature': {'warning': 85, 'critical': 100},
            'vibration': {'warning': 2.5, 'critical': 4.0},
            'rpm': {'warning_low': 1500, 'critical_low': 1200},
            'pressure': {'warning_low': 75, 'critical_low': 60}
        }
        
        # Fault patterns database
        self.fault_patterns = {
            'bearing_failure': {
                'indicators': ['temperature_high', 'vibration_high'],
                'severity': 'critical',
                'description': 'Bearing wear detected - immediate maintenance required',
                'rul_impact': -30  # Days reduction in RUL
            },
            'seal_leakage': {
                'indicators': ['pressure_low', 'temperature_high'],
                'severity': 'warning',
                'description': 'Seal degradation detected - schedule inspection',
                'rul_impact': -15
            },
            'imbalance': {
                'indicators': ['vibration_high', 'rpm_unstable'],
                'severity': 'warning',
                'description': 'Mechanical imbalance - rotor alignment needed',
                'rul_impact': -10
            },
            'fuel_system': {
                'indicators': ['rpm_low', 'pressure_low'],
                'severity': 'warning',
                'description': 'Fuel system issues - check fuel supply',
                'rul_impact': -7
            },
            'overheating': {
                'indicators': ['temperature_critical'],
                'severity': 'critical',
                'description': 'Critical overheating - shutdown recommended',
                'rul_impact': -45
            }
        }
        
        # Baseline RUL (days)
        self.baseline_rul = 180  # 6 months under normal conditions
    
    def analyze_sensor_state(self, reading):
        """
        Analyze current sensor readings to identify anomalies.
        
        Args:
            reading: Dict with temperature, vibration, rpm, pressure
            
        Returns:
            List of detected conditions
        """
        conditions = []
        
        # Temperature analysis
        if reading['temperature'] >= self.thresholds['temperature']['critical']:
            conditions.append('temperature_critical')
        elif reading['temperature'] >= self.thresholds['temperature']['warning']:
            conditions.append('temperature_high')
        
        # Vibration analysis
        if reading['vibration'] >= self.thresholds['vibration']['critical']:
            conditions.append('vibration_critical')
        elif reading['vibration'] >= self.thresholds['vibration']['warning']:
            conditions.append('vibration_high')
        
        # RPM analysis
        if reading['rpm'] <= self.thresholds['rpm']['critical_low']:
            conditions.append('rpm_critical_low')
        elif reading['rpm'] <= self.thresholds['rpm']['warning_low']:
            conditions.append('rpm_low')
        
        # Check RPM stability (would need historical data in real implementation)
        # Simplified version:
        if abs(reading['rpm'] - 2250) > 500:
            conditions.append('rpm_unstable')
        
        # Pressure analysis
        if reading['pressure'] <= self.thresholds['pressure']['critical_low']:
            conditions.append('pressure_critical_low')
        elif reading['pressure'] <= self.thresholds['pressure']['warning_low']:
            conditions.append('pressure_low')
        
        return conditions
    
    def root_cause_analysis(self, reading):
        """
        Perform root cause analysis based on sensor readings.
        
        Args:
            reading: Dict with sensor values
            
        Returns:
            Dict with identified faults and recommendations
        """
        # Get current conditions
        conditions = self.analyze_sensor_state(reading)
        
        if not conditions:
            return {
                'fault_detected': False,
                'root_causes': [],
                'recommendations': ['Continue normal operations'],
                'severity': 'normal',
                'confidence': 100
            }
        
        # Match conditions to fault patterns
        detected_faults = []
        max_severity = 'normal'
        
        for fault_name, fault_data in self.fault_patterns.items():
            # Check how many indicators match
            matches = sum(1 for indicator in fault_data['indicators'] 
                         if any(indicator in cond for cond in conditions))
            
            if matches > 0:
                confidence = (matches / len(fault_data['indicators'])) * 100
                
                detected_faults.append({
                    'fault_type': fault_name,
                    'description': fault_data['description'],
                    'severity': fault_data['severity'],
                    'confidence': confidence,
                    'matched_indicators': matches,
                    'total_indicators': len(fault_data['indicators'])
                })
                
                # Update max severity
                if fault_data['severity'] == 'critical':
                    max_severity = 'critical'
                elif fault_data['severity'] == 'warning' and max_severity != 'critical':
                    max_severity = 'warning'
        
        # Sort by confidence
        detected_faults.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detected_faults, conditions)
        
        return {
            'fault_detected': True,
            'root_causes': detected_faults,
            'recommendations': recommendations,
            'severity': max_severity,
            'conditions': conditions
        }
    
    def _generate_recommendations(self, faults, conditions):
        """Generate maintenance recommendations based on detected faults."""
        recommendations = []
        
        if not faults:
            return ['Continue monitoring system']
        
        # Critical recommendations
        if any(f['severity'] == 'critical' for f in faults):
            recommendations.append('🚨 IMMEDIATE ACTION REQUIRED')
            recommendations.append('Schedule emergency maintenance within 24 hours')
        
        # Specific recommendations based on fault types
        for fault in faults[:3]:  # Top 3 most likely faults
            fault_type = fault['fault_type']
            
            if fault_type == 'bearing_failure':
                recommendations.append('🔧 Replace turbine bearings')
                recommendations.append('📊 Inspect lubrication system')
            elif fault_type == 'seal_leakage':
                recommendations.append('🔧 Inspect and replace seals')
                recommendations.append('📊 Check for fluid contamination')
            elif fault_type == 'imbalance':
                recommendations.append('🔧 Perform rotor balancing')
                recommendations.append('📊 Check mounting and alignment')
            elif fault_type == 'fuel_system':
                recommendations.append('🔧 Inspect fuel lines and filters')
                recommendations.append('📊 Check fuel pressure regulator')
            elif fault_type == 'overheating':
                recommendations.append('🚨 Reduce load immediately')
                recommendations.append('🔧 Inspect cooling system')
        
        # Temperature-specific
        if 'temperature_high' in conditions or 'temperature_critical' in conditions:
            recommendations.append('🌡️ Monitor coolant levels')
        
        # Vibration-specific
        if 'vibration_high' in conditions or 'vibration_critical' in conditions:
            recommendations.append('📊 Conduct vibration analysis')
        
        return recommendations
    
    def predict_rul(self, reading, historical_data=None):
        """
        Predict Remaining Useful Life (RUL) based on current conditions.
        
        Args:
            reading: Current sensor reading
            historical_data: Optional historical degradation data
            
        Returns:
            Dict with RUL prediction
        """
        # Start with baseline RUL
        predicted_rul = self.baseline_rul
        
        # Analyze current state
        conditions = self.analyze_sensor_state(reading)
        rca_result = self.root_cause_analysis(reading)
        
        # Reduce RUL based on detected faults
        for fault in rca_result.get('root_causes', []):
            fault_type = fault['fault_type']
            if fault_type in self.fault_patterns:
                impact = self.fault_patterns[fault_type]['rul_impact']
                # Weight by confidence
                weighted_impact = impact * (fault['confidence'] / 100)
                predicted_rul += weighted_impact
        
        # Calculate degradation rate (simplified)
        degradation_factors = {
            'temperature': (reading['temperature'] - 70) / 50,  # Normalize
            'vibration': (reading['vibration'] - 1.2) / 3,
            'rpm_deviation': abs(reading['rpm'] - 2250) / 1000,
            'pressure_deviation': abs(reading['pressure'] - 100) / 40
        }
        
        # Average degradation
        avg_degradation = np.mean(list(degradation_factors.values()))
        degradation_penalty = avg_degradation * 20  # Max 20 days penalty
        
        predicted_rul -= degradation_penalty
        
        # Ensure RUL doesn't go negative
        predicted_rul = max(0, predicted_rul)
        
        # Calculate health percentage
        health_percentage = (predicted_rul / self.baseline_rul) * 100
        
        # Determine maintenance urgency
        if predicted_rul < 7:
            urgency = 'IMMEDIATE'
            urgency_color = 'red'
        elif predicted_rul < 30:
            urgency = 'HIGH'
            urgency_color = 'orange'
        elif predicted_rul < 90:
            urgency = 'MEDIUM'
            urgency_color = 'yellow'
        else:
            urgency = 'LOW'
            urgency_color = 'green'
        
        return {
            'rul_days': int(predicted_rul),
            'rul_hours': int(predicted_rul * 24),
            'health_percentage': round(health_percentage, 1),
            'maintenance_due': (datetime.now() + timedelta(days=predicted_rul)).strftime('%Y-%m-%d'),
            'urgency': urgency,
            'urgency_color': urgency_color,
            'degradation_rate': round(avg_degradation * 100, 2),
            'baseline_rul': self.baseline_rul
        }
    
    def generate_maintenance_schedule(self, rul_prediction, rca_result):
        """
        Generate smart maintenance schedule based on RUL and root cause analysis.
        
        Args:
            rul_prediction: RUL prediction dict
            rca_result: Root cause analysis result
            
        Returns:
            List of scheduled maintenance tasks
        """
        schedule = []
        
        # Immediate tasks (if critical)
        if rca_result['severity'] == 'critical' or rul_prediction['urgency'] == 'IMMEDIATE':
            schedule.append({
                'priority': 1,
                'type': 'EMERGENCY',
                'task': 'Emergency shutdown and inspection',
                'scheduled_date': datetime.now().strftime('%Y-%m-%d'),
                'estimated_duration': '4-8 hours',
                'required_parts': self._get_required_parts(rca_result),
                'technicians_needed': 2
            })
        
        # High priority tasks
        if rul_prediction['urgency'] in ['HIGH', 'IMMEDIATE']:
            for fault in rca_result.get('root_causes', [])[:2]:
                schedule.append({
                    'priority': 2,
                    'type': 'CORRECTIVE',
                    'task': fault['description'],
                    'scheduled_date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'estimated_duration': '2-4 hours',
                    'fault_type': fault['fault_type'],
                    'confidence': fault['confidence']
                })
        
        # Preventive maintenance
        if rul_prediction['rul_days'] > 30:
            schedule.append({
                'priority': 3,
                'type': 'PREVENTIVE',
                'task': 'Routine inspection and lubrication',
                'scheduled_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'estimated_duration': '1-2 hours',
                'recurring': 'monthly'
            })
        
        # Predictive maintenance based on RUL
        maintenance_date = datetime.now() + timedelta(days=int(rul_prediction['rul_days'] * 0.8))
        schedule.append({
            'priority': 4,
            'type': 'PREDICTIVE',
            'task': 'Scheduled component replacement',
            'scheduled_date': maintenance_date.strftime('%Y-%m-%d'),
            'estimated_duration': '4-6 hours',
            'notes': f"Scheduled at 80% of predicted RUL ({rul_prediction['rul_days']} days)"
        })
        
        return sorted(schedule, key=lambda x: x['priority'])
    
    def _get_required_parts(self, rca_result):
        """Determine required parts based on root cause analysis."""
        parts = []
        
        for fault in rca_result.get('root_causes', []):
            fault_type = fault['fault_type']
            
            if fault_type == 'bearing_failure':
                parts.extend(['Bearings (set of 4)', 'Lubricant'])
            elif fault_type == 'seal_leakage':
                parts.extend(['Seal kit', 'Gaskets'])
            elif fault_type == 'imbalance':
                parts.extend(['Balancing weights', 'Alignment tools'])
            elif fault_type == 'fuel_system':
                parts.extend(['Fuel filter', 'Fuel pump seals'])
        
        return list(set(parts))  # Remove duplicates
    
    def get_comprehensive_analysis(self, reading):
        """
        Get complete analysis including RCA, RUL, and maintenance schedule.
        
        Args:
            reading: Current sensor reading
            
        Returns:
            Complete analysis dict
        """
        # Perform all analyses
        rca_result = self.root_cause_analysis(reading)
        rul_prediction = self.predict_rul(reading)
        maintenance_schedule = self.generate_maintenance_schedule(rul_prediction, rca_result)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sensor_reading': reading,
            'root_cause_analysis': rca_result,
            'rul_prediction': rul_prediction,
            'maintenance_schedule': maintenance_schedule,
            'overall_status': self._get_overall_status(rca_result, rul_prediction)
        }
    
    def _get_overall_status(self, rca_result, rul_prediction):
        """Determine overall system status."""
        if rca_result['severity'] == 'critical' or rul_prediction['urgency'] == 'IMMEDIATE':
            return {
                'level': 'CRITICAL',
                'message': '🚨 Immediate attention required',
                'color': 'red'
            }
        elif rca_result['severity'] == 'warning' or rul_prediction['urgency'] == 'HIGH':
            return {
                'level': 'WARNING',
                'message': '⚠️ Maintenance needed soon',
                'color': 'orange'
            }
        elif rul_prediction['urgency'] == 'MEDIUM':
            return {
                'level': 'CAUTION',
                'message': '⚡ Monitor closely',
                'color': 'yellow'
            }
        else:
            return {
                'level': 'NORMAL',
                'message': '✅ Operating normally',
                'color': 'green'
            }


# ============ TESTING ============
if __name__ == "__main__":
    print("="*70)
    print("🧠 ADVANCED AI ENGINE - ROOT CAUSE ANALYSIS & RUL PREDICTION")
    print("="*70)
    
    # Initialize engine
    engine = AdvancedAIEngine()
    
    print("\n📊 Test Case 1: Normal Operation")
    print("-"*70)
    normal_reading = {
        'temperature': 72,
        'vibration': 1.3,
        'rpm': 2200,
        'pressure': 98
    }
    
    analysis = engine.get_comprehensive_analysis(normal_reading)
    print(f"Overall Status: {analysis['overall_status']['message']}")
    print(f"RUL: {analysis['rul_prediction']['rul_days']} days")
    print(f"Health: {analysis['rul_prediction']['health_percentage']}%")
    
    print("\n📊 Test Case 2: Bearing Failure Warning")
    print("-"*70)
    bearing_fault = {
        'temperature': 95,
        'vibration': 3.5,
        'rpm': 2150,
        'pressure': 96
    }
    
    analysis = engine.get_comprehensive_analysis(bearing_fault)
    print(f"Overall Status: {analysis['overall_status']['message']}")
    print(f"RUL: {analysis['rul_prediction']['rul_days']} days")
    print(f"Root Causes Detected: {len(analysis['root_cause_analysis']['root_causes'])}")
    
    if analysis['root_cause_analysis']['root_causes']:
        print("\nTop Root Cause:")
        top_cause = analysis['root_cause_analysis']['root_causes'][0]
        print(f"  - {top_cause['fault_type']}: {top_cause['description']}")
        print(f"  - Confidence: {top_cause['confidence']:.1f}%")
    
    print("\nRecommendations:")
    for rec in analysis['root_cause_analysis']['recommendations'][:3]:
        print(f"  • {rec}")
    
    print("\n📊 Test Case 3: Critical Overheating")
    print("-"*70)
    critical_reading = {
        'temperature': 115,
        'vibration': 5.2,
        'rpm': 1100,
        'pressure': 62
    }
    
    analysis = engine.get_comprehensive_analysis(critical_reading)
    print(f"Overall Status: {analysis['overall_status']['message']}")
    print(f"RUL: {analysis['rul_prediction']['rul_days']} days")
    print(f"Urgency: {analysis['rul_prediction']['urgency']}")
    
    print("\n🔧 Maintenance Schedule:")
    for task in analysis['maintenance_schedule'][:3]:
        print(f"\n  Priority {task['priority']} - {task['type']}")
        print(f"  Task: {task['task']}")
        print(f"  Date: {task['scheduled_date']}")
    
    print("\n" + "="*70)
    print("✅ Advanced AI Engine Test Complete!")
    print("="*70)