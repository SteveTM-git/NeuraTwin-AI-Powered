from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import sqlite3
from contextlib import asynccontextmanager
from advanced_ai_engine import AdvancedAIEngine

# ============ DATA MODELS ============
class SensorReading(BaseModel):
    temperature: float
    vibration: float
    rpm: float
    pressure: float
    timestamp: str = None

class PredictionResponse(BaseModel):
    status: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    timestamp: str
    reading: Dict[float, float]
    rul_prediction: Dict = None
    root_cause_analysis: Dict = None
    maintenance_schedule: List = None

# ============ DATABASE SETUP ============
def init_database():
    """Initialize SQLite database for storing sensor readings."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            vibration REAL NOT NULL,
            rpm REAL NOT NULL,
            pressure REAL NOT NULL,
            status TEXT,
            is_anomaly BOOLEAN,
            anomaly_score REAL,
            rul_days INTEGER,
            health_percentage REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized: turbine_data.db")

def save_reading_to_db(reading: SensorReading, prediction: dict):
    """Save sensor reading and prediction to database."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    rul_days = prediction.get('rul_prediction', {}).get('rul_days')
    health_pct = prediction.get('rul_prediction', {}).get('health_percentage')
    
    cursor.execute('''
        INSERT INTO sensor_readings 
        (timestamp, temperature, vibration, rpm, pressure, status, is_anomaly, anomaly_score, rul_days, health_percentage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        reading.timestamp or datetime.now().isoformat(),
        reading.temperature,
        reading.vibration,
        reading.rpm,
        reading.pressure,
        prediction['status'],
        prediction['is_anomaly'],
        prediction['anomaly_score'],
        rul_days,
        health_pct
    ))
    
    conn.commit()
    conn.close()

# ============ AI MODEL LOADER ============
class AIModelManager:
    """Manages loading and using trained AI models with advanced features."""
    
    def __init__(self):
        self.isolation_forest = None
        self.scaler = None
        self.advanced_engine = AdvancedAIEngine()
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models from disk."""
        try:
            self.isolation_forest = joblib.load('isolation_forest_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("✅ AI Models loaded successfully")
        except FileNotFoundError:
            print("⚠️  Warning: Model files not found. Train models first!")
    
    def predict(self, reading: SensorReading, include_advanced=True) -> dict:
        """Make prediction on sensor reading with advanced AI features."""
        if not self.isolation_forest or not self.scaler:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Prepare input
        reading_dict = {
            'temperature': reading.temperature,
            'vibration': reading.vibration,
            'rpm': reading.rpm,
            'pressure': reading.pressure
        }
        
        reading_array = np.array([[
            reading.temperature,
            reading.vibration,
            reading.rpm,
            reading.pressure
        ]])
        
        # Scale and predict
        reading_scaled = self.scaler.transform(reading_array)
        is_anomaly = self.isolation_forest.predict(reading_scaled)[0]
        anomaly_score = self.isolation_forest.score_samples(reading_scaled)[0]
        
        # Calculate confidence
        confidence = abs(anomaly_score) * 100
        confidence = min(confidence, 100)
        
        # Determine status
        if is_anomaly == -1:
            if anomaly_score < -0.5:
                status = "🚨 CRITICAL"
            else:
                status = "⚠️ WARNING"
        else:
            status = "✅ NORMAL"
        
        result = {
            'status': status,
            'is_anomaly': bool(is_anomaly == -1),
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'timestamp': reading.timestamp or datetime.now().isoformat(),
            'reading': reading_dict
        }
        
        # Add advanced AI features
        if include_advanced:
            comprehensive = self.advanced_engine.get_comprehensive_analysis(reading_dict)
            result['rul_prediction'] = comprehensive['rul_prediction']
            result['root_cause_analysis'] = comprehensive['root_cause_analysis']
            result['maintenance_schedule'] = comprehensive['maintenance_schedule']
            result['overall_status'] = comprehensive['overall_status']
        
        return result

# ============ SENSOR DATA SIMULATOR ============
class SensorDataStream:
    """Simulates real-time sensor data streaming."""
    
    def __init__(self, csv_file='turbine_sensor_data.csv'):
        self.df = pd.read_csv(csv_file)
        self.current_index = 0
    
    def get_next_reading(self) -> SensorReading:
        """Get next sensor reading from dataset."""
        if self.current_index >= len(self.df):
            self.current_index = 0
        
        row = self.df.iloc[self.current_index]
        self.current_index += 1
        
        return SensorReading(
            temperature=float(row['temperature']),
            vibration=float(row['vibration']),
            rpm=float(row['rpm']),
            pressure=float(row['pressure']),
            timestamp=datetime.now().isoformat()
        )

# ============ WEBSOCKET CONNECTION MANAGER ============
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

# ============ FASTAPI APP INITIALIZATION ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("\n" + "="*60)
    print("🚀 TURBINE DIGITAL TWIN - ENHANCED BACKEND SERVER")
    print("="*60)
    init_database()
    print("🔄 Starting real-time data stream...")
    
    asyncio.create_task(stream_sensor_data())
    
    yield
    
    print("\n🛑 Server shutting down...")

app = FastAPI(
    title="Turbine Digital Twin API - Advanced",
    description="AI-powered turbine monitoring with RUL prediction and root cause analysis",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ GLOBAL INSTANCES ============
model_manager = AIModelManager()
sensor_stream = SensorDataStream()
connection_manager = ConnectionManager()

# ============ BACKGROUND STREAMING TASK ============
async def stream_sensor_data():
    """Background task to continuously stream sensor data."""
    while True:
        try:
            reading = sensor_stream.get_next_reading()
            prediction = model_manager.predict(reading, include_advanced=True)
            save_reading_to_db(reading, prediction)
            
            await connection_manager.broadcast({
                'type': 'sensor_update',
                'data': prediction
            })
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            await asyncio.sleep(5)

# ============ REST API ENDPOINTS ============

@app.get("/")
async def root():
    """API health check."""
    return {
        "message": "🚀 Turbine Digital Twin API - Advanced",
        "version": "2.0.0",
        "features": ["Anomaly Detection", "RUL Prediction", "Root Cause Analysis", "Maintenance Scheduling"],
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_health(reading: SensorReading):
    """Predict turbine health with advanced AI analysis."""
    try:
        prediction = model_manager.predict(reading, include_advanced=True)
        save_reading_to_db(reading, prediction)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(limit: int = 100):
    """Get historical sensor readings from database."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM sensor_readings 
        ORDER BY id DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    columns = ['id', 'timestamp', 'temperature', 'vibration', 'rpm', 
               'pressure', 'status', 'is_anomaly', 'anomaly_score', 
               'rul_days', 'health_percentage']
    
    return {
        'count': len(rows),
        'data': [dict(zip(columns, row)) for row in rows]
    }

@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM sensor_readings')
    total_readings = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM sensor_readings WHERE is_anomaly = 1')
    total_anomalies = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT AVG(temperature), AVG(vibration), AVG(rpm), AVG(pressure), AVG(rul_days), AVG(health_percentage)
        FROM sensor_readings
        WHERE is_anomaly = 0
    ''')
    avg_normal = cursor.fetchone()
    
    cursor.execute('SELECT AVG(rul_days), MIN(rul_days) FROM sensor_readings WHERE rul_days IS NOT NULL')
    rul_stats = cursor.fetchone()
    
    conn.close()
    
    return {
        'total_readings': total_readings,
        'total_anomalies': total_anomalies,
        'anomaly_rate': (total_anomalies / total_readings * 100) if total_readings > 0 else 0,
        'average_rul_days': rul_stats[0] if rul_stats[0] else None,
        'minimum_rul_days': rul_stats[1] if rul_stats[1] else None,
        'average_normal_readings': {
            'temperature': avg_normal[0],
            'vibration': avg_normal[1],
            'rpm': avg_normal[2],
            'pressure': avg_normal[3],
            'rul_days': avg_normal[4],
            'health_percentage': avg_normal[5]
        } if avg_normal[0] else None
    }

@app.get("/rul")
async def get_rul_prediction():
    """Get latest RUL prediction."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT temperature, vibration, rpm, pressure, rul_days, health_percentage, timestamp
        FROM sensor_readings
        ORDER BY id DESC
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="No data available")
    
    return {
        'rul_days': row[4],
        'health_percentage': row[5],
        'timestamp': row[6],
        'current_reading': {
            'temperature': row[0],
            'vibration': row[1],
            'rpm': row[2],
            'pressure': row[3]
        }
    }

@app.get("/maintenance/schedule")
async def get_maintenance_schedule():
    """Get recommended maintenance schedule."""
    # Get latest reading
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT temperature, vibration, rpm, pressure
        FROM sensor_readings
        ORDER BY id DESC
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="No data available")
    
    reading_dict = {
        'temperature': row[0],
        'vibration': row[1],
        'rpm': row[2],
        'pressure': row[3]
    }
    
    # Get comprehensive analysis
    analysis = model_manager.advanced_engine.get_comprehensive_analysis(reading_dict)
    
    return {
        'schedule': analysis['maintenance_schedule'],
        'rul_prediction': analysis['rul_prediction'],
        'root_cause_analysis': analysis['root_cause_analysis']
    }

@app.delete("/history")
async def clear_history():
    """Clear all historical data from database."""
    conn = sqlite3.connect('turbine_data.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sensor_readings')
    conn.commit()
    conn.close()
    return {"message": "History cleared successfully"}

# ============ WEBSOCKET ENDPOINT ============

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sensor data streaming."""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎯 Starting Enhanced FastAPI server...")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   • REST API: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • WebSocket: ws://localhost:8000/ws/stream")
    print("   • RUL Prediction: http://localhost:8000/rul")
    print("   • Maintenance: http://localhost:8000/maintenance/schedule")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )