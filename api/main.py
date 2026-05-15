import asyncio
import json
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from core.detector import ONNXDetector
from core.tracker import RetailTracker

app = FastAPI()
CSV_PATH = "RetailSense_Final_Report.csv"
DB_PATH = "retail_analytics.db"

app.mount("/static", StaticFiles(directory="."), name="static")

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "DEMO_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            entries INTEGER,
            exits INTEGER,
            dwell_time REAL,
            gemini_insight TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(entries, exits, dwell_time, insight):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (entries, exits, dwell_time, gemini_insight)
        VALUES (?, ?, ?, ?)
    ''', (entries, exits, dwell_time, insight))
    conn.commit()
    conn.close()

def get_latest_stats():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT entries, exits, dwell_time, gemini_insight FROM events ORDER BY timestamp DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "total_entries": row[0],
            "total_exits": row[1],
            "avg_dwell_time": row[2],
            "gemini_insight": row[3],
            "active_count": 0
        }
    return None

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send latest stats on initial connection
        latest = get_latest_stats()
        if latest:
            await websocket.send_text(json.dumps(latest))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

def analyze_lost_conversion(dwell_time, track_id):
    try:
        prompt = f"Analyze a lost conversion: Person {track_id} dwell time {dwell_time}s, exits: 0. Provide a brief 1-sentence retail insight."
        if os.getenv("GEMINI_API_KEY") is None:
            return f"Customer {track_id} showing high interest ({dwell_time}s); possible assistance required or detection lag."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "High dwell time detected. Recommend immediate floor staff engagement."

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/history")
async def get_history():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, entries, exits, dwell_time FROM events ORDER BY timestamp DESC LIMIT 10')
    rows = cursor.fetchall()
    conn.close()
    return [{"timestamp": r[0], "entries": r[1], "exits": r[2], "dwell_time": r[3]} for r in rows]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def processing_loop():
    tracker = RetailTracker()
    
    while True:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            active_df = df[df['exit_time'].isna() | (df['exit_time'] == "")]
            
            total_in = len(df[df['status'] == 'inside']) + len(df[df['status'] == 'exited'])
            total_out = len(df[df['status'] == 'exited'])
            avg_dwell = df['dwell_time'].mean() if not df.empty else 0
            
            insight = ""
            if not active_df.empty:
                max_dwell_row = active_df.loc[active_df['dwell_time'].idxmax()]
                if max_dwell_row['dwell_time'] > 300:
                    insight = analyze_lost_conversion(max_dwell_row['dwell_time'], max_dwell_row['track_id'])
            
            save_to_db(int(total_in), int(total_out), float(avg_dwell), insight)
            
            stats = {
                "total_entries": int(total_in),
                "total_exits": int(total_out),
                "avg_dwell_time": float(avg_dwell),
                "active_count": len(active_df),
                "gemini_insight": insight
            }
        else:
            stats = {
                "total_entries": 0,
                "total_exits": 0,
                "avg_dwell_time": 0,
                "active_count": 0,
                "gemini_insight": ""
            }
            
        await manager.broadcast(json.dumps(stats))
        await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(processing_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
