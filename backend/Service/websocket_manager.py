"""
WebSocket Manager
Manages WebSocket connections and broadcasts for real-time updates.
"""
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket
import asyncio
import json
from datetime import datetime


class WebSocketManager:
    """
    Manages WebSocket connections for real-time dashboard updates.
    Supports multiple clients per session and broadcasts state changes.
    """
    
    def __init__(self):
        # session_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global connections (monitoring all sessions)
        self.global_connections: Set[WebSocket] = set()
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        async with self._lock:
            if session_id:
                if session_id not in self.active_connections:
                    self.active_connections[session_id] = set()
                self.active_connections[session_id].add(websocket)
            else:
                self.global_connections.add(websocket)
        
        # Send connection confirmation
        await self._send_json(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def disconnect(self, websocket: WebSocket, session_id: Optional[str] = None):
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id and session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            self.global_connections.discard(websocket)
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast a message to all clients connected to a specific session."""
        message["timestamp"] = datetime.utcnow().isoformat()
        
        async with self._lock:
            connections = self.active_connections.get(session_id, set()).copy()
            global_conns = self.global_connections.copy()
        
        # Send to session-specific connections
        for websocket in connections:
            await self._send_json(websocket, message)
        
        # Also send to global connections with session context
        global_message = {"session_id": session_id, **message}
        for websocket in global_conns:
            await self._send_json(websocket, global_message)
    
    async def broadcast_state_update(self, session_id: str, state: Dict[str, Any]):
        """Broadcast a pipeline state update."""
        await self.broadcast_to_session(session_id, {
            "type": "state_update",
            "state": state
        })
    
    async def broadcast_log(self, session_id: str, log_entry: Dict[str, Any]):
        """Broadcast a log entry."""
        await self.broadcast_to_session(session_id, {
            "type": "log",
            "entry": log_entry
        })
    
    async def broadcast_progress(
        self, 
        session_id: str, 
        stage: str, 
        progress: float,
        details: Optional[Dict] = None
    ):
        """Broadcast a progress update."""
        await self.broadcast_to_session(session_id, {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "details": details or {}
        })
    
    async def broadcast_decision(
        self,
        session_id: str,
        decision_type: str,
        decision: Dict[str, Any]
    ):
        """Broadcast a decision (for glass-box transparency)."""
        await self.broadcast_to_session(session_id, {
            "type": "decision",
            "decision_type": decision_type,
            "decision": decision
        })
    
    async def broadcast_intervention_available(
        self,
        session_id: str,
        intervention_type: str,
        options: Dict[str, Any]
    ):
        """Notify clients that an intervention opportunity is available."""
        await self.broadcast_to_session(session_id, {
            "type": "intervention_available",
            "intervention_type": intervention_type,
            "options": options
        })
    
    async def broadcast_error(
        self,
        session_id: str,
        error_message: str,
        error_details: Optional[Dict] = None
    ):
        """Broadcast an error notification."""
        await self.broadcast_to_session(session_id, {
            "type": "error",
            "message": error_message,
            "details": error_details or {}
        })
    
    async def broadcast_global(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        message["timestamp"] = datetime.utcnow().isoformat()
        
        async with self._lock:
            all_connections = self.global_connections.copy()
            for session_conns in self.active_connections.values():
                all_connections.update(session_conns)
        
        for websocket in all_connections:
            await self._send_json(websocket, message)
    
    async def _send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send JSON data to a WebSocket, handling errors."""
        try:
            await websocket.send_json(data)
        except Exception:
            # Connection might be closed, will be cleaned up
            pass
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get the number of active connections."""
        if session_id:
            return len(self.active_connections.get(session_id, set()))
        return sum(len(s) for s in self.active_connections.values()) + len(self.global_connections)
    
    def get_active_sessions(self) -> list:
        """Get list of sessions with active connections."""
        return list(self.active_connections.keys())


# Global WebSocket manager instance
ws_manager = WebSocketManager()
