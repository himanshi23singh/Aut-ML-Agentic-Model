"""
State Manager
Centralized pipeline state management with intervention hooks.
"""
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import asyncio
import uuid

try:
    from ..schemas.stream_state import StreamState, StageEnum, StatusEnum, LogLevel
except ImportError:
    from schemas.stream_state import StreamState, StageEnum, StatusEnum, LogLevel


class StateManager:
    """
    Manages pipeline state for all active sessions.
    Provides intervention hooks for pause, resume, abort, and override.
    """
    
    def __init__(self):
        self.sessions: Dict[str, StreamState] = {}
        self.intervention_handlers: Dict[str, Dict[str, Callable]] = {}
        self._state_change_callbacks: list = []
    
    def create_session(self, input_type: str = "text") -> StreamState:
        """Create a new pipeline session."""
        session_id = str(uuid.uuid4())
        
        state = StreamState(
            session_id=session_id,
            input_type=input_type,
            status=StatusEnum.PENDING
        )
        
        self.sessions[session_id] = state
        self.intervention_handlers[session_id] = {}
        
        return state
    
    def get_session(self, session_id: str) -> Optional[StreamState]:
        """Get the state for a session."""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> Dict[str, StreamState]:
        """Get all active sessions."""
        return self.sessions
    
    async def update_state(
        self, 
        session_id: str, 
        updates: Dict[str, Any],
        broadcast: bool = True
    ) -> Optional[StreamState]:
        """Update session state and optionally broadcast."""
        state = self.sessions.get(session_id)
        if not state:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        state.updated_at = datetime.utcnow()
        
        # Notify callbacks
        if broadcast:
            await self._notify_state_change(session_id, state)
        
        return state
    
    async def add_log(
        self,
        session_id: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
        metadata: Optional[Dict] = None
    ) -> Optional[StreamState]:
        """Add a log entry to a session."""
        state = self.sessions.get(session_id)
        if not state:
            return None
        
        state.add_log(message, level, metadata=metadata)
        await self._notify_state_change(session_id, state)
        
        return state
    
    async def add_decision_log(
        self,
        session_id: str,
        decision: str,
        rationale: str,
        metadata: Optional[Dict] = None
    ) -> Optional[StreamState]:
        """Add a decision log (for glass-box transparency)."""
        return await self.add_log(
            session_id,
            f"DECISION: {decision} - {rationale}",
            LogLevel.DECISION,
            metadata
        )
    
    async def transition_stage(
        self,
        session_id: str,
        new_stage: StageEnum,
        output_summary: Optional[str] = None
    ) -> Optional[StreamState]:
        """Transition to a new pipeline stage."""
        state = self.sessions.get(session_id)
        if not state:
            return None
        
        # Complete current stage if running
        if state.status == StatusEnum.RUNNING:
            state.complete_stage(state.current_stage, output_summary)
        
        # Start new stage
        state.update_stage(new_stage)
        
        await self._notify_state_change(session_id, state)
        
        return state
    
    # Intervention methods
    async def pause(self, session_id: str) -> bool:
        """Pause a running session."""
        state = self.sessions.get(session_id)
        if not state or state.status != StatusEnum.RUNNING:
            return False
        
        state.status = StatusEnum.PAUSED
        state.is_paused = True
        state.add_log("Pipeline paused by user", LogLevel.INFO)
        
        # Call intervention handler if registered
        handler = self.intervention_handlers.get(session_id, {}).get("pause")
        if handler:
            await handler()
        
        await self._notify_state_change(session_id, state)
        return True
    
    async def resume(self, session_id: str) -> bool:
        """Resume a paused session."""
        state = self.sessions.get(session_id)
        if not state or state.status != StatusEnum.PAUSED:
            return False
        
        state.status = StatusEnum.RUNNING
        state.is_paused = False
        state.add_log("Pipeline resumed by user", LogLevel.INFO)
        
        # Call intervention handler if registered
        handler = self.intervention_handlers.get(session_id, {}).get("resume")
        if handler:
            await handler()
        
        await self._notify_state_change(session_id, state)
        return True
    
    async def abort(self, session_id: str, reason: str = "User requested") -> bool:
        """Abort a session."""
        state = self.sessions.get(session_id)
        if not state:
            return False
        
        state.status = StatusEnum.ABORTED
        state.add_log(f"Pipeline aborted: {reason}", LogLevel.WARNING)
        
        # Call intervention handler if registered
        handler = self.intervention_handlers.get(session_id, {}).get("abort")
        if handler:
            await handler()
        
        await self._notify_state_change(session_id, state)
        return True
    
    async def override_task(
        self, 
        session_id: str, 
        new_task: str,
        reason: str = "User override"
    ) -> bool:
        """Override the detected task."""
        state = self.sessions.get(session_id)
        if not state:
            return False
        
        old_task = state.detected_task
        state.detected_task = new_task
        state.has_override = True
        state.override_details = {
            "type": "task",
            "original": old_task,
            "new": new_task,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        state.add_log(
            f"Task overridden: {old_task} → {new_task} ({reason})",
            LogLevel.DECISION
        )
        
        # Call intervention handler if registered
        handler = self.intervention_handlers.get(session_id, {}).get("override_task")
        if handler:
            await handler(new_task)
        
        await self._notify_state_change(session_id, state)
        return True
    
    async def override_tool(
        self,
        session_id: str,
        new_tool: str,
        reason: str = "User override"
    ) -> bool:
        """Override the selected tool."""
        state = self.sessions.get(session_id)
        if not state:
            return False
        
        old_tool = state.selected_tool
        state.selected_tool = new_tool
        state.has_override = True
        state.override_details = {
            "type": "tool",
            "original": old_tool,
            "new": new_tool,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        state.add_log(
            f"Tool overridden: {old_tool} → {new_tool} ({reason})",
            LogLevel.DECISION
        )
        
        # Call intervention handler if registered
        handler = self.intervention_handlers.get(session_id, {}).get("override_tool")
        if handler:
            await handler(new_tool)
        
        await self._notify_state_change(session_id, state)
        return True
    
    def register_intervention_handler(
        self,
        session_id: str,
        intervention_type: str,
        handler: Callable
    ):
        """Register a handler for an intervention type."""
        if session_id not in self.intervention_handlers:
            self.intervention_handlers[session_id] = {}
        self.intervention_handlers[session_id][intervention_type] = handler
    
    def register_state_change_callback(self, callback: Callable[[str, StreamState], Awaitable[None]]):
        """Register a callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    async def _notify_state_change(self, session_id: str, state: StreamState):
        """Notify all registered callbacks of a state change."""
        for callback in self._state_change_callbacks:
            try:
                await callback(session_id, state)
            except Exception:
                pass  # Don't let callback errors break the pipeline
    
    def cleanup_session(self, session_id: str):
        """Clean up a completed session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.intervention_handlers:
            del self.intervention_handlers[session_id]


# Global state manager instance
state_manager = StateManager()
