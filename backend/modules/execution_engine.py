"""
Execution Engine Module
Orchestrates NLP model execution with pause/resume/abort support.
"""
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import uuid


class ExecutionEngine:
    """
    Manages the execution of NLP tools with full control capabilities.
    Supports pause, resume, abort, and live progress streaming.
    """
    
    def __init__(self):
        self.active_executions: Dict[str, Dict] = {}
        self.tool_instances = {}
        self._pause_events: Dict[str, asyncio.Event] = {}
        self._abort_flags: Dict[str, bool] = {}
    
    async def execute(
        self,
        session_id: str,
        tool_id: str,
        text: str,
        parameters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[Dict], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Execute an NLP tool with the given text.
        
        Args:
            session_id: Current session ID
            tool_id: Tool to execute
            text: Processed text to analyze
            parameters: Optional tool parameters
            progress_callback: Async callback for progress updates
        
        Returns:
            Execution result with output and metadata
        """
        execution_id = str(uuid.uuid4())[:8]
        
        # Initialize control structures
        self._pause_events[execution_id] = asyncio.Event()
        self._pause_events[execution_id].set()  # Not paused initially
        self._abort_flags[execution_id] = False
        
        execution_record = {
            "execution_id": execution_id,
            "session_id": session_id,
            "tool_id": tool_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "intermediate_outputs": [],
            "parameters": parameters or {}
        }
        
        self.active_executions[execution_id] = execution_record
        
        try:
            # Get the appropriate tool
            tool = await self._get_tool(tool_id)
            
            # Execute with progress tracking
            result = await self._execute_tool(
                tool,
                tool_id,
                text,
                execution_id,
                parameters or {},
                progress_callback
            )
            
            execution_record["status"] = "completed"
            execution_record["completed_at"] = datetime.utcnow().isoformat()
            execution_record["result"] = result
            
            return {
                "success": True,
                "execution_id": execution_id,
                "result": result,
                "execution_record": execution_record
            }
            
        except asyncio.CancelledError:
            execution_record["status"] = "aborted"
            execution_record["completed_at"] = datetime.utcnow().isoformat()
            return {
                "success": False,
                "execution_id": execution_id,
                "error": "Execution aborted by user",
                "execution_record": execution_record
            }
            
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            execution_record["completed_at"] = datetime.utcnow().isoformat()
            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "execution_record": execution_record
            }
        finally:
            # Cleanup
            if execution_id in self._pause_events:
                del self._pause_events[execution_id]
            if execution_id in self._abort_flags:
                del self._abort_flags[execution_id]
    
    async def _get_tool(self, tool_id: str):
        """Get or initialize the requested tool."""
        if tool_id not in self.tool_instances:
            if tool_id == "sentiment_transformer":
                try:
                    from ..tools.sentiment_analyzer import SentimentAnalyzer
                except ImportError:
                    from tools.sentiment_analyzer import SentimentAnalyzer
                self.tool_instances[tool_id] = SentimentAnalyzer()
                
            elif tool_id == "spacy_ner":
                try:
                    from ..tools.ner_extractor import NERExtractor
                except ImportError:
                    from tools.ner_extractor import NERExtractor
                self.tool_instances[tool_id] = NERExtractor()
                
            elif tool_id == "zero_shot_classifier":
                try:
                    from ..tools.text_classifier import TextClassifier
                except ImportError:
                    from tools.text_classifier import TextClassifier
                self.tool_instances[tool_id] = TextClassifier()
                
            elif tool_id == "lda_topic_model":
                try:
                    from ..tools.topic_modeler import TopicModeler
                except ImportError:
                    from tools.topic_modeler import TopicModeler
                self.tool_instances[tool_id] = TopicModeler()
                
            elif tool_id == "general_analyzer":
                try:
                    from ..tools.general_analyzer import GeneralAnalyzer
                except ImportError:
                    from tools.general_analyzer import GeneralAnalyzer
                self.tool_instances[tool_id] = GeneralAnalyzer()
                
            else:
                raise ValueError(f"Unknown tool: {tool_id}")
        
        return self.tool_instances[tool_id]
    
    async def _execute_tool(
        self,
        tool,
        tool_id: str,
        text: str,
        execution_id: str,
        parameters: Dict,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute the tool with progress tracking and pause/abort support."""
        
        async def check_control_state():
            """Check for pause/abort requests."""
            if self._abort_flags.get(execution_id, False):
                raise asyncio.CancelledError("Aborted by user")
            
            # Wait if paused
            pause_event = self._pause_events.get(execution_id)
            if pause_event:
                await pause_event.wait()
        
        async def report_progress(progress: int, step: str, data: Optional[Dict] = None):
            """Report progress through callback."""
            if self.active_executions.get(execution_id):
                self.active_executions[execution_id]["progress"] = progress
                self.active_executions[execution_id]["current_step"] = step
                
                if data:
                    self.active_executions[execution_id]["intermediate_outputs"].append({
                        "step": step,
                        "progress": progress,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            if progress_callback:
                await progress_callback({
                    "execution_id": execution_id,
                    "progress": progress,
                    "step": step,
                    "data": data
                })
            
            await check_control_state()
        
        # Execute the tool with progress reporting
        result = await tool.analyze(
            text,
            parameters=parameters,
            progress_callback=report_progress
        )
        
        return result
    
    async def pause(self, execution_id: str) -> bool:
        """Pause an active execution."""
        if execution_id in self._pause_events:
            self._pause_events[execution_id].clear()
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "paused"
            return True
        return False
    
    async def resume(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        if execution_id in self._pause_events:
            self._pause_events[execution_id].set()
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "running"
            return True
        return False
    
    async def abort(self, execution_id: str) -> bool:
        """Abort an active execution."""
        if execution_id in self._abort_flags:
            self._abort_flags[execution_id] = True
            # Also resume if paused to allow abort to process
            if execution_id in self._pause_events:
                self._pause_events[execution_id].set()
            return True
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get the status of an execution."""
        return self.active_executions.get(execution_id)
    
    def get_all_active(self, session_id: Optional[str] = None) -> Dict[str, Dict]:
        """Get all active executions, optionally filtered by session."""
        if session_id:
            return {
                eid: info for eid, info in self.active_executions.items()
                if info.get("session_id") == session_id
            }
        return self.active_executions
