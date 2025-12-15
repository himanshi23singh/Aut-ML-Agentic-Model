"""
Pipeline Orchestrator
Coordinates the entire NLP pipeline execution.
"""
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

try:
    from ..schemas.stream_state import StreamState, StageEnum, StatusEnum, LogLevel
    from ..modules.input_handler import InputHandler
    from ..modules.preprocessor import Preprocessor
    from ..modules.task_detector import TaskDetector
    from ..modules.tool_selector import ToolSelector
    from ..modules.execution_engine import ExecutionEngine
    from ..modules.aggregator import Aggregator
    from ..modules.output_generator import OutputGenerator
    from .state_manager import StateManager
    from .websocket_manager import WebSocketManager
except ImportError:
    from schemas.stream_state import StreamState, StageEnum, StatusEnum, LogLevel
    from modules.input_handler import InputHandler
    from modules.preprocessor import Preprocessor
    from modules.task_detector import TaskDetector
    from modules.tool_selector import ToolSelector
    from modules.execution_engine import ExecutionEngine
    from modules.aggregator import Aggregator
    from modules.output_generator import OutputGenerator
    from services.state_manager import StateManager
    from services.websocket_manager import WebSocketManager


class PipelineOrchestrator:
    """
    Orchestrates the complete NLP pipeline from input to output.
    Provides glass-box transparency by streaming all decisions and state.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        ws_manager: WebSocketManager
    ):
        self.state_manager = state_manager
        self.ws_manager = ws_manager
        
        # Initialize pipeline components
        self.input_handler = InputHandler()
        self.preprocessor = Preprocessor()
        self.task_detector = TaskDetector()
        self.tool_selector = ToolSelector()
        self.execution_engine = ExecutionEngine()
        self.aggregator = Aggregator()
        self.output_generator = OutputGenerator()
    
    async def run_pipeline(
        self,
        session_id: str,
        text: Optional[str] = None,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None,
        force_task: Optional[str] = None,
        force_tool: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete NLP pipeline.
        
        Args:
            session_id: Session ID for tracking
            text: Raw text input
            file_content: File content bytes
            filename: Original filename
            force_task: Override task detection
            force_tool: Override tool selection
        
        Returns:
            Pipeline result with full trace
        """
        state = self.state_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        
        try:
            # Register intervention handlers
            self._register_session_handlers(session_id)
            
            # Stage 1: Input Processing
            await self._run_input_stage(session_id, state, text, file_content, filename)
            
            # Check for abort
            if await self._check_abort(session_id, state):
                return self._create_abort_result(session_id, state)
            
            # Stage 2: Preprocessing
            processed_text = await self._run_preprocessing_stage(session_id, state)
            
            if await self._check_abort(session_id, state):
                return self._create_abort_result(session_id, state)
            
            # Stage 3: Task Detection
            detected_task = await self._run_task_detection_stage(
                session_id, state, processed_text, force_task
            )
            
            if await self._check_abort(session_id, state):
                return self._create_abort_result(session_id, state)
            
            # Stage 4: Tool Selection
            selected_tool = await self._run_tool_selection_stage(
                session_id, state, detected_task, len(processed_text), force_tool
            )
            
            if await self._check_abort(session_id, state):
                return self._create_abort_result(session_id, state)
            
            # Stage 5: Execution
            execution_result = await self._run_execution_stage(
                session_id, state, selected_tool, processed_text
            )
            
            if await self._check_abort(session_id, state):
                return self._create_abort_result(session_id, state)
            
            # Stage 6: Aggregation
            aggregated_result = await self._run_aggregation_stage(
                session_id, state, [execution_result], detected_task
            )
            
            # Stage 7: Draft Output
            draft = await self._run_draft_output_stage(
                session_id, state, aggregated_result, detected_task
            )
            
            # Mark as waiting for approval
            state.status = StatusEnum.WAITING_APPROVAL
            await self._broadcast_state(session_id, state)
            
            return {
                "success": True,
                "session_id": session_id,
                "status": "waiting_approval",
                "draft": draft,
                "state": state.to_broadcast()
            }
            
        except Exception as e:
            state.status = StatusEnum.ERROR
            state.error_message = str(e)
            state.error_stage = state.current_stage
            state.add_log(f"Pipeline error: {str(e)}", LogLevel.ERROR)
            await self._broadcast_state(session_id, state)
            
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
                "stage": state.current_stage.value,
                "state": state.to_broadcast()
            }
    
    async def approve_draft(self, session_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Approve the draft and generate final output."""
        state = self.state_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        
        # Stage 8: Final Output
        await self.state_manager.transition_stage(session_id, StageEnum.FINAL_OUTPUT)
        
        final = await self.output_generator.approve_draft(session_id, notes)
        
        state.final_output = final
        state.status = StatusEnum.COMPLETED
        state.complete_stage(StageEnum.FINAL_OUTPUT, "Output finalized")
        state.add_log("Draft approved - final output generated", LogLevel.INFO)
        
        await self._broadcast_state(session_id, state)
        
        return {
            "success": True,
            "session_id": session_id,
            "final_output": final,
            "state": state.to_broadcast()
        }
    
    # Private stage methods
    async def _run_input_stage(
        self, 
        session_id: str, 
        state: StreamState,
        text: Optional[str],
        file_content: Optional[bytes],
        filename: Optional[str]
    ):
        """Run the input processing stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.INPUT)
        await self._broadcast_progress(session_id, "input", 0, "Starting input processing")
        
        extracted_text, metadata = await self.input_handler.process_input(
            text=text,
            file_content=file_content,
            filename=filename
        )
        
        # Store for later stages
        state.input_length = len(extracted_text)
        state._extracted_text = extracted_text  # Internal storage
        state._input_metadata = metadata
        
        await self.state_manager.add_log(
            session_id,
            f"Input processed: {metadata.get('input_type')} ({len(extracted_text)} chars)",
            LogLevel.INFO
        )
        
        state.complete_stage(StageEnum.INPUT, f"Processed {len(extracted_text)} characters")
        await self._broadcast_progress(session_id, "input", 100, "Input processing complete")
    
    async def _run_preprocessing_stage(
        self, 
        session_id: str, 
        state: StreamState
    ) -> str:
        """Run the preprocessing stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.PREPROCESSING)
        await self._broadcast_progress(session_id, "preprocessing", 0, "Starting preprocessing")
        
        processed_text, metadata = await self.preprocessor.preprocess(
            state._extracted_text,
            steps=['clean', 'normalize', 'detect_language']
        )
        
        state.detected_language = metadata.get("language_detected")
        
        # Log decisions
        for decision in metadata.get("decisions", []):
            await self.state_manager.add_decision_log(
                session_id,
                "Preprocessing",
                decision
            )
        
        state.complete_stage(
            StageEnum.PREPROCESSING,
            f"Language: {state.detected_language}, {metadata.get('length_reduction_pct')}% reduction"
        )
        await self._broadcast_progress(session_id, "preprocessing", 100, "Preprocessing complete")
        
        return processed_text
    
    async def _run_task_detection_stage(
        self,
        session_id: str,
        state: StreamState,
        text: str,
        force_task: Optional[str]
    ) -> str:
        """Run the task detection stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.TASK_DETECTION)
        await self._broadcast_progress(session_id, "task_detection", 0, "Detecting task")
        
        detected_task, metadata = await self.task_detector.detect_task(text, force_task)
        
        state.detected_task = detected_task
        state.task_confidence = metadata.get("confidence", 0)
        
        # Broadcast decision for glass-box transparency
        await self.ws_manager.broadcast_decision(
            session_id,
            "task_detection",
            {
                "selected_task": detected_task,
                "confidence": state.task_confidence,
                "rationale": metadata.get("rationale"),
                "alternatives": metadata.get("task_scores", {})
            }
        )
        
        await self.state_manager.add_decision_log(
            session_id,
            "Task Detection",
            metadata.get("rationale", f"Selected: {detected_task}")
        )
        
        # Broadcast intervention opportunity
        await self.ws_manager.broadcast_intervention_available(
            session_id,
            "task_override",
            {
                "current_task": detected_task,
                "available_tasks": self.task_detector.get_available_tasks()
            }
        )
        
        state.complete_stage(
            StageEnum.TASK_DETECTION,
            f"Detected: {detected_task} ({state.task_confidence:.0%})"
        )
        await self._broadcast_progress(session_id, "task_detection", 100, "Task detected")
        
        return detected_task
    
    async def _run_tool_selection_stage(
        self,
        session_id: str,
        state: StreamState,
        task: str,
        text_length: int,
        force_tool: Optional[str]
    ) -> str:
        """Run the tool selection stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.TOOL_SELECTION)
        await self._broadcast_progress(session_id, "tool_selection", 0, "Selecting tool")
        
        selected_tool, metadata = await self.tool_selector.select_tool(
            task, text_length, force_tool
        )
        
        state.selected_tool = selected_tool
        
        # Broadcast decision
        await self.ws_manager.broadcast_decision(
            session_id,
            "tool_selection",
            {
                "selected_tool": selected_tool,
                "tool_info": metadata.get("tool_info", {}),
                "reasoning": metadata.get("reasoning", [])
            }
        )
        
        await self.state_manager.add_decision_log(
            session_id,
            "Tool Selection",
            f"Selected {selected_tool} for {task}"
        )
        
        # Broadcast intervention opportunity
        await self.ws_manager.broadcast_intervention_available(
            session_id,
            "tool_override",
            {
                "current_tool": selected_tool,
                "available_tools": self.tool_selector.get_available_tools(task)
            }
        )
        
        state.complete_stage(
            StageEnum.TOOL_SELECTION,
            f"Selected: {metadata.get('tool_info', {}).get('name', selected_tool)}"
        )
        await self._broadcast_progress(session_id, "tool_selection", 100, "Tool selected")
        
        return selected_tool
    
    async def _run_execution_stage(
        self,
        session_id: str,
        state: StreamState,
        tool_id: str,
        text: str
    ) -> Dict[str, Any]:
        """Run the execution stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.EXECUTION)
        
        async def progress_callback(update: Dict):
            progress = update.get("progress", 0)
            step = update.get("step", "Processing")
            await self._broadcast_progress(
                session_id, 
                "execution", 
                progress, 
                step,
                update.get("data")
            )
        
        result = await self.execution_engine.execute(
            session_id=session_id,
            tool_id=tool_id,
            text=text,
            parameters={"text_length": len(text)},
            progress_callback=progress_callback
        )
        
        if result.get("success"):
            state.complete_stage(
                StageEnum.EXECUTION,
                f"Execution complete: {tool_id}"
            )
        else:
            state.add_log(
                f"Execution warning: {result.get('error', 'Unknown')}",
                LogLevel.WARNING
            )
        
        return result
    
    async def _run_aggregation_stage(
        self,
        session_id: str,
        state: StreamState,
        results: list,
        task: str
    ) -> Dict[str, Any]:
        """Run the aggregation stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.AGGREGATION)
        await self._broadcast_progress(session_id, "aggregation", 0, "Aggregating results")
        
        aggregated = await self.aggregator.aggregate(results, task, session_id)
        
        await self._broadcast_progress(session_id, "aggregation", 100, "Aggregation complete")
        state.complete_stage(StageEnum.AGGREGATION, aggregated.get("summary", "Results aggregated"))
        
        return aggregated
    
    async def _run_draft_output_stage(
        self,
        session_id: str,
        state: StreamState,
        aggregated: Dict,
        task: str
    ) -> Dict[str, Any]:
        """Run the draft output stage."""
        await self.state_manager.transition_stage(session_id, StageEnum.DRAFT_OUTPUT)
        await self._broadcast_progress(session_id, "draft_output", 0, "Generating draft")
        
        draft = await self.output_generator.generate_draft(
            session_id,
            aggregated,
            task,
            {"input_metadata": getattr(state, "_input_metadata", {})}
        )
        
        state.draft_output = draft
        state.draft_approved = False
        
        await self._broadcast_progress(session_id, "draft_output", 100, "Draft ready for review")
        state.complete_stage(StageEnum.DRAFT_OUTPUT, "Draft generated - awaiting approval")
        
        return draft
    
    # Helper methods
    def _register_session_handlers(self, session_id: str):
        """Register intervention handlers for a session."""
        # Handlers are connected to the execution engine
        self.state_manager.register_intervention_handler(
            session_id, "pause",
            lambda: self._handle_pause(session_id)
        )
        self.state_manager.register_intervention_handler(
            session_id, "resume",
            lambda: self._handle_resume(session_id)
        )
    
    async def _handle_pause(self, session_id: str):
        """Handle pause intervention."""
        # Get current execution and pause it
        executions = self.execution_engine.get_all_active(session_id)
        for exec_id in executions:
            await self.execution_engine.pause(exec_id)
    
    async def _handle_resume(self, session_id: str):
        """Handle resume intervention."""
        executions = self.execution_engine.get_all_active(session_id)
        for exec_id in executions:
            await self.execution_engine.resume(exec_id)
    
    async def _check_abort(self, session_id: str, state: StreamState) -> bool:
        """Check if the session has been aborted."""
        return state.status == StatusEnum.ABORTED
    
    def _create_abort_result(self, session_id: str, state: StreamState) -> Dict[str, Any]:
        """Create result for an aborted pipeline."""
        return {
            "success": False,
            "session_id": session_id,
            "status": "aborted",
            "stage": state.current_stage.value,
            "state": state.to_broadcast()
        }
    
    async def _broadcast_state(self, session_id: str, state: StreamState):
        """Broadcast state update via WebSocket."""
        await self.ws_manager.broadcast_state_update(session_id, state.to_broadcast())
    
    async def _broadcast_progress(
        self,
        session_id: str,
        stage: str,
        progress: float,
        step: str,
        details: Optional[Dict] = None
    ):
        """Broadcast progress update via WebSocket."""
        await self.ws_manager.broadcast_progress(session_id, stage, progress, {
            "step": step,
            **(details or {})
        })
