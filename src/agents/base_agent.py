"""
Base Agent - Abstract base class for all agents.

This module defines the abstract base class that all agents in the system
must inherit from. It provides common functionality like logging, metrics
tracking, and error handling.

All agents follow a consistent interface:
1. Receive AgentState
2. Process/transform state
3. Return updated AgentState

Example:
    >>> class MyAgent(BaseAgent):
    ...     def execute(self, state: AgentState) -> AgentState:
    ...         # Agent logic here
    ...         return state
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime

from src.models.agent_state import AgentState
from src.utils.logger import setup_logger
from src.utils.exceptions import AgentExecutionError


class BaseAgent(ABC):
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        log_level: str = "INFO"
    ):
        
        self.name = name
        self.version = version
        
        # Setup logger with agent-specific name
        self.logger = setup_logger(
            name=f"agent.{name}",
            level=log_level
        )
        
        # Initialize metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time_seconds": 0.0,
            "average_time_seconds": 0.0,
            "last_execution_time": None,
            "created_at": datetime.now().isoformat()
        }
        
        self.log(f"Agent '{name}' (v{version}) initialized", level="debug")
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        pass
    
    def run(self, state: AgentState) -> AgentState:
        
        start_time = time.time()
        
        try:
            # Log execution start
            self.log(
                f"Executing agent with query: {state.query[:50]}...",
                level="info"
            )
            
            # Execute agent logic
            updated_state = self.execute(state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update metrics for success
            self._update_metrics(
                success=True,
                execution_time=execution_time
            )
            
            # Log success
            self.log(
                f"Execution completed successfully in {execution_time:.2f}s",
                level="info"
            )
            
            return updated_state
            
        except Exception as e:
            # Calculate execution time even for failures
            execution_time = time.time() - start_time
            
            # Update metrics for failure
            self._update_metrics(
                success=False,
                execution_time=execution_time
            )
            
            # Log error
            self.log(
                f"Execution failed after {execution_time:.2f}s: {str(e)}",
                level="error"
            )
            
            # Wrap in AgentExecutionError if not already
            if isinstance(e, AgentExecutionError):
                raise
            else:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message=str(e),
                    details={
                        "execution_time": execution_time,
                        "query": state.query,
                        "original_error": type(e).__name__
                    }
                ) from e
    
    def log(self, message: str, level: str = "info") -> None:
        """
        Log message with agent context.
        
        Automatically prefixes message with agent name for easy tracking.
        
        Args:
            message: Log message
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        
        Example:
            >>> self.log("Processing started", level="info")
            >>> self.log("Invalid input detected", level="warning")
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{self.name}] {message}")
    
    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """
        Update agent performance metrics.
        
        Tracks calls, success rate, and timing statistics.
        
        Args:
            success: Whether execution was successful
            execution_time: Execution time in seconds
        
        Note:
            This is an internal method, use run() which calls this automatically.
        """
        self.metrics["total_calls"] += 1
        
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
        
        self.metrics["total_time_seconds"] += execution_time
        self.metrics["average_time_seconds"] = (
            self.metrics["total_time_seconds"] / self.metrics["total_calls"]
        )
        self.metrics["last_execution_time"] = execution_time
    
    
    def __repr__(self) -> str:
        """
        String representation of agent.
        
        Returns:
            String describing the agent
        
        Example:
            >>> print(agent)
            PlannerAgent(name='planner', version='1.0.0', calls=5)
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"calls={self.metrics['total_calls']})"
        )
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self.name} (v{self.version})"