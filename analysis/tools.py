"""Analytical Tool Architecture for Talking to Bridges (Phase 2 - Week 5).

Provides:
- Data Access Layer for dataset reading & metadata inspection.
- Base schemas & contracts for analytical tools (inputs, outputs, errors).
- Tool Registry for registering, discovering, validating, and executing tools.
- Deterministic Dummy Analytical Tools ready to be replaced by real implementations in Week 6+.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd

from app.core.config import settings
from app.core.logging_config import logger


# =============================================================================
# 1. DATA ACCESS LAYER
# =============================================================================

class DataAccessLayer:
    """Central data access abstraction so analytical tools do not reimplement file reading."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or settings.DOCUMENTS_DIR)

    def load_dataset(
        self,
        dataset_name_or_bytes: Union[str, bytes],
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads a DataFrame from raw bytes or disk path."""
        if isinstance(dataset_name_or_bytes, bytes):
            return self._parse_bytes(dataset_name_or_bytes, filename or "data.csv")

        path = Path(dataset_name_or_bytes)
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            # Fallback check under documents directory
            alt_path = self.data_dir / path.name
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Dataset file '{dataset_name_or_bytes}' not found.")

        return self._parse_bytes(path.read_bytes(), path.name)

    def _parse_bytes(self, content: bytes, filename: str) -> pd.DataFrame:
        for encoding in ("utf-8", "latin-1", "utf-8-sig"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception:
                continue
        raise ValueError(f"Could not parse CSV content for '{filename}'.")

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspects columns, types, numerical metrics, and timestamp candidates."""
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        
        timestamp_col = None
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                timestamp_col = col
                break

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "timestamp_column": timestamp_col,
        }


# Global Data Access Layer instance
data_access = DataAccessLayer()


# =============================================================================
# 2. TOOL CONTRACT SCHEMAS
# =============================================================================

class ToolExecutionError(Exception):
    """Exception raised during analytical tool execution."""
    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


@dataclass
class ToolDefinition:
    """Metadata definition for a registered analytical tool."""
    name: str
    description: str
    category: str  # e.g. "statistics", "time_series", "anomaly", "visualization", "prediction"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], Optional[pd.DataFrame]], Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


# =============================================================================
# 3. TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Central registry for analytical tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new analytical tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered analytical tool: '{tool.name}' [{tool.category}]")

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve a tool definition by name."""
        if name not in self._tools:
            raise ToolExecutionError(
                error_type="ToolNotFound",
                message=f"Analytical tool '{name}' is not registered.",
                details={"available_tools": list(self._tools.keys())}
            )
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List metadata of all registered tools."""
        return [tool.to_dict() for tool in self._tools.values()]

    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> None:
        """Validate arguments against tool input schema and optional DataFrame columns."""
        tool = self.get_tool(tool_name)
        schema = tool.input_schema
        required = schema.get("required", [])

        # 1. Check required parameters
        missing = [req for req in required if req not in arguments or arguments[req] is None]
        if missing:
            raise ToolExecutionError(
                error_type="InvalidParameter",
                message=f"Missing required parameter(s) for '{tool_name}': {', '.join(missing)}",
                details={"required": required, "provided": list(arguments.keys())}
            )

        # 2. Dataset-aware column validation if df is provided
        if df is not None:
            if "metric" in arguments and arguments["metric"] and arguments["metric"] not in df.columns:
                raise ToolExecutionError(
                    error_type="InvalidParameter",
                    message=f"Metric column '{arguments['metric']}' not found in dataset.",
                    details={"available_columns": list(df.columns)}
                )
            if "sensor_id" in arguments and arguments["sensor_id"] and "sensor_id" in df.columns:
                available_sensors = [str(s) for s in df["sensor_id"].dropna().unique()]
                if str(arguments["sensor_id"]) not in available_sensors:
                    raise ToolExecutionError(
                        error_type="InvalidParameter",
                        message=f"Sensor ID '{arguments['sensor_id']}' not found in dataset.",
                        details={"available_sensors": available_sensors}
                    )

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Execute a tool with structured parameters and error handling."""
        try:
            tool = self.get_tool(tool_name)
            self.validate_arguments(tool_name, arguments, df=df)
            result_data = tool.handler(arguments, df)
            
            return {
                "tool": tool_name,
                "status": "success",
                "data": result_data,
                "metadata": {
                    "category": tool.category,
                    "arguments": arguments,
                },
                "message": f"Tool '{tool_name}' executed successfully."
            }
        except ToolExecutionError as exc:
            return {
                "tool": tool_name,
                "status": "error",
                "error_type": exc.error_type,
                "message": exc.message,
                "details": exc.details,
            }
        except Exception as exc:
            logger.exception(f"Unexpected error executing tool '{tool_name}'")
            return {
                "tool": tool_name,
                "status": "error",
                "error_type": "InternalError",
                "message": f"Error executing tool '{tool_name}': {str(exc)}",
                "details": {}
            }


# Global tool registry instance
registry = ToolRegistry()


# =============================================================================
# 4. DETERMINISTIC DUMMY ANALYTICAL TOOL HANDLERS (WEEK 5 CONTRACTS)
# =============================================================================

def handle_summary_statistics(arguments: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Summary statistics tool handler (computes real stats if df provided, else dummy deterministic)."""
    sensor_id = arguments.get("sensor_id", "S01")
    metric = arguments.get("metric", "deflection")
    
    if df is not None and metric in df.columns:
        sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)] if "sensor_id" in df.columns and sensor_id else df
        series = pd.to_numeric(sub_df[metric], errors="coerce").dropna()
        if not series.empty:
            return {
                "sensor_id": sensor_id,
                "metric": metric,
                "count": int(series.count()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            }

    # Deterministic dummy fallback contract
    return {
        "sensor_id": sensor_id,
        "metric": metric,
        "count": 30,
        "min": 1.7,
        "max": 3.8,
        "mean": 2.54,
        "std": 0.65,
    }


def handle_anomaly_detection(arguments: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Anomaly detection tool handler (contract for Week 6 ML implementation)."""
    sensor_id = arguments.get("sensor_id", "S01")
    metric = arguments.get("metric", "vibration")
    threshold = float(arguments.get("threshold", 2.0))

    if df is not None and metric in df.columns:
        sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)] if "sensor_id" in df.columns and sensor_id else df
        series = pd.to_numeric(sub_df[metric], errors="coerce").dropna()
        if not series.empty:
            mean = series.mean()
            std = series.std() if len(series) > 1 else 1.0
            z_scores = (series - mean) / (std if std > 0 else 1.0)
            anomalies = sub_df[z_scores.abs() > threshold]
            
            return {
                "sensor_id": sensor_id,
                "metric": metric,
                "anomaly_count": len(anomalies),
                "threshold_z": threshold,
                "anomalous_timestamps": list(anomalies["timestamp"].astype(str)) if "timestamp" in anomalies.columns else [],
                "status_flag": "Alert" if len(anomalies) > 0 else "Normal",
            }

    # Deterministic dummy contract
    return {
        "sensor_id": sensor_id,
        "metric": metric,
        "anomaly_count": 2,
        "threshold_z": threshold,
        "anomalous_timestamps": ["2026-01-10", "2026-01-16"],
        "status_flag": "Alert",
    }


def handle_model_result(arguments: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Model result / prediction handler (contract for Week 7 model inference)."""
    sensor_id = arguments.get("sensor_id", "S01")
    target_variable = arguments.get("target_variable", "condition")

    if df is not None and target_variable in df.columns:
        sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)] if "sensor_id" in df.columns and sensor_id else df
        if not sub_df.empty:
            latest = sub_df.iloc[-1]
            return {
                "sensor_id": sensor_id,
                "predicted_class": str(latest[target_variable]),
                "confidence": 0.92,
                "features_used": [c for c in sub_df.columns if c not in ["timestamp", "sensor_id", target_variable]],
                "prediction_timestamp": str(latest.get("timestamp", "latest")),
            }

    # Deterministic dummy contract
    return {
        "sensor_id": sensor_id,
        "predicted_class": "Monitor",
        "confidence": 0.88,
        "features_used": ["temperature", "stress", "deflection", "vibration"],
        "prediction_timestamp": "2026-01-20",
    }


def handle_chart_data(arguments: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Chart data generation tool handler (wraps existing visualization service)."""
    from app.services.visualization_service import analyze_and_plot

    x_col = arguments.get("x_col", "timestamp")
    y_col = arguments.get("y_col", "deflection")

    if df is not None and x_col in df.columns and y_col in df.columns:
        fig, explanation = analyze_and_plot(df, x_col, y_col)
        return {
            "x_col": x_col,
            "y_col": y_col,
            "plot_json": fig.to_json(),
            "explanation": explanation,
        }

    # Deterministic dummy contract
    return {
        "x_col": x_col,
        "y_col": y_col,
        "plot_json": "{}",
        "explanation": f"Generated interactive plot for {y_col} over {x_col}.",
    }


# =============================================================================
# 5. REGISTER DUMMY TOOLS AT IMPORT TIME
# =============================================================================

registry.register(ToolDefinition(
    name="summary_statistics",
    description="Calculates summary statistics (count, min, max, mean, std) for a bridge sensor metric.",
    category="statistics",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "description": "ID of the sensor (e.g., S01)"},
            "metric": {"type": "string", "description": "Numerical measurement column (e.g., deflection, stress, vibration, temperature)"},
            "start_time": {"type": "string", "description": "Optional start date (YYYY-MM-DD)"},
            "end_time": {"type": "string", "description": "Optional end date (YYYY-MM-DD)"},
        },
        "required": ["metric"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "metric": {"type": "string"},
            "count": {"type": "integer"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "mean": {"type": "number"},
            "std": {"type": "number"},
        }
    },
    handler=handle_summary_statistics,
))

registry.register(ToolDefinition(
    name="anomaly_detection",
    description="Detects anomalous sensor readings exceeding statistical or threshold bounds.",
    category="anomaly",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "description": "ID of the sensor (e.g., S01)"},
            "metric": {"type": "string", "description": "Numerical measurement column"},
            "threshold": {"type": "number", "description": "Z-score or deviation threshold"},
        },
        "required": ["metric"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "metric": {"type": "string"},
            "anomaly_count": {"type": "integer"},
            "anomalous_timestamps": {"type": "array", "items": {"type": "string"}},
            "status_flag": {"type": "string"},
        }
    },
    handler=handle_anomaly_detection,
))

registry.register(ToolDefinition(
    name="model_result",
    description="Fetches model inference or structural condition predictions for bridge sensors.",
    category="prediction",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "description": "ID of the sensor (e.g., S01)"},
            "target_variable": {"type": "string", "description": "Target variable (e.g., condition)"},
        },
        "required": ["sensor_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "predicted_class": {"type": "string"},
            "confidence": {"type": "number"},
            "prediction_timestamp": {"type": "string"},
        }
    },
    handler=handle_model_result,
))

registry.register(ToolDefinition(
    name="chart_data",
    description="Generates interactive chart visualization payload for sensor measurements over time.",
    category="visualization",
    input_schema={
        "type": "object",
        "properties": {
            "x_col": {"type": "string", "description": "X-axis column (default timestamp)"},
            "y_col": {"type": "string", "description": "Y-axis numerical column (e.g., deflection)"},
        },
        "required": ["y_col"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "x_col": {"type": "string"},
            "y_col": {"type": "string"},
            "plot_json": {"type": "string"},
            "explanation": {"type": "string"},
        }
    },
    handler=handle_chart_data,
))
