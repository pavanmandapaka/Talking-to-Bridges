"""Pydantic schemas for analytical tool inputs and outputs (Phase 2 - Week 6).

Every tool that is registered in the ToolRegistry must have:
  - An Input schema  (validates arguments BEFORE calling the handler)
  - An Output schema (describes the structured result returned by the handler)

Design rules
------------
* All fields must be JSON-serializable (no numpy/pandas objects).
* Optional fields must carry a sensible default so callers can omit them.
* Sensor column names are grounded in the actual professor dataset schema:
    Processed by Kolla's pipeline  -> Sensor_1 .. Sensor_5, Relative_Time_Sec,
                                     Condition, Damage_Level, Specimen, Test_Type
    Uploaded fixture / CSV schema  -> sensor-specific names such as
                                     deflection, vibration, stress, temperature,
                                     crack_width, sensor_id, timestamp, condition
* The error envelope (ToolErrorOutput) is returned by the registry when a tool
  call fails; it is NOT a sub-class of the per-tool output schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared error envelope
# ---------------------------------------------------------------------------

class ToolErrorOutput(BaseModel):
    """Returned by the registry / dispatcher on any tool failure."""

    tool: str
    status: str = "error"
    error_type: str = Field(
        ...,
        description="UNKNOWN_TOOL | INVALID_INPUT | DATA_UNAVAILABLE | TOOL_EXECUTION_ERROR",
    )
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. summary_statistics
# ---------------------------------------------------------------------------

class SummaryStatisticsInput(BaseModel):
    """Input schema for the summary_statistics tool.

    Works with both the fixture CSV schema (sensor_id, metric by name) and the
    professor's processed dataset (Sensor_1 .. Sensor_5).
    """

    metric: str = Field(
        ...,
        description=(
            "Numerical measurement column name.  Examples from the fixture: "
            "'deflection', 'vibration', 'stress', 'temperature', 'crack_width'. "
            "Examples from the professor's processed dataset: 'Sensor_1' .. 'Sensor_5'."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Filter rows by sensor_id value when that column is present (e.g. 'S01').",
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 start date/time filter (e.g. '2026-01-01').",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 end date/time filter (e.g. '2026-01-31').",
    )


class SummaryStatisticsOutput(BaseModel):
    """Structured result from the summary_statistics tool."""

    sensor_id: Optional[str] = None
    metric: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    rms: Optional[float] = None   # Root-mean-square (from professor dataset)
    p2p: Optional[float] = None   # Peak-to-peak range (from professor dataset)


# ---------------------------------------------------------------------------
# 2. anomaly_detection
# ---------------------------------------------------------------------------

class AnomalyDetectionInput(BaseModel):
    """Input schema for the anomaly_detection tool.

    Uses z-score thresholding as a placeholder; Kolla's ML-based detector
    should replace the handler while keeping this schema intact.
    """

    metric: str = Field(
        ...,
        description="Numerical column to scan for anomalies (e.g. 'vibration', 'Sensor_3').",
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Filter to a single sensor before anomaly detection.",
    )
    threshold: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Z-score magnitude at which a reading is flagged as anomalous.",
    )


class AnomalyDetectionOutput(BaseModel):
    """Structured result from the anomaly_detection tool."""

    sensor_id: Optional[str] = None
    metric: str
    threshold_z: float
    total_rows_checked: int
    anomaly_count: int
    anomaly_fraction: float
    anomalous_indices: List[int] = Field(default_factory=list)
    anomalous_timestamps: List[str] = Field(default_factory=list)
    status_flag: str   # "Normal" | "Alert"


# ---------------------------------------------------------------------------
# 3. trend_analysis
# ---------------------------------------------------------------------------

class TrendAnalysisInput(BaseModel):
    """Input schema for the trend_analysis tool.

    Fits a linear trend to the metric over time and reports slope direction
    and strength.  Works on Relative_Time_Sec (professor dataset) or any
    numeric time column (timestamp index on the fixture dataset).
    """

    metric: str = Field(
        ...,
        description="Numerical column to analyse for trend (e.g. 'deflection', 'Sensor_2').",
    )
    time_column: Optional[str] = Field(
        default=None,
        description=(
            "Name of the time / X-axis column.  Defaults to 'Relative_Time_Sec' if "
            "present, otherwise falls back to the DataFrame integer index."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor_id filter when that column is present.",
    )


class TrendAnalysisOutput(BaseModel):
    """Structured result from the trend_analysis tool."""

    metric: str
    time_column: str
    slope: float
    intercept: float
    r_squared: float
    trend_direction: str   # "increasing" | "decreasing" | "flat"
    data_points: int
    start_value: float
    end_value: float
    change_magnitude: float


# ---------------------------------------------------------------------------
# 4. correlation_analysis
# ---------------------------------------------------------------------------

class CorrelationAnalysisInput(BaseModel):
    """Input schema for the correlation_analysis tool.

    Computes Pearson correlation coefficients between pairs of numeric columns
    (e.g. Sensor_1 vs Sensor_5, deflection vs vibration).
    """

    columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of column names to correlate.  When absent, all numeric columns "
            "in the dataset are used.  Must contain at least 2 names."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor_id filter applied before computing correlations.",
    )


class CorrelationAnalysisOutput(BaseModel):
    """Structured result from the correlation_analysis tool."""

    columns_used: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    strongest_pair: Optional[str] = None
    weakest_pair: Optional[str] = None


# ---------------------------------------------------------------------------
# 5. model_result  (placeholder - Krishna's Week 7 ML inference)
# ---------------------------------------------------------------------------

class ModelResultInput(BaseModel):
    """Input schema for the model_result tool."""

    sensor_id: str = Field(
        ...,
        description="Sensor / specimen identifier for which a prediction is requested.",
    )
    target_variable: str = Field(
        default="condition",
        description="Target column/label the model predicts (e.g. 'condition', 'Damage_Level').",
    )


class ModelResultOutput(BaseModel):
    """Structured result from the model_result tool."""

    sensor_id: str
    target_variable: str
    predicted_class: str
    confidence: float
    features_used: List[str] = Field(default_factory=list)
    prediction_timestamp: str
    model_note: str = "Placeholder result - replace with Krishna's trained model in Week 7."


# ---------------------------------------------------------------------------
# 6. chart_data  (wraps Nagarjun's visualization_service)
# ---------------------------------------------------------------------------

class ChartDataInput(BaseModel):
    """Input schema for the chart_data tool."""

    y_col: str = Field(
        ...,
        description="Numerical column to plot on the Y axis (e.g. 'deflection', 'Sensor_1').",
    )
    x_col: str = Field(
        default="timestamp",
        description="Column to use for the X axis.  Defaults to 'timestamp' / 'Relative_Time_Sec'.",
    )


class ChartDataOutput(BaseModel):
    """Structured result from the chart_data tool."""

    x_col: str
    y_col: str
    plot_json: str
    explanation: str


# ---------------------------------------------------------------------------
# Schema registries - used by the dispatcher for pre-validation
# ---------------------------------------------------------------------------

TOOL_INPUT_SCHEMAS: Dict[str, type] = {
    "summary_statistics": SummaryStatisticsInput,
    "anomaly_detection": AnomalyDetectionInput,
    "trend_analysis": TrendAnalysisInput,
    "correlation_analysis": CorrelationAnalysisInput,
    "model_result": ModelResultInput,
    "chart_data": ChartDataInput,
}

TOOL_OUTPUT_SCHEMAS: Dict[str, type] = {
    "summary_statistics": SummaryStatisticsOutput,
    "anomaly_detection": AnomalyDetectionOutput,
    "trend_analysis": TrendAnalysisOutput,
    "correlation_analysis": CorrelationAnalysisOutput,
    "model_result": ModelResultOutput,
    "chart_data": ChartDataOutput,
}
