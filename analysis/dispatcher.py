"""Tool Dispatcher for the Talking to Bridges LLM/Agent layer (Phase 2 - Week 6).

Responsibilities
----------------
1. Intent detection: map a plain-English query to a registered tool name +
   best-guess arguments using keyword heuristics (no LLM required).
2. Dataset resolution: find the active uploaded DataFrame (if any) so tools
   receive real data rather than placeholder results.
3. Tool invocation: delegate cleanly to the ToolRegistry and return a
   JSON-serializable result envelope.
4. Fallback: return None when no analytical intent is detected so the
   caller (routes.py) can fall back to pure RAG retrieval.

Architecture
------------

User query (plain text)
    |
    v
ToolDispatcher.detect_intent(query, df?)
    |
    +-- match found --> ToolDispatcher.dispatch(tool_name, args, df?)
    |                       |
    |                       v
    |                  ToolRegistry.execute(name, args, df)
    |                       |
    |                       v
    |                  handler(args, df) -> structured dict
    |                       |
    |                       v
    |                  DispatchResult(tool, status, data, llm_summary)
    |
    +-- no match --> None (caller uses pure RAG)

This module does NOT import routes.py (no circular dependency).
The routes.py file imports from here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.logging_config import logger
from analysis.tools import data_access, registry


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Returned by ToolDispatcher.dispatch().

    Attributes:
        tool_name:   Name of the executed tool.
        status:      "success" or "error".
        data:        Dict from the tool handler (on success).
        error_type:  Error code string (on failure).
        message:     Human-readable description of the result/error.
        metadata:    Category and arguments used.
        llm_summary: A short prose summary injected into the LLM context block.
    """

    tool_name: str
    status: str
    data: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_summary: str = ""


# ---------------------------------------------------------------------------
# Intent detection rules
# ---------------------------------------------------------------------------

# Each rule is a (tool_name, keyword_set) tuple.
# The first rule whose keywords intersect the query terms wins.
_INTENT_RULES: List[tuple] = [
    # Chart / visualization  (check BEFORE statistics to avoid 'plot average')
    # NOTE: 'show' and 'display' are intentionally omitted because they appear
    # in many non-chart queries (e.g. 'show statistics', 'show trend').
    ("chart_data", {
        "chart", "plot", "graph", "visualize", "visualise",
        "draw", "figure",
    }),
    # Anomaly
    ("anomaly_detection", {
        "anomaly", "anomalies", "outlier", "outliers",
        "alert", "spike", "spikes", "abnormal", "unusual",
    }),
    # Trend
    ("trend_analysis", {
        "trend", "trends", "increasing", "decreasing", "rising",
        "falling", "drift", "change over time", "slope",
    }),
    # Correlation
    ("correlation_analysis", {
        "correlat", "correlation", "relationship", "relation",
        "compare sensors", "compare columns",
    }),
    # Baseline / reference state
    ("baseline", {
        "baseline", "healthy", "healthy baseline", "deviation",
        "deviation ranking", "normal baseline", "reference state",
    }),
    # Prediction / model
    ("model_result", {
        "predict", "condition", "classify", "classification",
        "model", "forecast", "ml", "damage level", "damage class", "severity",
    }),
    # Summary statistics  (broadest, check LAST)
    ("summary_statistics", {
        "stat", "statistic", "statistics", "average", "mean",
        "min", "minimum", "max", "maximum", "std", "standard deviation",
        "summary", "describe", "distribution",
    }),
]

# Metric keywords matched against column names in the active dataset or known aliases
_SENSOR_ALIASES = {
    "deflection": "deflection",
    "vibration":  "vibration",
    "stress":     "stress",
    "temperature": "temperature",
    "crack":      "crack_width",
    "sensor1":    "Sensor_1",
    "sensor_1":   "Sensor_1",
    "sensor2":    "Sensor_2",
    "sensor_2":   "Sensor_2",
    "sensor3":    "Sensor_3",
    "sensor_3":   "Sensor_3",
    "sensor4":    "Sensor_4",
    "sensor_4":   "Sensor_4",
    "sensor5":    "Sensor_5",
    "sensor_5":   "Sensor_5",
}


# Comparison charts (healthy vs uploaded). A comparison word is required; the other
# words only pick which comparison chart is meant.
_COMPARE_WORDS = {"healthy", "baseline", "compare", "comparison", "compared", "versus", "vs", "against"}
_FEATURE_WORDS = {
    "feature", "features", "rms", "kurtosis", "skewness", "variance", "crest", "fft",
    "deviation", "deviations", "deviate", "range",
}
_HEATMAP_WORDS = {"heatmap", "heat"}


_SENSOR_PHRASE = re.compile(r"\bsensor[\s_-]*([1-9])\b")


def _named_sensor(query: str, df: Optional[pd.DataFrame]) -> Optional[str]:
    """'sensor 3' / 'Sensor-3' typed in a query -> the 'Sensor_3' column, when it exists."""
    match = _SENSOR_PHRASE.search(query.lower())
    if match and df is not None and f"Sensor_{match.group(1)}" in df.columns:
        return f"Sensor_{match.group(1)}"
    return None


def _detect_comparison(tokens: List[str]) -> Optional[str]:
    """'signals' | 'features' | 'heatmap' when the query asks to compare with the healthy state."""
    words = set(tokens)
    if not words & _COMPARE_WORDS:
        return None
    if words & _HEATMAP_WORDS:
        return "heatmap"
    if words & _FEATURE_WORDS:
        return "features"
    return "signals"


def _tokenize(text: str) -> List[str]:
    """Lower-case, alphanumeric + underscore tokenization."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _extract_metric(tokens: List[str], df: Optional[pd.DataFrame]) -> Optional[str]:
    """Return the best matching metric column from the query tokens.

    Priority:
    1. Any token that is a direct column name in df.
    2. Any token matching a known alias.
    3. First numeric column in df (fallback).
    """
    # Build alias map from the actual df columns too
    if df is not None:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        for tok in tokens:
            if tok in numeric_cols:
                return tok
            # Match via known aliases
            alias = _SENSOR_ALIASES.get(tok)
            if alias and alias in df.columns:
                return alias
        if numeric_cols:
            return numeric_cols[0]

    # Fallback without a df
    for tok in tokens:
        alias = _SENSOR_ALIASES.get(tok)
        if alias:
            return alias

    return None


def _extract_sensor_id(tokens: List[str], df: Optional[pd.DataFrame]) -> Optional[str]:
    """Extract a sensor_id from the query (e.g. 's01', 's1', 'sensor 1')."""
    # Look for patterns like s01, s1, sensor1 in tokenized query
    sensor_id_pattern = re.compile(r"^s\d+$")
    for tok in tokens:
        if sensor_id_pattern.match(tok):
            return tok.upper()
    return None


def _detect_tool(query: str) -> Optional[str]:
    """Return the matched tool name or None."""
    tokens = set(_tokenize(query))
    for tool_name, keywords in _INTENT_RULES:
        # Support multi-word keywords like "change over time"
        for kw in keywords:
            kw_tokens = set(_tokenize(kw))
            if kw_tokens.issubset(tokens) or any(tok.startswith(kw) for tok in tokens):
                return tool_name
    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Routes user queries to the correct analytical tool.

    This is the bridge between the LLM/chat pipeline and the ToolRegistry.
    The LLM does not call this directly; routes.py calls detect_and_dispatch()
    before building the LLM prompt.
    """

    def __init__(self):
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_dispatch(
        self,
        query: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[DispatchResult]:
        """Detect analytical intent in query and, if found, execute the tool.

        Args:
            query: The raw user query string.
            df:    Optional active DataFrame (from the most recently uploaded file).

        Returns:
            DispatchResult if an analytical tool was invoked, else None.
        """
        tool_name = _detect_tool(query)
        if tool_name is None:
            return None

        arguments = self._build_arguments(tool_name, query, df)
        return self.dispatch(tool_name, arguments, df)

    def dispatch(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
    ) -> DispatchResult:
        """Execute tool_name with arguments and wrap the result in a DispatchResult.

        Args:
            tool_name:  Registered tool name.
            arguments:  Dict of tool arguments (validated by the registry).
            df:         Optional DataFrame to pass to the tool handler.

        Returns:
            DispatchResult with status, data or error, and llm_summary.
        """
        raw = self._registry.execute(tool_name, arguments, df=df)

        if raw["status"] == "success":
            summary = self._build_llm_summary(tool_name, raw["data"], arguments)
            return DispatchResult(
                tool_name=tool_name,
                status="success",
                data=raw["data"],
                message=raw.get("message", ""),
                metadata=raw.get("metadata", {}),
                llm_summary=summary,
            )
        else:
            return DispatchResult(
                tool_name=tool_name,
                status="error",
                error_type=raw.get("error_type"),
                message=raw.get("message", "Tool execution failed."),
                metadata={},
                llm_summary=(
                    f"[Tool Error: {raw.get('error_type', 'UNKNOWN')}] "
                    f"{raw.get('message', '')}"
                ),
            )

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tool definitions (JSON-safe)."""
        return self._registry.list_tools()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_arguments(
        self,
        tool_name: str,
        query: str,
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build the best-guess argument dict for tool_name from the query."""
        tokens = _tokenize(query)
        metric = _extract_metric(tokens, df)
        sensor_id = _extract_sensor_id(tokens, df)

        if tool_name == "summary_statistics":
            args: Dict[str, Any] = {"metric": metric or "deflection"}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "anomaly_detection":
            # Look for threshold hints in query (e.g. "threshold 3", "3 sigma")
            threshold = 5.0
            for tok in tokens:
                try:
                    v = float(tok)
                    if 0.5 <= v <= 10.0:
                        threshold = v
                        break
                except ValueError:
                    pass
            args = {"metric": metric or "vibration", "threshold": threshold}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "trend_analysis":
            args = {"metric": metric or "deflection"}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "correlation_analysis":
            # If specific columns mentioned, use them; else use all numeric cols
            args = {}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "model_result":
            return {
                "sensor_id": sensor_id or "S01",
                "target_variable": "condition",
            }

        if tool_name == "baseline":
            args = {}
            if sensor_id:
                args["sensor_id"] = sensor_id
            if metric:
                args["metric"] = metric
            return args

        if tool_name == "chart_data":
            y_col = metric or "deflection"
            # X column: prefer Relative_Time_Sec if available in df
            x_col = "timestamp"
            if df is not None and "Relative_Time_Sec" in df.columns:
                x_col = "Relative_Time_Sec"
            y_col = _named_sensor(query, df) or y_col
            args = {"y_col": y_col, "x_col": x_col}
            comparison = _detect_comparison(tokens)
            if comparison:
                args["comparison"] = comparison
            return args

        return {}

    def _build_llm_summary(
        self,
        tool_name: str,
        data: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> str:
        """Build a concise prose summary of the tool result for injection into
        the LLM context.  The LLM uses this to compose its natural-language
        reply to the user.
        """
        metric = data.get("metric") or arguments.get("metric") or arguments.get("y_col", "")
        sensor = data.get("sensor_id") or arguments.get("sensor_id", "")
        sensor_str = f" for sensor '{sensor}'" if sensor else ""

        if tool_name == "summary_statistics":
            return (
                f"[ANALYTICAL RESULT: summary_statistics]\n"
                f"Metric: {metric}{sensor_str}\n"
                f"Count: {data.get('count')}  |  "
                f"Min: {data.get('min')}  |  Max: {data.get('max')}  |  "
                f"Mean: {data.get('mean')}  |  Median: {data.get('median')}  |  "
                f"Std: {data.get('std')}  |  RMS: {data.get('rms')}  |  "
                f"Peak-to-Peak: {data.get('p2p')}\n"
                f"Note: {data.get('_note', 'Real data result.')}"
            )

        if tool_name == "anomaly_detection":
            event_count = data.get("event_count", 0)
            max_abs_z = data.get("max_abs_z")
            events = data.get("events", [])

            lines = [
                "[ANALYTICAL RESULT: anomaly_detection]",
                f"Metric: {metric}{sensor_str}  |  Z-score threshold: {data.get('threshold_z')}",
                f"Total rows checked: {data.get('total_rows_checked')}  |  Anomalies found: {data.get('anomaly_count')} ({data.get('anomaly_fraction', 0)*100:.1f}%)",
                f"Status: {data.get('status_flag')}",
            ]
            if max_abs_z is not None:
                lines.append(f"Peak score: {max_abs_z} (max z-score)")
            if event_count > 0 or events:
                lines.append(f"Event count: {event_count} detected event(s)")
                if events:
                    ev_previews = []
                    for ev in events[:3]:
                        pz = ev.get("peak_z", ev.get("peak_score", ""))
                        pv = ev.get("peak_value", "")
                        ev_previews.append(f"Event (start: {ev.get('start_time', ev.get('start_index'))}, peak Z: {pz}, peak val: {pv})")
                    lines.append(f"Event details: {'; '.join(ev_previews)}")
            elif data.get("anomalous_timestamps"):
                lines.append(f"Anomalous timestamps (first 5): {data.get('anomalous_timestamps', [])[:5]}")
            return "\n".join(lines)

        if tool_name == "trend_analysis":
            return (
                f"[ANALYTICAL RESULT: trend_analysis]\n"
                f"Metric: {metric}{sensor_str}  |  "
                f"Time axis: {data.get('time_column')}\n"
                f"Trend direction: {data.get('trend_direction')}  |  "
                f"Slope: {data.get('slope')}  |  R-squared: {data.get('r_squared')}\n"
                f"Start value: {data.get('start_value')}  |  "
                f"End value: {data.get('end_value')}  |  "
                f"Change magnitude: {data.get('change_magnitude')}"
            )

        if tool_name == "correlation_analysis":
            return (
                f"[ANALYTICAL RESULT: correlation_analysis]\n"
                f"Columns analysed: {', '.join(data.get('columns_used', []))}\n"
                f"Strongest correlation: {data.get('strongest_pair')}\n"
                f"Weakest correlation:   {data.get('weakest_pair')}"
            )

        if tool_name == "baseline":
            rankings = data.get("deviation_ranking", [])
            rank_str = ", ".join([f"{r.get('sensor')}: {r.get('deviation_score')} ({r.get('status')})" for r in rankings[:3]])
            return (
                f"[ANALYTICAL RESULT: baseline]\n"
                f"Status: {data.get('status')}  |  "
                f"Sensors evaluated: {', '.join(data.get('sensors_evaluated', []))}\n"
                f"Deviation ranking: {rank_str}\n"
                f"Note: {data.get('note', '')}"
            )

        if tool_name == "model_result":
            damage_str = f"  |  Damage Class: {data.get('damage_class')}" if data.get('damage_class') else ""
            sev_str = f"  |  Severity: {data.get('severity')}" if data.get('severity') else ""
            contrib = data.get('contributing_parameters', [])
            contrib_str = f"\nContributing parameters: {', '.join(contrib)}" if contrib else ""
            return (
                f"[ANALYTICAL RESULT: model_result]\n"
                f"Sensor: {data.get('sensor_id')}  |  "
                f"Target: {data.get('target_variable')}\n"
                f"Predicted condition: {data.get('predicted_class')}{damage_str}{sev_str}  |  "
                f"Confidence: {data.get('confidence', 0)*100:.0f}%"
                f"{contrib_str}\n"
                f"Note: {data.get('model_note', '')}"
            )

        if tool_name == "chart_data":
            shown = (
                "\nThe interactive chart is already displayed to the user below your reply. "
                "Describe what it shows in plain words; do not write plotting code, image "
                "links or instructions for drawing it."
                if data.get("plot_json") not in (None, "", "{}")
                else ""
            )
            return (
                f"[ANALYTICAL RESULT: chart_data]\n"
                f"Chart generated for '{data.get('y_col')}' over '{data.get('x_col')}'.\n"
                f"{data.get('explanation', '')}{shown}"
            )

        return f"[ANALYTICAL RESULT: {tool_name}]\n{data}"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

dispatcher = ToolDispatcher()
