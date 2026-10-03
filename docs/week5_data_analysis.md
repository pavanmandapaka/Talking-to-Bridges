# Data Formats & Engineering Variables Analysis
**Prepared for Week 5 (Phase 2)**

I have successfully run a local analysis script against the `damaged_bridges.zip` and `undamaged data of 2 cantilevers.zip` files (278MB total) directly on your machine. 

Here are the results you need to share with Kolla (for the ingestion pipeline) and Nagarjun (for the dashboards) by Tuesday.

---

### 1. The Raw Data Formats
Both datasets are provided as **headerless Excel files (`.xlsx`)**. Kolla’s ingestion pipeline must be programmed to assign column names manually because the raw files do not have header rows.

**Undamaged Data Format (9 Columns):**
- **Cols 0, 1, 2:** Absolute Timestamp split into Hour, Minute, and Seconds respectively (e.g., `10`, `44`, `16.881`).
- **Col 3:** Relative Test Time in seconds (starting at `0.000` and incrementing by `0.003s`, which indicates a **333 Hz** sampling rate).
- **Cols 4, 5, 6, 7, 8:** Five identical sensor channels (likely Accelerometers or Strain Gauges placed along the cantilever).

**Damaged Data Format (7 Columns):**
- **Col 0:** Full Absolute Timestamp string (e.g., `11:55:24.624000`).
- **Col 1:** Relative Test Time in seconds (incrementing by `0.0033s`, indicating a **~300 Hz** sampling rate).
- **Cols 2, 3, 4, 5, 6:** The same five sensor channels.

> **Action Item for Kolla:** The ingestion pipeline must normalize these two different timestamp formats into a single standard `DateTime` column, and map the sensor columns as `Sensor_1` through `Sensor_5`.

---

### 2. Useful Engineering Variables to Extract
To make the data useful for the AI Assistant and the Machine Learning models (Krishna's task), we cannot just feed in raw 300Hz vibration data. The pipeline should extract the following **Engineering Variables** from those 5 sensor columns:

1. **Peak Amplitude:** The maximum absolute displacement or acceleration recorded during the `1cm`, `2cm`, or `Multi-hit` tests.
2. **Signal Variance / RMS:** The Root Mean Square of the vibration signal, which is a classic indicator of structural energy and potential damage.
3. **Fundamental Frequency (Hz):** Kolla's pipeline should run a Fast Fourier Transform (FFT) over the relative time and sensor data to find the natural frequency of the cantilever. A drop in natural frequency is a primary indicator of damage.
4. **Damping Ratio:** How quickly the vibration decays after the initial hit or displacement release.

> **Action Item for Nagarjun:** The visualization dashboard should allow the user to plot `Sensor_1` through `Sensor_5` against the `Relative_Time` variable to visualize the raw vibration wave, and also provide summary cards for Peak Amplitude and Frequency.
