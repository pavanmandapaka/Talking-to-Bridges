from app.services.tts_service import clean_text_for_speech


def test_clean_asterisks_bold_and_italic():
    text = "The beam is classified as **damaged** with **100% confidence**."
    cleaned = clean_text_for_speech(text)
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert cleaned == "The beam is classified as damaged with 100% confidence."


def test_clean_mode_frequencies_and_parentheses():
    text = "Its **mode 2 frequency** is 5.2 % higher than the healthy baseline (39.85 Hz vs. 37.87 Hz)."
    cleaned = clean_text_for_speech(text)
    assert "**" not in cleaned
    assert cleaned == "Its mode 2 frequency is 5.2 % higher than the healthy baseline (39.85 Hz vs. 37.87 Hz)."


def test_clean_sensor_names_for_tts():
    text = "Signals from Sensor_1, Sensor_2 and Sensor_3 show anomalies."
    cleaned = clean_text_for_speech(text)
    assert "Sensor_1" not in cleaned
    assert "Sensor 1, Sensor 2 and Sensor 3 show anomalies." in cleaned


def test_clean_markdown_headers_and_code():
    text = "### Diagnostic Summary\nLook at `Sensor_4` value: ***critical***."
    cleaned = clean_text_for_speech(text)
    assert "#" not in cleaned
    assert "`" not in cleaned
    assert "*" not in cleaned
    assert "Diagnostic Summary" in cleaned
    assert "Sensor 4 value: critical." in cleaned


def test_clean_markdown_tables():
    text = "**Why?**\n| Reason | Detail |\n|---|---|\n| **No data** | Please upload. |"
    cleaned = clean_text_for_speech(text)
    assert "|" not in cleaned
    assert "*" not in cleaned
    assert "---" not in cleaned
    assert "Why?" in cleaned
    assert "No data" in cleaned
    assert "Please upload." in cleaned


def test_clean_empty_and_whitespace():
    assert clean_text_for_speech("") == ""
    assert clean_text_for_speech(None) == ""
    assert clean_text_for_speech("   ") == ""
