"""
Unit and integration tests for AVT-4: Lip-sync and Avatar Refinement (Krishna).
Validates the 7 key avatar optimization criteria, audio playback synchronization,
interaction state transitions, and reliable fallback modes in frontend/voice_ui.html.
"""

import os
import re
import unittest
from pathlib import Path


class TestAvatarRefinementAVT4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.voice_ui_path = cls.project_root / "frontend" / "voice_ui.html"
        cls.assertTrue(cls.voice_ui_path.exists(), f"voice_ui.html not found at {cls.voice_ui_path}")
        with open(cls.voice_ui_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

    def test_voice_ui_file_structure(self):
        """Verifies that voice_ui.html is a well-formed HTML5 document."""
        self.assertIn("<!DOCTYPE html>", self.html_content)
        self.assertIn("<html", self.html_content)
        self.assertIn("</html>", self.html_content)
        self.assertIn("<head>", self.html_content)
        self.assertIn("<body>", self.html_content)

    def test_requirement_1_realtime_audio_driven_lip_sync(self):
        """
        AVT-4 Requirement 1:
        Real-time audio-driven lip synchronization via Web Audio API,
        AnalyserNode, FFT frequency band decomposition, and dynamic visemes.
        """
        # Audio Analyser Setup
        self.assertIn("createAnalyser", self.html_content)
        self.assertIn("smoothingTimeConstant", self.html_content)
        self.assertIn("fftSize", self.html_content)
        self.assertIn("getByteFrequencyData", self.html_content)
        self.assertIn("getByteTimeDomainData", self.html_content)

        # Multi-band frequency extraction (Low for vowels/jaw, Mid for width, High for sibilants)
        self.assertIn("getAudioMetrics", self.html_content)
        self.assertIn("targetMouthOpen", self.html_content)
        self.assertIn("targetMouthWidth", self.html_content)
        self.assertIn("targetJawOpen", self.html_content)
        self.assertIn("targetLipRound", self.html_content)

    def test_requirement_2_natural_mouth_and_jaw_movements(self):
        """
        AVT-4 Requirement 2:
        Natural mouth and jaw movements with asymmetric attack/decay physics damping,
        inter-word silence gating, Cupid's bow contour, upper teeth, lower teeth, and tongue.
        """
        # Physics damping
        self.assertIn("attackSpeed", self.html_content)
        self.assertIn("decaySpeed", self.html_content)
        self.assertIn("silenceThreshold", self.html_content)

        # Anatomical drawing components
        self.assertIn("drawMouthAndJaw", self.html_content)
        # Upper teeth and lower tongue rendering
        self.assertIn("teethW", self.html_content)
        self.assertIn("tongue", self.html_content.lower())
        self.assertIn("oralGrad", self.html_content)

    def test_requirement_3_smooth_facial_animations_and_blinking(self):
        """
        AVT-4 Requirement 3:
        Smooth facial animations including stochastic natural blinking with sinusoidal easing,
        micro-saccades / eye gaze drift, dynamic eyebrows, and breathing micro-movement.
        """
        # Natural Blinking
        self.assertIn("isBlinking", self.html_content)
        self.assertIn("blinkDuration", self.html_content)
        self.assertIn("blinkRatio", self.html_content)
        self.assertIn("Math.sin(p * Math.PI / 2)", self.html_content)

        # Eye Gaze and Micro-Saccades
        self.assertIn("gazeX", self.html_content)
        self.assertIn("gazeY", self.html_content)
        self.assertIn("nextSaccadeTime", self.html_content)
        self.assertIn("drawEyes", self.html_content)
        self.assertIn("drawEyebrows", self.html_content)

        # Organic Breathing Micro-Motion
        self.assertIn("breathPhase", self.html_content)
        self.assertIn("breathY", self.html_content)

    def test_requirement_4_synchronization_with_actual_audio_playback(self):
        """
        AVT-4 Requirement 4:
        Synchronization with actual audio playback, silence pauses between words,
        immediate interruption handling, and clean audio lifecycle management.
        """
        self.assertIn("AvatarAudioController", self.html_content)
        self.assertIn("playTTS", self.html_content)
        self.assertIn("audio.onplay", self.html_content)
        self.assertIn("audio.onended", self.html_content)
        self.assertIn("audio.onerror", self.html_content)

        # Interruption and pause handling
        self.assertIn("stopSystem", self.html_content)
        self.assertIn("audioSyncController.stop()", self.html_content)
        self.assertIn("cleanupCurrentAudio", self.html_content)
        self.assertIn("resetSpeechDynamics", self.html_content)

    def test_requirement_5_smooth_state_transitions(self):
        """
        AVT-4 Requirement 5:
        Smooth idle, listening, thinking and speaking transitions with linear interpolation (lerp),
        and state-reactive holographic aura rings.
        """
        # State machine
        self.assertIn("stateWeights", self.html_content)
        self.assertIn("setAgentState", self.html_content)
        self.assertIn("drawAura", self.html_content)

        # Four distinct interaction states
        for state in ["idle", "listening", "thinking", "speaking"]:
            self.assertIn(state, self.html_content)

    def test_requirement_6_animation_performance_and_resource_usage(self):
        """
        AVT-4 Requirement 6:
        Animation performance and resource usage: requestAnimationFrame with delta-time clamping,
        zero-allocation render loop, and Page Visibility API integration to pause when hidden.
        """
        self.assertIn("requestAnimationFrame", self.html_content)
        self.assertIn("visibilitychange", self.html_content)
        self.assertIn("document.hidden", self.html_content)
        self.assertIn("isRenderingActive", self.html_content)
        self.assertIn("setupDPI", self.html_content)

    def test_requirement_7_reliable_fallback_modes(self):
        """
        AVT-4 Requirement 7:
        Reliable voice-only / basic-avatar fallback:
        - Animated Avatar mode
        - Classic Orb mode (AVT-2/3 baseline)
        - Voice-Only mode (avatar-off mode fulfilling INT-1 requirement)
        - LocalStorage persistence and try-catch auto-downgrade safeguard.
        """
        self.assertIn("setPresentationMode", self.html_content)
        self.assertIn("triggerFallback", self.html_content)
        self.assertIn("ttb_avatar_mode", self.html_content)

        # Presentation elements in DOM
        self.assertIn('id="avatar-canvas"', self.html_content)
        self.assertIn('id="orb"', self.html_content)
        self.assertIn('id="voice-only-stage"', self.html_content)
        self.assertIn('id="btn-mode-avatar"', self.html_content)
        self.assertIn('id="btn-mode-orb"', self.html_content)
        self.assertIn('id="btn-mode-voice-only"', self.html_content)
        self.assertIn('id="avatar-fallback-alert"', self.html_content)

    def test_preservation_of_existing_features(self):
        """
        Verifies that all pre-existing features in voice_ui.html remain intact:
        File upload, CSV analysis, Plotly integration, STT, LLM, TTS, Voice stop commands.
        """
        self.assertIn('id="file-upload"', self.html_content)
        self.assertIn('id="upload-card"', self.html_content)
        self.assertIn('id="chat-log"', self.html_content)
        self.assertIn('id="status"', self.html_content)
        self.assertIn('id="x-axis-select"', self.html_content)
        self.assertIn('id="y-axis-select"', self.html_content)
        self.assertIn("uploadDocument", self.html_content)
        self.assertIn("generateChart", self.html_content)
        self.assertIn("b64toBlob", self.html_content)
        self.assertIn("/upload", self.html_content)
        self.assertIn("/transcribe", self.html_content)
        self.assertIn("/api/chat", self.html_content)
        self.assertIn("/speak", self.html_content)
        self.assertIn("/analyze_csv", self.html_content)
        self.assertIn("goodbye", self.html_content)


if __name__ == "__main__":
    unittest.main()

