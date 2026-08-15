# Spotify Lyrics Translation Bridge

An ultra-fast, local translation bridge server built with Flask that intercepts lyrics requests (e.g., from Spicetify / Spotify lyrics extensions) and translates them using local LLM backends via LM Studio.

Optimized for speed, efficiency, and accuracy using multi-level caching, parallel chunk processing, and active OS media playback context integration.

---

Features

- Parallel Processing Engine: Breaks lyrics into configurable chunks and translates them simultaneously using thread pools (ThreadPoolExecutor) for minimal latency.
- Spotify / System Media Context Awareness:
  - Automatically detects the currently playing artist and song title via Windows SMTC, macOS AppleScript, or Linux playerctl.
  - Passes track context to the LLM to resolve grammatical gender agreement, tone, and poetic nuance (especially useful for languages like Italian, Spanish, French, German, etc.).
- Multi-Layer Persistent Caching:
  - Global Song Cache: Full lyrics hash lookup for instantaneous response on repeated tracks.
  - Line-Level Cache: Caches repeated verses and choruses across different songs.
  - Atomic JSON disk writes with automatic rotating backups to prevent cache corruption.
- Smart Pre-Filtering & Anti-Hallucination:
  - Instantly bypasses AI translation for symbols, numbers, and structural non-text lines.
  - Bypasses Latin-script lyrics when target language is English.
  - Custom output parsing with strict line-numbering regex and stop tokens to prevent LLM loops and hallucinations.
- Flexible External Configuration: Easily customizable via config.json or .env environment variables without touching the code.
- Monitoring & Health Checks: Built-in /status and /cache/stats endpoints, desktop notifications for slow translations, and colored console log diagnostics.

---

Getting Started

Prerequisites
- Python 3.8 or higher
- LM Studio (or any OpenAI-compatible local endpoint) running a translation model (e.g., translategemma-12b-it).

Installation

1. Clone the repository:

   
2. Install dependencies with requirements.txt:
   pip install -r requirements.txt
   
   or install them manually:
       - Install core dependencies:
         pip install flask requests python-dotenv

        - Install platform-specific dependencies (Recommended for Media Context):
          - Windows (for reliable SMTC media detection & toast notifications):
            pip install winsdk win10toast
          - Linux (for media detection):
            sudo apt install playerctl libnotify-bin  # Debian/Ubuntu

---

Configuration

Download the config.json file from the repository
or
Create a config.json file in the root directory to customize default settings:

{
  "lm_studio_url": "http://127.0.0.1:1234/v1/chat/completions",
  "model_name": "translategemma-12b-it",
  "chunk_size": 10,
  "max_workers": 4,
  "lm_timeout": 120,
  "default_target_lang": "en",
  "server": {
    "host": "127.0.0.1",
    "port": 5000
  },
  "logging": {
    "console_verbose": true,
    "color_console": true
  },
  "cache_backup": {
    "enabled": true,
    "every_n_saves": 20,
    "keep_backups": 3
  },
  "notifications": {
    "enabled": false,
    "slow_translation_threshold_seconds": 25
  }
}

Environment Variables

You can also override parameters using environment variables or a .env file:

- BRIDGE_LM_STUDIO_URL: Endpoint for the LLM server (Default: http://127.0.0.1:1234/v1/chat/completions)
- BRIDGE_MODEL_NAME: Loaded LLM model name (Default: translategemma-12b-it)
- BRIDGE_CHUNK_SIZE: Lines per translation block (Default: 10)
- BRIDGE_MAX_WORKERS: Max concurrent worker threads (Default: 4)
- BRIDGE_HOST: Server host IP (Default: 127.0.0.1)
- BRIDGE_PORT: Server port (Default: 5000)

---

Usage

1. Start your LM Studio local server and ensure your model is loaded.
2. Run the bridge server:
   python bridge.py
3. Configure your Spicetify lyrics extension to point its custom translation endpoint to:
   http://127.0.0.1:5000/

---

API Endpoints

POST /
Main translation proxy endpoint.

Payload:
{
  "q": ["Line 1 of song lyrics", "Line 2 of song lyrics"],
  "target": "it"
}

Response:
{
  "translatedText": ["Riga 1 del testo della canzone", "Riga 2 del testo della canzone"],
  "translated_text": "Riga 1 del testo della canzone\nRiga 2 del testo della canzone"
}

GET /status
Returns server uptime, LM Studio health status, cache entry counts, and stats for the last served translation.

GET /cache/stats
Returns detailed disk and entry statistics for global_cache.json and line_cache.json.

POST /cache/clear?confirm=yes
Clears the in-memory and disk caches (creates an automatic backup prior to deletion).
Optional query param: &which=global|line|all (default: all).

---

License:
This project is licensed under the GNU Lesser General Public License v3.0 (LGPLv3).
You are free to link or integrate this tool into proprietary software. However, any modifications or improvements made to the source code of this bridge must remain Open Source under the same LGPLv3 license. See the LICENSE and LICENSE.LESSER files for details.
