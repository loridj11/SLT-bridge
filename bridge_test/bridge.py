import hashlib
import time
import re
import os
import json
import tempfile
import sys
import shutil
import threading
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


# =====================================================================
# CONFIGURAZIONE ESTERNA (config.json + variabili d'ambiente / .env)
# =====================================================================
# Tutti i parametri modificabili senza toccare il codice vivono qui.
# Priorità (dal più basso al più alto): DEFAULT_CONFIG -> config.json -> env/.env

DEFAULT_CONFIG = {
    "lm_studio_url": "http://127.0.0.1:1234/v1/chat/completions",
    "model_name": "translategemma-12b-it",
    "chunk_size": 10,
    "max_workers": 4,
    "lm_timeout": 120,
    "default_target_lang": "en",
    "smtc_fallback_to_sole_playing_session": True,
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
    },
    "logging": {
        "log_dir": "logs",
        "log_file": "bridge.log",
        "level": "INFO",
        "max_bytes": 5 * 1024 * 1024,
        "backup_count": 3,
    },
    "cache_backup": {
        "enabled": True,
        "every_n_saves": 20,
        "keep_backups": 3,
    },
    "notifications": {
        "enabled": False,
        "slow_translation_threshold_seconds": 25,
    },
    "prefetch": {
        "enabled": False,
        "poll_interval_seconds": 5,
    },
}


def _deep_merge(base, override):
    """Merge ricorsivo di due dict: 'override' vince sui duplicati,
    ma i dict annidati vengono uniti campo per campo invece di essere
    sovrascritti in blocco (così un config.json parziale, es. solo
    {"chunk_size": 5}, non cancella il resto dei default)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Carica la configurazione da config.json (se presente, accanto allo
    script) sopra i default, poi applica eventuali override da variabili
    d'ambiente (utile per Docker/servizi) o da un file .env locale.
    NOTA: qui si usa 'print' invece del logger perché il logging non è
    ancora stato inizializzato (dipende proprio da questa configurazione)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy semplice

    config_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg = _deep_merge(cfg, user_cfg)
            print(f"[CONFIG] Caricato {config_path}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN CONFIG] Impossibile leggere config.json, uso i valori di default: {e}")
    else:
        print("[CONFIG] Nessun config.json trovato, uso i valori di default.")

    # Supporto opzionale a un file .env (richiede 'python-dotenv'; se non
    # installato viene semplicemente ignorato senza errori).
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
    except ImportError:
        pass

    env_map = {
        "BRIDGE_LM_STUDIO_URL": ("lm_studio_url", str),
        "BRIDGE_MODEL_NAME": ("model_name", str),
        "BRIDGE_CHUNK_SIZE": ("chunk_size", int),
        "BRIDGE_MAX_WORKERS": ("max_workers", int),
        "BRIDGE_LM_TIMEOUT": ("lm_timeout", int),
        "BRIDGE_DEFAULT_TARGET_LANG": ("default_target_lang", str),
    }
    for env_var, (cfg_key, caster) in env_map.items():
        raw_val = os.environ.get(env_var)
        if raw_val is not None:
            try:
                cfg[cfg_key] = caster(raw_val)
            except ValueError:
                print(f"[WARN CONFIG] Valore non valido per {env_var}={raw_val!r}, ignorato.")

    # host/port vivono in un sotto-dict, quindi vanno gestiti separatamente
    # dal resto della env_map (che scrive solo chiavi di primo livello).
    raw_host = os.environ.get("BRIDGE_HOST")
    if raw_host is not None:
        cfg.setdefault("server", {})["host"] = raw_host
    raw_port = os.environ.get("BRIDGE_PORT")
    if raw_port is not None:
        try:
            cfg.setdefault("server", {})["port"] = int(raw_port)
        except ValueError:
            print(f"[WARN CONFIG] Valore non valido per BRIDGE_PORT={raw_port!r}, ignorato.")

    return cfg


CONFIG = load_config()


# =====================================================================
# LOGGING SU FILE (con rotazione) + CONSOLE
# =====================================================================

def setup_logging(cfg):
    log_cfg = cfg.get("logging", {})
    log_dir = os.path.join(BASE_DIR, log_cfg.get("log_dir", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_cfg.get("log_file", "bridge.log"))
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)

    lg = logging.getLogger("bridge")
    lg.setLevel(level)
    lg.propagate = False
    lg.handlers.clear()  # evita handler duplicati se il modulo viene ricaricato

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=log_cfg.get("max_bytes", 5 * 1024 * 1024),
        backupCount=log_cfg.get("backup_count", 3),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    lg.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    lg.addHandler(console_handler)

    return lg


logger = setup_logging(CONFIG)

LM_STUDIO_URL = CONFIG["lm_studio_url"]
MODEL_NAME = CONFIG["model_name"]

# --- OTTIMIZZAZIONI PER MASSIMA VELOCITÀ ---
CHUNK_SIZE = CONFIG["chunk_size"]        # righe per blocco
MAX_WORKERS = CONFIG["max_workers"]      # slot paralleli sfruttati su LM Studio
LM_TIMEOUT = CONFIG["lm_timeout"]        # evita timeout durante i picchi di lavoro

DEFAULT_TARGET_LANG = CONFIG.get("default_target_lang", "en")
SMTC_FALLBACK_TO_SOLE_PLAYING_SESSION = CONFIG.get("smtc_fallback_to_sole_playing_session", True)
SERVER_CONFIG = CONFIG.get("server", {})
CACHE_BACKUP_CONFIG = CONFIG.get("cache_backup", {})
NOTIFICATIONS_CONFIG = CONFIG.get("notifications", {})
PREFETCH_CONFIG = CONFIG.get("prefetch", {})

logger.info(
    f"[CONFIG] lm_studio_url={LM_STUDIO_URL} | model={MODEL_NAME} | "
    f"chunk_size={CHUNK_SIZE} | max_workers={MAX_WORKERS} | timeout={LM_TIMEOUT}s | "
    f"target_default={DEFAULT_TARGET_LANG} | "
    f"server={SERVER_CONFIG.get('host', '127.0.0.1')}:{SERVER_CONFIG.get('port', 5000)}"
)

# --- PERSISTENZA CACHE SU DISCO ---
CACHE_DIR = os.path.join(BASE_DIR, "cache")
GLOBAL_CACHE_FILE = os.path.join(CACHE_DIR, "global_cache.json")
LINE_CACHE_FILE = os.path.join(CACHE_DIR, "line_cache.json")

CACHE_DATA_LOCK = Lock()  # Protegge OGNI accesso (lettura e scrittura) a GLOBAL_CACHE e LINE_CACHE,
                           # non solo il salvataggio su disco: i thread worker scrivono su
                           # LINE_CACHE in parallelo, e richieste diverse leggono/scrivono
                           # GLOBAL_CACHE contemporaneamente. Senza questo lock si rischiano
                           # RuntimeError da "dictionary changed size during iteration" e
                           # cache miss/dati incompleti nella finestra tra clear() e update().


def _load_json_cache(path):
    """Carica un file JSON di cache dal disco, se presente e valido."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[WARN CACHE] Impossibile leggere {path}: {e}")
    return {}


def _save_json_cache(path, data):
    """Scrive un dict su file JSON in modo atomico (evita corruzioni in caso di crash).
    NOTA: 'data' deve già essere una copia stabile (es. dict(...) fatto sotto lock),
    non un riferimento diretto a un dizionario condiviso mutabile da altri thread."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(f"[WARN CACHE] Impossibile scrivere {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def global_cache_get(key):
    """Lettura thread-safe di una voce da GLOBAL_CACHE."""
    with CACHE_DATA_LOCK:
        return GLOBAL_CACHE.get(key)


def global_cache_set(key, value):
    """Scrittura thread-safe di una voce in GLOBAL_CACHE."""
    with CACHE_DATA_LOCK:
        GLOBAL_CACHE[key] = value


def line_cache_get(key):
    """Lettura thread-safe di una voce da LINE_CACHE."""
    with CACHE_DATA_LOCK:
        return LINE_CACHE.get(key)


def line_cache_set(key, value):
    """Scrittura thread-safe di una voce in LINE_CACHE (chiamata dai thread worker)."""
    with CACHE_DATA_LOCK:
        LINE_CACHE[key] = value


# =====================================================================
# BACKUP PERIODICO DELLE CACHE (rete di sicurezza extra oltre al merge)
# =====================================================================

def _rotate_backup_file(path, keep):
    """Ruota le copie .bak1..bakN di un file (.bak(N-1) -> .bakN, ecc.),
    poi copia il file corrente in .bak1. 'keep' è il numero massimo di
    copie storiche mantenute."""
    if not os.path.exists(path) or keep <= 0:
        return
    for i in range(keep, 1, -1):
        src = f"{path}.bak{i - 1}"
        dst = f"{path}.bak{i}"
        if os.path.exists(src):
            try:
                os.replace(src, dst)
            except OSError as e:
                logger.warning(f"[BACKUP] Impossibile ruotare {src} -> {dst}: {e}")
    try:
        shutil.copy2(path, f"{path}.bak1")
    except OSError as e:
        logger.warning(f"[BACKUP] Impossibile creare backup di {path}: {e}")


def backup_cache_files_now(keep=None):
    """Forza subito un backup di entrambi i file di cache, a prescindere
    dal contatore periodico (usato es. prima di /cache/clear)."""
    keep = keep if keep is not None else CACHE_BACKUP_CONFIG.get("keep_backups", 3)
    _rotate_backup_file(GLOBAL_CACHE_FILE, keep)
    _rotate_backup_file(LINE_CACHE_FILE, keep)


def _maybe_periodic_backup():
    """Da chiamare ad ogni save_caches(): ogni N salvataggi (config
    'cache_backup.every_n_saves') crea automaticamente un backup a rotazione."""
    if not CACHE_BACKUP_CONFIG.get("enabled", True):
        return
    with STATE_LOCK:
        STATE["cache_saves"] += 1
        n = STATE["cache_saves"]
    every = CACHE_BACKUP_CONFIG.get("every_n_saves", 20)
    if every > 0 and n % every == 0:
        backup_cache_files_now()
        logger.info(f"[BACKUP] Backup periodico delle cache eseguito (salvataggio #{n}).")


def save_caches():
    """Salva entrambe le cache su disco, UNENDO quello che c'è già su disco con
    quello in memoria invece di sovrascrivere (protegge da riavvii/race tra processi),
    ed eseguendo l'intera operazione (merge in memoria + scrittura su file) sotto
    un unico lock, così nessun altro thread può leggere/scrivere i dizionari mentre
    vengono svuotati e ripopolati.

    Questo evita che un processo con una cache in memoria "più vecchia" (es. un
    vecchio processo ancora in chiusura durante un riavvio) cancelli per errore
    le voci più recenti già scritte su disco da un altro salvataggio: qualunque
    voce presente sul file JSON viene sempre preservata, mai persa."""
    with CACHE_DATA_LOCK:
        disk_global = _load_json_cache(GLOBAL_CACHE_FILE)
        disk_global.update(GLOBAL_CACHE)   # la memoria vince sui duplicati (è la più recente)
        GLOBAL_CACHE.clear()
        GLOBAL_CACHE.update(disk_global)   # riallinea anche la memoria (recupera eventuali voci esterne)
        _save_json_cache(GLOBAL_CACHE_FILE, disk_global)

        disk_line = _load_json_cache(LINE_CACHE_FILE)
        disk_line.update(LINE_CACHE)
        LINE_CACHE.clear()
        LINE_CACHE.update(disk_line)
        _save_json_cache(LINE_CACHE_FILE, disk_line)

    _maybe_periodic_backup()


def clear_caches_on_disk(which="all"):
    """Svuota le cache in memoria E su disco (a differenza di save_caches(),
    qui NON si fa merge col disco: è una cancellazione esplicita, sempre
    preceduta da un backup di sicurezza). which: 'all' | 'global' | 'line'."""
    backup_cache_files_now()
    with CACHE_DATA_LOCK:
        if which in ("all", "global"):
            GLOBAL_CACHE.clear()
            _save_json_cache(GLOBAL_CACHE_FILE, {})
        if which in ("all", "line"):
            LINE_CACHE.clear()
            _save_json_cache(LINE_CACHE_FILE, {})


GLOBAL_CACHE = _load_json_cache(GLOBAL_CACHE_FILE)   # Cache per intere canzoni
LINE_CACHE = _load_json_cache(LINE_CACHE_FILE)       # Cache per singoli versi/ritornelli ripetuti
PENDING_LOCKS = {}
GLOBAL_LOCK = Lock()

# --- STATO RUNTIME (per /status e il riepilogo statistico) ---
STATE_LOCK = Lock()
STATE = {
    "start_time": time.time(),
    "total_requests": 0,
    "cache_saves": 0,
    "last_translation": None,       # dict con i dettagli dell'ultima traduzione servita
    "lm_studio_reachable": None,    # ultimo esito noto del controllo di salute
    "lm_studio_last_check": None,
}

logger.info(f"[INFO] Cache caricata da disco: {len(GLOBAL_CACHE)} canzoni, {len(LINE_CACHE)} righe singole.")


# =====================================================================
# NOTIFICHE DESKTOP (toast di Windows / notify-send su Linux)
# =====================================================================
_toast_notifier = None
_notify_warned = False
_notify_send_checked = False
_notify_send_available = False

if sys.platform.startswith("win"):
    try:
        from win10toast import ToastNotifier
        _toast_notifier = ToastNotifier()
    except ImportError:
        _toast_notifier = None


def _check_notify_send_available():
    """Verifica UNA VOLA SOLA (risultato cacheato) se 'notify-send' è disponibile
    sul sistema Linux corrente (pacchetto 'libnotify-bin' o equivalente)."""
    global _notify_send_checked, _notify_send_available
    if not _notify_send_checked:
        _notify_send_available = shutil.which("notify-send") is not None
        _notify_send_checked = True
    return _notify_send_available


def notify_desktop(title, message):
    """Mostra una notifica desktop, se abilitata in config:
    - Windows: toast tramite 'win10toast' (richiede 'pip install win10toast').
    - Linux: tramite il comando di sistema 'notify-send' (pacchetto 'libnotify-bin'
      o equivalente sulla propria distro; nessuna dipendenza pip necessaria).
    macOS non è supportato da questa funzione (nessuna notifica mostrata, nessun errore).
    In tutti i casi di dipendenza mancante, avvisa UNA VOLA SOLA nei log invece di
    ripetere l'avviso ad ogni chiamata."""
    global _notify_warned
    if not NOTIFICATIONS_CONFIG.get("enabled", False):
        return

    if sys.platform.startswith("win"):
        if _toast_notifier is None:
            if not _notify_warned:
                logger.warning(
                    "[NOTIFY] Pacchetto 'win10toast' non installato: notifiche desktop disattivate. "
                    "Esegui: pip install win10toast"
                )
                _notify_warned = True
            return
        try:
            _toast_notifier.show_toast(title, message, duration=6, threaded=True)
        except Exception as e:
            logger.warning(f"[NOTIFY] Impossibile mostrare la notifica desktop: {e}")

    elif sys.platform.startswith("linux"):
        if not _check_notify_send_available():
            if not _notify_warned:
                logger.warning(
                    "[NOTIFY] Comando 'notify-send' non trovato: notifiche desktop disattivate. "
                    "Installa il pacchetto 'libnotify-bin' (Debian/Ubuntu) o l'equivalente della "
                    "tua distribuzione per abilitarle."
                )
                _notify_warned = True
            return
        try:
            subprocess.run(
                ["notify-send", title, message],
                capture_output=True, timeout=3
            )
        except Exception as e:
            logger.warning(f"[NOTIFY] Impossibile mostrare la notifica desktop: {e}")

    # macOS e altre piattaforme: nessuna notifica (supporto non richiesto).


def make_cors_response(data, status_code=200):
    response = jsonify(data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'authorization, content-type, x-api-key'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    return response, status_code

def get_hash(text_list, target):
    raw = f"{target}::" + "\n".join(text_list)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def make_line_cache_key(line, target_lang):
    """Costruisce la chiave della LINE_CACHE includendo la lingua di destinazione,
    così una riga tradotta verso 'en' non viene mai restituita per una richiesta verso 'it'."""
    return f"{target_lang}::{line}"

# Mappa dei codici lingua più comuni verso un nome esplicito e non ambiguo.
# Necessaria perché passare al modello un codice grezzo come "it" può essere
# frainteso (es. "it" è anche un pronome inglese), causando output in lingue
# sbagliate o miste da un blocco all'altro.
LANGUAGE_NAMES = {
    "en": "English",
    "it": "Italian (Italiano)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "pt-br": "Brazilian Portuguese (Português do Brasil)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "ru": "Russian (Русский)",
    "tr": "Turkish (Türkçe)",
    "ko": "Korean (한국어)",
    "zh": "Chinese (中文)",
    "zh-cn": "Simplified Chinese (简体中文)",
    "zh-tw": "Traditional Chinese (繁體中文)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "sv": "Swedish (Svenska)",
    "da": "Danish (Dansk)",
    "no": "Norwegian (Norsk)",
    "fi": "Finnish (Suomi)",
    "el": "Greek (Ελληνικά)",
    "ar": "Arabic (العربية)",
    "hi": "Hindi (हिन्दी)",
}


def resolve_language_name(target_lang):
    """Converte un codice lingua (es. 'it') in un nome esplicito per il prompt
    (es. 'Italian (Italiano)'). Se il codice non è nella mappa, usa il valore
    originale così com'è (permette comunque di passare un nome già completo)."""
    if not target_lang:
        return "English"
    key = target_lang.strip().lower()
    return LANGUAGE_NAMES.get(key, target_lang)


# --- CONTESTO SPOTIFY (Artista/Titolo) PER TRADUZIONI PIÙ ACCURATE ---
# Fornire all'AI il nome dell'artista e del brano aiuta soprattutto con
# l'accordo di genere grammaticale (es. aggettivi/participi in italiano,
# francese, spagnolo...), che il giapponese spesso omette del tutto.
#
# Se True, quando NESSUNA sessione multimediale risulta chiaramente di Spotify
# (es. Spotify aperto nel browser come tab normale, non installato come PWA:
# in quel caso l'app id riportato è quello del browser, non "spotify"), e
# c'è ESATTAMENTE UNA sessione di sistema in riproduzione attiva, la si userà
# comunque come ultima risorsa (loggando chiaramente che è una supposizione).
# Impostare a False (in config.json) se si preferisce non rischiare mai un match sbagliato.


def _get_spotify_track_windows_smtc():
    """Interroga le API multimediali di sistema di Windows (SMTC - System Media
    Transport Controls), le stesse che alimentano il centro notifiche e la
    preview sulla taskbar. È il metodo AFFIDABILE su Windows 10/11: funziona
    anche con le versioni recenti di Spotify, che non scrivono più
    artista/titolo nel testo reale della finestra (solo nella preview di sistema).
    Richiede il pacchetto 'winsdk' (pip install winsdk); se non è installato,
    solleva ImportError e il chiamante ripiega sul metodo legacy.

    Gestisce esplicitamente il caso di PIÙ sessioni multimediali attive
    contemporaneamente (es. un browser che riproduce audio insieme a Spotify):
    tra le sessioni che sembrano essere Spotify, dà priorità a quella
    ATTUALMENTE IN RIPRODUZIONE rispetto a una eventualmente in pausa/stale,
    invece di prendere semplicemente la prima trovata."""
    import asyncio
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
    )

    async def _extract(session):
        info = await session.try_get_media_properties_async()
        artist = (info.artist or "").strip()
        title = (info.title or "").strip()
        if artist or title:
            return artist, title
        return None

    async def _query():
        manager = await MediaManager.request_async()
        sessions = list(manager.get_sessions())

        spotify_playing = []
        spotify_other = []
        other_playing = []

        for session in sessions:
            try:
                app_id = (session.source_app_user_model_id or "").lower()
            except Exception:
                app_id = ""
            try:
                playback_info = session.get_playback_info()
                is_playing = (
                    playback_info is not None
                    and playback_info.playback_status == PlaybackStatus.PLAYING
                )
            except Exception:
                is_playing = False

            # Copre sia l'eseguibile classico ("Spotify.exe") sia la versione
            # Microsoft Store/PWA (app id tipo "SpotifyAB.SpotifyMusic_...").
            is_spotify = "spotify" in app_id

            if is_spotify and is_playing:
                spotify_playing.append(session)
            elif is_spotify:
                spotify_other.append(session)
            elif is_playing:
                other_playing.append(session)

        # Priorità 1: sessione di Spotify ATTUALMENTE in riproduzione.
        for session in spotify_playing:
            result = await _extract(session)
            if result:
                return result

        # Priorità 2: sessione di Spotify presente ma non in riproduzione attiva
        # (es. messa in pausa un istante prima) — meglio di nessun contesto.
        for session in spotify_other:
            result = await _extract(session)
            if result:
                return result

        # Priorità 3 (best-effort, disattivabile): nessuna sessione riconosciuta
        # esplicitamente come Spotify (tipico di Spotify aperto come tab
        # "normale" del browser, non installato come app/PWA), ma esiste
        # un'UNICA altra sessione di sistema in riproduzione attiva: la usiamo
        # come ipotesi ragionevole, segnalandolo chiaramente nei log.
        if SMTC_FALLBACK_TO_SOLE_PLAYING_SESSION and len(other_playing) == 1:
            result = await _extract(other_playing[0])
            if result:
                logger.info(
                    "[INFO SPOTIFY CONTEXT] Nessuna sessione riconosciuta esplicitamente "
                    "come Spotify: uso l'unica sessione multimediale in riproduzione come ipotesi."
                )
                return result

        return None

    try:
        return asyncio.run(_query())
    except RuntimeError:
        # Nel raro caso in cui esista già un event loop attivo nel thread corrente
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_query())
        finally:
            loop.close()


def _get_spotify_track_windows_title_fallback():
    """[METODO LEGACY] Legge il titolo della finestra di Spotify su Windows
    (formato 'Artista - Titolo') tramite le API Win32 via ctypes. Funziona solo
    con vecchie versioni di Spotify che scrivevano artista/titolo nel titolo
    reale della finestra; usato solo come ripiego se 'winsdk' non è disponibile."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if hproc:
            exe_buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleBaseNameW(hproc, None, exe_buf, 260)
            kernel32.CloseHandle(hproc)
            if exe_buf.value.lower() == "spotify.exe":
                found.append(title)
        return True

    user32.EnumWindows(enum_proc, 0)

    for title in found:
        clean_title = title.strip()
        if clean_title and " - " in clean_title and clean_title.lower() not in (
            "spotify", "spotify free", "spotify premium"
        ):
            artist, _, track = clean_title.partition(" - ")
            return artist.strip(), track.strip()
    return None


_SMTC_HINT_SHOWN = False  # Evita di stampare il suggerimento di installazione ad ogni richiesta


def _get_spotify_track_windows():
    """Prova prima con le API di sistema SMTC (affidabili, funzionano con le
    versioni recenti di Spotify). Se il pacchetto 'winsdk' non è installato,
    ripiega sul parsing del titolo della finestra (funziona solo con vecchie
    versioni di Spotify)."""
    global _SMTC_HINT_SHOWN
    try:
        return _get_spotify_track_windows_smtc()
    except ImportError:
        if not _SMTC_HINT_SHOWN:
            logger.info(
                "[INFO SPOTIFY CONTEXT] Pacchetto 'winsdk' non installato: uso il "
                "metodo di rilevamento legacy (meno affidabile con Spotify recente). "
                "Per il rilevamento affidabile esegui: pip install winsdk"
            )
            _SMTC_HINT_SHOWN = True
    except Exception as e:
        logger.warning(f"[WARN SPOTIFY CONTEXT] Metodo SMTC fallito, uso il fallback: {e}")

    return _get_spotify_track_windows_title_fallback()


def _get_spotify_track_macos():
    """Legge artista e titolo dalla app Spotify su macOS tramite AppleScript."""
    script = (
        'if application "Spotify" is running then\n'
        '  tell application "Spotify"\n'
        '    if player state is playing then\n'
        '      artist of current track & "||" & name of current track\n'
        '    end if\n'
        '  end tell\n'
        'end if'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=3
    )
    output = (result.stdout or "").strip()
    if output and "||" in output:
        artist, _, track = output.partition("||")
        return artist.strip(), track.strip()
    return None


_playerctl_checked = False
_playerctl_available = False


def _check_playerctl_available():
    """Verifica UNA VOLTA SOLA (risultato cacheato) se 'playerctl' è installato,
    invece di scoprirlo ad ogni singola richiesta con un fallimento silenzioso."""
    global _playerctl_checked, _playerctl_available
    if not _playerctl_checked:
        _playerctl_available = shutil.which("playerctl") is not None
        _playerctl_checked = True
        if not _playerctl_available:
            logger.info(
                "[INFO SPOTIFY CONTEXT] Comando 'playerctl' non trovato: il rilevamento del "
                "brano corrente su Linux resterà disattivato. Installa 'playerctl' "
                "(es. 'sudo apt install playerctl' su Debian/Ubuntu, o l'equivalente della "
                "tua distribuzione) per abilitarlo."
            )
    return _playerctl_available


_wsl_checked = False
_wsl_detected = False


def _is_running_under_wsl():
    """Rileva UNA VOLTA SOLA (risultato cacheato) se lo script sta girando dentro WSL
    (Windows Subsystem for Linux), leggendo /proc/version."""
    global _wsl_checked, _wsl_detected
    if not _wsl_checked:
        try:
            with open("/proc/version", "r") as f:
                _wsl_detected = "microsoft" in f.read().lower()
        except OSError:
            _wsl_detected = False
        _wsl_checked = True
    return _wsl_detected


_wsl_warned = False


def _get_spotify_track_linux():
    """Legge artista e titolo su Linux tramite playerctl (se installato)."""
    global _wsl_warned
    if not _check_playerctl_available():
        return None

    if _is_running_under_wsl() and not _wsl_warned:
        logger.warning(
            "[WARN SPOTIFY CONTEXT] Sembra che lo script sia in esecuzione dentro WSL: "
            "'playerctl' non ha visibilità sulla sessione multimediale di Windows (sono "
            "ambienti separati). Se Spotify gira nativamente su Windows (fuori da WSL), "
            "il rilevamento qui non funzionerà mai: esegui il bridge direttamente su "
            "Windows per usare il metodo SMTC, invece che dentro WSL."
        )
        _wsl_warned = True

    artist = subprocess.run(
        ["playerctl", "-p", "spotify", "metadata", "artist"],
        capture_output=True, text=True, timeout=3
    ).stdout.strip()
    track = subprocess.run(
        ["playerctl", "-p", "spotify", "metadata", "title"],
        capture_output=True, text=True, timeout=3
    ).stdout.strip()
    if artist and track:
        return artist, track
    return None


_unsupported_platform_warned = False


def get_current_spotify_track():
    """Restituisce una tupla (artista, titolo) del brano attualmente in
    riproduzione su Spotify, oppure None se non disponibile (Spotify chiuso,
    in pausa, playerctl non installato, piattaforma non supportata, ecc.).
    Non solleva mai eccezioni: qualunque errore viene ignorato e si ritorna
    semplicemente None, così la traduzione prosegue comunque senza il
    contesto extra, senza mai bloccare o rallentare la richiesta."""
    global _unsupported_platform_warned
    try:
        if sys.platform.startswith("win"):
            return _get_spotify_track_windows()
        elif sys.platform == "darwin":
            return _get_spotify_track_macos()
        elif sys.platform.startswith("linux"):
            return _get_spotify_track_linux()
        else:
            if not _unsupported_platform_warned:
                logger.warning(
                    f"[WARN SPOTIFY CONTEXT] Piattaforma '{sys.platform}' non riconosciuta: "
                    "il rilevamento del brano Spotify è supportato ufficialmente solo su "
                    "Windows e Linux. Il contesto Spotify non sarà mai disponibile su questo sistema."
                )
                _unsupported_platform_warned = True
    except Exception as e:
        logger.warning(f"[WARN SPOTIFY CONTEXT] Impossibile leggere il brano corrente: {e}")
    return None

def clean_translated_line(text):
    """PULIZIA SANITIZZANTE: Rimuove etichette, numeri e punti singoli, MANTIENE (...)."""
    if not text:
        return text
    # 1. Rimuove numeri isolati dentro o fuori parentesi, es: (10.), [11.], (11)
    text = re.sub(r'[\(\[]\s*\d+[\.\)]?\s*[\)\]]', '', text)
    
    # 2. Rimuove prefissi tipo Verse 1:, Chorus:, Intro:, Outro:
    text = re.sub(r'^\s*(?:\(?Verse|\(?Chorus|\(?Intro|\(?Outro|\(?Bridge)\s*\d*[\:\)]?\s*', '', text, flags=re.IGNORECASE)
    
    # 3. Rimuove due punti e punti e virgola isolati a fine riga
    text = re.sub(r'[\:;]+(?=\s*[\)\>]?$|$)', '', text)

    # 4. Rimuove i punti fermi finali SINGOLI o DOPPI, lasciando intatti i 3 o più punti di sospensione (...)
    text = re.sub(r'(?<!\.)\.\.?(?!\.)(?=\s*[\)\>]?$|$)', '', text)
    
    # 5. Rimuove eventuali punti fermi singoli attaccati a parole isolate (escluso dentro ...)
    text = re.sub(r'(?<=\w)\.(?!\.)(?=\s|$)', '', text)
    
    # 6. Rimuove doppi spazi o spazi residui all'inizio/fine
    text = re.sub(r' +', ' ', text).strip()
    return text

def is_symbols_only(text):
    """Vero se la riga, tolti spazi/numeri/punteggiatura/simboli, non contiene più nulla
    (quindi non c'è testo da tradurre in nessuna lingua)."""
    cleaned = re.sub(r'[\s\d\t\r\n.,!?\'"\(\)\-\–\—\_\:\;\~\★\☆\♪]', '', text)
    return not cleaned


# Intervalli Unicode dei principali "script" non latini che compaiono nei
# testi delle canzoni. Generalizza il vecchio has_japanese(): qualunque
# lingua sorgente (giapponese, coreano, cinese, russo, arabo, ecc.) che usi
# uno di questi alfabeti viene rilevata, non solo il giapponese.
_NON_LATIN_SCRIPT_RANGES = [
    (0x3040, 0x30FF),   # Hiragana + Katakana (giapponese)
    (0x3400, 0x4DBF),   # CJK Ideografi estesi
    (0x4E00, 0x9FFF),   # CJK Ideografi unificati (kanji / hanzi)
    (0xAC00, 0xD7A3),   # Hangul (coreano) - sillabe complete
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3130, 0x318F),   # Hangul compatibility Jamo
    (0x0400, 0x04FF),   # Cirillico (russo, ucraino, ecc.)
    (0x0600, 0x06FF),   # Arabo
    (0x0750, 0x077F),   # Arabo supplemento
    (0x0590, 0x05FF),   # Ebraico
    (0x0370, 0x03FF),   # Greco
    (0x0E00, 0x0E7F),   # Thai
    (0x0900, 0x097F),   # Devanagari (hindi, ecc.)
]
_NON_LATIN_SCRIPT_PATTERN = re.compile(
    "[" + "".join(f"\\u{start:04x}-\\u{end:04x}" for start, end in _NON_LATIN_SCRIPT_RANGES) + "]"
)


def has_non_latin_script(text):
    """Vero se il testo contiene caratteri di uno script non latino
    (giapponese, coreano, cinese, russo, arabo, ebraico, greco, thai,
    devanagari...). Sostituisce il vecchio has_japanese(), che presumeva
    sempre e solo giapponese."""
    return bool(_NON_LATIN_SCRIPT_PATTERN.search(text))


# Alias di compatibilità con il nome precedente, nel caso qualcosa lo referenzi ancora.
has_japanese = has_non_latin_script

def is_target_english(target_lang):
    if not target_lang:
        return True
    return target_lang.strip().lower() in ("en", "en-us", "en-gb", "english")

def can_bypass_ai(text, target_lang):
    """LIVELLO 1: Decide se una riga può saltare l'AI.
    - Righe fatte solo di simboli/numeri/spazi: bypass sempre (nulla da tradurre).
    - Righe senza uno script non latino: bypass SOLO se la lingua target è
      l'inglese (altrimenti il testo, pur non essendo in uno script "esotico",
      potrebbe comunque non essere nella lingua target richiesta e va comunque
      tradotto)."""
    if is_symbols_only(text):
        return True
    if has_non_latin_script(text):
        return False
    return is_target_english(target_lang)


# Regex che riconosce una riga numerata in output, tipo "12. testo", "[12] testo", "12) testo".
_LINE_NUM_RE = re.compile(r'^(?:\[?(\d+)\]?[\.\)\:\s]+)(.*)')


def parse_translated_chunk(raw_output, start_idx, expected_count):
    """Analizza l'output grezzo del modello per un singolo blocco, applicando
    la logica anti-allucinazione osservata in produzione:
    - i numeri di riga fuori dal range atteso vengono ignorati;
    - in caso di numero di riga ripetuto (il modello che allucina un finto
      nuovo turno e ricomincia a rigenerare le stesse righe), vince SEMPRE la
      PRIMA occorrenza: le successive sono quasi certamente spazzatura;
    - una volta raccolte tutte le 'expected_count' righe attese, il resto
      dell'output (tipicamente proprio l'inizio dell'allucinazione) viene
      ignorato.
    Estratta in una funzione a parte (pura, senza side-effect su cache o
    stato globale) per poter essere testata con pytest senza dover mockare
    LM Studio.
    Ritorna un dict {indice_relativo_al_blocco: testo_pulito}."""
    output_lines = raw_output.split("\n")
    translated_dict = {}

    for line in output_lines:
        line_str = line.strip()
        if not line_str:
            continue

        match = _LINE_NUM_RE.match(line_str)
        if match:
            line_num = int(match.group(1))
            raw_text_content = match.group(2).strip()
            cleaned_content = clean_translated_line(raw_text_content)

            relative_idx = line_num - 1 - start_idx
            if 0 <= relative_idx < expected_count:
                if relative_idx not in translated_dict:
                    translated_dict[relative_idx] = cleaned_content

        if len(translated_dict) >= expected_count:
            break

    return translated_dict


def translate_chunk_worker(args):
    """Worker in esecuzione nei thread per elaborare i blocchi in parallelo.
    Ritorna sempre (start_idx, righe_tradotte, tempo_impiegato_in_secondi),
    anche nei percorsi di errore/timeout, per poter calcolare statistiche
    accurate (tempo medio per blocco) a fine traduzione."""
    chunk_lines, target_lang, start_idx, spotify_context = args
    end_idx = start_idx + len(chunk_lines)
    numbered_lines = [f"{start_idx + i + 1}. {line}" for i, line in enumerate(chunk_lines)]
    full_text = "\n".join(numbered_lines)
    lang_name = resolve_language_name(target_lang)

    context_rule = ""
    context_line = ""
    if spotify_context:
        artist, track = spotify_context
        context_rule = (
            f"8. SONG CONTEXT: This is the song \"{track}\" performed by {artist}. "
            f"Use this context (e.g. the likely gender of the singer/narrator, and the "
            f"tone/genre suggested by the artist) to choose natural, grammatically correct "
            f"wording in {lang_name} — especially for languages requiring gender agreement "
            f"on adjectives, past participles, or pronouns (e.g. Italian, French, Spanish, "
            f"Portuguese, German, Russian, Arabic, Hindi). Do not mention the artist or song "
            f"title in the translated output itself.\n"
        )
        context_line = f"Song: \"{track}\" by {artist}.\n"

    system_prompt = (
    f"You are a professional song lyrics translator. Translate the provided lyrics into natural, expressive {lang_name}.\n"
    f"STRICT RULES:\n"
    f"1. Output lines strictly numbered from {start_idx + 1} to {end_idx}.\n"
    f"2. Do NOT change or reset line numbering.\n"
    f"3. Do NOT add section headers, labels, extra numbers, or single end-of-line periods.\n"
    f"4. Preserve emotional tone and poetic meaning.\n"
    f"5. Every single line MUST be written in {lang_name} ONLY. "
    f"Never switch to English, Spanish, or any other language, even for a single line.\n"
    f"6. STYLISTIC & RHYTHMIC PATTERNS: Respect intentional, expressive repetitions "
    f"(such as repeated words, verbs, or exclamations used for emotional or rhythmic impact). "
    f"Mirror this repetition in {lang_name} rather than summarizing or flattening it into descriptive prose.\n"
    f"7. KEYWORDS & POETIC VOCABULARY: Do NOT over-simplify rare, evocative, or thematic words "
    f"(especially terms tied to the song's title or central motif). Use equally strong, "
    f"expressive, or literal terms in {lang_name} to preserve the author's original word choice.\n"
    f"{context_rule}"
)

    user_prompt = (
        f"{context_line}"
        f"Target language: {lang_name}.\n"
        f"Translate these lines ({start_idx + 1}-{end_idx}) into {lang_name} only:\n{full_text}"
    )

    lm_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.05,
        "max_tokens": 384,
        # "AI:"/"Assistant:" intercettano il caso osservato nei log: il modello,
        # dopo aver tradotto correttamente le righe richieste, a volte allucina
        # un finto nuovo turno etichettato "AI:" e ricomincia a rigenerare (male,
        # spesso troncato) le stesse righe da capo. Senza questi stop token il
        # modello continuava a generare fino a max_tokens o al timeout del client.
        "stop": ["\n\n\n", "User:", "Human:", "AI:", "\nAI:", "Assistant:", "\nAssistant:"]
    }

    t0 = time.time()
    try:
        res = requests.post(
            LM_STUDIO_URL,
            json=lm_payload,
            headers={'Content-Type': 'application/json'},
            timeout=LM_TIMEOUT
        )
        elapsed = time.time() - t0

        if res.status_code == 200:
            res_json = res.json()
            raw_output = res_json['choices'][0]['message']['content'].strip()
            expected_count = len(chunk_lines)

            translated_dict = parse_translated_chunk(raw_output, start_idx, expected_count)

            final_lines = []
            for i in range(len(chunk_lines)):
                if i in translated_dict:
                    final_lines.append(translated_dict[i])
                    line_cache_set(make_line_cache_key(chunk_lines[i], target_lang), translated_dict[i])
                else:
                    final_lines.append(clean_translated_line(chunk_lines[i]))

            logger.info(f" -> Blocco {start_idx + 1}-{end_idx} completato in {elapsed:.2f}s")
            return start_idx, final_lines, elapsed
        else:
            logger.warning(f"[WARN LM STUDIO] Status {res.status_code}")
            return start_idx, [clean_translated_line(l) for l in chunk_lines], elapsed

    except Exception as e:
        elapsed = time.time() - t0
        logger.warning(f"[WARN TIMEOUT/ERRORE] Blocco {start_idx + 1}-{end_idx}: {e}")
        return start_idx, [clean_translated_line(l) for l in chunk_lines], elapsed

def process_hybrid_translation(lines, target_lang, spotify_context=None):
    """Gestore principale del flusso ibrido con elaborazione parallela.
    spotify_context: tupla opzionale (artista, titolo) del brano corrente,
    inviata come contesto aggiuntivo ad ogni blocco tradotto dall'AI.

    Ritorna (final_output, stats), dove stats è un dict pensato per il
    riepilogo di fine traduzione e per l'endpoint /status: quante righe sono
    state bypassate, quante recuperate dalla cache di riga, quante inviate
    all'AI, in quanti blocchi, e il tempo medio per blocco."""
    total_lines = len(lines)
    final_output = [None] * total_lines
    lines_needing_ai = []
    indices_needing_ai = []

    bypass_count = 0
    line_cache_hit_count = 0

    # PASSAGGIO 1: Filtro rapido (righe bypassabili e cache dei singoli versi)
    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if can_bypass_ai(clean_line, target_lang):
            final_output[idx] = clean_translated_line(line)
            bypass_count += 1
        else:
            cached_value = line_cache_get(make_line_cache_key(clean_line, target_lang))
            if cached_value is not None:
                final_output[idx] = cached_value
                line_cache_hit_count += 1
            else:
                lines_needing_ai.append(line)
                indices_needing_ai.append(idx)

    ai_count = len(lines_needing_ai)
    logger.info(f"[INFO] Totale righe: {total_lines} | Inviate all'AI: {ai_count} | Bypassate: {total_lines - ai_count}")

    if ai_count == 0:
        stats = {
            "total_lines": total_lines,
            "bypassed": bypass_count,
            "from_line_cache": line_cache_hit_count,
            "from_ai": 0,
            "ai_blocks": 0,
            "avg_block_seconds": 0,
        }
        return final_output, stats

    # PASSAGGIO 2: Creazione dei task per i blocchi da processare
    tasks = []
    for i in range(0, ai_count, CHUNK_SIZE):
        chunk_lines = lines_needing_ai[i:i + CHUNK_SIZE]
        chunk_start_orig_idx = indices_needing_ai[i]
        tasks.append((chunk_lines, target_lang, chunk_start_orig_idx, spotify_context))

    # PASSAGGIO 3: Esecuzione PARALLELA fino a MAX_WORKERS task simultanei
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(translate_chunk_worker, tasks))

    # PASSAGGIO 4: Riassemblaggio delle righe tradotte mantenendo l'ordine originale
    block_elapsed_times = []
    for start_orig_idx, translated_chunk, elapsed in results:
        block_elapsed_times.append(elapsed)
        for sub_i, translated_text in enumerate(translated_chunk):
            target_original_idx = indices_needing_ai[indices_needing_ai.index(start_orig_idx) + sub_i]
            final_output[target_original_idx] = translated_text

    for i in range(total_lines):
        if final_output[i] is None:
            final_output[i] = clean_translated_line(lines[i])

    stats = {
        "total_lines": total_lines,
        "bypassed": bypass_count,
        "from_line_cache": line_cache_hit_count,
        "from_ai": ai_count,
        "ai_blocks": len(tasks),
        "avg_block_seconds": round(sum(block_elapsed_times) / len(block_elapsed_times), 2) if block_elapsed_times else 0,
    }

    logger.info(
        f"[RIEPILOGO] {total_lines} righe totali | bypass: {bypass_count} | "
        f"da cache di riga: {line_cache_hit_count} | da AI: {ai_count} "
        f"(in {stats['ai_blocks']} blocchi, media {stats['avg_block_seconds']:.2f}s/blocco)"
    )

    return final_output, stats


# =====================================================================
# HEALTH CHECK LM STUDIO
# =====================================================================

def check_lm_studio_health(timeout=5):
    """Ping veloce a LM Studio (endpoint /v1/models, sempre disponibile su
    un server LM Studio attivo) per sapere subito se il backend è
    raggiungibile, invece di scoprirlo solo alla prima richiesta di
    traduzione. Aggiorna anche lo STATE globale per l'endpoint /status."""
    base_url = LM_STUDIO_URL.rsplit("/chat/completions", 1)[0]
    models_url = f"{base_url}/models"
    reachable = False
    try:
        r = requests.get(models_url, timeout=timeout)
        if r.status_code == 200:
            try:
                n_models = len(r.json().get("data", []))
            except ValueError:
                n_models = "?"
            logger.info(f"[HEALTH] LM Studio raggiungibile su {LM_STUDIO_URL} (modelli disponibili: {n_models}).")
            reachable = True
        else:
            logger.warning(f"[HEALTH] LM Studio ha risposto con status {r.status_code} su {models_url}.")
    except requests.exceptions.RequestException as e:
        logger.warning(
            f"[HEALTH] LM Studio NON raggiungibile su {LM_STUDIO_URL} ({e}). "
            f"Verifica che LM Studio sia avviato con il server locale attivo."
        )

    with STATE_LOCK:
        STATE["lm_studio_reachable"] = reachable
        STATE["lm_studio_last_check"] = datetime.now().isoformat()

    return reachable


# =====================================================================
# PREFETCH IN BACKGROUND (best-effort)
# =====================================================================

def fetch_lyrics_lrclib(artist, title):
    """Recupera il testo (plain, non sincronizzato) di un brano da lrclib.net,
    un'API pubblica gratuita senza autenticazione, usata comunemente anche da
    altre app di lyrics per Spicetify. Best-effort: ritorna None se il brano
    non è trovato o in caso di errore di rete."""
    try:
        r = requests.get(
            "https://lrclib.net/api/get",
            params={"artist_name": artist, "track_name": title},
            timeout=6,
        )
        if r.status_code == 200:
            data = r.json()
            plain = data.get("plainLyrics")
            if plain:
                return plain.split("\n")
    except requests.exceptions.RequestException as e:
        logger.debug(f"[PREFETCH] lrclib.net non raggiungibile: {e}")
    return None


_last_prefetched_track = None


def prefetch_worker():
    """Thread in background: sondaggio periodico di SMTC per rilevare un
    cambio di brano su Spotify, e pre-traduzione anticipata verso la lingua
    target di default ('default_target_lang' in config.json), PRIMA che
    Spicetify la richieda esplicitamente.

    LIMITE NOTO (importante): la cache è indicizzata sull'hash esatto delle
    righe inviate da Spicetify (get_hash su testo+lingua). Il testo recuperato
    qui da lrclib.net potrebbe non corrispondere carattere per carattere
    (spaziatura, righe vuote, differenze di sorgente) a quello che Spicetify
    invierà davvero: in quel caso il prefetch non produce un cache-hit e la
    traduzione verrà comunque rifatta alla richiesta reale. Resta comunque
    utile quando i testi combaciano (caso frequente), e non ha costo se il
    brano non viene mai richiesto. Disattivato di default: abilitalo con
    "prefetch": {"enabled": true} in config.json."""
    global _last_prefetched_track
    interval = PREFETCH_CONFIG.get("poll_interval_seconds", 5)
    target_lang = DEFAULT_TARGET_LANG
    logger.info(f"[PREFETCH] Thread di prefetch avviato (poll ogni {interval}s, target='{target_lang}').")

    while True:
        try:
            track = get_current_spotify_track()
            if track and track != _last_prefetched_track:
                artist, title = track
                _last_prefetched_track = track
                logger.info(f"[PREFETCH] Nuovo brano rilevato: {artist} - {title}. Recupero testo da lrclib.net...")

                lyric_lines = fetch_lyrics_lrclib(artist, title)
                if lyric_lines:
                    req_hash = get_hash(lyric_lines, target_lang)
                    if global_cache_get(req_hash) is None:
                        logger.info(f"[PREFETCH] Pre-traduzione in background di {len(lyric_lines)} righe...")
                        translated, _stats = process_hybrid_translation(lyric_lines, target_lang, track)
                        global_cache_set(req_hash, translated)
                        save_caches()
                        logger.info(f"[PREFETCH] Pre-traduzione completata e messa in cache per '{title}'.")
                    else:
                        logger.info(f"[PREFETCH] '{title}' già in cache, nulla da fare.")
                else:
                    logger.info(f"[PREFETCH] Testo non trovato su lrclib.net per '{artist} - {title}'.")
        except Exception as e:
            logger.warning(f"[PREFETCH] Errore nel ciclo di prefetch: {e}")

        time.sleep(interval)


# =====================================================================
# ROUTE FLASK
# =====================================================================

@app.route('/status', methods=['GET'])
def status_handler():
    """Vista d'insieme rapida da browser: dimensione cache, uptime,
    raggiungibilità di LM Studio, ultima traduzione fatta."""
    with CACHE_DATA_LOCK:
        global_count = len(GLOBAL_CACHE)
        line_count = len(LINE_CACHE)

    reachable = check_lm_studio_health()

    with STATE_LOCK:
        uptime = time.time() - STATE["start_time"]
        last_translation = STATE.get("last_translation")
        total_requests = STATE.get("total_requests")

    return make_cors_response({
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "lm_studio_reachable": reachable,
        "lm_studio_url": LM_STUDIO_URL,
        "model_name": MODEL_NAME,
        "cache": {"songs_cached": global_count, "lines_cached": line_count},
        "total_requests_served": total_requests,
        "last_translation": last_translation,
    })


@app.route('/cache/stats', methods=['GET'])
def cache_stats_handler():
    """Ispeziona la cache (dimensioni, file su disco) senza doverla aprire a mano."""
    with CACHE_DATA_LOCK:
        global_count = len(GLOBAL_CACHE)
        line_count = len(LINE_CACHE)

    def file_info(path):
        if os.path.exists(path):
            st = os.stat(path)
            return {
                "exists": True,
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            }
        return {"exists": False}

    return make_cors_response({
        "global_cache": {"entries": global_count, "file": file_info(GLOBAL_CACHE_FILE)},
        "line_cache": {"entries": line_count, "file": file_info(LINE_CACHE_FILE)},
    })


@app.route('/cache/clear', methods=['GET', 'POST'])
def cache_clear_handler():
    """Svuota la cache su richiesta esplicita (richiede ?confirm=yes per
    evitare cancellazioni accidentali, es. da un prefetch del browser).
    Query param opzionale 'which=global|line|all' (default: all).
    Un backup viene sempre creato prima di cancellare."""
    confirm = request.args.get('confirm', '').strip().lower() in ('1', 'true', 'yes', 'y')
    which = request.args.get('which', 'all').strip().lower()
    if which not in ('all', 'global', 'line'):
        which = 'all'

    if not confirm:
        return make_cors_response({
            "error": (
                "Conferma richiesta. Aggiungi ?confirm=yes all'URL "
                "(opzionale &which=global|line|all) per procedere."
            ),
        }, 400)

    clear_caches_on_disk(which)
    logger.info(f"[CACHE] Cache svuotata via /cache/clear (which={which}) su richiesta esplicita.")
    return make_cors_response({"status": "cleared", "which": which})


@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS', 'GET'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS', 'GET'])
def proxy_handler(path):
    if request.method == 'OPTIONS':
        return make_cors_response({})

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        lines = data.get("q") or data.get("text") or []
        target_lang = data.get("target") or data.get("target_lang") or DEFAULT_TARGET_LANG

        if not lines:
            return make_cors_response({"translatedText": []})

        if isinstance(lines, str):
            lines = [lines]

        with STATE_LOCK:
            STATE["total_requests"] += 1

        req_hash = get_hash(lines, target_lang)

        # Controllo Cache Globale per l'intera canzone
        cached_song = global_cache_get(req_hash)
        if cached_song is not None:
            logger.info("[CACHE HIT] Canzone restituita istantaneamente!")
            return make_cors_response({
                "translatedText": cached_song,
                "translated_text": "\n".join(cached_song)
            })

        # Lock per evitare che due richieste identiche contemporanee lavorino due volte
        with GLOBAL_LOCK:
            if req_hash not in PENDING_LOCKS:
                PENDING_LOCKS[req_hash] = Lock()
            request_lock = PENDING_LOCKS[req_hash]

        try:
            with request_lock:
                cached_song = global_cache_get(req_hash)
                if cached_song is not None:
                    return make_cors_response({
                        "translatedText": cached_song,
                        "translated_text": "\n".join(cached_song)
                    })

                t_start = time.time()
                logger.info(f"[INFO] Ricevuta canzone ({len(lines)} righe)...")

                spotify_context = get_current_spotify_track()
                if spotify_context:
                    logger.info(f"[INFO] Contesto Spotify rilevato: {spotify_context[0]} - {spotify_context[1]}")
                else:
                    logger.info("[INFO] Nessun contesto Spotify disponibile (finestra non trovata, pausa, o piattaforma non supportata).")

                try:
                    translated_lines, stats = process_hybrid_translation(lines, target_lang, spotify_context)
                    global_cache_set(req_hash, translated_lines)
                    save_caches()
                    total_time = time.time() - t_start
                    logger.info(f"[SUCCESS] Canzone completata in {total_time:.2f}s!")

                    with STATE_LOCK:
                        STATE["last_translation"] = {
                            "timestamp": datetime.now().isoformat(),
                            "target_lang": target_lang,
                            "duration_seconds": round(total_time, 2),
                            **stats,
                        }

                    slow_threshold = NOTIFICATIONS_CONFIG.get("slow_translation_threshold_seconds", 25)
                    if total_time > slow_threshold:
                        notify_desktop(
                            "Traduzione lenta",
                            f"{len(lines)} righe in {total_time:.1f}s (soglia: {slow_threshold}s)."
                        )

                    return make_cors_response({
                        "translatedText": translated_lines,
                        "translated_text": "\n".join(translated_lines)
                    })
                except Exception as err:
                    logger.error(f"[FALLBACK INTERNO] {err}")
                    notify_desktop("Traduzione fallita", f"Errore durante la traduzione: {err}")
                    return make_cors_response({
                        "translatedText": lines,
                        "translated_text": "\n".join(lines)
                    })
        finally:
            # Rimuove il lock della richiesta completata per evitare che PENDING_LOCKS
            # cresca all'infinito nel tempo.
            with GLOBAL_LOCK:
                PENDING_LOCKS.pop(req_hash, None)

    return "Bridge ibrido attivo (Parallel Engine v2)!", 200

if __name__ == '__main__':
    logger.info("==================================================")
    logger.info(" Bridge Spotify Lyrics -> LM Studio (Parallel 4)")
    logger.info(" In ascolto su: http://127.0.0.1:5000")
    logger.info("==================================================")

    check_lm_studio_health()

    if PREFETCH_CONFIG.get("enabled", False):
        threading.Thread(target=prefetch_worker, daemon=True).start()
    else:
        logger.info("[PREFETCH] Disattivato (abilita 'prefetch.enabled' in config.json per attivarlo).")

    app.run(
        host=SERVER_CONFIG.get("host", "127.0.0.1"),
        port=SERVER_CONFIG.get("port", 5000),
        threaded=True,
    )
