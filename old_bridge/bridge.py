import hashlib
import time
import re
import os
import json
import tempfile
import sys
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "translategemma-12b-it"

# --- OTTIMIZZAZIONI PER MASSIMA VELOCITÀ ---
CHUNK_SIZE = 10      # 10 righe per blocco
MAX_WORKERS = 4      # Sfrutta appieno i 4 slot paralleli di LM Studio
LM_TIMEOUT = 120      # Evita timeout durante i picchi di lavoro

# --- PERSISTENZA CACHE SU DISCO ---
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
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
            print(f"[WARN CACHE] Impossibile leggere {path}: {e}")
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
        print(f"[WARN CACHE] Impossibile scrivere {path}: {e}")
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


GLOBAL_CACHE = _load_json_cache(GLOBAL_CACHE_FILE)   # Cache per intere canzoni
LINE_CACHE = _load_json_cache(LINE_CACHE_FILE)       # Cache per singoli versi/ritornelli ripetuti
PENDING_LOCKS = {}
GLOBAL_LOCK = Lock()

print(f"[INFO] Cache caricata da disco: {len(GLOBAL_CACHE)} canzoni, {len(LINE_CACHE)} righe singole.")

def make_cors_response(data, status_code=200):
    response = jsonify(data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'authorization, content-type, x-api-key'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
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

# Se True, quando NESSUNA sessione multimediale risulta chiaramente di Spotify
# (es. Spotify aperto nel browser come tab normale, non installato come PWA:
# in quel caso l'app id riportato è quello del browser, non "spotify"), e
# c'è ESATTAMENTE UNA sessione di sistema in riproduzione attiva, la si userà
# comunque come ultima risorsa (loggando chiaramente che è una supposizione).
# Impostare a False se si preferisce non rischiare mai un match sbagliato.
SMTC_FALLBACK_TO_SOLE_PLAYING_SESSION = True


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
                print(
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
            print(
                "[INFO SPOTIFY CONTEXT] Pacchetto 'winsdk' non installato: uso il "
                "metodo di rilevamento legacy (meno affidabile con Spotify recente). "
                "Per il rilevamento affidabile esegui: pip install winsdk"
            )
            _SMTC_HINT_SHOWN = True
    except Exception as e:
        print(f"[WARN SPOTIFY CONTEXT] Metodo SMTC fallito, uso il fallback: {e}")

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


def _get_spotify_track_linux():
    """Legge artista e titolo su Linux tramite playerctl (se installato)."""
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


def get_current_spotify_track():
    """Restituisce una tupla (artista, titolo) del brano attualmente in
    riproduzione su Spotify, oppure None se non disponibile (Spotify chiuso,
    in pausa, playerctl non installato, piattaforma non supportata, ecc.).
    Non solleva mai eccezioni: qualunque errore viene ignorato e si ritorna
    semplicemente None, così la traduzione prosegue comunque senza il
    contesto extra, senza mai bloccare o rallentare la richiesta."""
    try:
        if sys.platform.startswith("win"):
            return _get_spotify_track_windows()
        elif sys.platform == "darwin":
            return _get_spotify_track_macos()
        elif sys.platform.startswith("linux"):
            return _get_spotify_track_linux()
    except Exception as e:
        print(f"[WARN SPOTIFY CONTEXT] Impossibile leggere il brano corrente: {e}")
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

def has_japanese(text):
    return bool(re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]', text))

def is_target_english(target_lang):
    if not target_lang:
        return True
    return target_lang.strip().lower() in ("en", "en-us", "en-gb", "english")

def can_bypass_ai(text, target_lang):
    """LIVELLO 1: Decide se una riga può saltare l'AI.
    - Righe fatte solo di simboli/numeri/spazi: bypass sempre (nulla da tradurre).
    - Righe senza giapponese: bypass SOLO se la lingua target è l'inglese
      (altrimenti il testo, pur non essendo giapponese, potrebbe comunque non
      essere nella lingua target richiesta e va comunque tradotto)."""
    if is_symbols_only(text):
        return True
    if has_japanese(text):
        return False
    return is_target_english(target_lang)

def translate_chunk_worker(args):
    """Worker in esecuzione nei thread per elaborare i blocchi in parallelo."""
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

    try:
        t0 = time.time()
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

            output_lines = raw_output.split("\n")
            translated_dict = {}
            expected_count = len(chunk_lines)

            for line in output_lines:
                line_str = line.strip()
                if not line_str:
                    continue

                match = re.match(r'^(?:\[?(\d+)\]?[\.\)\:\s]+)(.*)', line_str)
                if match:
                    line_num = int(match.group(1))
                    raw_text_content = match.group(2).strip()
                    cleaned_content = clean_translated_line(raw_text_content)

                    relative_idx = line_num - 1 - start_idx
                    if 0 <= relative_idx < expected_count:
                        # IMPORTANTE: non sovrascrivere una traduzione già presente.
                        # Se il modello allucina un finto turno successivo e ripete
                        # lo stesso numero di riga (visto nei log: "AI:\n21. ..."),
                        # la ripetizione va IGNORATA — la prima occorrenza è quella
                        # legittima, le successive sono quasi sempre spazzatura
                        # generata dal loop di ripetizione.
                        if relative_idx not in translated_dict:
                            translated_dict[relative_idx] = cleaned_content

                # Una volta raccolte tutte le righe attese, il resto dell'output
                # (tipicamente proprio l'inizio dell'allucinazione) viene ignorato.
                if len(translated_dict) >= expected_count:
                    break

            final_lines = []
            for i in range(len(chunk_lines)):
                if i in translated_dict:
                    final_lines.append(translated_dict[i])
                    line_cache_set(make_line_cache_key(chunk_lines[i], target_lang), translated_dict[i])
                else:
                    final_lines.append(clean_translated_line(chunk_lines[i]))

            print(f" -> Blocco {start_idx + 1}-{end_idx} completato in {elapsed:.2f}s")
            return start_idx, final_lines
        else:
            print(f"[WARN LM STUDIO] Status {res.status_code}")
            return start_idx, [clean_translated_line(l) for l in chunk_lines]

    except Exception as e:
        print(f"[WARN TIMEOUT/ERRORE] Blocco {start_idx + 1}-{end_idx}: {e}")
        return start_idx, [clean_translated_line(l) for l in chunk_lines]

def process_hybrid_translation(lines, target_lang, spotify_context=None):
    """Gestore principale del flusso ibrido con elaborazione parallela.
    spotify_context: tupla opzionale (artista, titolo) del brano corrente,
    inviata come contesto aggiuntivo ad ogni blocco tradotto dall'AI."""
    total_lines = len(lines)
    final_output = [None] * total_lines
    lines_needing_ai = []
    indices_needing_ai = []

    # PASSAGGIO 1: Filtro rapido (Inglese/simboli e cache dei singoli versi)
    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if can_bypass_ai(clean_line, target_lang):
            final_output[idx] = clean_translated_line(line)
        else:
            cached_value = line_cache_get(make_line_cache_key(clean_line, target_lang))
            if cached_value is not None:
                final_output[idx] = cached_value
            else:
                lines_needing_ai.append(line)
                indices_needing_ai.append(idx)

    ai_count = len(lines_needing_ai)
    print(f"[INFO] Totale righe: {total_lines} | Inviate all'AI: {ai_count} | Bypassate: {total_lines - ai_count}")

    if ai_count == 0:
        return final_output

    # PASSAGGIO 2: Creazione dei task per i blocchi da processare
    tasks = []
    for i in range(0, ai_count, CHUNK_SIZE):
        chunk_lines = lines_needing_ai[i:i + CHUNK_SIZE]
        chunk_start_orig_idx = indices_needing_ai[i]
        tasks.append((chunk_lines, target_lang, chunk_start_orig_idx, spotify_context))

    # PASSAGGIO 3: Esecuzione PARALLELA fino a 4 task simultanei
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(translate_chunk_worker, tasks)

    # PASSAGGIO 4: Riassemblaggio delle righe tradotte mantenendo l'ordine originale
    for start_orig_idx, translated_chunk in results:
        for sub_i, translated_text in enumerate(translated_chunk):
            target_original_idx = indices_needing_ai[indices_needing_ai.index(start_orig_idx) + sub_i]
            final_output[target_original_idx] = translated_text

    for i in range(total_lines):
        if final_output[i] is None:
            final_output[i] = clean_translated_line(lines[i])

    return final_output

@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS', 'GET'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS', 'GET'])
def proxy_handler(path):
    if request.method == 'OPTIONS':
        return make_cors_response({})

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        lines = data.get("q") or data.get("text") or []
        target_lang = data.get("target") or data.get("target_lang") or "en"

        if not lines:
            return make_cors_response({"translatedText": []})

        if isinstance(lines, str):
            lines = [lines]

        req_hash = get_hash(lines, target_lang)

        # Controllo Cache Globale per l'intera canzone
        cached_song = global_cache_get(req_hash)
        if cached_song is not None:
            print("[CACHE HIT] Canzone restituita istantaneamente!")
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
                print(f"\n[INFO] Ricevuta canzone ({len(lines)} righe)...")

                spotify_context = get_current_spotify_track()
                if spotify_context:
                    print(f"[INFO] Contesto Spotify rilevato: {spotify_context[0]} - {spotify_context[1]}")
                else:
                    print("[INFO] Nessun contesto Spotify disponibile (finestra non trovata, pausa, o piattaforma non supportata).")

                try:
                    translated_lines = process_hybrid_translation(lines, target_lang, spotify_context)
                    global_cache_set(req_hash, translated_lines)
                    save_caches()
                    total_time = time.time() - t_start
                    print(f"[SUCCESS] Canzone completata in {total_time:.2f}s!")

                    return make_cors_response({
                        "translatedText": translated_lines,
                        "translated_text": "\n".join(translated_lines)
                    })
                except Exception as err:
                    print(f"[FALLBACK INTERNO] {err}")
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
    print("==================================================")
    print(" Bridge Spotify Lyrics -> LM Studio (Parallel 4)")
    print(" In ascolto su: http://127.0.0.1:5000")
    print("==================================================")
    app.run(host='127.0.0.1', port=5000, threaded=True)