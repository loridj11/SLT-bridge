"""
Test per le funzioni "pure" di bridge.py (nessuna chiamata di rete, nessun
side-effect su cache/disco): clean_translated_line, can_bypass_ai,
resolve_language_name, has_non_latin_script e il parsing anti-allucinazione
(parse_translated_chunk).

Esecuzione:
    pytest test_bridge.py -v

Nota: importare bridge.py carica la configurazione, imposta il logging su
file (crea la cartella logs/) e legge/crea la cartella cache/ accanto allo
script: è un side-effect noto e voluto, dato che bridge.py non è pensato
come libreria pura. Non vengono fatte chiamate di rete all'import.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge import (
    clean_translated_line,
    can_bypass_ai,
    resolve_language_name,
    has_non_latin_script,
    is_symbols_only,
    parse_translated_chunk,
)


class TestCleanTranslatedLine:
    def test_removes_leading_number_in_parens(self):
        assert clean_translated_line("(10.) Ciao mondo") == "Ciao mondo"

    def test_removes_leading_number_in_brackets(self):
        assert clean_translated_line("[11] Ciao mondo") == "Ciao mondo"

    def test_keeps_ellipsis(self):
        assert clean_translated_line("Aspetta...") == "Aspetta..."

    def test_removes_trailing_single_period(self):
        assert clean_translated_line("Ciao mondo.") == "Ciao mondo"

    def test_removes_trailing_double_period(self):
        assert clean_translated_line("Ciao mondo..") == "Ciao mondo"

    def test_removes_verse_label(self):
        assert clean_translated_line("Verse 1: Ciao") == "Ciao"

    def test_removes_chorus_label_case_insensitive(self):
        assert clean_translated_line("chorus: Ciao") == "Ciao"

    def test_collapses_extra_whitespace(self):
        assert clean_translated_line("Ciao   mondo  ") == "Ciao mondo"

    def test_empty_string_passthrough(self):
        assert clean_translated_line("") == ""

    def test_none_passthrough(self):
        assert clean_translated_line(None) is None


class TestIsSymbolsOnly:
    def test_pure_symbols(self):
        assert is_symbols_only("♪ ♪ ... ★") is True

    def test_pure_numbers_and_punct(self):
        assert is_symbols_only("12, 34 - 56") is True

    def test_has_real_text(self):
        assert is_symbols_only("Hello ♪") is False


class TestHasNonLatinScript:
    def test_japanese_hiragana_katakana(self):
        assert has_non_latin_script("こんにちは") is True

    def test_japanese_kanji(self):
        assert has_non_latin_script("愛してる") is True

    def test_korean_hangul(self):
        assert has_non_latin_script("안녕하세요") is True

    def test_chinese_hanzi(self):
        assert has_non_latin_script("你好世界") is True

    def test_russian_cyrillic(self):
        assert has_non_latin_script("Привет мир") is True

    def test_arabic(self):
        assert has_non_latin_script("مرحبا") is True

    def test_greek(self):
        assert has_non_latin_script("Γειά σου") is True

    def test_latin_text_is_false(self):
        assert has_non_latin_script("Hello world, comment ça va?") is False

    def test_empty_string_is_false(self):
        assert has_non_latin_script("") is False


class TestCanBypassAi:
    def test_symbols_only_always_bypasses(self):
        assert can_bypass_ai("♪ ♪ ...", "it") is True
        assert can_bypass_ai("♪ ♪ ...", "en") is True

    def test_japanese_never_bypasses(self):
        assert can_bypass_ai("こんにちは", "en") is False

    def test_korean_never_bypasses(self):
        assert can_bypass_ai("안녕하세요", "en") is False

    def test_chinese_never_bypasses(self):
        assert can_bypass_ai("你好", "it") is False

    def test_latin_text_bypasses_only_for_english_target(self):
        assert can_bypass_ai("Hello world", "en") is True
        assert can_bypass_ai("Hello world", "it") is False
        assert can_bypass_ai("Hello world", "fr") is False


class TestResolveLanguageName:
    def test_known_code(self):
        assert resolve_language_name("it") == "Italian (Italiano)"

    def test_case_and_whitespace_insensitive(self):
        assert resolve_language_name("  IT  ") == "Italian (Italiano)"

    def test_unknown_code_passthrough(self):
        assert resolve_language_name("xx") == "xx"

    def test_empty_defaults_to_english(self):
        assert resolve_language_name("") == "English"

    def test_none_defaults_to_english(self):
        assert resolve_language_name(None) == "English"


class TestParseTranslatedChunk:
    """Copre la logica anti-allucinazione: numeri fuori range, righe
    ripetute (finti nuovi turni "AI:"), e interruzione anticipata una volta
    raccolte tutte le righe attese."""

    def test_basic_parsing(self):
        raw = "1. Ciao\n2. Mondo"
        assert parse_translated_chunk(raw, start_idx=0, expected_count=2) == {
            0: "Ciao",
            1: "Mondo",
        }

    def test_ignores_hallucinated_repeat_turn(self):
        raw = "1. Ciao\n2. Mondo\nAI:\n1. Ciao ripetuto male\n2. Mondo troncat"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=2)
        assert result == {0: "Ciao", 1: "Mondo"}

    def test_first_occurrence_wins_on_duplicate_line_number(self):
        raw = "1. Prima versione (corretta)\n1. Seconda versione (allucinata)"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=1)
        assert result[0] == "Prima versione (corretta)"

    def test_out_of_range_line_numbers_are_ignored(self):
        raw = "5. Fuori range\n1. Dentro range"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=1)
        assert result == {0: "Dentro range"}

    def test_offset_start_idx(self):
        raw = "11. Riga undici\n12. Riga dodici"
        result = parse_translated_chunk(raw, start_idx=10, expected_count=2)
        assert result == {0: "Riga undici", 1: "Riga dodici"}

    def test_stops_collecting_once_expected_count_reached(self):
        raw = "1. Uno\n2. Due\n1. Uno ripetuto (allucinato, deve essere ignorato)"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=2)
        assert result == {0: "Uno", 1: "Due"}

    def test_unnumbered_lines_are_skipped(self):
        raw = "premessa senza numero\n1. Ciao\nun'altra riga senza numero\n2. Mondo"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=2)
        assert result == {0: "Ciao", 1: "Mondo"}

    def test_bracket_style_numbering(self):
        raw = "[1] Ciao\n[2] Mondo"
        result = parse_translated_chunk(raw, start_idx=0, expected_count=2)
        assert result == {0: "Ciao", 1: "Mondo"}

    def test_empty_output(self):
        assert parse_translated_chunk("", start_idx=0, expected_count=3) == {}
