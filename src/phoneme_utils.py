"""Phoneme lexicon utilities for soft CTC alignment experiments."""
import re
from collections import OrderedDict
from pathlib import Path


_WORD_RE = re.compile(r"[a-zA-Z']+")


def normalize_word(word):
    word = word.lower().strip("'")
    if word.endswith("'s") and len(word) > 2:
        word = word[:-2]
    return word


def words_from_text(text):
    return [normalize_word(w) for w in _WORD_RE.findall(text) if normalize_word(w)]


def strip_stress(phone):
    return re.sub(r"\d+$", "", phone)


def load_lexicon(path=None, strip_stress_marks=True):
    """Load a CMU-style lexicon: WORD PH1 PH2 ...

    If path is omitted, tries NLTK's cmudict corpus. The returned dict maps a
    lowercase word to the first pronunciation found.
    """
    entries = OrderedDict()
    if path:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith((";", "#")):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = normalize_word(re.sub(r"\(\d+\)$", "", parts[0]))
                phones = parts[1:]
                if strip_stress_marks:
                    phones = [strip_stress(p) for p in phones]
                if word and word not in entries:
                    entries[word] = phones
        return dict(entries)

    try:
        import nltk
        local_nltk = Path(__file__).resolve().parents[1] / "nltk_data"
        if local_nltk.exists() and str(local_nltk) not in nltk.data.path:
            nltk.data.path.insert(0, str(local_nltk))
        from nltk.corpus import cmudict
        for word, phones in cmudict.entries():
            word = normalize_word(word)
            if strip_stress_marks:
                phones = [strip_stress(p) for p in phones]
            if word and word not in entries:
                entries[word] = list(phones)
    except LookupError as exc:
        raise RuntimeError(
            "No phoneme lexicon was provided and NLTK cmudict is not installed. "
            "Pass --lexicon PATH with CMU format lines like: WORD PH1 PH2 ..."
        ) from exc
    return dict(entries)


def make_phone_vocab(lexicon):
    phones = sorted({p for pron in lexicon.values() for p in pron})
    phone_to_id = {p: i for i, p in enumerate(phones)}
    id_to_phone = {i: p for p, i in phone_to_id.items()}
    blank_id = len(phones)
    phone_to_id["<blank>"] = blank_id
    id_to_phone[blank_id] = "<blank>"
    return phone_to_id, id_to_phone, blank_id


def text_to_phones(text, lexicon, unk_policy="skip"):
    """Convert text to a flat ARPABET phone sequence.

    unk_policy:
      skip  : return None if any word is missing.
      drop  : silently drop missing words.
    """
    out = []
    missing = []
    for word in words_from_text(text):
        pron = lexicon.get(word)
        if pron is None:
            missing.append(word)
            if unk_policy == "skip":
                return None, missing
            continue
        out.extend(pron)
    return out, missing


def text_to_phone_ids(text, lexicon, phone_to_id, unk_policy="skip"):
    phones, missing = text_to_phones(text, lexicon, unk_policy=unk_policy)
    if phones is None:
        return None, missing
    return [phone_to_id[p] for p in phones], missing


def save_phone_vocab(path, id_to_phone):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(len(id_to_phone)):
            f.write(f"{i}\t{id_to_phone[i]}\n")


def load_phone_vocab(path):
    id_to_phone = {}
    with open(path) as f:
        for line in f:
            idx, phone = line.rstrip("\n").split("\t", 1)
            id_to_phone[int(idx)] = phone
    phone_to_id = {p: i for i, p in id_to_phone.items()}
    blank_id = phone_to_id["<blank>"]
    return phone_to_id, id_to_phone, blank_id
