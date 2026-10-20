"""Nối lại âm tiết tiếng Việt bị `pypdf` tách bằng khoảng trắng thừa ("c ủa" -> "của", "thi ết" -> "thiết").

Chỉ nối khi chữ ghép lại là một âm tiết hợp lệ theo chính tả (phụ âm đầu + nguyên âm + phụ âm cuối, đúng một dấu
thanh), và các mảnh phía sau bắt đầu bằng chữ có dấu. Nhờ vậy cặp từ thật như "sống ở", "học ở", "tôi ở" không bị
nối nhầm: "sốngở", "họcở", "tôiở" không phải âm tiết.
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

VOCAB_FILE = Path(__file__).parent / "vi_syllables.txt"  # tạo bằng build_vocab.py từ chính kho tài liệu

_TONES = "̣̀́̃̉"  # huyền, sắc, ngã, hỏi, nặng
_ONSET = r"ngh|ng|nh|ph|th|tr|ch|kh|gh|gi|qu|[bcdđghklmnpqrstvx]"
_NUCLEUS = (
    r"iêu|yêu|oai|oao|oay|oeo|uây|uôi|uya|uyê|ươi|ươu|"
    r"ai|ao|au|ay|âu|ây|eo|êu|ia|iê|yê|iu|oa|oă|oe|oi|ôi|ơi|ua|uô|uâ|uê|ui|ưa|ươ|ưi|ưu|uy|"
    r"[aăâeêiyoôơuư]"
)
_CODA = r"ch|ng|nh|[cmnpt]"
_SYLLABLE = re.compile(rf"(?:{_ONSET})?({_NUCLEUS})({_CODA})?")

_VI_LETTER = re.compile(r"[a-zđăâêôơưẠ-ỹàáãảạèéẽẻẹìíĩỉịòóõỏọùúũủụỳýỹỷỵ]", re.I)


def _is_syllable(word: str) -> bool:
    w = unicodedata.normalize("NFC", word.lower())
    nfd = unicodedata.normalize("NFD", w)
    tones = [c for c in nfd if c in _TONES]
    if len(tones) > 1:
        return False
    plain = unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _TONES))
    m = _SYLLABLE.fullmatch(plain)
    if not m:
        return False
    nucleus, coda = m.group(1), m.group(2)
    if nucleus in ("ă", "â") and not coda:
        return False  # ă, â không đứng cuối âm tiết
    if nucleus in ("iê", "yê", "uyê", "uô", "ươ") and not coda:
        return False  # các vần này luôn có phụ âm cuối (không có coda thì viết ia/ya/ua/ưa)
    if nucleus in ("ia", "ua", "ưa", "uya") and coda:
        return False  # ngược lại ia/ua/ưa/uya chỉ là vần mở
    if coda in ("c", "ch", "p", "t") and tones and tones[0] not in "̣́":
        return False  # vần kết thúc c/ch/p/t chỉ mang dấu sắc hoặc nặng
    return True


def _has_accent(ch: str) -> bool:
    """Chữ cái tiếng Việt không phải ASCII (có dấu hoặc là đ, ă, â, ê, ô, ơ, ư)."""
    return ch.isalpha() and not ch.isascii() and _VI_LETTER.match(ch) is not None


@lru_cache(maxsize=1)
def _vocab() -> frozenset:
    """Các âm tiết thật, xuất hiện nguyên vẹn nhiều lần trong kho tài liệu."""
    try:
        return frozenset(VOCAB_FILE.read_text(encoding="utf-8").split())
    except OSError:
        return frozenset()


def _known(word: str) -> bool:
    return unicodedata.normalize("NFC", word.lower()) in _vocab()


_TOKEN = re.compile(r"(\W*)([^\W\d_]+)(\W*)")


def _split(token: str):
    m = _TOKEN.fullmatch(token)
    return m.groups() if m else None


def merge_split_syllables(text: str) -> str:
    """Nối các mảnh âm tiết bị tách bằng khoảng trắng; giữ nguyên xuống dòng và dấu câu."""
    return "\n".join(_merge_line(line) for line in text.split("\n"))


def _merge_line(line: str) -> str:
    tokens = line.split(" ")
    out = []
    i = 0
    while i < len(tokens):
        head = _split(tokens[i])
        merged = None
        # head: không có dấu câu phía sau, chữ cuối không viết hoa (tránh từ viết tắt)
        # "5 m ở": m là đơn vị đo (m, g, l, s), không phải mảnh âm tiết; các mảnh khác sau số ("2 ph ương") vẫn nối
        unit = i > 0 and tokens[i - 1][-1:].isdigit() and head is not None and head[1] in ("m", "g", "l", "s")
        if head and not head[2] and not head[1][-1].isupper() and not unit:
            for take in (3, 2):  # thử nối 3 mảnh trước ("tr ư ờng"), rồi 2 mảnh ("c ủa")
                parts = [_split(t) for t in tokens[i + 1:i + take]]
                if len(parts) < take - 1 or any(p is None for p in parts):
                    continue
                if any(p[0] for p in parts):  # mảnh sau không được mở đầu bằng dấu câu
                    continue
                if any(p[2] for p in parts[:-1]):  # dấu câu chỉ được ở cuối mảnh cuối
                    continue
                first = parts[0][1]
                if not _has_accent(first[0]):
                    continue
                # mảnh thứ ba (nếu có) chỉ là phần vần tiếp theo của mảnh thứ hai là một chữ có dấu đơn lẻ
                if take == 3 and len(first) != 1:
                    continue
                core = head[1] + "".join(p[1] for p in parts)
                if not _is_syllable(core):
                    continue
                # cả các mảnh cũng đều là âm tiết hợp lệ ("cho ảnh" -> "choảnh"?): chỉ nối khi chữ ghép là từ thật
                ambiguous = _is_syllable(head[1]) and all(_is_syllable(p[1]) for p in parts)
                if (not ambiguous and take == 2) or _known(core):  # nối 3 mảnh rủi ro hơn: luôn đòi từ có thật
                    merged = (head[0] + core + parts[-1][2], take)
                    break
        if merged:
            out.append(merged[0])
            i += merged[1]
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)
