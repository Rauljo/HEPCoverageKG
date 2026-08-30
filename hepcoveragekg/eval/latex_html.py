"""LaTeX to readable HTML, for the sentences a physicist is asked to judge.

`aliases.context.clean_latex` exists and is not this. It normalises for MATCHING
-- it folds `$t\\bar{t}$` to "ttbar" so two spellings of one entity collide -- and
it leaves `\\ell`, `\\eta`, `\\to` and `\\penalty 10000` sitting in the text,
which is fine for a dictionary key and wrong for something someone reads.

70 of the 104 rows in batch 2 carry a maths span. Asking a supervisor to read

    $120<m_{\\ell\\ell j}<135\\penalty 10000\\ \\text{Ge\\kern-1.00006ptV}$

is asking him to do the typesetting himself, forty times, before he can judge
anything.

WHY NOT KATEX OR MATHJAX. An artifact is served under a strict CSP with no
external requests, so the library and its fonts would have to be inlined --
several hundred kilobytes to render expressions that are almost all a symbol
with a sub- and superscript. Sub/sup tags and Unicode cover this corpus.

WHAT IT IS BUILT FOR. The commands actually present, counted: mathup 132, text
73, mathrm 68, hskip 25, ell 20, to 16, mu 16, bar 14, chi 14, tilde 14, plus
123 subscripts and 74 superscripts. LaTeXML emits the typesetting noise --
`\\kern`, `\\penalty`, `\\hskip` -- which is why "GeV" arrives as
`Ge\\kern-1.00006ptV`. Anything unrecognised keeps its own text rather than
vanishing: a reader seeing `\\foo` knows something was not handled, where a
silent deletion would look like the paper never said it.
"""
from __future__ import annotations

import html
import re

#: Typesetting instructions with no meaning to a reader. Dropped entirely.
_NOISE = [
    # A DOUBLED BACKSLASH before a letter is an escaping artifact, not a LaTeX
    # line break. `\\Q` reaches us from the stored JSON where the paper wrote
    # `\Q`, the same escaping family as D-067's `\bar` becoming a backspace.
    # Normalised first so every later rule sees one backslash.
    (re.compile(r"\\\\(?=[a-zA-Z])"), r"\\"),
    (re.compile(r"\\penalty\s*-?\d+"), ""),
    # `mu` as well as `pt`, and `\mkern` as well as `\kern`: the diphoton rows
    # carry `^{\mkern 3.0mu\text{miss}}` and an unhandled spacing command
    # leaks its own name into a superscript.
    (re.compile(r"\\m?(?:kern|hskip|vskip|raise|lower)\s*-?[\d.]+\s*(?:pt|mu|em|ex)"), ""),
    (re.compile(r"\\(?:displaystyle|scriptstyle|scriptscriptstyle|textstyle)\b"), ""),
    (re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)\b"), ""),
    (re.compile(r"\\[,;:!>]"), " "),          # thin/medium spaces
    (re.compile(r"\\ (?=\S)"), " "),
    # An ESCAPED brace is a literal brace, not grouping. Left alone it meets the
    # final brace-strip, which removes the `{` and leaves the backslash orphaned:
    # `\left\{\mathcal{Q}\right\}` came out as `\Q\`.
    (re.compile(r"\\([{}])"), r"\1"),
]

#: The same wrappers used WITHOUT braces, LaTeXML's house style: it emits
#: `{\mathup Z}` rather than `\mathup{Z}`, so the font switch applies to the
#: rest of the group. Deleting the command and keeping what follows is correct,
#: and not doing it rendered 21 quotes as "mathupZ", "mathupH", "mathupb".
_UNWRAP_BARE = re.compile(
    r"\\(?:text|mathrm|mathup|mathbf|mathit|mathsf|textrm|mathcal|mathbb|rm|it|bf)"
    r"(?![a-zA-Z])\s*")

#: Wrappers whose only job is upright type. The content is what matters.
_UNWRAP = re.compile(
    r"\\(?:text|mathrm|mathup|mathbf|mathit|mathsf|hbox|textrm|operatorname"
    r"|mathcal|mathbb|mathfrak|mbox|emph)\s*\{([^{}]*)\}")

_SYMBOLS = {
    r"\ell": "ℓ", r"\mu": "μ", r"\nu": "ν", r"\tau": "τ", r"\eta": "η",
    r"\chi": "χ", r"\psi": "ψ", r"\phi": "φ", r"\varphi": "φ", r"\rho": "ρ",
    r"\sigma": "σ", r"\gamma": "γ", r"\alpha": "α", r"\beta": "β",
    r"\delta": "δ", r"\epsilon": "ε", r"\varepsilon": "ε", r"\theta": "θ",
    r"\lambda": "λ", r"\pi": "π", r"\omega": "ω", r"\zeta": "ζ", r"\xi": "ξ",
    r"\Delta": "Δ", r"\Lambda": "Λ", r"\Sigma": "Σ", r"\Omega": "Ω",
    r"\Gamma": "Γ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Upsilon": "Υ",
    r"\to": " → ", r"\rightarrow": " → ", r"\leftarrow": " ← ",
    r"\approx": " ≈ ", r"\sim": "~", r"\times": "×", r"\pm": "±", r"\mp": "∓",
    r"\geq": " ≥ ", r"\ge": " ≥ ", r"\leq": " ≤ ", r"\le": " ≤ ",
    r"\neq": " ≠ ", r"\cdot": "·", r"\infty": "∞", r"\propto": " ∝ ",
    r"\lvert": "|", r"\rvert": "|", r"\vert": "|", r"\mid": "|",
    r"\%": "%", r"\&": "&", r"\#": "#",
}

#: Accents rendered with a combining character, so they need no CSS.
_ACCENTS = {"bar": "\u0304", "overline": "\u0304", "tilde": "\u0303",
            "hat": "\u0302", "vec": "\u20d7", "dot": "\u0307"}


def _flatten_braces(text: str) -> str:
    """`{{{X}}}` to `{X}`.

    LaTeXML nests groups freely -- `{\mathup{{{Z}}}}`, `^{\scriptstyle{+}}` --
    and every pattern here matches a body of `[^{}]*`, which a nested group
    defeats silently. Two things came out wrong because of it: `\overline{\ell}`
    rendered as "overlineℓ" with the accent never applied, and `e^{+}e^{-}`
    kept literal carets instead of becoming superscripts.

    Collapsing first is safe because these groups carry no content of their
    own; they are grouping, and one level of grouping means the same as three.
    """
    for _ in range(6):
        new = re.sub(r"\{\{([^{}]*)\}\}", r"{\1}", text)
        if new == text:
            return text
        text = new
    return text


def _accents(text: str) -> str:
    def one(m):
        body = m.group(2)
        combining = _ACCENTS[m.group(1)]
        # After the FIRST character, so t-bar reads t̄ rather than ̄t.
        return (body[0] + combining + body[1:]) if body else body
    return re.sub(r"\\(" + "|".join(_ACCENTS) + r")\s*\{([^{}]*)\}", one, text)


def _scripts(text: str) -> str:
    """`_{x}` and `^{x}` to real sub/sup tags. BRACED ONLY.

    The bare forms were handled too and had to be removed. A field can mix
    English with maths -- the value rows read "what signal efficiency does the
    pT^miss cut retain?  WE ANSWER: $p_{\mathrm{T}}^{\text{miss}}<40$..." -- and
    a bare rule cannot tell the prose `^m` from the maths one. It turned
    "pT^miss" into "pT<sup>m</sup>iss": markup damage on the half of the string
    that had no markup.

    Braced scripts are the ones LaTeX actually writes. A bare `p_T` survives as
    "p_T", which reads fine and is what the paper's own plain-text rendering
    would show anyway.
    """
    text = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", text)
    return text


def looks_like_latex(text: str) -> bool:
    """Whether a string is worth running through the converter.

    Question text is ENGLISH and writes notation in plain ASCII: "what signal
    efficiency does the pT^miss < 40 GeV cut retain?". Run that through the
    maths rules and `^m` becomes a superscript, giving "pT<sup>m</sup>iss" --
    markup damage on a string that had no markup. So conversion is opt-in on
    evidence of LaTeX, not applied to everything with a caret in it.
    """
    return "$" in text or "\\" in text


def to_html(text: str) -> str:
    """One sentence, LaTeX and all, as HTML safe to drop into a page.

    Escaped FIRST, so a paper's own `<` in "120 < m < 135" cannot become a tag,
    and only then are our own sub/sup tags introduced.
    """
    if not text:
        return ""
    out = html.escape(text)

    for pattern, repl in _NOISE:
        out = pattern.sub(repl, out)
    for _ in range(4):                      # \text{\mathrm{x}} nests a little
        out = _flatten_braces(out)
        new = _UNWRAP.sub(r"\1", out)
        if new == out:
            break
        out = new
    out = _flatten_braces(out)
    # SYMBOLS BEFORE ACCENTS. The other way round, `\tilde{\chi}` puts the
    # combining mark on the backslash and renders as "\̃chi" -- the accent
    # handler takes the first CHARACTER of the body, and the body still began
    # with a command.
    for command, char in sorted(_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(command, char)
    out = _UNWRAP_BARE.sub("", out)
    out = _accents(out)
    out = _scripts(out)

    out = out.replace("$", "")
    # An unrecognised command keeps its NAME and loses its backslash, so
    # `\mathcal{Q}` reads "Q" rather than "\mathcalQ". Dropping the name too
    # would delete content the paper actually contains; keeping the backslash
    # shows a reader typesetting they cannot act on.
    out = re.sub(r"\\([a-zA-Z]+)\s*\{([^{}]*)\}", r"\2", out)
    out = re.sub(r"\\([a-zA-Z]+)", r"\1", out)
    # Braces that survived carried grouping, not content.
    out = re.sub(r"[{}]", "", out)

    # PARAGRAPH BREAKS SURVIVE. The value rows put our answer inside the
    # question -- "...cut retain?\n\nWE ANSWER: ...\n\nIs that correct?" -- and
    # collapsing every run of whitespace would fuse the three parts into one
    # paragraph, which is the readability problem this file exists to fix.
    # Quotes are already single-line by then, so they are unaffected.
    out = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*", "\x00\x00", out)
    out = re.sub(r"[ \t]*\n[ \t]*", "\x00", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = out.replace("\x00\x00", "<br><br>").replace("\x00", "<br>")
    return out.strip()
