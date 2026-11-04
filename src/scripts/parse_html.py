#!/usr/bin/env python3
"""parser_html.py - Phase 1: parsing the consolidated AI Act (EUR-Lex).

Reads the HTML downloaded from EUR-Lex (.html or .html.txt), isolates
<div id="document1"> and produces the hierarchical structure:
chapters -> sections -> articles -> paragraphs, plus the annexes.
Tracks the amendment markers (▼M1/▼B) in amendments and deletions.
No external dependencies.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_HTML = DATA_RAW / "32024R1689_consolidated_2026-07-27.html.txt"

VOID = {"br", "hr", "meta", "link", "img", "col", "input", "area", "base"}
INLINE = {"span", "a", "i", "b", "em", "strong", "sup", "sub", "u", "small"}
JUNK = re.compile(r"^[;.,\s]*$")
RE_ART = re.compile(r"^art_(\d+)([a-z]?)$")
RE_CH = re.compile(r"^cpt_([IVXL]+)$")
RE_SEC = re.compile(r"^cpt_([IVXL]+)\.sct_(\d+)$")
RE_ANX = re.compile(r"^anx_([IVXL]+)$")
RE_POINT = re.compile(r"^\(\d+[a-z]*\)$")
RE_FN = re.compile(r"^E\d+$")


class Node:
    __slots__ = ("attrs", "children", "tag")

    def __init__(self, tag, attrs=None):
        self.tag, self.attrs, self.children = tag, dict(attrs or {}), []

    @property
    def cls(self):
        return set((self.attrs.get("class") or "").split())

    @property
    def id(self):
        return self.attrs.get("id") or ""

    def el(self):
        return [c for c in self.children if isinstance(c, Node)]

    def iter(self, tag=None):
        for c in self.children:
            if isinstance(c, Node):
                if tag is None or c.tag == tag:
                    yield c
                yield from c.iter(tag)

    def find(self, pred):
        return next((n for n in self.iter() if pred(n)), None)

    def child(self, tag=None, cls=None):
        return next(
            (
                c
                for c in self.el()
                if (tag is None or c.tag == tag) and (cls is None or cls in c.cls)
            ),
            None,
        )

    def text(self):
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)


class Build(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].children.append(n)
        if tag not in VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def dom(html):
    b = Build()
    b.feed(html)
    b.close()
    return b.root


def norm(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


class Ctx:
    """Tracks the current ▼M1/▼B marker and collects amendments/deletions."""

    def __init__(self):
        self.state = None
        self.seen = set()
        self.container = None
        self.last_label = None
        self.doc_deletions = []

    def marker(self, node: Node):
        a = node.find(lambda n: n.tag == "a" and n.attrs.get("title"))
        title = (a.attrs.get("title") if a else "") or ""
        anchor = node.find(lambda n: n.tag == "a" and "anchorarrow" in n.cls)
        m = re.match(r"([A-Z]+\d*)-", (anchor.attrs.get("id") if anchor else "") or "")
        marker = m.group(1) if m else None
        act, _, status = title.partition(":")
        act, status = act.strip(), status.strip()
        if not status:
            self.state = None
        elif status == "DELETED":
            rec = {"marker": marker, "act": act, "status": "DELETED", "after": self.last_label}
            tgt = self.container if self.container is not None else self.doc_deletions
            if isinstance(tgt, list):
                tgt.append(rec)
            else:
                tgt.setdefault("deletions", []).append(rec)
        else:
            self.state = (marker, act, status)

    def begin(self):
        self.seen = set()

    def touch(self):
        self.seen.add(self.state)

    def summary(self):
        amend = sorted(
            (s for s in self.seen if s is not None), key=lambda s: (s[0] or "", s[1], s[2])
        )
        return (
            [{"marker": m, "act": a, "status": st} for m, a, st in amend],
            bool(amend) and None in self.seen,
        )


def lines(node: Node, ctx: Ctx, depth=0) -> list[str]:
    out, buf = [], []
    ind = "  " * depth

    def flush():
        t = norm("".join(buf))
        buf.clear()
        if t and not JUNK.match(t):
            ctx.touch()
            out.append(ind + t)

    for ch in node.children:
        if isinstance(ch, str):
            buf.append(ch)
            continue
        c = ch.cls
        if "modref" in c:
            flush()
            ctx.marker(ch)
        elif ch.tag in INLINE:
            buf.append(ch.text())
        elif ch.tag == "br":
            flush()
        elif ch.tag == "table":
            flush()
            for tr in ch.iter("tr"):
                row = " | ".join(
                    norm(td.text()) for td in tr.el() if td.tag in ("td", "th") and norm(td.text())
                )
                if row:
                    ctx.touch()
                    out.append(ind + row)
        elif "grid-container" in c:
            flush()
            lab = ch.child(cls="grid-list-column-1")
            body = ch.child(cls="grid-list-column-2")
            label = norm(lab.text()) if lab else ""
            sub = lines(body, ctx, depth + 1) if body else []
            if sub:
                sub[0] = f"{ind}{label} {sub[0].lstrip()}".rstrip()
            elif label:
                ctx.touch()
                sub = [ind + label]
            out.extend(sub)
        elif ch.tag in ("hr", "script", "style"):
            flush()
        else:
            flush()
            out.extend(lines(ch, ctx, depth))
    flush()
    return out


def parse_article(div: Node, ctx: Ctx) -> dict:
    m = RE_ART.match(div.id)
    art = {
        "id": div.id,
        "number": (m.group(1) + m.group(2)) if m else None,
        "base_number": int(m.group(1)) if m else None,
        "suffix": (m.group(2) if m else ""),
        "title": "",
        "paragraphs": [],
    }
    ctx.container, ctx.last_label = art, None
    art_seen, paras, has_num = set(), [], False

    def new_para(num, txt, seen):
        paras.append({"number": num, "text_lines": list(txt), "_seen": set(seen)})
        ctx.last_label = num

    for ch in div.el():
        c = ch.cls
        if "modref" in c:
            ctx.marker(ch)
        elif "eli-title" in c:
            ctx.begin()
            art["title"] = " ".join(l.strip() for l in lines(ch, ctx))
            art_seen.update(ctx.seen)
        elif ch.tag == "div" and "norm" in c and ch.child("span", "no-parag"):
            has_num = True
            label = norm(ch.child("span", "no-parag").text())
            body = next((x for x in ch.el() if x.tag == "div"), None)
            ctx.begin()
            txt = lines(body, ctx) if body else []
            mm = re.search(r"(\d+[a-z]*)", label)
            new_para(mm.group(1) if mm else None, txt, ctx.seen)
        else:
            ctx.begin()
            txt = lines(ch, ctx)
            if not txt:
                continue
            label = (
                norm(ch.child(cls="grid-list-column-1").text()) if "grid-container" in c else None
            )
            if label and RE_POINT.match(label) and not has_num:
                new_para(label.strip("()"), txt, ctx.seen)
            elif paras:
                paras[-1]["text_lines"] += txt
                paras[-1]["_seen"].update(ctx.seen)
            else:
                new_para(None, txt, ctx.seen)
    for i, p in enumerate(paras, 1):
        art_seen.update(p["_seen"])
        amend, mixed = ctx.summary()
        art["paragraphs"].append(
            {
                "id": f"{art['id']}.p{p['number'] or 'u' + str(i)}",
                "number": p["number"],
                "text": "\n".join(p["text_lines"]),
                "amendments": amend,
                "mixed": mixed,
            }
        )
    art["amendments"], art["mixed"] = ctx.summary()
    art.pop("deletions", None)
    ctx.container = None
    return art


def parse_annex(div: Node, ctx: Ctx) -> dict:
    m = RE_ANX.match(div.id)
    an = {"id": div.id, "number": m.group(1) if m else None, "title": "", "blocks": []}
    ctx.container, ctx.last_label = an, None
    seen_all, titles = set(), []
    for ch in div.el():
        c = ch.cls
        if "modref" in c:
            ctx.marker(ch)
        elif c & {"title-annex-1", "title-annex-2"}:
            t = norm(ch.text())
            if an["number"] is None and "title-annex-1" in c:
                an["number"] = re.sub(r"^ANNEX\s*", "", t)
            else:
                titles.append(t)
        elif ch.tag == "hr" or (ch.tag == "p" and not norm(ch.text())):
            continue
        else:
            ctx.begin()
            txt = lines(ch, ctx)
            if txt:
                amend, mixed = ctx.summary()
                seen_all.update(ctx.seen)
                an["blocks"].append(
                    {
                        "n": len(an["blocks"]) + 1,
                        "heading": bool(c & {"title-gr-seq-level-1", "title-gr-seq-level-2"}),
                        "text": "\n".join(txt),
                        "amendments": amend,
                        "mixed": mixed,
                    }
                )
                ctx.last_label = len(an["blocks"])
    an["title"] = " - ".join(titles)
    an["amendments"], an["mixed"] = ctx.summary()
    ctx.container = None
    return an


def _titles(div: Node):
    p1, p2 = div.child("p", "title-division-1"), div.child("p", "title-division-2")
    num = re.sub(r"^(CHAPTER|SECTION)\s*", "", norm(p1.text()) if p1 else "", flags=re.I)
    return num, norm(p2.text()) if p2 else ""


def parse_document(html: str, sha="", source="") -> dict:
    root = dom(html)
    doc_node = root.find(lambda n: n.id == "document1")
    if doc_node is None:
        raise SystemExit("ERROR: div#document1 not found")
    ctx = Ctx()
    doc = {
        "source": {"file": source, "sha256": sha},
        "title": "",
        "amended_by": [],
        "chapters": [],
        "annexes": [],
        "concluding_formula": "",
        "footnotes": {},
        "deletions": [],
        "warnings": [],
    }
    canon = root.find(lambda n: n.tag == "link" and n.attrs.get("rel") == "canonical")
    if canon:
        doc["source"]["canonical_url"] = canon.attrs.get("href")
    tit = root.find(lambda n: n.id == "tit_1")
    if tit:
        doc["title"] = " ".join(norm(p.text()) for p in tit.iter("p"))
    ref = root.find(lambda n: "reference" in n.cls)
    if ref:
        doc["source"]["reference"] = norm(ref.text())
    for tr in root.iter("tr"):
        a = tr.find(lambda n: n.tag == "a" and n.attrs.get("title", "").count(":"))
        if a and tr.find(lambda n: "title-fam-member-star" in n.cls):
            cells = [norm(td.text()) for td in tr.el() if td.tag == "td"]
            tail = [c for c in cells if c][-3:]
            doc["amended_by"].append(
                {
                    "celex": a.attrs["title"],
                    "title": norm(a.text()),
                    "oj": tail[0] if len(tail) == 3 else None,
                    "page": tail[1] if len(tail) == 3 else None,
                    "date": tail[2] if len(tail) == 3 else None,
                }
            )
    st = {"ch": None, "sec": None}

    def walk(node):
        for ch in node.el():
            cid, c = ch.id, ch.cls
            if "modref" in c:
                ctx.marker(ch)
            elif ch.tag == "div" and RE_CH.match(cid):
                num, title = _titles(ch)
                st["ch"] = {
                    "id": cid,
                    "number": num,
                    "title": title,
                    "sections": [],
                    "articles": [],
                }
                st["sec"] = None
                doc["chapters"].append(st["ch"])
                walk(ch)
            elif ch.tag == "div" and RE_SEC.match(cid) and st["ch"]:
                num, title = _titles(ch)
                st["sec"] = {"id": cid, "number": num, "title": title, "articles": []}
                st["ch"]["sections"].append(st["sec"])
                walk(ch)
            elif ch.tag == "div" and RE_ART.match(cid):
                if not st["ch"]:
                    doc["warnings"].append(f"{cid}: article outside a chapter")
                    continue
                (st["sec"] or st["ch"])["articles"].append(parse_article(ch, ctx))
            elif ch.tag == "div" and RE_ANX.match(cid):
                st["ch"] = st["sec"] = None
                doc["annexes"].append(parse_annex(ch, ctx))
            elif cid == "fnp_1":
                doc["concluding_formula"] = " ".join(lines(ch, ctx))
            elif "footnote" in c:
                a = ch.find(lambda n: n.tag == "a" and RE_FN.match(n.attrs.get("id", "")))
                if a:
                    doc["footnotes"][a.attrs["id"]] = norm(ch.text())
            else:
                walk(ch)

    walk(doc_node)
    doc["deletions"] = ctx.doc_deletions
    n_bad = html.count("\ufffd")
    if n_bad:
        doc["warnings"].append(f"{n_bad} U+FFFD characters in the source: check the encoding.")
    n_art = sum(
        len(c["articles"]) + sum(len(s["articles"]) for s in c["sections"]) for c in doc["chapters"]
    )
    doc["stats"] = {
        "chapters": len(doc["chapters"]),
        "articles": n_art,
        "annexes": len(doc["annexes"]),
    }
    exp = {"chapters": 13, "articles": 119, "annexes": 14}
    if doc["stats"] != exp:
        doc["warnings"].append(f"unexpected stats: {doc['stats']} (expected: {exp}).")
    return doc



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parse the consolidated AI Act (EUR-Lex HTML).")
    ap.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_HTML,
        help=f"HTML file (default: {DEFAULT_HTML})",
    )
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args(argv)
    out = a.output or DATA_PROCESSED / "ai_act.json"
    if str(a.output or "") == "-":
        out = None
    if not a.input.exists():
        ap.error(f"file not found: {a.input}")
    raw = a.input.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("utf-8", errors="replace")
    doc = parse_document(html, sha, a.input.name)
    payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    print(f"sha256={sha}  stats={doc['stats']}", file=sys.stderr)
    for w in doc["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0