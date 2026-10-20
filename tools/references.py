"""Incorporation by reference: list the documents a contract points to,
count them, and (optionally) measure their size as archived on the
contract's effective date.

Typical use:

    from tools.references import extract_references, extract_corpus_references
    refs = extract_references(text, platform='fb')                  # one contract
    refs, contracts = extract_corpus_references('data/tous_data')   # whole corpus

    # optional, needs the network (Wayback Machine):
    refs, sizes, summary = analyse_contract_references(
        'data/tous_data/facebook/fb_2020_10_01.txt', measure=True,
        cache_dir='processed_data/references_cache')

The rules below were written after reading a sample of three contracts per
platform (earliest, middle and latest version) and revised after checking a
second, random sample of three further contracts per platform. The
extraction is rule-based, so it errs in known ways:
  * a named document is only found when it is written in Title Case
    ('Community Guidelines') or belongs to the lexicon of common document
    types (KNOWN_DOCUMENTS, matched in any case: 'our privacy policy');
  * references to the contract itself ('these Terms', 'this Agreement') and
    to its own parts ('Arbitration Agreement', 'Special State Terms',
    'This license') are dropped;
  * ALL-CAPS phrases are dropped unless they are a known document type,
    because in this corpus they are almost always disclaimers or headings.

The text files in data/tous_data have no hyperlinks, so URLs come from two
places: URLs written out in the contract itself and the hand-maintained
REFERENCE_URLS registry. Documents without a URL get status 'no_url' when
measured, so the registry can be extended.
"""
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Hand-maintained URL registry: platform -> document key -> URL.
# Platform keys follow the filename prefix (fb_2020_10_01.txt -> 'fb'); document
# keys are the `key` column of extract_references. Check new entries against
# the Wayback Machine before relying on them.
REFERENCE_URLS = {
    "fb": {
        "data policy": "https://www.facebook.com/policy.php",
        "community standards": "https://www.facebook.com/communitystandards/",
        "commercial terms": "https://www.facebook.com/legal/commercial_terms",
        "advertising policies": "https://www.facebook.com/policies/ads/",
        "self-serve ad terms": "https://www.facebook.com/legal/self_service_ads_terms",
        "pages, groups and events policy": "https://www.facebook.com/policies/pages_groups_events/",
        "facebook platform policy": "https://developers.facebook.com/policy/",
        "developer payment terms": "https://www.facebook.com/payments_terms/developer",
        "community payment terms": "https://www.facebook.com/payments_terms",
        "commerce policies": "https://www.facebook.com/policies/commerce",
        "music guidelines": "https://www.facebook.com/legal/music_guidelines_fb.php",
        "brand usage guidelines": "https://en.facebookbrand.com/guidelines/brand",
    },
    "google": {
        "privacy policy": "https://policies.google.com/privacy",
        "service-specific additional terms": "https://policies.google.com/terms/service-specific",
        "copyright help center": "https://support.google.com/legal/answer/3463239",
        "safety center": "https://safety.google/",
    },
    "twitter": {
        "privacy policy": "https://twitter.com/en/privacy",
        "rules and policies": "https://help.twitter.com/en/rules-and-policies",
    },
}

# Names a platform uses for itself, keyed by filename prefix. Used to drop
# self-references ('WhatsApp Terms of Service' in a WhatsApp contract) and to
# merge 'Grindrs Privacy Notice' with 'privacy notice'. The platform key is
# always included.
OWN_BRANDS = {
    "fb": ["facebook", "meta"], "twitter": ["twitter", "x"], "wechat": ["wechat", "weixin"],
    "myspace": ["myspace", "myspace.com", "myspace.com.s"], "match": ["match", "match.com", "matchcom"],
    "twitch": ["twitch", "justin.tv"], "snapchat": ["snap", "snapchat", "snap group limited"],
    "bluesky": ["bluesky", "bluesky social"], "truthsocial": ["truth social"], "imo": ["imo"],
    "likee": ["likee", "bigo"], "zoom": ["zoom", "zoom voice communications, inc."],
    "linkedin": ["linkedin"], "meetup": ["meetup"], "onlyfans": ["onlyfans"],
}
# Companies in the same group: their documents count as 'affiliate' rather than 'third_party'.
AFFILIATES = {
    "fb": ["instagram", "whatsapp", "messenger", "threads", "oculus"],
    "instagram": ["facebook", "meta", "threads"], "threads": ["instagram", "meta", "facebook"],
    "whatsapp": ["facebook", "meta"], "youtube": ["google"], "google": ["youtube"],
    "venmo": ["paypal"], "twitch": ["amazon"], "tiktok": ["bytedance", "capcut"],
    "capcut": ["bytedance", "tiktok"], "wechat": ["tencent"], "match": ["tinder", "hinge", "match group"],
    "tinder": ["match group", "match"], "hinge": ["match group", "match"], "bumble": ["badoo"],
    "linkedin": ["slideshare", "microsoft"], "twitter": ["periscope", "vine"],
}
# Other companies whose documents contracts cite; together with the platform
# names (folder names and OWN_BRANDS) they mark a document as someone else's.
OTHER_BRANDS = ["google", "youtube", "twitter", "apple", "amazon", "paypal", "stripe", "synchrony", "bancorp",
                "automattic", "key-systems", "bytedance", "tencent", "microsoft", "runway", "tmg", "dosh",
                "american express", "visa", "mastercard"]
THIRD_PARTY = re.compile(
    r"\b(?:aaa'?s?|american arbitration|jams'?s?|nams?|adr services|siac|lcia|hkiac|icc|"
    r"singapore international|hong kong international|"
    r"apple|app store|google play|amazon|appstore|gnu|creative commons|mit|"
    r"apache|creativeml|openrail|runway|federal trade|ftc|card scheme|payment processors?|registrar|"
    r"domain registration|third[- ]party|app distributors?|optional rules|supplementary procedures)\b")

# Head words that end a document name. Strong heads name a document on their
# own ('Community Guidelines'); weak heads ('Agreement', 'Notice', 'License')
# usually name a part of the contract, so they need a QUALIFIER.
STRONG_HEADS = (r"terms(?: of (?:service|use|sale))?(?: and conditions)?|polic(?:y|ies)|guidelines|standards|"
                r"rules(?: (?:and|&|&amp;) procedures)?|principles|code of conduct|faqs?|eula|charter")
WEAK_HEADS = (r"agreements?|conditions|licen[cs]es?|notices?|statements?|addend(?:um|a)|"
              r"disclosures?|guides?|cent(?:er|re)|procedures|program(?:me)?s?")
# Heads after which a run-on heading may start a new name ('Privacy Policy Acceptable Use Policy');
# not 'Centre' ('Hong Kong International Arbitration Centre Administered Arbitration Rules').
SPLIT_HEADS = STRONG_HEADS + r"|agreements?|addend(?:um|a)|statements?|notices?"
HEADS = STRONG_HEADS + "|" + WEAK_HEADS
QUALIFIER = re.compile(
    r"\b(?:privacy|cookies?|developer|api|publisher|registrar|registration agreement|cardholder|card|credit|wallet|"
    r"commercial|master services|stored credential|billing|software|materials|merchant|seller|"
    r"data processing|data protection|gnu|general public|creative commons|mit|apache|openrail|open source|"
    r"help|safety|transparency|e-sign|electronic (?:communications|disclosures?)|operating|"
    r"business accounts?|services? agreement|licen[cs]e agreement|\w+ user (?:service )?agreement|"
    r"copyright cent(?:er|re)|disclosures guide|endorsement guides?|supplementary procedures|"

    r"mastercard|american express|domain|consumer rights|vat|tax)\b")

# Programs are documents only when they come with terms of their own.
PROGRAM_QUALIFIER = re.compile(
    r"\b(?:partner|referral|vulnerability reporting|bug bounty|purchase protection|seller protection|purchase|"
    r"rewards|loyalty|creator|ambassador|monetization|beta) program")

# Title-Case names: up to eight capitalised words (lower-case connectors allowed) and a head word.
_WORD = r"[A-Z][\w'&.-]*"
_JOIN = r"(?:,? (?:and|&|of|for|on|to|the|in) |, | )"
NAMED_DOC = re.compile(r"\b((?:" + _WORD + _JOIN + r"){0,8}(?i:" + HEADS + r"))\b")
# Rules named after their institution: 'Arbitration Rules of the Singapore International Arbitration Centre'.
RULES_OF = re.compile(r"\b((?:[A-Z][\w'-]* ){0,3}(?:Rules|Procedures) of (?:the )?"
                      r"(?=(?:[A-Z][\w'-]* ){0,5}(?:Arbitration|Association|Cent(?:er|re)|Chamber|Institute|"
                      r"Mediation|Forum|JAMS|ADR))(?:[A-Z][\w'-]*(?: |$|(?=[,.;)]))){2,7})")

# Common document types, matched in any case ('our privacy policy').
KNOWN_DOCUMENTS = re.compile(
    r"\b(privacy (?:policy|notice|statement)|data policy|cookies? (?:policy|notice)|"
    r"community (?:guidelines|standards|rules)|acceptable use policy|content (?:policy|standards)|"
    r"copyright policy|intellectual property policy|dmca policy|trademark policy|brand guidelines|"
    r"usage rules|help cent(?:er|re)|safety cent(?:er|re)|code of conduct|refund policy|"
    r"cancellation policy|payment terms|terms of sale|developer (?:agreement|policy|terms)|"
    r"api terms(?: of use)?|advertising (?:policies|guidelines)|ads? policies|commercial terms|"
    r"branded content policy|music guidelines|card scheme rules|supplementary procedures for "
    r"consumer[- ]related disputes)\b", re.I)

# Open-ended references: they name no particular document.
GENERIC_DOC = re.compile(
    r"\b(service[- ]specific (?:additional )?terms|supplemental terms(?: of service)?|"
    r"additional terms(?: and (?:conditions|policies))?|other terms(?: and (?:conditions|policies))?|"
    r"other (?:policies|rules|guidelines|agreements)|incorporated policies|other conditions or policies|"
    r"documents? (?:that may be )?posted|other documents expressly incorporated|"
    r"(?:our|its) (?:other )?(?:policies|rules|guidelines))\b", re.I)
GENERIC_NAME = re.compile(
    r"^(?:(?:any|all|such|other|additional|supplemental|separate|applicable|relevant|specific|"
    r"service[- ]specific|product[- ]specific|our|its)\s+)*"
    r"(?:(?:(?:additional|supplemental|service[- ]specific|other|separate|incorporated|specific)\s+)"
    r"(?:terms|policies|rules|guidelines|conditions|agreements?|documents)"
    r"(?:\s+(?:and|or)\s+(?:conditions|policies|guidelines|rules))?(?: of (?:service|use))?|"
    r"(?:policy|policies|rules|guidelines|standards|principles)"
    r"(?:\s+(?:and|or)\s+(?:policies|guidelines|rules))?|terms and policies)$")

# The contract itself, or one of its own sections.
SELF_NAME = re.compile(
    r"^(?:(?:updated|revised|new|amended|modified|final|general|legal|current|complete|whole|following|"
    r"applicable|entire|binding|these|this|the|our|your|previous|prior|key|other legal|miscellaneous|"
    r"site|website|service|legally binding|"
    r"archive of (?:previous|prior)|summary of key)\s+)*"
    r"(?:terms|terms of (?:service|use)|terms and conditions(?: of use)?|terms of (?:service|use) agreement|"
    r"(?:user |customer |member )?agreement|conditions(?: of use)?|statement of rights and responsibilities|"
    r"eula|end user license agreement|user terms|legal terms|terms of service and terms|agreements|term, terms)$")
INTERNAL_NAME = re.compile(
    r"^(?:(?:special|specific|country|state|special state|special country|dispute|final|basic|arbitration|"
    r"limited|content|subscriber|user content|ip|scope of|duration of|transfer of|your|our)\s+)+"
    r"(?:terms|rules|license|agreement|notice)$|"
    r"^(?:changes?|updates?|agreement to|acceptance|ability|survival|applicability|scope|duration|transfer|"
    r"termination|modifications?|amendments?|usage of|dispute|eligibility|age|interpretation|ending|"
    r"severability|relation|capitalized|waiver|resolving|excluded|limitation|governing|jurisdiction|"
    r"class action|table of contents|use restrictions|that may apply|notification|mass filings?|"
    r"terms of (?:use|service) for|"
    r"arbitration procedures)\b")
# Section-heading words that glue onto a following name ('Eligibility Account and Registration Licenses',
# 'Violate SlideShare's Community Guidelines'): the name starts after the last of them.
HEADING_WORDS = {"prohibitions", "termination", "eligibility", "capacity", "contents", "documents",
                 "dos", "donts", "waiver", "conducting", "violate", "violates", "using", "accessing", "users",
                 "deactivation", "ownership"}

URL = re.compile(r"https?://[^\s)\]]+")
LEADING_NOISE = re.compile(
    r"^(?:(?:Please|Read|Review|Learn|More|About|See|If|You|We|In|By|Additionally|Besides|For|And|"
    r"Under|Accessing|Welcome|Note|Also|Unless|When|With|Together)\s+)+")
NAVIGATION = re.compile(r"\b(?:Welcome to|Go to|Back to|Please|Table of Contents|PDF|Read|See|Review)\s+")
CONNECTOR = re.compile(r"^(?:(?:and|&|of|for|on|to|the|in),?\s+)+")
DETERMINER = re.compile(r"\b(?:the|The|THE|These|THESE|This|THIS|Our|OUR|Your|YOUR|Its|Any|All|Such|Those)\s+")

BINDING_CUES = re.compile(
    r"(?:terms|policies|guidelines|rules) apply|part of (?:these terms|this agreement|our agreement|the user agreement|"
    r"this user agreement)|incorporat|made a part|you agree|you accept|agree to be bound|must (?:comply|follow|agree|abide)|"
    r"governed by|govern your|subject to|comprises|applies? to your use|become part|you must|apply to you|"
    r"abide by|in accordance with", re.I)
INCORPORATION_CUES = re.compile(r"incorporat|made a part|become part|form(?:s)? part|part of (?:these|this|our|the)", re.I)
CONDITIONAL_CUES = re.compile(r"\b(?:if you|when you|depending on|apply (?:if|when)|to the extent you|"
                              r"in connection with your use of (?:those|these|such))\b", re.I)
INFORMATIVE_CUES = re.compile(r"learn more|explains|describes|for more information|resources like|"
                              r"\bsee\b|to answer common questions|further information|you can find", re.I)
PRECEDENCE_CUES = re.compile(r"prevail|take precedence|shall govern|will govern|govern(?:s)? (?:with respect|to the extent)|"
                             r"override|control(?:s)? (?:to the extent|in the event)", re.I)
AMENDABLE_CUES = re.compile(r"as amended|from time to time|as updated|may be (?:updated|amended|modified|changed|revised)|"
                            r"current version|we may (?:update|change|modify|revise)", re.I)

CATEGORIES = [
    ("privacy", r"privacy|cookie|data polic|data processing|data protection|data use|personal information|"
                r"ccpa|consumer rights"),
    ("dispute resolution", r"arbitra|\baaa|\bjams|\bnams?\b|adr services|dispute resolution|mass filing|mediation|"
                           r"\bsiac\b|\blcia\b|\bicc\b|optional rules|supplementary procedures|consumer[- ]related disputes"),
    ("law enforcement & compliance", r"law enforcement|government|sanction|security|data request|transparency notice|"
                                     r"vulnerability|bug bounty"),
    ("ai features", r"\bai\b|generative|chatbot|avatar|machine learning"),
    ("app stores & third-party services", r"apple|app store|appstore|google play|amazon|third[- ]party|"
                                          r"payment processor|registrar|registration agreement|app distributor"),
    ("intellectual property", r"copyright|trademark|dmca|\bbrand\b|intellectual property|\bip\b|takedown|"
                              r"infring|music"),
    ("developer & api", r"developer|\bapi\b|^platform (?:polic|terms|application)|sdk|embed"),
    ("advertising & branded content", r"advertis|\bads?\b|branded|sponsor|promotion|campaign|endorsement|ftc|federal trade"),
    ("commerce & payments", r"payment|purchas|refund|cancellation|\bsale\b|subscription|virtual|billing|card|wallet|"
                            r"merchant|seller|buyer|commerc|paid|credit|\btax\b|\bvat\b|fee|pricing|stash|business|"
                            r"electronic disclosure|e-sign|money|rewards|monetization|earn|econ|charity|crypto|referral|"
                            r"\bach\b|nacha|operating guide|partner program|loyalty"),
    ("community & conduct", r"communit|acceptable use|content|usage|conduct|safety|guidelines|rules|standards|"
                            r"moderator|profile|fair us|reasonable use|spam|manipulation|appeal|complaint|reporting|"
                            r"username|termination|strike|recommendation|principles|inactive|groups?\b|events|\blive\b|"
                            r"organizer|hosts?\b"),
    ("open source & licences", r"open source|gnu|general public|creative commons|\bmit\b|apache|openrail|licen"),
    ("help & information", r"help|faq|guide|center|transparency"),
    ("other terms & conditions", r"terms(?: of (?:service|use))?$|terms and conditions$"),
]


def split_sentences(text: str) -> list:
    """Split contract text into sentences on terminal punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text) if s]


def parse_contract_filename(fn: str) -> tuple:
    """Return (platform, effective date) from a name like fb_2020_10_01.txt.

    Unlike utils.parse_filename this keeps the day, which matters when
    picking the closest Wayback snapshot.
    """
    el = Path(fn).stem.split("_")
    parts = (el[1:] + ["01", "01"])[:3]
    return el[0], datetime.strptime("".join(parts), "%Y%m%d")


def _brands(platform: str) -> list:
    names = set(OWN_BRANDS.get(platform, [])) | {platform}
    return sorted(names, key=len, reverse=True)


@lru_cache(maxsize=None)
def _all_brands() -> tuple:
    names = set(OTHER_BRANDS) | set(OWN_BRANDS) | {b for v in OWN_BRANDS.values() for b in v} \
        | {b for v in AFFILIATES.values() for b in v} | set(PLATFORMS)
    names -= {"x", "fb", "match", "snap", "imo"}  # too ambiguous inside a sentence
    return tuple(sorted(names, key=len, reverse=True))


# Folder names in data/tous_data, so brand names of every platform are recognised.
PLATFORMS = ["bereal", "bluesky", "bumble", "capcut", "discord", "faceapp", "facebook", "gab", "gettr", "google",
             "grindr", "hinge", "imo", "instagram", "likee", "linkedin", "match", "medium", "meetup", "myspace",
             "nextdoor", "onlyfans", "patreon", "pinterest", "reddit", "signal", "snapchat", "substack", "telegram",
             "threads", "tiktok", "tinder", "truthsocial", "tumblr", "twitch", "twitter", "venmo", "viber",
             "wechat", "whatsapp", "youtube", "zoom"]


def _brand_prefix(key: str, brands: list) -> str:
    """The brand a lower-cased name starts with ('grindrs privacy notice' -> 'grindr'), or None."""
    for brand in brands:
        if re.match(re.escape(brand) + r"(?:'?s|\.s)?,?\s", key):
            return brand
    return None


def _is_brand_start(key: str) -> bool:
    return _brand_prefix(key, _all_brands()) is not None


def _normalise(name: str) -> str:
    key = name.lower().replace("'", "").replace("’", "")
    key = re.sub(r"\blicence", "license", key)
    key = re.sub(r"\bcentre\b", "center", key)
    key = re.sub(r"\bprogramme", "program", key)
    key = re.sub(r"\bcookies\b", "cookie", key)
    key = re.sub(r"\b(aaa|jams|nam|ftc)s\b", r"\1", key)  # possessives without apostrophe: 'AAAs rules'
    key = re.sub(r"\brules (?:and|&|&amp;) procedures\b", "rules", key)  # 'JAMS Streamlined Arbitration Rules & Procedures'
    key = re.sub(r"\b(agreement|license|addend|guide|notice|statement|disclosure)(?:s|a)$",
                 lambda m: m.group(1) + ("um" if m.group(1) == "addend" else ""), key)
    return re.sub(r"\s+", " ", key).strip(" .,;:")


def _strip_brand(key: str, brands: list) -> tuple:
    """Remove a leading own-brand name ('grindrs privacy notice' -> 'privacy notice').

    Returns (key without brand, whether a brand was removed).
    """
    for brand in brands:
        m = re.match(re.escape(brand) + r"(?:s|\.s)?,?\s+(.*)$", key)
        if m:
            return m.group(1), True
    return key, False


def _strip_prefix(raw: str) -> str:
    """Drop sentence and heading text in front of a name."""
    raw = LEADING_NOISE.sub("", raw.strip(" ,"))
    # Keep what follows the last determiner: 'Miscellaneous These Terms' -> 'Terms'.
    raw = DETERMINER.split(raw)[-1]
    raw = CONNECTOR.sub("", raw)
    # ... and what follows the last heading word, if two or more words follow it
    # ('Eligibility Account and Registration Licenses'; but 'Account Termination Policy' stays).
    words = raw.split()
    cut = max((i for i, w in enumerate(words[:-2]) if w.lower().strip(",") in HEADING_WORDS), default=-1)
    return CONNECTOR.sub("", " ".join(words[cut + 1:]))


def _clean_candidate(raw: str) -> list:
    """Turn a raw Title-Case match into zero or more candidate names."""
    tokens = raw.split()
    # Drop a leading run of two or more ALL-CAPS words (a section heading).
    caps = 0
    while caps < len(tokens) and tokens[caps].isupper() and len(tokens[caps]) > 1:
        caps += 1
    if 2 <= caps < len(tokens):
        raw = " ".join(tokens[caps:])
    # Navigation text glued to a name: 'Go to Medium Policy', 'Welcome to Bumbles Terms'.
    raw = NAVIGATION.split(raw)[-1]
    # 'Terms of Service, the Privacy Policy' -> two names; 'Terms of Service and Privacy Policy' too,
    # but not 'Rules and Policies' or 'Pages, Groups and Events Policy'.
    ends_in_head = lambda p: re.search(r"(?i:" + HEADS + r")$", p) is not None
    parts = [p for p in re.split(r",\s*(?:and\s+)?", raw) if p.strip()]
    if len(parts) > 1 and not all(ends_in_head(p) for p in parts):
        parts = [raw]
    names = []
    for part in parts:
        part = _strip_prefix(part)
        halves = re.split(r" (?i:and) ", part, maxsplit=1)
        if len(halves) == 2 and ends_in_head(halves[0]) and ends_in_head(halves[1]) and len(halves[1].split()) >= 2:
            names += [_strip_prefix(h) for h in halves]
        else:
            names.append(part)
    # Headings run together: 'Privacy Policy Acceptable Use Policy' -> two names.
    split = []
    for name in names:
        split += re.sub(r"\b((?i:" + SPLIT_HEADS + r"))\s+(?=[A-Z])(?!(?:And|Or|Of|For|On|To)\b)",
                        "\\1\x00", name).split("\x00")
    # A brand after the first word starts the name: 'Privacy Reddits Privacy Policy'.
    out = []
    for name in split:
        words = name.split()
        start = next((i for i in range(1, len(words)) if _is_brand_start(" ".join(words[i:]).lower())), 0)
        out.append(" ".join(words[start:]).strip(" ,"))
    return [n for n in out if n]


def _classify_name(name: str, platform: str, from_lexicon: bool) -> tuple:
    """Return (key, kind) where kind is 'named', 'generic' or None (drop)."""
    if name.isupper() and not KNOWN_DOCUMENTS.fullmatch(name.lower()) and not from_lexicon:
        return None, None
    key = _normalise(name)
    bare, had_brand = _strip_brand(key, _brands(platform))
    if SELF_NAME.match(bare) or INTERNAL_NAME.search(bare):
        return None, None
    if GENERIC_NAME.match(key):
        return re.sub(r"^(?:(?:any|all|such|our|its|the)\s+)+", "", key), "generic"
    if RULES_OF.fullmatch(name) or (from_lexicon and KNOWN_DOCUMENTS.fullmatch(name) and not had_brand):
        return key, "named"
    other_brand = None if had_brand else _brand_prefix(key, _all_brands())
    head = re.search(r"(?:" + HEADS + r")$", bare)
    if head is None:
        return None, None
    if re.fullmatch(r"programs?|programmes?", head.group(0)):
        if not PROGRAM_QUALIFIER.search(_normalise(bare).replace("programme", "program")):
            return None, None
    elif re.fullmatch(r"(?:" + WEAK_HEADS + r")", head.group(0)) and not QUALIFIER.search(bare) \
            and not KNOWN_DOCUMENTS.fullmatch(bare):
        return None, None
    single_head = bare == head.group(0) and " " not in bare
    if had_brand or other_brand:
        brand = platform if had_brand else other_brand
        rest = bare if had_brand else _strip_brand(key, [other_brand])[0]
        if had_brand and (KNOWN_DOCUMENTS.fullmatch(bare) or " " in bare):
            # 'Reddit Econ Terms' and 'Econ Terms' are the same document; 'Reddit Rules' keeps its brand.
            key = bare
        elif rest == head.group(0) and " " not in rest and name.split()[-1].islower():
            # 'TikToks policy', 'Zoom policies': a lower-case head after a brand is generic.
            return f"{brand} {rest}", "generic"
    elif single_head and bare not in ("faq", "faqs"):
        # A bare head word ('Policies', 'Guidelines') names no particular document; 'FAQ' does.
        return None, None
    return key, "named"


def _source(key: str, platform: str) -> str:
    """'own', 'affiliate' (same corporate group) or 'third_party', from the brand
    the name starts with or, failing that, known third-party names in it."""
    brand = _brand_prefix(key, _all_brands())
    if brand in _brands(platform) or (platform == "fb" and brand == "facebook"):
        return "own"
    if brand in AFFILIATES.get(platform, []):
        return "affiliate"
    if brand is not None or THIRD_PARTY.search(key):
        return "third_party"
    # Arbitration rules are an arbitration provider's rules even when the provider is not named.
    if re.search(r"arbitra|dispute resolution", key) and re.search(r"rules|procedures|minimum standards", key):
        return "third_party"
    return "own"


def _category(key: str, kind: str) -> str:
    if kind == "generic":
        return "open-ended"
    for label, pattern in CATEGORIES:
        if re.search(pattern, key):
            return label
    return "other"


def _sentence_candidates(sent: str, platform: str) -> list:
    """All (start, end, name, from_lexicon) candidates in a sentence, longest span first,
    without overlapping spans."""
    spans = []
    for m in NAMED_DOC.finditer(sent):
        if m.group(1)[0].isupper():
            spans.append((m.start(1), m.end(1), m.group(1), False))
    for m in KNOWN_DOCUMENTS.finditer(sent):
        spans.append((m.start(1), m.end(1), m.group(1), True))
    for m in RULES_OF.finditer(sent):
        spans.append((m.start(1), m.end(1), m.group(1).strip(), True))
    for m in GENERIC_DOC.finditer(sent):
        spans.append((m.start(1), m.end(1), m.group(1), True))
    spans.sort(key=lambda s: (s[0] - s[1], s[0]))
    kept = []
    for span in spans:
        if all(span[1] <= k[0] or span[0] >= k[1] for k in kept):
            kept.append(span)
    return sorted(kept)


def extract_mentions(text: str, platform: str = "") -> pd.DataFrame:
    """One row per mention of a referenced document, with the sentence-level cues."""
    rows = []
    for i, sent in enumerate(split_sentences(text)):
        urls = [(m.start(), m.group(0).rstrip(".,;")) for m in URL.finditer(sent)]
        for start, end, raw, from_lexicon in _sentence_candidates(sent, platform):
            names = [raw] if from_lexicon else _clean_candidate(raw)
            for name in names:
                key, kind = _classify_name(name, platform, from_lexicon)
                if key is None:
                    continue
                # A URL belongs to this document when it follows the name within 120 characters.
                url = next((u for pos, u in urls if 0 <= pos - end < 120), None)
                rows.append({
                    "sentence_id": i, "name": name, "key": key, "kind": kind, "url": url,
                    "binding": bool(BINDING_CUES.search(sent)),
                    "incorporated": bool(INCORPORATION_CUES.search(sent)),
                    "conditional": bool(CONDITIONAL_CUES.search(sent)),
                    "informative": bool(INFORMATIVE_CUES.search(sent)),
                    "precedence": bool(PRECEDENCE_CUES.search(sent)),
                    "amendable": bool(AMENDABLE_CUES.search(sent)),
                    "sentence": sent,
                })
    return pd.DataFrame(rows, columns=["sentence_id", "name", "key", "kind", "url", "binding", "incorporated",
                                       "conditional", "informative", "precedence", "amendable", "sentence"])


REFERENCE_COLUMNS = ["document", "key", "kind", "category", "source", "n_mentions", "binding", "incorporated",
                     "conditional", "informative", "precedence", "amendable", "url", "example"]


def extract_references(text: str, platform: str = "") -> pd.DataFrame:
    """List the documents a contract refers to.

    Arguments:
        text: contract text
        platform: filename prefix of the contract ('fb', 'twitter', ...), used to
            recognise the platform's own name; '' works but keeps more self-references

    Returns one row per referenced document, aggregated over its mentions:
        document      most frequent surface form
        key           normalised name (lower case, own brand removed for common types)
        kind          'named' (a particular document) or 'generic' (open-ended:
                      'additional terms', 'other policies', bare 'Policies')
        category      privacy, community & conduct, intellectual property, ...
                      ('open-ended' for generic references)
        source        'own', 'affiliate' (same corporate group) or 'third_party'
        n_mentions    number of mentions
        binding       a mention uses binding language ('you agree', 'subject to', ...)
        incorporated  a mention says it is incorporated / made part of the contract
        conditional   a mention makes it apply only to some uses ('if you', ...)
        informative   a mention only points to it ('learn more', 'see', ...)
        precedence    a mention says it prevails over the contract
        amendable     a mention says it can change ('as amended', 'from time to time')
        url           first URL written next to it, if any
        example       first sentence mentioning it
    """
    mentions = extract_mentions(text, platform)
    if mentions.empty:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    out = mentions.groupby("key", sort=False).agg(
        document=("name", lambda s: Counter(s).most_common(1)[0][0]),
        kind=("kind", "first"),
        n_mentions=("name", "size"),
        binding=("binding", "any"),
        incorporated=("incorporated", "any"),
        conditional=("conditional", "any"),
        informative=("informative", "any"),
        precedence=("precedence", "any"),
        amendable=("amendable", "any"),
        url=("url", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        example=("sentence", "first"),
    ).reset_index()
    out["category"] = [_category(k, kind) for k, kind in zip(out["key"], out["kind"])]
    out["source"] = [_source(k, platform) for k in out["key"]]
    return out[REFERENCE_COLUMNS]


def extract_corpus_references(data_dir: str, platforms: list = None) -> tuple:
    """Run extract_references over every contract in data_dir/<platform>/*.txt.

    Returns:
        tuple: (refs, contracts)
            refs: one row per (contract, referenced document), the columns of
                extract_references plus contract_id, platform, folder, date, year
            contracts: one row per contract with counts of referenced documents
                (see summarise_contracts); contracts without references are kept
    """
    ref_frames, contract_rows = [], []
    for path in sorted(Path(data_dir).glob("*/*.txt")):
        if platforms and path.parent.name not in platforms:
            continue
        platform, date = parse_contract_filename(path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta = {"contract_id": path.stem, "platform": platform, "folder": path.parent.name,
                "date": date, "year": date.year}
        contract_rows.append({**meta, "contract_words": len(re.findall(r"\w+", text)),
                              "contract_sentences": len(split_sentences(text))})
        refs = extract_references(text, platform)
        if len(refs):
            ref_frames.append(refs.assign(**meta))
    refs = pd.concat(ref_frames, ignore_index=True) if ref_frames else pd.DataFrame(columns=REFERENCE_COLUMNS)
    meta_cols = ["contract_id", "platform", "folder", "date", "year"]
    refs = refs[meta_cols + [c for c in refs.columns if c not in meta_cols]]
    return refs, summarise_contracts(refs, pd.DataFrame(contract_rows))


def summarise_contracts(refs: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """Per-contract counts of referenced documents.

    n_documents counts named documents only; open-ended references are counted
    separately in n_open_ended. The flag counts (n_binding, ...) are over named
    documents, and n_cat_<category> gives the number of named documents per category.
    """
    named = refs[refs["kind"] == "named"]
    g = named.groupby("contract_id")
    counts = pd.DataFrame({
        "n_documents": g.size(),
        "n_binding": g["binding"].sum(),
        "n_incorporated": g["incorporated"].sum(),
        "n_conditional": g["conditional"].sum(),
        "n_precedence": g["precedence"].sum(),
        "n_amendable": g["amendable"].sum(),
        "n_third_party": g["source"].apply(lambda s: (s == "third_party").sum()),
        "n_affiliate": g["source"].apply(lambda s: (s == "affiliate").sum()),
        "n_mentions": g["n_mentions"].sum(),
    })
    generic = refs[refs["kind"] == "generic"].groupby("contract_id")
    counts["n_open_ended"] = generic.size()
    counts["open_ended_precedence"] = generic["precedence"].any()
    by_cat = named.pivot_table(index="contract_id", columns="category", values="key", aggfunc="count")
    by_cat.columns = ["n_cat_" + re.sub(r"\W+", "_", c).strip("_") for c in by_cat.columns]
    out = contracts.set_index("contract_id").join(counts).join(by_cat)
    count_cols = [c for c in out.columns if c.startswith("n_")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["open_ended_precedence"] = out["open_ended_precedence"].fillna(False).astype(bool)
    out["documents_per_1k_words"] = 1000 * out["n_documents"] / out["contract_words"]
    return out.reset_index()


def _get(url: str, params: dict = None, retries: int = 4, pause: float = 5.0) -> requests.Response:
    """GET with linear back-off; None when every attempt failed (the Wayback
    CDX server often times out or throttles under repeated queries)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(pause * (attempt + 1))
    return None


# Domains a platform moved away from: archived copies may sit under the old name.
DOMAIN_ALIASES = {"x.com": "twitter.com", "transparency.meta.com": "transparency.fb.com"}


def candidate_urls(url: str) -> list:
    """The URL without its #fragment, plus the same path on any older domain
    (with and without an /en/ language prefix, which twitter.com used)."""
    url = url.split("#")[0]
    candidates = [url]
    for new, old in DOMAIN_ALIASES.items():
        if re.search(r"//(?:[\w-]+\.)?" + re.escape(new) + r"\b", url):
            moved = re.sub(r"(//(?:[\w-]+\.)?)" + re.escape(new), r"\g<1>" + old, url)
            candidates.append(moved)
            if "/en/" not in moved:
                candidates.append(re.sub(r"(//[^/]+)/", r"\1/en/", moved, count=1))
    return candidates


def find_snapshot(url: str, date: datetime, cache_dir: Path = None) -> tuple:
    """Return (timestamp, original_url) of the Wayback snapshot closest to
    `date` across the candidate URLs, preferring the latest one on or before
    `date` for each candidate, else the earliest one after it; (None, None)
    when nothing is archived.

    Raises LookupError when a CDX query failed and nothing was found, so a
    server time-out is not mistaken for 'not archived'.
    """
    ymd = date.strftime("%Y%m%d")
    if cache_dir is not None:
        cached = Path(cache_dir) / f"cdx_{hashlib.md5(f'{url}{ymd}'.encode()).hexdigest()}.json"
        if cached.exists():
            return tuple(json.loads(cached.read_text()))
    best = (None, None, None)
    failed = False
    for candidate in candidate_urls(url):
        base = {"url": candidate, "output": "json", "fl": "timestamp,original",
                "filter": "statuscode:200", "collapse": "digest"}
        for extra in ({"to": ymd, "limit": -1}, {"from": ymd, "limit": 1}):
            r = _get(WAYBACK_CDX, {**base, **extra})
            failed = failed or r is None
            rows = r.json() if r is not None and r.text.strip() else []
            if len(rows) > 1:
                timestamp, original = rows[1]
                gap = abs((datetime.strptime(timestamp[:8], "%Y%m%d") - date).days)
                if best[0] is None or gap < best[0]:
                    best = (gap, timestamp, original)
                break
    if best[1] is None and failed:
        raise LookupError(f"Wayback CDX lookup failed for {url}")
    if cache_dir is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps([best[1], best[2]]))
    return best[1], best[2]


def html_to_text(html: str) -> str:
    """Visible text of a page, without scripts, styles and site chrome."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return re.sub(r"\s+", " ", main.get_text(" ")).strip()


def fetch_snapshot_text(timestamp: str, original: str, cache_dir: Path = None) -> str:
    """Download an archived page (raw, without the Wayback banner) and return its text."""
    if cache_dir is not None:
        key = hashlib.md5(f"{timestamp}{original}".encode()).hexdigest()
        cached = Path(cache_dir) / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    r = _get(f"https://web.archive.org/web/{timestamp}id_/{original}")
    text = html_to_text(r.text) if r is not None else ""
    if cache_dir is not None and text:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
    return text


def measure_size(text: str) -> dict:
    """Word, sentence and character counts of a text."""
    return {"words": len(re.findall(r"\w+", text)),
            "sentences": len(split_sentences(text)),
            "characters": len(text)}


def measure_references(refs: pd.DataFrame, platform: str, date: datetime,
                       url_registry: dict = None, cache_dir: Path = None,
                       min_words: int = 200, max_gap_days: int = 365,
                       pause: float = 1.0) -> pd.DataFrame:
    """Add archived size to each referenced document.

    URLs written in the contract win over the registry. Added columns:
        url, snapshot, gap_days (snapshot date minus contract date; negative = before),
        words, sentences, characters, status, text.
    status is 'ok', 'no_url', 'not_archived', 'lookup_failed' (Wayback did
    not answer; rerun later, successful lookups are cached), 'stale' (closest snapshot more
    than `max_gap_days` away from the contract date), or 'too_short' (fewer
    than `min_words` words, usually a login wall or a JavaScript-only page).
    """
    registry = (url_registry if url_registry is not None else REFERENCE_URLS).get(platform, {})
    rows = []
    for _, ref in refs.iterrows():
        row = ref.to_dict()
        row["url"] = ref["url"] or registry.get(ref["key"])
        row.update(snapshot=None, gap_days=None, words=None, sentences=None, characters=None, text=None)
        if not row["url"]:
            row["status"] = "no_url"
            rows.append(row)
            continue
        try:
            timestamp, original = find_snapshot(row["url"], date, cache_dir)
        except LookupError:
            row["status"] = "lookup_failed"
            rows.append(row)
            continue
        gap = (datetime.strptime(timestamp[:8], "%Y%m%d") - date).days if timestamp else None
        if timestamp is None or abs(gap) > max_gap_days:
            row.update(snapshot=timestamp, gap_days=gap)
            row["status"] = "not_archived" if timestamp is None else "stale"
            rows.append(row)
            continue
        text = fetch_snapshot_text(timestamp, original, cache_dir)
        row.update(snapshot=timestamp, gap_days=gap, text=text, **measure_size(text))
        row["status"] = "ok" if row["words"] >= min_words else "too_short"
        rows.append(row)
        time.sleep(pause)
    return pd.DataFrame(rows)


def analyse_contract_references(contract: str, platform: str = None, date: datetime = None,
                                measure: bool = False, url_registry: dict = None,
                                cache_dir: str = None) -> tuple:
    """List the documents a contract refers to and, optionally, measure their archived size.

    Arguments:
        contract: path to a contract .txt file (platform and date are read from
            its name) or the contract text itself (then pass platform and date)
        platform: filename prefix, e.g. 'fb'; overrides the filename
        date: effective date of the contract; overrides the filename
        measure: also look up each document in the Wayback Machine and count
            its words (slow; needs the network)
        url_registry: platform -> document key -> URL; defaults to REFERENCE_URLS
        cache_dir: folder for downloaded page texts, so reruns skip the network

    Returns:
        tuple: (refs, sizes, summary)
            refs: one row per referenced document (see extract_references)
            sizes: refs plus URL, snapshot and size columns (see
                measure_references); None unless measure=True
            summary: dict with counts of referenced documents and, when
                measured, the ratio of measured referenced text to contract
                text, overall and for binding, non-conditional documents only
    """
    path = Path(contract)
    if len(contract) < 300 and path.exists():
        text = path.read_text(encoding="utf-8")
        file_platform, file_date = parse_contract_filename(path.name)
        platform, date = platform or file_platform, date or file_date
    else:
        text = contract
    if platform is None or date is None:
        raise ValueError("pass platform and date when giving the contract as text")

    refs = extract_references(text, platform)
    named = refs[refs["kind"] == "named"]
    contract_words = measure_size(text)["words"]
    summary = {
        "platform": platform,
        "date": date.date().isoformat(),
        "contract_words": contract_words,
        "n_documents": len(named),
        "n_open_ended": int((refs["kind"] == "generic").sum()),
        "n_binding": int(named["binding"].sum()),
        "n_incorporated": int(named["incorporated"].sum()),
        "n_precedence": int(refs["precedence"].sum()),
    }
    if not measure:
        return refs, None, summary

    sizes = measure_references(named, platform, date, url_registry, Path(cache_dir) if cache_dir else None)
    ok = sizes[sizes["status"] == "ok"] if len(sizes) else sizes
    baseline = ok[ok["binding"] & ~ok["conditional"]] if len(ok) else ok
    summary.update({
        "n_measured": len(ok),
        "referenced_words": int(ok["words"].sum()) if len(ok) else 0,
        "baseline_binding_words": int(baseline["words"].sum()) if len(baseline) else 0,
    })
    summary["iceberg_ratio"] = summary["referenced_words"] / contract_words
    summary["baseline_iceberg_ratio"] = summary["baseline_binding_words"] / contract_words
    return refs, sizes, summary
