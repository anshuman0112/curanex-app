"""Deterministic red-flag rules. Runs before any AI and never calls an LLM or a model."""
import re

RED_FLAGS: list[tuple[str, str]] = [
    (r"(chest|heart).{0,40}(pain|pressure|tight|crush).{0,60}(arm|jaw|neck|back|sweat|breath)"
     r"|crushing chest|heart attack", "Possible heart attack"),
    (r"(face|mouth).{0,20}droop|slurred speech|can'?t (move|feel) (my )?(arm|leg|side)|stroke", "Possible stroke"),
    (r"can'?t breathe|cannot breathe|struggling to breathe|choking|lips (are )?(turning )?blue", "Severe breathing difficulty"),
    (r"unconscious|passed out|not responding|unresponsive", "Loss of consciousness"),
    (r"(heavy|severe|won'?t stop|uncontrolled) bleeding|bleeding (a lot|heavily)", "Severe bleeding"),
    (r"seizure|convulsion", "Seizure"),
    (r"(kill|end|hurt) (myself|my life)|suicid|don'?t want to (live|be alive)", "Thoughts of self-harm"),
    (r"overdose|took too many (pills|tablets)|poison(ed|ing)", "Possible overdose or poisoning"),
]


def red_flag(text: str) -> str | None:
    for pattern, reason in RED_FLAGS:
        if re.search(pattern, text, re.I):
            return reason
    return None
