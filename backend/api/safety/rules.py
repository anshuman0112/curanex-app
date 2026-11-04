"""Deterministic red-flag rules. Runs before any AI and never calls an LLM or a model."""
import re

RED_FLAGS: list[tuple[str, str]] = [
    (r"(chest|heart).{0,40}(pain|pian|pressure|tight|crush|hurt).{0,60}(arm|jaw|neck|back|sweat|breath)|(pain|pressure|tight\w*).{0,30}chest.{0,60}(arm|jaw|neck|back|sweat|breath)|(severe|intense|terrible|unbearable).{0,20}chest pain|crushing chest|heart attack|seene (mein|me|main).{0,20}dard", "Possible heart attack"),
    (r"(face|mouth).{0,20}droop|slurred speech|can'?t (move|feel) (my )?(arm|leg|side)|(arm|leg|face|side).{0,20}(went|gone|going) (weak|numb)|stroke", "Possible stroke"),
    (r"worst headache|thunderclap headache", "Sudden severe headache"),
    (r"can'?t breathe|cannot breathe|struggling to breathe|not breathing|stopped breathing|can'?t catch my breath|choking|lips (are )?(turning )?blue|sa+ns nahi", "Severe breathing difficulty"),
    (r"unconscious|passed out|not responding|unresponsive|fainted|collapsed|(won'?t|not) wak(e|ing) up|behosh", "Loss of consciousness"),
    (r"(heavy|severe|won'?t stop|uncontrolled) bleeding|bleeding (a lot|heavily|won'?t stop|will not stop|isn'?t stopping)|blood.{0,20}(pouring|gushing|spurting|won'?t stop)|(vomit\w*|cough\w*|throw\w* up).{0,20}blood|khoon.{0,10}(ruk nahi|nahi ruk)", "Severe bleeding"),
    (r"seizure|convulsion|(having|getting|got) fits", "Seizure"),
    (r"(kill|end|hurt) (myself|my life)|suicid|don'?t want to (live|be alive)|(want|wanna|going) to die|wanna die|better off dead|end(ing)? it all|no reason to (go on|live|keep)|marna chah|mar jana chah|jeena nahi chah|khudkushi|aatmahatya", "Thoughts of self-harm"),
    (r"overdose|took too many (pills|tablets)|poison(ed|ing)?\b|(drank|drunk|swallowed|ate) .{0,15}(bleach|pesticide|acid|phenyl)|\bze(he|ha)r\b", "Possible overdose or poisoning"),
]


def red_flag(text: str) -> str | None:
    for pattern, reason in RED_FLAGS:
        if re.search(pattern, text, re.I):
            return reason
    return None
