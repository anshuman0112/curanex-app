/**
 * On-device red-flag check. Deterministic: no AI. The backend runs the same
 * idea server-side; this copy makes escalation work with no network.
 */
const RULES: [RegExp, string][] = [
  [/(chest|heart).{0,40}(pain|pressure|tight|crush).{0,60}(arm|jaw|neck|back|sweat|breath)|crushing chest|heart attack/i, 'Possible heart attack'],
  [/(face|mouth).{0,20}droop|slurred speech|can'?t (move|feel) (my )?(arm|leg|side)|stroke/i, 'Possible stroke'],
  [/can'?t breathe|cannot breathe|struggling to breathe|choking|lips (are )?(turning )?blue/i, 'Severe breathing difficulty'],
  [/unconscious|passed out|not responding|unresponsive|fainted and/i, 'Loss of consciousness'],
  [/(heavy|severe|won'?t stop|uncontrolled) bleeding|bleeding (a lot|heavily)/i, 'Severe bleeding'],
  [/seizure|convulsion|fits/i, 'Seizure'],
  [/(kill|end|hurt) (myself|my life)|suicid|don'?t want to (live|be alive)/i, 'Thoughts of self-harm'],
  [/overdose|took too many (pills|tablets)|poison(ed|ing)/i, 'Possible overdose or poisoning'],
];

export function redFlag(text: string): string | null {
  for (const [re, reason] of RULES) if (re.test(text)) return reason;
  return null;
}
