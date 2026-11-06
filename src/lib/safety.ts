/**
 * On-device red-flag check. Deterministic: no AI. The backend runs the same
 * idea server-side; this copy makes escalation work with no network.
 */
const RULES: [RegExp, string][] = [
  [/(chest|heart).{0,40}(pain|pian|pressure|tight|crush|hurt).{0,60}(arm|jaw|neck|back|sweat|breath)|(pain|pressure|tight\w*).{0,30}chest.{0,60}(arm|jaw|neck|back|sweat|breath)|(severe|intense|terrible|unbearable).{0,20}chest pain|crushing chest|heart attack|seene (mein|me|main).{0,20}dard|सीने में .{0,20}दर्द|दिल का दौरा/i, 'Possible heart attack'],
  [/(face|mouth).{0,20}droop|slurred speech|can'?t (move|feel) (my )?(arm|leg|side)|(arm|leg|face|side).{0,20}(went|gone|going) (weak|numb)|stroke|लकवा|मुँह टेढ़ा|मुंह टेढ़ा/i, 'Possible stroke'],
  [/worst headache|thunderclap headache/i, 'Sudden severe headache'],
  [/can'?t breathe|cannot breathe|struggling to breathe|not breathing|stopped breathing|can'?t catch my breath|choking|lips (are )?(turning )?blue|sa+ns nahi|साँस नहीं|सांस नहीं|साँस लेने में .{0,15}(तकलीफ|दिक्कत)|सांस लेने में .{0,15}(तकलीफ|दिक्कत)/i, 'Severe breathing difficulty'],
  [/unconscious|passed out|not responding|unresponsive|fainted|collapsed|(won'?t|not) wak(e|ing) up|behosh|बेहोश/i, 'Loss of consciousness'],
  [/(heavy|severe|won'?t stop|uncontrolled) bleeding|bleeding (a lot|heavily|won'?t stop|will not stop|isn'?t stopping)|blood.{0,20}(pouring|gushing|spurting|won'?t stop)|(vomit\w*|cough\w*|throw\w* up).{0,20}blood|khoon.{0,10}(ruk nahi|nahi ruk)|खून .{0,12}(नहीं रुक|रुक नहीं)/i, 'Severe bleeding'],
  [/seizure|convulsion|(having|getting|got) fits|मिर्गी|दौरा पड़/i, 'Seizure'],
  [/(kill|end|hurt) (myself|my life)|suicid|don'?t want to (live|be alive)|(want|wanna|going) to die|wanna die|better off dead|end(ing)? it all|no reason to (go on|live|keep)|marna chah|mar jana chah|jeena nahi chah|khudkushi|aatmahatya|मरना चाहत|मर जाना चाहत|जीना नहीं चाहत|आत्महत्या|खुदकुशी/i, 'Thoughts of self-harm'],
  [/overdose|took too many (pills|tablets)|poison(ed|ing)?\b|(drank|drunk|swallowed|ate) .{0,15}(bleach|pesticide|acid|phenyl)|\bze(he|ha)r\b|ज़हर|जहर/i, 'Possible overdose or poisoning'],
];

export function redFlag(text: string): string | null {
  for (const [re, reason] of RULES) if (re.test(text)) return reason;
  return null;
}
