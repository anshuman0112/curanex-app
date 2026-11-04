import { Platform } from 'react-native';
import { redFlag } from './safety';

/**
 * Talks to the Python (FastAPI) backend. Set EXPO_PUBLIC_API_URL in .env
 * (e.g. http://192.168.1.20:8000 when testing on a phone on the same Wi-Fi).
 * If the backend is unreachable, the app falls back to an OFFLINE DEMO that
 * shows the UI flows with clearly-labelled sample output.
 */
export const API_URL =
  process.env.EXPO_PUBLIC_API_URL ?? (Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000');

// ---- /chat "blocks" contract (ARCHITECTURE.md §6) ----
export type FormField = { key: string; label: string; unit?: string; type: 'number' | 'text'; optional?: boolean };
export type Block =
  | { type: 'text'; text: string }
  | { type: 'escalate'; reason: string }
  | { type: 'form_request'; model_id: string; title: string; fields: FormField[]; prefilled?: Record<string, string | number> }
  | { type: 'upload_request'; model_id: string; kind: 'skin_photo' | 'dental_photo' | 'chest_xray' | 'knee_xray' | 'fundus' | 'report'; instructions: string }
  | { type: 'model_result'; model_id: string; model_name: string; label: string; probabilities: Record<string, number>; trained_on: string; limitations: string; demo?: boolean;
      heatmap_url?: string | null; warnings?: string[]; credit?: string; verified?: boolean; explanation?: { heatmap_finding?: string; flagged?: string[]; top_factors?: [string, number][] } };

export type ChatRequest = {
  session_id: string; message: string; context: unknown;
  history: { role: 'user' | 'assistant'; content: string }[];
  form_values?: Record<string, string> | null; image_id?: string | null; model_id?: string | null;
};
export type ChatResponse = { blocks: Block[]; demo?: boolean };

async function post<T>(path: string, body: unknown, timeoutMs = 20000): Promise<T> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as T;
  } finally { clearTimeout(t); }
}

export async function health(): Promise<boolean> {
  try { const ctl = new AbortController(); setTimeout(() => ctl.abort(), 2500); const r = await fetch(`${API_URL}/health`, { signal: ctl.signal }); return r.ok; } catch { return false; }
}

/** Live registry for the Model Library. Null when the backend can't be reached (the screen then uses the saved list). */
export async function fetchModels<T>(): Promise<T[] | null> {
  try {
    const ctl = new AbortController(); setTimeout(() => ctl.abort(), 6000);
    const r = await fetch(`${API_URL}/models`, { signal: ctl.signal });
    if (!r.ok) return null;
    return ((await r.json()) as { models: T[] }).models;
  } catch { return null; }
}

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  // Safety runs on-device first too, so escalation works even with no network.
  const flag = req.message ? redFlag(req.message) : null;
  if (flag) return { blocks: [{ type: 'escalate', reason: flag }] };
  try { return await post<ChatResponse>('/chat', req); }
  catch { return { ...offlineDemo(req), demo: true }; }
}

export async function uploadImage(uri: string): Promise<string> {
  try {
    const form = new FormData();
    if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, 'upload.jpg'); }
    else form.append('file', { uri, name: 'upload.jpg', type: 'image/jpeg' } as unknown as Blob);
    const r = await fetch(`${API_URL}/upload`, { method: 'POST', body: form });
    if (!r.ok) throw new Error();
    return ((await r.json()) as { image_id: string }).image_id;
  } catch { return 'demo-image'; }
}

// ---------------- Offline demo (UI preview only) ----------------
const DEMO_NOTE = 'Offline demo: the AI backend is not connected, so this is sample output to preview the flow.';

function offlineDemo(req: ChatRequest): ChatResponse {
  const m = (req.message || '').toLowerCase();
  if (req.model_id && (req.form_values || req.image_id)) {
    const isImg = !!req.image_id;
    return { blocks: [
      { type: 'model_result', model_id: req.model_id, demo: true,
        model_name: isImg ? 'Chest X-ray screen (torchxrayvision)' : 'Type 2 diabetes risk',
        label: isImg ? 'No strong finding' : 'Elevated risk',
        probabilities: isImg ? { 'No finding': 0.71, Pneumonia: 0.12, Effusion: 0.09, Other: 0.08 } : { 'Lower risk': 0.34, 'Elevated risk': 0.66 },
        trained_on: isImg ? 'Public chest X-ray datasets' : 'Pima Indians Diabetes dataset',
        limitations: isImg ? 'Phone photos of X-ray film reduce accuracy. Not validated for clinical use.' : 'Trained on a small, non-Indian population. Not a diagnosis.' },
      { type: 'text', text: `${DEMO_NOTE}\n\nWith the real backend, the AI doctor explains what this estimate means for you, what it cannot tell, and which test confirms it (for example an HbA1c test).` },
    ] };
  }
  if (/(thirst|urinat|pee a lot|sugar|diabet)/.test(m)) return { blocks: [
    { type: 'text', text: 'Frequent thirst and urination have several possible causes, and blood sugar is one worth checking. A quick risk check can help decide what to test next.' },
    { type: 'form_request', model_id: 'moizah1_diabetes_prediction', title: 'Quick diabetes risk check',
      fields: [{ key: 'glucose', label: 'Fasting glucose', unit: 'mg/dL', type: 'number' }, { key: 'bp', label: 'Blood pressure (diastolic)', unit: 'mmHg', type: 'number', optional: true }],
      prefilled: { age: (req.context as any)?.profile?.age ?? '—', bmi: (req.context as any)?.profile?.bmi ?? '—' } },
  ] };
  if (/(x-?ray|chest scan|lung)/.test(m)) return { blocks: [
    { type: 'text', text: 'I can run a screening model on a chest X-ray image. It flags possible findings for you to discuss with a doctor; it cannot diagnose.' },
    { type: 'upload_request', model_id: 'mlmed_torchxrayvision', kind: 'chest_xray', instructions: 'Upload the original image file if you have it. If you photograph the film, hold it flat against a bright screen.' },
  ] };
  if (/(rash|skin|itch|acne|eczema|patch)/.test(m)) return { blocks: [
    { type: 'text', text: 'A photo helps here. The skin model can suggest which common condition it resembles, and I will tell you when it needs a dermatologist.' },
    { type: 'upload_request', model_id: 'smon2006_skin_disease_detection', kind: 'skin_photo', instructions: 'Daylight, 10–15 cm away, the patch in focus and centred.' },
  ] };
  if (/(tooth|teeth|cavity|gum)/.test(m)) return { blocks: [
    { type: 'upload_request', model_id: 'andreygermanov_yolov8_caries_detector', kind: 'dental_photo', instructions: 'Open wide in good light and photograph the painful area.' },
  ] };
  return { blocks: [{ type: 'text', text: `${DEMO_NOTE}\n\nTry: "I'm always thirsty", "Here is my chest X-ray", "I have a rash on my arm", or "crushing chest pain spreading to my left arm".` }] };
}
