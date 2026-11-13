import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { redFlag } from './safety';

/**
 * Talks to the Python (FastAPI) backend. Set EXPO_PUBLIC_API_URL in .env
 * (e.g. http://192.168.1.20:8000 when testing on a phone on the same Wi-Fi).
 * If the backend is unreachable, the app falls back to an OFFLINE DEMO that
 * shows the UI flows with clearly-labelled sample output.
 */
const CONFIGURED = process.env.EXPO_PUBLIC_API_URL ?? (Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000');
// Opened from another device (e.g. an iPhone on the same Wi-Fi/hotspot at http://<PC-IP>:8081): "localhost" would mean the phone,
// so talk to the same computer that served the page.
const pageHost = Platform.OS === 'web' && typeof window !== 'undefined' ? window.location.hostname : '';
export const API_URL = pageHost && pageHost !== 'localhost' && /\/\/(localhost|127\.0\.0\.1)/.test(CONFIGURED)
  ? CONFIGURED.replace(/\/\/(localhost|127\.0\.0\.1)/, `//${pageHost}`)
  : CONFIGURED;

// ---- /chat "blocks" contract (ARCHITECTURE.md §6) ----
export type FormField = { key: string; label: string; unit?: string; type: 'number' | 'text'; optional?: boolean; options?: string[] };
export type Block =
  | { type: 'text'; text: string }
  | { type: 'escalate'; reason: string }
  | { type: 'questions'; questions: { q: string; options: string[] }[] }
  | { type: 'form_request'; model_id: string; title: string; fields: FormField[]; prefilled?: Record<string, string | number>; defaults?: Record<string, string | number> }
  | { type: 'upload_request'; model_id: string; kind: 'skin_photo' | 'dental_photo' | 'chest_xray' | 'knee_xray' | 'bone_xray' | 'brain_mri' | 'blood_cell' | 'endoscopy' | 'ct_scan' | 'fundus' | 'report' | 'ultrasound' | 'dermoscopy'; instructions: string }
  | { type: 'care_plan'; urgency: 'self_care' | 'doctor_soon' | 'doctor_today' | 'emergency'; urgency_label: string; causes: { name: string; why: string; likelihood: string }[];
      tests: { name: string; why: string }[]; medicines: { name: string; use: string; caution: string }[]; medicine_note?: string | null; self_care: string[]; see_doctor_if: string[];
      sources: { title: string; url: string }[]; basis: string }
  | { type: 'model_result'; model_id: string; model_name: string; label: string; probabilities: Record<string, number>; trained_on: string; limitations: string; demo?: boolean; flag?: 'green' | 'amber';
      heatmap_url?: string | null; warnings?: string[]; credit?: string; verified?: boolean; explanation?: { heatmap_finding?: string; flagged?: string[]; top_factors?: [string, number][] } };

export type ChatRequest = {
  session_id: string; message: string; context: unknown;
  history: { role: 'user' | 'assistant'; content: string }[];
  form_values?: Record<string, string> | null; image_id?: string | null; model_id?: string | null; via?: 'text' | 'voice'; report_text?: string | null;
};
export type ChatResponse = { blocks: Block[]; demo?: boolean };

// A model run plus the explanation can take most of a minute on a laptop; a shorter limit would look like being offline.
/** The logged-in patient's token (set on the Record screen). When present, the server saves chats, uploads and results to that account. */
async function auth(): Promise<Record<string, string>> {
  try { const t = await AsyncStorage.getItem('curanex.session.patient'); return t ? { Authorization: `Bearer ${t}` } : {}; } catch { return {}; }
}

async function post<T>(path: string, body: unknown, timeoutMs = 60000): Promise<T> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(await auth()) }, body: JSON.stringify(body), signal: ctl.signal });
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

export type ExperimentalModel = { id: string; title: string; labels: string[]; test: { n: number; accuracy: number; auc: number }; notice: string };
export type ExperimentalResult = { label: string; top: [string, number][]; notice: string };

export async function fetchExperimental(): Promise<ExperimentalModel[]> {
  try { const r = await fetch(`${API_URL}/experimental`); return r.ok ? ((await r.json()) as { models: ExperimentalModel[] }).models : []; } catch { return []; }
}

export async function runExperimental(id: string, uri: string): Promise<ExperimentalResult> {
  const image_id = await uploadImage(uri);
  return post<ExperimentalResult>(`/experimental/${id}`, { image_id });
}

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  // Safety runs on-device first too, so escalation works even with no network.
  const flag = req.message ? redFlag(req.message) : null;
  if (flag) return { blocks: [{ type: 'escalate', reason: flag }] };
  try { return await post<ChatResponse>('/chat', req); }
  catch { return { ...offlineDemo(req), demo: true }; }
}

export async function uploadImage(uri: string): Promise<string> {
  let rejected = '';
  try {
    const form = new FormData();
    if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, 'upload.jpg'); }
    else form.append('file', { uri, name: 'upload.jpg', type: 'image/jpeg' } as unknown as Blob);
    const r = await fetch(`${API_URL}/upload`, { method: 'POST', body: form, headers: await auth() });
    if (r.status === 415) rejected = ((await r.json()) as { detail: string }).detail;
    else if (r.ok) return ((await r.json()) as { image_id: string }).image_id;
  } catch { /* backend unreachable: the offline demo takes over below */ }
  if (rejected) throw new Error(rejected);
  return 'demo-image';
}

/** Reads a PDF or photographed report on the backend. Throws a message the screen can show. */
export async function readReport(uri: string, name: string, mime: string): Promise<string> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, name); }
  else form.append('file', { uri, name, type: mime } as unknown as Blob);
  let r: Response;
  try { r = await fetch(`${API_URL}/report`, { method: 'POST', body: form, headers: await auth() }); }
  catch { throw new Error('Reading a report needs the Curanex server, which is not connected.'); }
  if (!r.ok) throw new Error(r.status === 415 ? ((await r.json()) as { detail: string }).detail : 'That report could not be read. Please try again.');
  return ((await r.json()) as { text: string }).text;
}

/** Sends a recording to the backend's speech-to-text. Throws when it can't be understood, so the screen can say so. */
export async function transcribe(uri: string): Promise<{ text: string; language: string | null }> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, 'voice.webm'); }
  else form.append('file', { uri, name: 'voice.m4a', type: 'audio/m4a' } as unknown as Blob);
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 30000);
  try {
    const r = await fetch(`${API_URL}/transcribe`, { method: 'POST', body: form, signal: ctl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as { text: string; language: string | null };
  } finally { clearTimeout(t); }
}

export type VoiceCheckResult = {
  label: string; flagged: boolean; score: number; measures: Record<string, string>; notice: string; trained_on: string; credit: string;
  test: { test_people: { n: number; sensitivity: number; specificity: number; auc: number } };
};

/** Experimental voice health check. Throws a message the screen can show (bad recording, server missing). */
export async function voiceCheck(uri: string, sex: string): Promise<VoiceCheckResult> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, 'voice.webm'); }
  else form.append('file', { uri, name: 'voice.m4a', type: 'audio/m4a' } as unknown as Blob);
  form.append('sex', sex);
  let r: Response;
  try { r = await fetch(`${API_URL}/voice-check`, { method: 'POST', body: form }); }
  catch { throw new Error('The voice check needs the Curanex server, which is not connected.'); }
  if (r.status === 422 || r.status === 413) { const d = ((await r.json()) as { detail: unknown }).detail; throw new Error(typeof d === 'string' ? d : 'That recording could not be used. Please record again.'); }
  if (!r.ok) throw new Error('The voice check failed. Please try again.');
  return (await r.json()) as VoiceCheckResult;
}

export type AreaAlert = {
  district: string; place: string; weather: { temperature_c: number; humidity_pct: number; rain_4_weeks_mm: number; until: string };
  mosquito_weather: boolean; headline: string; advice: string[]; notice: string; source: string;
  estimate: { level: string; experimental: true; accuracy_note: string; trained_on: string } | null;
};

/** Dengue-season weather for a district. Returns a message string when there is nothing to show. */
export async function fetchAlert(district: string): Promise<AreaAlert | string> {
  try {
    const r = await fetch(`${API_URL}/alerts?district=${encodeURIComponent(district)}`);
    if (r.ok) return (await r.json()) as AreaAlert;
    const d = ((await r.json()) as { detail: unknown }).detail;
    return typeof d === 'string' ? d : 'Local alerts are not available right now.';
  } catch { return 'Local alerts need the Curanex server, which is not connected.'; }
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

export type Identified =
  | { route: 'report'; text: string; values: Record<string, number>; kind?: string }
  | { route: 'image'; image_id: string; kind: string; body_part: string; findings: string[]; model_id: string | null };

/** Upload anything (report, X-ray, ECG, MRI, scan, photo): the server works out what it is and which check fits. */
export async function identifyUpload(uri: string, name: string, mime: string): Promise<Identified> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, name); }
  else form.append('file', { uri, name, type: mime } as unknown as Blob);
  let r: Response;
  try { r = await fetch(`${API_URL}/v1/identify`, { method: 'POST', body: form, headers: await auth() }); }
  catch { throw new Error('Checking a file needs the Curanex server, which is not connected.'); }
  if (!r.ok) throw new Error(r.status === 415 ? ((await r.json()) as { detail: string }).detail : 'That file could not be read. Please try again.');
  return (await r.json()) as Identified;
}

// ---- Nutrition ----
export type Nutrients = Record<'kcal' | 'protein_g' | 'carbs_g' | 'fat_g' | 'fibre_g' | 'sugar_g' | 'iron_mg' | 'calcium_mg' | 'vitamin_c_mg' | 'vitamin_a_ug' |
  'vitamin_b12_ug' | 'vitamin_d_ug' | 'folate_ug' | 'sodium_mg' | 'potassium_mg', number>;
export type FoodItem = Nutrients & { name: string; portion: string; grams: number };
export type FoodResult = { is_food: boolean; items: FoodItem[]; totals: Partial<Nutrients>; confidence?: string; note?: string; error?: string };
export type Targets = { kcal: number; protein_g: number; carbs_g: number; fat_g: number; fibre_g: number; iron_mg: number; calcium_mg: number; vitamin_c_mg: number; vitamin_a_ug: number; vitamin_b12_ug: number; sodium_mg: number; source: string };
export type PlanMeal = { time: string; name: string; why: string; kcal: number; protein: number; carbs: number; fat: number };
export type MealPlan = { targets: Targets; meals: PlanMeal[]; tips: string[]; error: string | null };

/** Food photo -> dishes, portions and nutrients (AI estimate). Throws a message the screen can show. */
export async function analyseFood(uri: string): Promise<FoodResult> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, 'meal.jpg'); }
  else form.append('file', { uri, name: 'meal.jpg', type: 'image/jpeg' } as unknown as Blob);
  let r: Response;
  try { r = await fetch(`${API_URL}/v1/food-photo`, { method: 'POST', body: form, headers: await auth() }); }
  catch { throw new Error('Reading a food photo needs the Curanex server, which is not connected.'); }
  if (!r.ok) throw new Error(r.status === 415 ? ((await r.json()) as { detail: string }).detail : 'That photo could not be read. Please try again.');
  return (await r.json()) as FoodResult;
}

export async function getMealPlan(profile: unknown): Promise<MealPlan> {
  return post<MealPlan>('/v1/meal-plan', { profile }, 120000);
}

export async function getTargets(profile: unknown): Promise<Targets | null> {
  try { return await post<Targets>('/v1/nutrition/targets', { profile }, 15000); } catch { return null; }
}

// ---- Smartwatch ----
export type WatchReply = { days: import('../store/health').DaySummary[]; sources: string[]; via: string; latest: string | null;
  alerts: import('../store/health').WatchAlert[] };

/** iPhone: the Health app's export.zip (watch app -> Apple Health). Big files: up to 10 minutes. */
export async function uploadHealthExport(uri: string, name: string, mime: string): Promise<WatchReply> {
  const form = new FormData();
  if (Platform.OS === 'web') { const blob = await (await fetch(uri)).blob(); form.append('file', blob, name); }
  else form.append('file', { uri, name, type: mime } as unknown as Blob);
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 600000);
  let r: Response;
  try { r = await fetch(`${API_URL}/v1/wearables/apple-health`, { method: 'POST', body: form, headers: await auth(), signal: ctl.signal }); }
  catch { throw new Error('Reading your health export needs the Curanex server, which is not connected.'); }
  finally { clearTimeout(t); }
  if (!r.ok) throw new Error(r.status === 415 ? ((await r.json()) as { detail: string }).detail : 'That file could not be read. Please try again.');
  return (await r.json()) as WatchReply;
}

/** Android: daily summaries read from Health Connect on the phone. */
export async function syncWatchDays(days: import('../store/health').DaySummary[], apps: string[]): Promise<WatchReply> {
  return post<WatchReply>('/v1/wearables/sync', { days, apps, source: 'Health Connect' }, 60000);
}
