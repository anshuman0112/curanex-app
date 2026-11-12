// Doctor module API (doctor-only). Every call sends the doctor key; the server refuses without it (403).
import { API_URL } from './api';

export type Source = { id: string; source: string; title: string; section: string; url: string; licence: string };
export type Patient = {
  age?: number; sex?: string; pregnant?: boolean; weight_kg?: number; kidney_disease?: boolean; liver_disease?: boolean;
  allergies?: string[]; current_medicines?: string[];
};
export type Case = { patient: Patient; symptoms: string; history?: string; report?: Record<string, string> };
export type TestsReply = { emergency: string | null; missing_info: string[]; tests: { test: string; why: string; sources: Source[] }[]; dropped: number; note?: string };
export type DiffReply = {
  emergency: string | null; missing_info: string[]; dropped: number; note?: string;
  supported: { disease: string; support: number; evidence: { text: string; sources: Source[] }[] }[]; also_consider: string[];
};
export type LabelCard = {
  medicine: string; found: boolean; note: string; label_title?: string; label_url?: string; label_date?: string; warnings?: string[];
  sections?: { section: string; text: string }[];
};
export type TreatReply = {
  disease: string; emergency: string | null; missing_info: string[]; dropped: number; note?: string;
  plan: { text: string; sources: Source[] }[]; referral: { text: string; sources: Source[] }[];
  medicines: { why: string; sources: Source[]; label: LabelCard }[];
};

async function call<T>(path: string, key: string, body: unknown): Promise<T> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 240000); // library answers take up to a few minutes on the free AI tier
  try {
    const r = await fetch(`${API_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Doctor-Key': key }, body: JSON.stringify(body), signal: ctl.signal });
    if (r.status === 403) throw new Error('Doctor access only. Check the doctor key.');
    if (!r.ok) throw new Error(`The server could not answer (HTTP ${r.status}).`);
    return (await r.json()) as T;
  } finally { clearTimeout(t); }
}

export const suggestTests = (key: string, c: Case) => call<TestsReply>('/doctor/tests', key, c);
export const findDiseases = (key: string, c: Case) => call<DiffReply>('/doctor/differential', key, c);
export const planTreatment = (key: string, c: Case, disease: string) => call<TreatReply>('/doctor/treatment', key, { case: c, disease });
