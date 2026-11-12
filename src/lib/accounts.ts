// Accounts and health records (doctor module). Each area keeps its own session: patient record, doctor portal, admin.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useEffect, useState } from 'react';
import { API_URL } from './api';

export type User = { id: number; email: string; name: string; role: 'patient' | 'doctor' | 'admin'; status: string; details?: Record<string, string> };
export type DoctorPlan = { doctor_id: number; doctor_name: string; confirmed: number; diagnosis: string; tests: string[]; medicines: string[]; advice: string; referral?: string | null };
export type RecordCase = {
  id: number; title: string; created: number; symptoms: string; history?: string | null; report?: Record<string, string> | null;
  patient?: Record<string, unknown> | null; shared_with: { id: number; name: string; email: string }[]; doctor_plans: DoctorPlan[];
};

export async function api<T>(path: string, token: string | null, method = 'GET', body?: unknown): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, {
    method, headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : `Request failed (HTTP ${r.status}).`);
  return data as T;
}

/** A remembered login for one area of the app ("patient", "doctor" or "admin"). */
export function useSession(area: string) {
  const key = `curanex.session.${area}`;
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    AsyncStorage.getItem(key).then(async t => {
      if (t) {
        try { setUser(await api<User>('/auth/me', t)); setToken(t); } catch { await AsyncStorage.removeItem(key).catch(() => {}); }
      }
    }).catch(() => {}).finally(() => setReady(true));
  }, [key]);
  const save = async (t: string, u: User) => { setToken(t); setUser(u); await AsyncStorage.setItem(key, t).catch(() => {}); };
  const logout = async () => { if (token) await api('/auth/logout', token, 'POST').catch(() => {}); setToken(null); setUser(null); await AsyncStorage.removeItem(key).catch(() => {}); };
  const refresh = async () => { if (token) setUser(await api<User>('/auth/me', token)); };
  return { token, user, ready, save, logout, refresh };
}

export const login = (email: string, password: string) => api<{ token: string; user: User }>('/auth/login', null, 'POST', { email, password });
export const signup = (b: { email: string; password: string; name: string; role: 'patient' | 'doctor'; clinic?: string; specialty?: string; phone?: string }) =>
  api<{ token: string; user: User }>('/auth/signup', null, 'POST', b);
