// Android: read the user's smartwatch data from Health Connect, where their watch app (NoiseFit -> Google Fit, Nothing X / CMF Watch,
// Samsung Health, Zepp ...) already writes it. Needs the Curanex Android app build; in Expo Go the native module is missing, so we say so.
import Constants from 'expo-constants';
import type { DaySummary } from '../store/health';

export type LiveRead = { days: DaySummary[]; apps: string[] };

type HC = typeof import('react-native-health-connect');
let hc: HC | null = null;
function load(): HC | null {
  if (hc) return hc;
  if (Constants.executionEnvironment === 'storeClient') return null; // Expo Go has no Health Connect module
  try { hc = require('react-native-health-connect') as HC; } catch { hc = null; }
  return hc;
}

export function liveUnavailableReason(): string | null {
  return load() ? null : 'Live watch sync needs the Curanex Android app (not Expo Go). Install the Curanex APK, then tap Connect.';
}

// Package name of the app that wrote the data -> a name people recognise
const APPS: Record<string, string> = {
  'com.nothing.smartcenter': 'Nothing X', 'com.nothing.cmf.watch': 'CMF Watch', 'com.noisefit': 'NoiseFit', 'com.google.android.apps.fitness': 'Google Fit',
  'com.sec.android.app.shealth': 'Samsung Health', 'com.huami.watch.hmwatchmanager': 'Zepp', 'com.fitbit.FitbitMobile': 'Fitbit', 'com.oneplus.health.international': 'OHealth',
  'com.google.android.apps.healthdata': 'Health Connect',
};
const appName = (pkg: string) => APPS[pkg] ?? pkg.split('.').slice(-2).join('.');

const TYPES = ['Steps', 'HeartRate', 'RestingHeartRate', 'SleepSession', 'OxygenSaturation', 'HeartRateVariabilityRmssd', 'ActiveCaloriesBurned', 'Weight', 'BloodPressure'] as const;
const DAYS = 30;
const dayKey = (iso: string) => { const d = new Date(iso); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; };

export function openHealthConnect(): void { load()?.openHealthConnectSettings(); }

export async function readHealthConnect(): Promise<LiveRead> {
  const H = load();
  if (!H) throw new Error(liveUnavailableReason() ?? 'Health Connect is not available.');
  const status = await H.getSdkStatus();
  if (status !== H.SdkAvailabilityStatus.SDK_AVAILABLE) {
    throw new Error(status === H.SdkAvailabilityStatus.SDK_UNAVAILABLE_PROVIDER_UPDATE_REQUIRED
      ? 'Please update the Health Connect app from the Play Store, then try again.'
      : 'Health Connect is not on this phone. Install "Health Connect" from the Play Store (built in from Android 14).');
  }
  await H.initialize();
  const granted = await H.requestPermission(TYPES.map(recordType => ({ accessType: 'read' as const, recordType })));
  const ok = new Set(granted.map(g => ('recordType' in g ? g.recordType : '')));
  if (!ok.size) throw new Error('No permission given. Tap Connect again and allow Curanex to read your watch data.');

  const start = new Date(); start.setHours(0, 0, 0, 0); start.setDate(start.getDate() - (DAYS - 1));
  const range = { operator: 'between' as const, startTime: start.toISOString(), endTime: new Date().toISOString() };
  const slicer = { duration: 'DAYS' as const, length: 1 };
  const byDay = new Map<string, DaySummary>();
  const row = (iso: string) => { const k = dayKey(iso); if (!byDay.has(k)) byDay.set(k, { date: k }); return byDay.get(k)!; };
  const apps = new Set<string>();
  const safe = async (fn: () => Promise<void>) => { try { await fn(); } catch { /* a type the user did not allow or the watch does not record */ } };

  // Totals: Health Connect removes double counting between phone and watch
  if (ok.has('Steps')) await safe(async () => {
    for (const g of await H.aggregateGroupByDuration({ recordType: 'Steps', timeRangeFilter: range, timeRangeSlicer: slicer })) {
      if (g.result.COUNT_TOTAL) row(g.startTime).steps = g.result.COUNT_TOTAL; g.result.dataOrigins.forEach(o => apps.add(appName(o)));
    }
  });
  if (ok.has('ActiveCaloriesBurned')) await safe(async () => {
    for (const g of await H.aggregateGroupByDuration({ recordType: 'ActiveCaloriesBurned', timeRangeFilter: range, timeRangeSlicer: slicer })) {
      const k = g.result.ACTIVE_CALORIES_TOTAL?.inKilocalories; if (k) row(g.startTime).kcal_active = Math.round(k);
    }
  });
  if (ok.has('HeartRate')) await safe(async () => {
    for (const g of await H.aggregateGroupByDuration({ recordType: 'HeartRate', timeRangeFilter: range, timeRangeSlicer: slicer })) {
      if (g.result.BPM_AVG) { const r = row(g.startTime); r.hr_avg = Math.round(g.result.BPM_AVG); r.hr_max = Math.round(g.result.BPM_MAX); }
      g.result.dataOrigins.forEach(o => apps.add(appName(o)));
    }
  });
  if (ok.has('RestingHeartRate')) await safe(async () => {
    for (const g of await H.aggregateGroupByDuration({ recordType: 'RestingHeartRate', timeRangeFilter: range, timeRangeSlicer: slicer })) {
      if (g.result.BPM_AVG) row(g.startTime).hr_resting = Math.round(g.result.BPM_AVG);
    }
  });
  if (ok.has('SleepSession')) await safe(async () => {
    const { records } = await H.readRecords('SleepSession', { timeRangeFilter: range, pageSize: 500 });
    for (const s of records) {
      const h = (new Date(s.endTime).getTime() - new Date(s.startTime).getTime()) / 3_600_000;
      const r = row(s.endTime); r.sleep_h = Math.round(Math.min(16, (r.sleep_h ?? 0) + h) * 10) / 10;
      if (s.metadata?.dataOrigin) apps.add(appName(s.metadata.dataOrigin));
    }
  });
  // Single readings: average / lowest per day
  const perDay = async <T extends 'OxygenSaturation' | 'HeartRateVariabilityRmssd' | 'Weight' | 'BloodPressure'>(type: T, put: (r: DaySummary, vals: number[][]) => void, pick: (rec: any) => number[]) => {
    if (!ok.has(type)) return;
    await safe(async () => {
      const { records } = await H.readRecords(type, { timeRangeFilter: range, pageSize: 2000 });
      const groups = new Map<string, number[][]>();
      for (const rec of records as any[]) { const k = rec.time as string; const key = dayKey(k); groups.set(key, [...(groups.get(key) ?? []), pick(rec)]); }
      groups.forEach((vals, k) => { if (!byDay.has(k)) byDay.set(k, { date: k }); put(byDay.get(k)!, vals); });
    });
  };
  const mean = (a: number[]) => Math.round((a.reduce((x, y) => x + y, 0) / a.length) * 10) / 10;
  await perDay('OxygenSaturation', (r, v) => { const p = v.map(x => x[0]); r.spo2_min = Math.round(Math.min(...p)); r.spo2_avg = Math.round(mean(p)); }, rec => [rec.percentage]);
  await perDay('HeartRateVariabilityRmssd', (r, v) => { r.hrv_ms = mean(v.map(x => x[0])); }, rec => [rec.heartRateVariabilityMillis]);
  await perDay('Weight', (r, v) => { r.weight_kg = mean(v.map(x => x[0])); }, rec => [rec.weight?.inKilograms ?? rec.weight?.value ?? 0]);
  await perDay('BloodPressure', (r, v) => { r.bp_sys = mean(v.map(x => x[0])); r.bp_dia = mean(v.map(x => x[1])); },
    rec => [rec.systolic?.inMillimetersOfMercury ?? rec.systolic?.value ?? 0, rec.diastolic?.inMillimetersOfMercury ?? rec.diastolic?.value ?? 0]);

  const days = [...byDay.values()].filter(d => Object.keys(d).length > 1).sort((a, b) => a.date.localeCompare(b.date));
  apps.delete('Health Connect');
  return { days, apps: [...apps] };
}
