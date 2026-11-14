// The user's weekly workout plan: generated on the server from the exercise catalog + health details,
// stored on this device. Days are Mon..Sun (index 0..6).
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useState } from 'react';
import { workoutPlan, type PlanDay, type PlanPrefs, type WorkoutPlan } from './api';
import type { Profile, Watch } from '../store/health';

const KEY = 'curanex.workoutPlan.v1';
export type SavedPlan = WorkoutPlan & { createdAt: number; profileKey: string };

/** Changes when the health details that shape the plan change, so we can ask to regenerate. */
export const profileKey = (p: Profile | null) => JSON.stringify([p?.age, p?.heightCm, p?.weightKg, p?.goal, [...(p?.conditions ?? [])].sort()]);
export const todayIndex = () => (new Date().getDay() + 6) % 7; // Mon = 0
export const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export const GOAL_OPTIONS: { id: PlanPrefs['goal']; label: string }[] = [
  { id: 'weight_loss', label: 'Weight loss' }, { id: 'strength', label: 'Gym strength' }, { id: 'yoga', label: 'Yoga' },
  { id: 'flexibility', label: 'Flexibility' }, { id: 'general', label: 'General fitness' }, { id: 'low_impact', label: 'Low-impact / seniors' },
];
export const EQUIPMENT_OPTIONS = [{ id: 'mat', label: 'Mat' }, { id: 'dumbbells', label: 'Dumbbells' }, { id: 'chair', label: 'Chair' }, { id: 'wall', label: 'Wall' }];

export function defaultPrefs(p: Profile | null): PlanPrefs {
  const goal = p?.goal === 'Lose weight' ? 'weight_loss' : p?.goal === 'Build muscle' ? 'strength' : p?.goal === 'Manage a condition' ? 'low_impact' : 'general';
  return { goal, days: 4, minutes: 30, level: 'beginner', equipment: ['mat', 'chair', 'wall'] };
}

/** Latest smartwatch readings that affect safety (BP, resting HR), if any. */
export function latestVitals(w?: Watch | null) {
  const days = [...(w?.days ?? [])].reverse();
  const pick = <K extends 'bp_sys' | 'bp_dia' | 'hr_resting'>(k: K) => days.find(d => d[k] != null)?.[k];
  const v = { bp_sys: pick('bp_sys'), bp_dia: pick('bp_dia'), hr_resting: pick('hr_resting') };
  return Object.values(v).some(x => x != null) ? v : null;
}

export function usePlan() {
  const [plan, setPlan] = useState<SavedPlan | null | undefined>(undefined); // undefined = loading
  useEffect(() => { AsyncStorage.getItem(KEY).then(r => setPlan(r ? JSON.parse(r) : null)).catch(() => setPlan(null)); }, []);
  const save = useCallback((p: SavedPlan | null) => {
    setPlan(p);
    (p ? AsyncStorage.setItem(KEY, JSON.stringify(p)) : AsyncStorage.removeItem(KEY)).catch(() => {});
  }, []);
  const generate = useCallback(async (profile: Profile | null, watch: Watch | null | undefined, prefs: PlanPrefs) => {
    const body = profile ? { age: profile.age, sex: profile.sex, heightCm: profile.heightCm, weightKg: profile.weightKg, goal: profile.goal, conditions: profile.conditions } : {};
    const out = await workoutPlan(body, latestVitals(watch), prefs);
    const saved: SavedPlan = { ...out, createdAt: Date.now(), profileKey: profileKey(profile) };
    save(saved);
    return saved;
  }, [save]);
  return { plan, save, generate };
}

export const dayOf = (plan: SavedPlan | null | undefined, i: number): PlanDay | undefined => plan?.days?.[i];
