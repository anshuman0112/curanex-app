import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';

/** Shared health store: every module reads and writes here. This is what makes Curanex "integrated". */

export type Contact = { name: string; phone: string; relation: string };
export type Profile = {
  name: string; age: string; sex: 'Female' | 'Male' | 'Other' | ''; heightCm: string; weightKg: string; bloodGroup: string;
  conditions: string[]; allergies: string[]; medications: string[];
  diet: 'Vegetarian' | 'Non-veg' | 'Vegan' | 'Jain' | 'Eggetarian' | '';
  goal: 'Lose weight' | 'Build muscle' | 'Stay fit' | 'Manage a condition' | '';
  contacts: Contact[];
};
export type Meal = { id: string; name: string; time: string; kcal: number; protein: number; carbs: number; fat: number; at: number };
export type Workout = { id: string; title: string; minutes: number; at: number };
export type State = {
  profile: Profile | null;
  meals: Meal[];
  workouts: Workout[];
  waterMl: { date: string; ml: number };
  medsTaken: { date: string; names: string[] };
};

const today = () => new Date().toISOString().slice(0, 10);
const initial: State = { profile: null, meals: [], workouts: [], waterMl: { date: today(), ml: 0 }, medsTaken: { date: today(), names: [] } };
const KEY = 'curanex.state.v1';

type Ctx = State & {
  ready: boolean;
  setProfile: (p: Profile) => void;
  addMeal: (m: Omit<Meal, 'id' | 'at'>) => void;
  addWorkout: (w: Omit<Workout, 'id' | 'at'>) => void;
  addWater: (ml: number) => void;
  toggleMed: (name: string) => void;
  loadDemo: () => void;
  reset: () => void;
  water: number;
  takenToday: string[];
};
const C = createContext<Ctx | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const [st, setSt] = useState<State>(initial);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try { const raw = await AsyncStorage.getItem(KEY); if (raw) setSt({ ...initial, ...JSON.parse(raw) }); } catch {}
      setReady(true);
    })();
  }, []);
  useEffect(() => { if (ready) AsyncStorage.setItem(KEY, JSON.stringify(st)).catch(() => {}); }, [st, ready]);

  const id = () => Math.random().toString(36).slice(2, 9);
  const setProfile = useCallback((profile: Profile) => setSt(s => ({ ...s, profile })), []);
  const addMeal = useCallback((m: Omit<Meal, 'id' | 'at'>) => setSt(s => ({ ...s, meals: [{ ...m, id: id(), at: Date.now() }, ...s.meals].slice(0, 100) })), []);
  const addWorkout = useCallback((w: Omit<Workout, 'id' | 'at'>) => setSt(s => ({ ...s, workouts: [{ ...w, id: id(), at: Date.now() }, ...s.workouts].slice(0, 100) })), []);
  const addWater = useCallback((ml: number) => setSt(s => {
    const base = s.waterMl.date === today() ? s.waterMl.ml : 0;
    return { ...s, waterMl: { date: today(), ml: Math.max(0, base + ml) } };
  }), []);
  const toggleMed = useCallback((name: string) => setSt(s => {
    const names = s.medsTaken.date === today() ? s.medsTaken.names : [];
    return { ...s, medsTaken: { date: today(), names: names.includes(name) ? names.filter(n => n !== name) : [...names, name] } };
  }), []);
  const loadDemo = useCallback(() => setSt({ ...initial, ...demoState() }), []);
  const reset = useCallback(() => setSt(initial), []);

  const value = useMemo<Ctx>(() => ({
    ...st, ready, setProfile, addMeal, addWorkout, addWater, toggleMed, loadDemo, reset,
    water: st.waterMl.date === today() ? st.waterMl.ml : 0,
    takenToday: st.medsTaken.date === today() ? st.medsTaken.names : [],
  }), [st, ready, setProfile, addMeal, addWorkout, addWater, toggleMed, loadDemo, reset]);
  return <C.Provider value={value}>{children}</C.Provider>;
}

export const useHealth = () => {
  const c = useContext(C); if (!c) throw new Error('useHealth outside provider'); return c;
};

export const bmi = (p: Profile | null) => {
  if (!p) return null; const h = Number(p.heightCm) / 100, w = Number(p.weightKg);
  return h > 0 && w > 0 ? Math.round((w / (h * h)) * 10) / 10 : null;
};

/** The shared context sent with every AI request (ARCHITECTURE.md §2 step 2). */
export function getUserContext(s: State) {
  const startOfDay = new Date(); startOfDay.setHours(0, 0, 0, 0);
  return {
    profile: s.profile ? { ...s.profile, contacts: undefined, bmi: bmi(s.profile) } : null,
    recent_meals: s.meals.slice(0, 3).map(m => ({ name: m.name, kcal: m.kcal })),
    last_workout: s.workouts[0] ? { title: s.workouts[0].title, minutes: s.workouts[0].minutes, at: new Date(s.workouts[0].at).toISOString() } : null,
    water_ml_today: s.waterMl.date === today() ? s.waterMl.ml : 0,
    kcal_today: s.meals.filter(m => m.at >= startOfDay.getTime()).reduce((a, m) => a + m.kcal, 0),
  };
}

export const todaysMeals = (meals: Meal[]) => {
  const d = new Date(); d.setHours(0, 0, 0, 0); return meals.filter(m => m.at >= d.getTime());
};

function demoState(): Partial<State> {
  const now = Date.now(); const h = 3600_000;
  return {
    profile: {
      name: 'Aarav', age: '34', sex: 'Male', heightCm: '174', weightKg: '82', bloodGroup: 'B+',
      conditions: ['Hypertension'], allergies: ['Penicillin'], medications: ['Amlodipine 5 mg'],
      diet: 'Vegetarian', goal: 'Manage a condition',
      contacts: [{ name: 'Priya (sister)', phone: '+919800000000', relation: 'Sister' }],
    },
    meals: [
      { id: 'd1', name: 'Poha with peanuts', time: 'Breakfast', kcal: 320, protein: 8, carbs: 52, fat: 9, at: now - 5 * h },
      { id: 'd2', name: 'Dal, 2 rotis, bhindi sabzi', time: 'Lunch', kcal: 610, protein: 22, carbs: 84, fat: 18, at: now - 1 * h },
    ],
    workouts: [{ id: 'w1', title: 'Low-impact cardio', minutes: 25, at: now - 26 * h }],
    waterMl: { date: today(), ml: 1250 },
    medsTaken: { date: today(), names: [] },
  };
}
