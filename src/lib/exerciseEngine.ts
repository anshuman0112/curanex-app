// Generic pose engine for every exercise in src/data/exercises.json.
// Each catalog entry is pure data (angle thresholds, form rules, hold targets); this file turns it into
// live rep counting, hold timing and corrections. Deterministic: no AI decides whether form is right.
// Rep logic generalises the detectors of shradha-khapra/ai-gym-coach (two-threshold state machine).
import catalogJson from '../data/exercises.json';

export type Lm = { x: number; y: number; z?: number; visibility?: number };
type Test = { lt?: number; gt?: number };
export type Rule = { metric: string; test: Test; issue: string; cue: string; phase: 'any' | 'in' | 'out' };
export type Target = { metric: string; min: number; max: number; cue: string };
export type Mode = 'reps' | 'hold' | 'timed';
export type Exercise = {
  id: string; name: string; category: string; equipment: string; level: 'beginner' | 'intermediate' | 'advanced'; impact: 'low' | 'high';
  muscles: string[]; view: 'side' | 'front'; met: number; mode: Mode; camera: boolean; steps: string[]; avoid: string[]; caution: string[];
  defaults: { sets: number; reps?: number; seconds?: number }; range: { sets: [number, number]; reps?: [number, number]; seconds?: [number, number] };
  count?: { metric: string; enter: Test; exit: Test; partial?: { at: number; cue: string } };
  rules?: Rule[]; hold?: { targets: Target[] }; show?: [string, string][]; source?: string;
};
type Catalog = { version: number; categories: Record<string, string>; conditions: Record<string, string[]>; conditionLabels: Record<string, string>; exercises: Exercise[] };

export const CATALOG = catalogJson as unknown as Catalog;
export const EXERCISES = CATALOG.exercises;
export const CATEGORIES = CATALOG.categories;
const byId = new Map(EXERCISES.map(e => [e.id, e]));
const byName = new Map(EXERCISES.map(e => [e.name.toLowerCase(), e]));
export const getExercise = (id: string) => byId.get(id);
export function findExercise(nameOrId: string): Exercise | undefined {
  const k = nameOrId.toLowerCase();
  return byId.get(k) ?? byName.get(k) ?? EXERCISES.find(e => e.name.toLowerCase().includes(k) || k.includes(e.name.toLowerCase()));
}

// ---------- Health tags (same keyword table the backend uses) ----------
export type Person = { age?: string | number; conditions?: string[]; heightCm?: string | number; weightKg?: string | number };
export function healthTags(p: Person | null | undefined): string[] {
  if (!p) return [];
  const text = (p.conditions ?? []).join(' | ').toLowerCase();
  const tags = Object.entries(CATALOG.conditions).filter(([, words]) => words.some(w => text.includes(w))).map(([t]) => t);
  if (Number(p.age) >= 65) tags.push('senior');
  const h = Number(p.heightCm) / 100, w = Number(p.weightKg);
  if (h > 0 && w > 0 && w / (h * h) >= 35) tags.push('obesity_severe');
  return tags;
}
export const tagLabel = (t: string) => CATALOG.conditionLabels[t] ?? t;
export function suitability(e: Exercise, tags: string[]): { level: 'ok' | 'caution' | 'avoid'; reasons: string[] } {
  const avoid = e.avoid.filter(t => tags.includes(t));
  if (avoid.length) return { level: 'avoid', reasons: avoid.map(tagLabel) };
  const caution = e.caution.filter(t => tags.includes(t));
  return caution.length ? { level: 'caution', reasons: caution.map(tagLabel) } : { level: 'ok', reasons: [] };
}

// ---------- Metrics ----------
const J = {
  L: { sh: 11, el: 13, wr: 15, hip: 23, knee: 25, ank: 27 },
  R: { sh: 12, el: 14, wr: 16, hip: 24, knee: 26, ank: 28 },
};
type Side = typeof J.L;
const TRIPLETS: Record<string, (s: Side) => [number, number, number]> = {
  knee: s => [s.hip, s.knee, s.ank],
  hip: s => [s.sh, s.hip, s.knee],
  elbow: s => [s.sh, s.el, s.wr],
  shoulder: s => [s.hip, s.sh, s.el],
  armRaise: s => [s.hip, s.sh, s.wr],
  body: s => [s.sh, s.hip, s.ank],
};

export function angle(a: Lm, b: Lm, c: Lm) {
  const ax = a.x - b.x, ay = a.y - b.y, cx = c.x - b.x, cy = c.y - b.y;
  const m = Math.hypot(ax, ay) * Math.hypot(cx, cy);
  if (!m) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, (ax * cx + ay * cy) / m))) * 180) / Math.PI;
}
const vis = (lm: Lm[], i: number) => lm[i]?.visibility ?? 1;
const mid = (a: Lm, b: Lm) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a: Lm, b: Lm) => Math.hypot(a.x - b.x, a.y - b.y);

/** Joints a metric needs (for the "can't see you" check). */
export function metricJoints(metric: string): number[] {
  const [name] = metric.split(':');
  if (TRIPLETS[name]) return [...TRIPLETS[name](J.L), ...TRIPLETS[name](J.R)];
  switch (name) {
    case 'torsoLean': case 'hipLevel': return [11, 12, 23, 24];
    case 'footSpread': case 'legSpread': return [11, 12, 23, 24, 27, 28];
    case 'wristsUp': return [0, 15, 16];
    case 'elbowDrift': return [11, 12, 13, 14, 23, 24];
    case 'hipSag': return [11, 12, 23, 24, 27, 28];
    default: return [];
  }
}

/** Value of one metric, e.g. "knee:best", "armRaise:min", "torsoLean". Angles in degrees, distances normalised. */
export function metric(lm: Lm[], m: string): number {
  const [name, sel = 'best'] = m.split(':');
  const trip = TRIPLETS[name];
  if (trip) {
    const [la, lb, lc] = trip(J.L), [ra, rb, rc] = trip(J.R);
    const l = angle(lm[la], lm[lb], lm[lc]), r = angle(lm[ra], lm[rb], lm[rc]);
    const lv = Math.min(vis(lm, la), vis(lm, lb), vis(lm, lc)), rv = Math.min(vis(lm, ra), vis(lm, rb), vis(lm, rc));
    if (sel === 'min') return Math.min(l, r);
    if (sel === 'max') return Math.max(l, r);
    if (sel === 'L') return l;
    if (sel === 'R') return r;
    return lv >= rv ? l : r;
  }
  const sh = mid(lm[11], lm[12]), hp = mid(lm[23], lm[24]);
  const torso = Math.max(dist(sh, hp), 1e-3);
  const shoulderW = Math.max(dist(lm[11], lm[12]), torso * 0.35); // side view: shoulders overlap
  switch (name) {
    case 'torsoLean': return (Math.atan2(Math.abs(sh.x - hp.x), Math.abs(sh.y - hp.y)) * 180) / Math.PI;
    case 'hipLevel': return Math.abs(lm[23].y - lm[24].y) / torso;
    case 'footSpread': return Math.abs(lm[27].x - lm[28].x) / shoulderW;
    case 'legSpread': return angle(lm[27], hp, lm[28]);
    case 'wristsUp': return Math.max(lm[15].y, lm[16].y) < lm[0].y ? 1 : 0;
    case 'elbowDrift': {
      const s = vis(lm, 13) >= vis(lm, 14) ? J.L : J.R;
      return Math.abs(lm[s.el].x - lm[s.sh].x) / torso;
    }
    case 'hipSag': {
      const s = vis(lm, 23) >= vis(lm, 24) ? J.L : J.R;
      // + = hips below the shoulder-ankle line (sagging), - = hips above it (piked). Image y grows downward.
      return (lm[s.hip].y - (lm[s.sh].y + lm[s.ank].y) / 2) / torso;
    }
    default: return 0;
  }
}
const pass = (v: number, t: Test) => (t.lt === undefined || v < t.lt) && (t.gt === undefined || v > t.gt);

// ---------- Engine ----------
export type Issue = { issue: string; cue: string };
export type EngineFrame = {
  visible: boolean;          // key joints in view
  reps: number;
  holdSec: number;           // seconds held in the correct pose (hold mode)
  inPose: boolean;           // hold mode: all targets met
  issues: Issue[];
  status: string;
  values: Record<string, number>;
};

const STABLE_FRAMES = 6; // a problem must persist ~0.2 s before we report it (avoids flicker)

export class ExerciseEngine {
  reps = 0; holdSec = 0;
  private stage: 'in' | 'out' | null = null;
  private extreme: number | null = null;   // furthest point of the current attempt (partial-rep check)
  private partialUntil = 0;
  private lastT = 0;
  private streak = new Map<string, number>();
  private joints: number[];
  constructor(public ex: Exercise) {
    const ms = [ex.count?.metric, ...(ex.hold?.targets.map(t => t.metric) ?? [])].filter(Boolean) as string[];
    this.joints = [...new Set(ms.flatMap(metricJoints))];
  }
  reset() { this.reps = 0; this.holdSec = 0; this.stage = null; this.extreme = null; this.partialUntil = 0; this.lastT = 0; this.streak.clear(); }

  private stable(key: string, on: boolean) {
    const n = on ? (this.streak.get(key) ?? 0) + 1 : 0; this.streak.set(key, n); return n >= STABLE_FRAMES;
  }

  /** Joints visible enough? Uses the better side for side-view exercises. */
  private visible(lm: Lm[]) {
    if (!this.joints.length) return [11, 12, 23, 24].some(i => vis(lm, i) > 0.5);
    const left = this.joints.filter(i => [0, 11, 13, 15, 23, 25, 27].includes(i));
    const right = this.joints.filter(i => [0, 12, 14, 16, 24, 26, 28].includes(i));
    const ok = (ids: number[]) => ids.every(i => vis(lm, i) > 0.5);
    return this.ex.view === 'side' ? ok(left) || ok(right) : ok(this.joints);
  }

  process(lm: Lm[] | undefined, tMs: number): EngineFrame {
    const dt = this.lastT ? Math.min(0.25, (tMs - this.lastT) / 1000) : 0; this.lastT = tMs;
    const base = { reps: this.reps, holdSec: this.holdSec, values: {} as Record<string, number> };
    if (!lm || !this.visible(lm)) {
      const cue = this.ex.view === 'side' ? 'Turn side-on to the camera and step back so your whole body is in frame.' : 'Face the camera and step back so your whole body is in frame.';
      return { ...base, visible: false, inPose: false, issues: [{ issue: 'The user is not fully visible to the camera.', cue }], status: '' };
    }
    const values: Record<string, number> = {};
    const val = (m: string) => (values[m] ??= Math.round(metric(lm, m) * 100) / 100);
    const issues: Issue[] = [];
    let status = '';

    // Counting (reps mode, and optional movement count in timed mode)
    const c = this.ex.count;
    if (c) {
      const v = val(c.metric);
      if (pass(v, c.enter)) this.stage = 'in';
      else if (pass(v, c.exit)) {
        if (this.stage === 'in') { this.reps++; }
        else if (c.partial && this.extreme !== null && this.attempted(this.extreme, c)) this.partialUntil = tMs + 2500;
        this.stage = 'out'; this.extreme = null;
      }
      if (!pass(v, c.exit) && this.stage !== 'in') this.extreme = this.extreme === null ? v : (c.enter.lt !== undefined ? Math.min(this.extreme, v) : Math.max(this.extreme, v));
      if (tMs < this.partialUntil && c.partial) issues.push({ issue: `Partial repetition: the user did not reach full range on ${this.ex.name}.`, cue: c.partial.cue });
      status = this.stage === 'in' ? 'Good range, now return' : 'Ready';
    }

    // Hold targets
    let inPose = false;
    if (this.ex.mode === 'hold' && this.ex.hold) {
      const miss = this.ex.hold.targets.find(t => { const v = val(t.metric); return v < t.min || v > t.max; });
      inPose = !miss;
      if (this.stable('hold', !inPose) && miss) issues.push({ issue: `The user is not in the correct ${this.ex.name} position.`, cue: miss.cue });
      if (inPose) this.holdSec += dt;
      status = inPose ? 'Hold it, breathe' : 'Get into the pose';
    }

    // Form rules
    for (const [i, r] of (this.ex.rules ?? []).entries()) {
      const phaseOk = r.phase === 'any' || (r.phase === 'in' ? this.stage === 'in' : this.stage !== 'in');
      const bad = phaseOk && pass(val(r.metric), r.test);
      if (this.stable(`r${i}`, bad)) issues.push({ issue: r.issue, cue: r.cue });
    }
    for (const [m] of this.ex.show ?? []) val(m);
    if (!status) status = 'Tracking, keep going';
    return { visible: true, reps: this.reps, holdSec: this.holdSec, inPose, issues, status, values };
  }

  private attempted(extreme: number, c: NonNullable<Exercise['count']>) {
    const at = c.partial!.at;
    return c.enter.lt !== undefined ? extreme < at : extreme > at;
  }
}

/** Seconds a planned item takes (for plan length estimates). */
export function itemSeconds(e: Exercise, sets: number, reps?: number, seconds?: number) {
  const work = e.mode === 'reps' ? (reps ?? e.defaults.reps ?? 10) * 3 : (seconds ?? e.defaults.seconds ?? 30);
  return sets * work + Math.max(0, sets - 1) * 30;
}
export const kcal = (e: Exercise, minutes: number, weightKg: number) => Math.round((e.met * 3.5 * (weightKg || 65) / 200) * minutes);

export const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];
