// Adapter used by the guided workout's "Live form check" (LiveCoach.web.tsx): looks the exercise up in the
// catalog by name and runs the shared pose engine (src/lib/exerciseEngine.ts).
import { ExerciseEngine, findExercise, type Lm } from './exerciseEngine';
export { angle, POSE_CONNECTIONS, type Lm } from './exerciseEngine';

export type CoachFrame = {
  reps: number;
  ok: boolean;
  angle?: { label: string; value: number };
  issues: string[];
  status: string;
};
export interface Detector { process(lm: Lm[]): CoachFrame; reset(): void; counts: boolean }

export function detectorFor(name: string): Detector {
  const ex = findExercise(name);
  if (!ex) {
    return { counts: false, reset() {}, process: () => ({ reps: 0, ok: true, issues: [], status: 'Tracking. Move slowly and breathe.' }) };
  }
  const eng = new ExerciseEngine(ex);
  return {
    counts: ex.mode === 'reps' || !!ex.count,
    reset: () => eng.reset(),
    process(lm: Lm[]): CoachFrame {
      const f = eng.process(lm, performance.now());
      const [m, label] = ex.show?.[0] ?? (ex.count ? [ex.count.metric, 'Angle'] : ['', '']);
      const v = m ? f.values[m] : undefined;
      const status = ex.mode === 'hold' ? `${f.inPose ? 'In pose' : 'Get into the pose'} · ${Math.floor(f.holdSec)}s held` : f.status;
      return { reps: f.reps, ok: f.visible, issues: f.issues.map(i => i.cue), status,
        angle: v !== undefined && m.split(':')[0] !== 'hipSag' ? { label, value: Math.round(v) } : undefined };
    },
  };
}
