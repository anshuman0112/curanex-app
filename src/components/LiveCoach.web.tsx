// Web: live camera form coach. MediaPipe Pose Landmarker (lite, GPU) runs fully in the
// browser; frames never leave the device. Rep/form rules live in src/lib/formCoach.ts.
import { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors, radius, type } from '../theme';
import { detectorFor, POSE_CONNECTIONS, type CoachFrame, type Lm } from '../lib/formCoach';
import type { LiveCoachProps } from './LiveCoach';

export const LIVE_COACH_SUPPORTED = true;

const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export type Landmarker = { detectForVideo(v: HTMLVideoElement, t: number): { landmarks: Lm[][] }; close(): void };
let landmarkerP: Promise<Landmarker> | null = null;

function loadVision(): Promise<any> {
  const w = window as any;
  if (w.__mpVision) return Promise.resolve(w.__mpVision);
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.type = 'module';
    s.textContent = `import * as v from '${MP}/vision_bundle.mjs'; window.__mpVision = v; window.dispatchEvent(new Event('mpvision'));`;
    window.addEventListener('mpvision', () => res(w.__mpVision), { once: true });
    s.onerror = () => rej(new Error('Could not load pose model'));
    setTimeout(() => !w.__mpVision && rej(new Error('Pose model took too long to load. Check your connection.')), 30000);
    document.head.appendChild(s);
  });
}

export function getLandmarker(): Promise<Landmarker> {
  if (!landmarkerP) landmarkerP = (async () => {
    const v = await loadVision();
    const files = await v.FilesetResolver.forVisionTasks(`${MP}/wasm`);
    const opts = (delegate: 'GPU' | 'CPU') => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numPoses: 1 });
    try { return await v.PoseLandmarker.createFromOptions(files, opts('GPU')); }
    catch { return await v.PoseLandmarker.createFromOptions(files, opts('CPU')); }
  })().catch(e => { landmarkerP = null; throw e; });
  return landmarkerP;
}

export default function LiveCoach({ exercise, target, active, onReps, say }: LiveCoachProps) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [err, setErr] = useState('');
  const [frame, setFrame] = useState<CoachFrame | null>(null);
  const det = useRef(detectorFor(exercise));
  const activeRef = useRef(active);
  const badRef = useRef(false);
  const cb = useRef({ onReps, say }); cb.current = { onReps, say };
  activeRef.current = active;

  // New exercise -> fresh detector
  useEffect(() => { det.current = detectorFor(exercise); setFrame(null); }, [exercise]);

  useEffect(() => {
    let stream: MediaStream | null = null; let raf = 0; let stopped = false;
    let lastT = -1; let lastSpoken = ''; let lastSpokeAt = 0; let lastReps = 0; let lastUi = 0;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access.');
        const [lmk, s] = await Promise.all([
          getLandmarker(),
          navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }),
        ]);
        stream = s;
        if (stopped) { s.getTracks().forEach(t => t.stop()); return; }
        const v = video.current!; v.srcObject = s; await v.play();
        setState('ready');
        const loop = () => {
          if (stopped) return;
          raf = requestAnimationFrame(loop);
          if (v.readyState < 2 || v.currentTime === lastT) return;
          lastT = v.currentTime;
          const res = lmk.detectForVideo(v, performance.now());
          const lm = res.landmarks?.[0];
          draw(lm);
          if (!activeRef.current) return;
          const now = performance.now();
          const f: CoachFrame = lm ? det.current.process(lm) : { reps: lastReps, ok: false, issues: ['Stand in front of the camera'], status: '' };
          if (det.current.counts && f.reps !== lastReps) {
            lastReps = f.reps; cb.current.onReps?.(f.reps);
            cb.current.say(String(f.reps)); lastSpokeAt = now;
          } else if (f.issues[0] && (f.issues[0] !== lastSpoken ? now - lastSpokeAt > 2500 : now - lastSpokeAt > 6000)) {
            lastSpoken = f.issues[0]; lastSpokeAt = now; cb.current.say(f.issues[0]);
          }
          badRef.current = f.issues.length > 0;
          if (now - lastUi > 120) { lastUi = now; setFrame(f); } // throttle React renders
        };
        loop();
      } catch (e: any) {
        if (stopped) return;
        setErr(e?.name === 'NotAllowedError' ? 'Camera permission was blocked. Allow it in the address bar to use live form check.' : (e?.message ?? 'Camera unavailable'));
        setState('error');
      }
    })();
    return () => { stopped = true; cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); };
  }, []);

  function draw(lm?: Lm[]) {
    const c = canvas.current, v = video.current; if (!c || !v) return;
    if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
    const g = c.getContext('2d')!; g.clearRect(0, 0, c.width, c.height);
    if (!lm) return;
    const bad = activeRef.current && badRef.current;
    g.lineWidth = 4; g.strokeStyle = bad ? colors.warn : colors.accentFrom; g.fillStyle = colors.onDark;
    for (const [a, b] of POSE_CONNECTIONS) {
      const p = lm[a], q = lm[b]; if ((p.visibility ?? 1) < 0.5 || (q.visibility ?? 1) < 0.5) continue;
      g.beginPath(); g.moveTo(p.x * c.width, p.y * c.height); g.lineTo(q.x * c.width, q.y * c.height); g.stroke();
    }
    for (const i of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
      const p = lm[i]; if ((p.visibility ?? 1) < 0.5) continue;
      g.beginPath(); g.arc(p.x * c.width, p.y * c.height, 5, 0, Math.PI * 2); g.fill();
    }
  }

  const issue = frame?.issues[0];
  return (
    <View style={s.wrap}>
      <View style={s.stage}>
        <video ref={video} playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: 'block' }} />
        <canvas ref={canvas} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' } as any} />
        {state !== 'ready' && (
          <View style={s.overlay}>
            <Text style={[type.small, { color: state === 'error' ? colors.onDark : colors.onDarkMuted, textAlign: 'center' }]}>
              {state === 'loading' ? 'Starting camera and pose model…' : err}
            </Text>
          </View>
        )}
        {state === 'ready' && det.current.counts && active && (
          <View style={s.reps}>
            <Text style={[type.h1, { color: colors.onDark }]}>{frame?.reps ?? 0}{target ? <Text style={[type.small, { color: colors.onDarkMuted }]}> / {target}</Text> : null}</Text>
            <Text style={[type.small, { color: colors.onDarkMuted }]}>reps</Text>
          </View>
        )}
        {state === 'ready' && frame?.angle && active && (
          <View style={s.angle}><Text style={[type.small, { color: colors.onDark }]}>{frame.angle.label} {frame.angle.value}°</Text></View>
        )}
      </View>
      {state === 'ready' && (
        <View style={[s.feedback, { backgroundColor: !active ? colors.inkSoft : issue ? colors.warn : colors.good }]}>
          <Text style={[type.small, { color: colors.onDark, fontWeight: '600' }]}>
            {!active ? 'Coach paused during rest' : issue ?? (frame?.status || 'Looking good')}
          </Text>
        </View>
      )}
      <Text style={[type.small, { color: colors.onDarkMuted, fontSize: 11 }]}>
        Live form check runs on this device (MediaPipe Pose). Rule-based guidance, not medical advice. Logic adapted from ai-gym-coach by Shradha Khapra.
      </Text>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { gap: 8, width: '100%', maxWidth: 520, alignSelf: 'center' },
  stage: { width: '100%', aspectRatio: 4 / 3, borderRadius: radius.cardSm, overflow: 'hidden', backgroundColor: '#000', position: 'relative' },
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', padding: 20, backgroundColor: 'rgba(34,23,9,0.85)' },
  reps: { position: 'absolute', top: 10, left: 10, backgroundColor: 'rgba(34,23,9,0.75)', borderRadius: radius.control, paddingHorizontal: 12, paddingVertical: 6, alignItems: 'center' },
  angle: { position: 'absolute', top: 10, right: 10, backgroundColor: 'rgba(34,23,9,0.75)', borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4 },
  feedback: { borderRadius: radius.control, paddingHorizontal: 14, paddingVertical: 10 },
});
