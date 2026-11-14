// Web: camera + MediaPipe pose (in the browser, frames never leave the device) + the shared exercise engine.
// Draws a mirrored skeleton, a "NO POSE DETECTED" warning and a live status line on a canvas, like the
// original ai-gym-coach video processor. Frames are reported ~4x a second.
import { useEffect, useRef, useState } from 'react';
import { Text, View } from 'react-native';
import { colors, radius, type } from '../theme';
import { getLandmarker } from './LiveCoach.web';
import { POSE_CONNECTIONS, type EngineFrame, type Lm } from '../lib/exerciseEngine';
import type { GymCameraProps } from './GymCamera';

export const GYM_CAMERA_SUPPORTED = true;

export default function GymCamera({ exercise, engine, active, onFrame, onError }: GymCameraProps) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [err, setErr] = useState('');
  const live = useRef({ exercise, engine, active, onFrame, onError });
  live.current = { exercise, engine, active, onFrame, onError };

  useEffect(() => {
    let stream: MediaStream | null = null; let raf = 0; let stopped = false; let lastT = -1; let lastEmit = 0;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access. Use Chrome or Edge.');
        const [lmk, s] = await Promise.all([
          getLandmarker(),
          navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } }, audio: false }),
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
          const now = performance.now();
          const lm = lmk.detectForVideo(v, now).landmarks?.[0];
          const L = live.current;
          const f: EngineFrame | null = L.active ? L.engine.process(lm, now) : null;
          draw(lm, f);
          if (f && now - lastEmit >= 250) { lastEmit = now; L.onFrame(f); }
        };
        loop();
      } catch (e: any) {
        if (stopped) return;
        const msg = e?.name === 'NotAllowedError' ? 'Camera permission was blocked. Click the camera icon in the address bar, allow it, then press Start again.'
          : e?.name === 'NotFoundError' ? 'No camera found on this device.' : (e?.message ?? 'Camera unavailable');
        setErr(msg); setState('error'); live.current.onError?.(msg);
      }
    })();
    return () => { stopped = true; cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); };
  }, []);

  function draw(lm: Lm[] | undefined, f: EngineFrame | null) {
    const c = canvas.current, vd = video.current; if (!c || !vd) return;
    if (c.width !== vd.videoWidth) { c.width = vd.videoWidth; c.height = vd.videoHeight; }
    const g = c.getContext('2d')!; const W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    const X = (x: number) => (1 - x) * W; // mirror to match the selfie video
    const font = Math.max(16, Math.round(W / 32));
    const label = (text: string, x: number, y: number, color: string) => {
      g.font = `600 ${font}px Onest, system-ui, sans-serif`;
      const w = g.measureText(text).width;
      g.fillStyle = 'rgba(34,23,9,0.72)'; g.fillRect(x - 8, y - font, w + 16, font * 1.35);
      g.fillStyle = color; g.fillText(text, x, y);
    };
    if (!lm) {
      label('NO POSE DETECTED', 24, font + 20, colors.warn);
      label(live.current.exercise.view === 'side' ? 'TURN SIDE-ON, WHOLE BODY IN FRAME' : 'FACE THE CAMERA, WHOLE BODY IN FRAME', 24, font * 2 + 36, colors.onDark);
      return;
    }
    const bad = !!f && f.issues.length > 0;
    g.lineWidth = Math.max(4, W / 120); g.lineCap = 'round'; g.strokeStyle = bad ? colors.warn : f?.inPose ? '#9BC46A' : colors.accentFrom;
    for (const [a, b] of POSE_CONNECTIONS) {
      const p = lm[a], q = lm[b]; if ((p.visibility ?? 0) < 0.6 || (q.visibility ?? 0) < 0.6) continue;
      g.beginPath(); g.moveTo(X(p.x), p.y * H); g.lineTo(X(q.x), q.y * H); g.stroke();
    }
    g.fillStyle = colors.onDark;
    for (let i = 11; i < lm.length; i++) {
      const p = lm[i]; if ((p.visibility ?? 0) < 0.6) continue;
      g.beginPath(); g.arc(X(p.x), p.y * H, Math.max(5, W / 140), 0, Math.PI * 2); g.fill();
    }
    if (!f) { label('REST', 20, H - 24, colors.onDark); return; }
    const ex = live.current.exercise;
    const shown = (ex.show ?? (ex.count ? [[ex.count.metric, 'Angle']] : [])).filter(([m]) => f.values[m] !== undefined && !m.startsWith('hipSag'))
      .map(([m, l]) => `${l.toUpperCase()}: ${Math.round(f.values[m])}°`).join(' | ');
    const text = f.issues[0] ? f.issues[0].cue.toUpperCase() : ex.mode === 'hold' ? (f.inPose ? `IN POSE · ${Math.floor(f.holdSec)}s` : 'GET INTO THE POSE') : (shown || f.status.toUpperCase());
    label(text.length > 60 ? text.slice(0, 57) + '…' : text, 20, H - 24, bad ? colors.warn : '#9BC46A');
  }

  return (
    <View style={{ width: '100%', aspectRatio: 4 / 3, borderRadius: radius.cardSm, overflow: 'hidden', backgroundColor: '#000' }}>
      <video ref={video} playsInline muted style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' } as any} />
      <canvas ref={canvas} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' } as any} />
      {state !== 'ready' && (
        <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', padding: 24, backgroundColor: 'rgba(34,23,9,0.88)' }}>
          <Text style={[type.body, { color: colors.onDark, textAlign: 'center' }]}>{state === 'loading' ? 'Starting camera and pose model…' : err}</Text>
        </View>
      )}
    </View>
  );
}
