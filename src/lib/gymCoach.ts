// AI Gym Coach voice: port of shradha-khapra/ai-gym-coach services/coaching/voice_pipeline.py.
// Major events (workout started, set done, workout done) always speak; form corrections at most every 5 s.
// The server grounds the LLM in the exercise's catalog steps and rejects drifting replies; if it is
// unreachable, the deterministic cue from the pose engine is spoken instead (labelled "Offline cue").
import { Platform } from 'react-native';
import * as Speech from 'expo-speech';
import { gymCoach, type GymEvent } from './api';
import type { Exercise, Issue } from './exerciseEngine';

export type CoachMessage = { text: string; source: 'llm' | 'fallback' | 'offline' | 'local' };

const OFFLINE: Record<GymEvent, string> = {
  workout_started: "Let's go! Get into position and start your first rep.",
  set_completed: 'Set complete, great work! Take a short rest.',
  workout_completed: 'Workout complete. Excellent effort today.',
  no_pose_detected: 'I cannot see you. Step back so your whole body is in the camera.',
  ongoing_form_check: 'Looking strong, keep that steady pace.',
};

export class VoicePipeline {
  private lastSpokenAt = 0;
  private busy = false;
  private history: { role: 'assistant'; content: string }[] = [];
  private audio: HTMLAudioElement | null = null;
  constructor(private onMessage: (m: CoachMessage) => void, public voiceOn = true) {}

  async processEvent(event: GymEvent, ex: Exercise | null, issue?: Issue | null) {
    const now = Date.now();
    const major = event === 'workout_started' || event === 'set_completed' || event === 'workout_completed';
    if (!major) {
      if (!issue) return;
      if (now - this.lastSpokenAt < 5000 || this.busy) return;
    }
    this.lastSpokenAt = now; this.busy = true;
    let msg: CoachMessage; let audio: string | null = null;
    try {
      const r = await gymCoach({ event, exercise: ex?.name ?? '', exercise_id: ex?.id, issue: issue?.issue ?? null, cue: issue?.cue ?? null,
        history: this.history.slice(-10), voice: this.voiceOn && Platform.OS === 'web' });
      msg = { text: r.text, source: r.source }; audio = r.audio_b64;
      this.history.push({ role: 'assistant', content: r.text });
    } catch {
      msg = { text: issue?.cue ?? OFFLINE[event], source: 'offline' };
    } finally { this.busy = false; this.lastSpokenAt = Date.now(); }
    this.onMessage(msg);
    if (this.voiceOn) this.play(msg.text, audio);
  }

  /** Deterministic announcement (exercise intro, countdowns): no LLM, so nothing can be made up. */
  say(text: string, show = true) {
    this.lastSpokenAt = Date.now();
    if (show) this.onMessage({ text, source: 'local' });
    if (this.voiceOn) this.play(text, null);
  }

  private play(text: string, b64: string | null) {
    this.stop();
    if (b64 && typeof Audio !== 'undefined') {
      this.audio = new Audio(`data:audio/mpeg;base64,${b64}`);
      this.audio.play().catch(() => { try { Speech.speak(text); } catch {} });
    } else { try { Speech.speak(text); } catch {} }
  }
  stop() { try { this.audio?.pause(); this.audio = null; Speech.stop(); } catch {} }
}

export type GymHistoryRow = { exercise: string; exerciseId?: string; reps: number; sets: number; seconds: number; at: number };
export const GYM_HISTORY_KEY = 'curanex.gymHistory';
