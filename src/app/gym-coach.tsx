// AI Gym Coach: live camera coaching for every exercise in the catalog (gym, bodyweight, yoga, cardio /
// weight loss, core, mobility, low-impact, breathing). Two modes:
//   - Library (/gym-coach): browse, filtered by the user's health details, pick one exercise, start.
//   - Session (/gym-coach?day=N): run that day of the weekly plan, exercise by exercise, set by set.
// Started as a port of shradha-khapra/ai-gym-coach (plan -> camera -> reps/sets -> LLM voice -> history).
import { useEffect, useMemo, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { router, useLocalSearchParams } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Card, Chip, DarkCard, Disclaimer, Eyebrow, GhostButton, PrimaryButton, ProgressBar, RoundIcon, Screen, SectionTitle, useWide } from '../components/ui';
import { ArrowL, Check, Mic, Sparkle } from '../components/icons';
import GymCamera, { GYM_CAMERA_SUPPORTED } from '../components/GymCamera';
import { useHealth } from '../store/health';
import { colors, radius, type } from '../theme';
import { CATEGORIES, EXERCISES, ExerciseEngine, getExercise, healthTags, kcal, suitability, tagLabel, type EngineFrame, type Exercise } from '../lib/exerciseEngine';
import { GYM_HISTORY_KEY, VoicePipeline, type CoachMessage, type GymHistoryRow } from '../lib/gymCoach';
import { DAY_NAMES, usePlan } from '../lib/workoutPlan';

type QItem = { ex: Exercise; sets: number; target: number; caution?: string[] }; // target = reps or seconds
const REST_BETWEEN_SETS = 30;
const REST_BETWEEN_EXERCISES = 20;
const MODE_LABEL = { reps: 'Reps', hold: 'Hold', timed: 'Timed' } as const;
const unit = (q: QItem) => (q.ex.mode === 'reps' ? `${q.target} reps` : `${q.target}s`);

export default function GymCoach() {
  const params = useLocalSearchParams<{ day?: string; id?: string }>();
  const wide = useWide();
  const { profile, addWorkout } = useHealth();
  const { plan } = usePlan();
  const tags = useMemo(() => healthTags(profile), [profile]);

  // ---------- library state ----------
  const [cat, setCat] = useState<string>('all');
  const [q, setQ] = useState('');
  const [picked, setPicked] = useState<Exercise | null>(params.id ? getExercise(String(params.id)) ?? null : null);
  const [inline, setInline] = useState(false); // details shown in place of the tapped tile (vs on top)
  const [cfg, setCfg] = useState({ sets: 3, target: 10 });
  useEffect(() => {
    if (!picked) return;
    const careful = suitability(picked, tags).level === 'caution';
    let target = picked.defaults.reps ?? picked.defaults.seconds ?? 30;
    if (careful && picked.mode === 'hold' && tags.includes('hypertension')) target = Math.min(target, 30);
    setCfg({ sets: careful ? Math.min(2, picked.defaults.sets) : picked.defaults.sets, target });
  }, [picked, tags]);

  // ---------- session state ----------
  const [queue, setQueue] = useState<QItem[] | null>(null);
  const [title, setTitle] = useState('');
  const [idx, setIdx] = useState(0);
  const [setNo, setSetNo] = useState(0);
  const [phase, setPhase] = useState<'work' | 'rest' | 'done'>('work');
  const [paused, setPaused] = useState(false);
  const [elapsed, setElapsed] = useState(0);       // seconds in this set (timed / camera-less)
  const [restLeft, setRestLeft] = useState(0);
  const [frame, setFrame] = useState<EngineFrame | null>(null);
  const [coach, setCoach] = useState<CoachMessage | null>(null);
  const [voiceOn, setVoiceOn] = useState(true);
  const [history, setHistory] = useState<GymHistoryRow[] | null>(null);
  const [summary, setSummary] = useState<{ title: string; sets: number; reps: number; seconds: number; kcal: number } | null>(null);
  const stats = useRef({ startedAt: 0, setStartedAt: 0, sets: 0, reps: 0, kcal: 0, completing: false });

  const pipeline = useRef<VoicePipeline | null>(null);
  if (!pipeline.current) pipeline.current = new VoicePipeline(m => setCoach(m));
  pipeline.current.voiceOn = voiceOn;

  useEffect(() => {
    AsyncStorage.getItem(GYM_HISTORY_KEY).then(r => setHistory(r ? JSON.parse(r) : [])).catch(() => setHistory([]));
    return () => pipeline.current?.stop();
  }, []);

  // Plan day from the URL: build the queue once the plan has loaded
  const dayParam = params.day !== undefined ? Number(params.day) : null;
  const planDay = dayParam !== null && plan ? plan.days?.[dayParam] : undefined;
  const dayQueue = useMemo<QItem[] | null>(() => {
    if (!planDay || planDay.rest) return null;
    return planDay.items.map(it => ({ ex: getExercise(it.id)!, sets: it.sets, target: it.reps ?? it.seconds ?? 30, caution: it.caution })).filter(x => x.ex);
  }, [planDay]);

  const cur = queue?.[idx];
  const engine = useMemo(() => (cur ? new ExerciseEngine(cur.ex) : null), [cur, setNo]);
  const clockDriven = !!cur && (cur.ex.mode === 'timed' || !cur.ex.camera || !GYM_CAMERA_SUPPORTED);

  function start(items: QItem[], name: string) {
    const now = Date.now();
    stats.current = { startedAt: now, setStartedAt: now, sets: 0, reps: 0, kcal: 0, completing: false };
    setQueue(items); setTitle(name); setIdx(0); setSetNo(0); setPhase('work'); setElapsed(0); setFrame(null); setSummary(null); setPaused(false);
    pipeline.current!.processEvent('workout_started', items[0].ex);
    setTimeout(() => announce(items[0], 0), 3500);
  }

  function announce(item: QItem, set: number) {
    const s = item.ex;
    const what = s.mode === 'reps' ? `${item.target} reps` : s.mode === 'hold' ? `hold for ${item.target} seconds` : `${item.target} seconds`;
    pipeline.current!.say(set === 0 ? `${s.name}. ${item.sets > 1 ? `${item.sets} sets, ` : ''}${what}. ${s.steps[0]}` : `Set ${set + 1} of ${item.sets}. ${what}. Go.`);
  }

  // Clock: work time (timed exercises) and rest countdown
  useEffect(() => {
    if (!queue || paused || phase === 'done') return;
    const t = setInterval(() => {
      if (phase === 'rest') setRestLeft(r => r - 1);
      else if (clockDriven) setElapsed(e => e + 1);
    }, 1000);
    return () => clearInterval(t);
  }, [queue, paused, phase, clockDriven]);

  // Set completion
  useEffect(() => {
    if (!cur || phase !== 'work' || paused) return;
    const doneNow = cur.ex.mode === 'reps' && !clockDriven ? (frame?.reps ?? 0) >= cur.target
      : cur.ex.mode === 'hold' && !clockDriven ? (frame?.holdSec ?? 0) >= cur.target
      : elapsed >= cur.target;
    if (doneNow) completeSet();
  }, [frame, elapsed]);

  // Rest finished -> next set / exercise
  useEffect(() => {
    if (phase !== 'rest' || restLeft > 0 || !queue) return;
    const lastSet = setNo + 1 >= queue[idx].sets;
    const ni = lastSet ? idx + 1 : idx, ns = lastSet ? 0 : setNo + 1;
    setIdx(ni); setSetNo(ns); setElapsed(0); setFrame(null); setPhase('work');
    stats.current.setStartedAt = Date.now(); stats.current.completing = false;
    announce(queue[ni], ns);
  }, [restLeft, phase]);

  function completeSet() {
    if (!queue || !cur || stats.current.completing) return;
    stats.current.completing = true;
    const secs = Math.round((Date.now() - stats.current.setStartedAt) / 1000);
    const reps = cur.ex.mode === 'reps' ? Math.max(frame?.reps ?? 0, clockDriven ? cur.target : 0) : 0;
    const row: GymHistoryRow = { exercise: cur.ex.name, exerciseId: cur.ex.id, reps, sets: 1, seconds: secs, at: Date.now() };
    setHistory(h => { const next = [row, ...(h ?? [])].slice(0, 1000); AsyncStorage.setItem(GYM_HISTORY_KEY, JSON.stringify(next)).catch(() => {}); return next; });
    stats.current.sets++; stats.current.reps += reps; stats.current.kcal += kcal(cur.ex, secs / 60, Number(profile?.weightKg));
    const last = idx + 1 >= queue.length && setNo + 1 >= cur.sets;
    if (last) { finish(); return; }
    const nextEx = setNo + 1 >= cur.sets;
    pipeline.current!.processEvent('set_completed', cur.ex);
    setRestLeft(nextEx ? REST_BETWEEN_EXERCISES : REST_BETWEEN_SETS); setPhase('rest');
  }

  function finish() {
    const s = stats.current; const secs = Math.round((Date.now() - s.startedAt) / 1000);
    setPhase('done'); setQueue(null);
    pipeline.current!.processEvent('workout_completed', cur?.ex ?? null);
    setSummary({ title, sets: s.sets, reps: s.reps, seconds: secs, kcal: Math.round(s.kcal) });
    if (s.sets > 0) addWorkout({ title: `AI Coach · ${title}`, minutes: Math.max(1, Math.round(secs / 60)) });
  }

  function onFrame(f: EngineFrame) {
    setFrame(f);
    if (phase !== 'work' || paused || !cur) return;
    if (!f.visible) pipeline.current!.processEvent('no_pose_detected', cur.ex, f.issues[0]);
    else if (f.issues[0]) pipeline.current!.processEvent('ongoing_form_check', cur.ex, f.issues[0]);
  }

  // ---------- views ----------
  const header = (
    <View style={{ paddingTop: 12, flexDirection: 'row', alignItems: 'center', gap: 12 }}>
      <RoundIcon onPress={() => { pipeline.current?.stop(); queue ? finish() : router.back(); }}><ArrowL size={18} /></RoundIcon>
      <View style={{ flex: 1 }}>
        <Eyebrow>Fitness · Live coach</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 4 }]}>{queue ? title : 'AI Gym Coach'}</Text>
        <Text style={[type.small, { color: colors.muted }]}>Real-time pose detection with AI voice coaching · gym, yoga, cardio, mobility</Text>
      </View>
      <Pressable onPress={() => { setVoiceOn(v => !v); if (voiceOn) pipeline.current?.stop(); }} style={s.voice} accessibilityLabel="Toggle coach voice">
        <Mic size={16} color={voiceOn ? colors.accent : colors.muted} />
        <Text style={[type.small, { color: voiceOn ? colors.ink : colors.muted }]}>Voice {voiceOn ? 'on' : 'off'}</Text>
      </Pressable>
    </View>
  );

  const coachBanner = coach && (
    <View style={[s.coach, frame?.issues[0] && phase === 'work' && { borderColor: colors.warn }]}>
      <Sparkle size={14} color={colors.accent} />
      <Text style={[type.body, { color: colors.ink, flex: 1 }]}><Text style={{ fontWeight: '700' }}>Coach: </Text>{coach.text}</Text>
      {(coach.source === 'offline' || coach.source === 'fallback') && <Text style={[type.small, { color: colors.muted }]}>{coach.source === 'offline' ? 'Offline cue' : 'Basic cue'}</Text>}
    </View>
  );

  // ===== Session =====
  if (queue && cur) {
    const next = phase === 'rest' ? (setNo + 1 >= cur.sets ? queue[idx + 1] : cur) : null;
    const progress = cur.ex.mode === 'reps' && !clockDriven ? (frame?.reps ?? 0) / cur.target
      : cur.ex.mode === 'hold' && !clockDriven ? (frame?.holdSec ?? 0) / cur.target : elapsed / cur.target;
    const issue = phase === 'work' ? frame?.issues[0] : undefined;
    const big = cur.ex.mode === 'reps' && !clockDriven ? `${frame?.reps ?? 0} / ${cur.target}` : cur.ex.mode === 'hold' && !clockDriven ? `${Math.floor(frame?.holdSec ?? 0)}s / ${cur.target}s` : `${Math.max(0, cur.target - elapsed)}s`;

    const camera = cur.ex.camera ? (
      <GymCamera exercise={cur.ex} engine={engine!} active={phase === 'work' && !paused} onFrame={onFrame} />
    ) : (
      <DarkCard style={{ gap: 10, alignItems: 'center', paddingVertical: 32 }}>
        <Text style={[type.label, { color: colors.onDarkMuted }]}>No camera needed</Text>
        <Text style={[type.hero, { color: colors.onDark }]}>{phase === 'rest' ? `${restLeft}s` : big}</Text>
        {cur.ex.steps.map((t, i) => <Text key={i} style={[type.body, { color: colors.onDark, textAlign: 'center' }]}>{t}</Text>)}
      </DarkCard>
    );

    const side = (
      <View style={{ gap: 12 }}>
        <Card style={{ gap: 10 }}>
          <Text style={[type.label, { color: colors.accent }]}>Exercise {idx + 1} of {queue.length} · Set {setNo + 1} of {cur.sets}</Text>
          <Text style={[type.h2, { color: colors.ink }]}>{cur.ex.name}</Text>
          <Text style={[type.small, { color: colors.muted }]}>{CATEGORIES[cur.ex.category]} · {MODE_LABEL[cur.ex.mode]} · {cur.ex.view === 'side' ? 'Side-on to camera' : 'Face the camera'}</Text>
          {cur.caution && <Text style={[type.small, { color: colors.warn }]}>Take care: {cur.caution.join(', ')}. Go gently and stop if it hurts.</Text>}
          <Text style={[type.h1, { color: colors.ink }]}>{phase === 'rest' ? `Rest ${restLeft}s` : big}</Text>
          <ProgressBar value={phase === 'rest' ? 1 : Math.min(1, progress)} />
          {cur.ex.count && cur.ex.mode === 'timed' && <Text style={[type.small, { color: colors.muted }]}>Movements counted: {frame?.reps ?? 0}</Text>}
          {(cur.ex.show ?? []).map(([m, l]) => frame?.values[m] !== undefined && (
            <View key={m} style={s.metric}><Text style={[type.small, { color: colors.muted }]}>{l}</Text><Text style={[type.h3, { color: colors.ink }]}>{Math.round(frame.values[m])}°</Text></View>
          ))}
          <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>
            <GhostButton label={paused ? 'Resume' : 'Pause'} onPress={() => setPaused(p => !p)} />
            {phase === 'work' ? <GhostButton label="Done set" onPress={completeSet} /> : <GhostButton label="Skip rest" onPress={() => setRestLeft(0)} />}
            <GhostButton label="End" onPress={finish} />
          </View>
        </Card>
        <Card style={{ gap: 6 }}>
          <Text style={[type.label, { color: colors.accent }]}>{phase === 'rest' && next ? `Up next: ${next.ex.name}` : 'How to do it'}</Text>
          {(phase === 'rest' && next ? next.ex : cur.ex).steps.map((t, i) => (
            <View key={i} style={{ flexDirection: 'row', gap: 8 }}><Text style={[type.small, { color: colors.accent, width: 14 }]}>{i + 1}</Text><Text style={[type.small, { color: colors.fg, flex: 1 }]}>{t}</Text></View>
          ))}
        </Card>
        <Card style={{ gap: 6 }}>
          <Text style={[type.label, { color: colors.muted }]}>Today's list</Text>
          {queue.map((q2, i) => (
            <View key={q2.ex.id + i} style={{ flexDirection: 'row', justifyContent: 'space-between', gap: 8 }}>
              <Text style={[type.small, { color: i < idx ? colors.muted : i === idx ? colors.ink : colors.fg, fontWeight: i === idx ? '700' : '400', flex: 1 }]}>{i < idx ? '✓ ' : ''}{q2.ex.name}</Text>
              <Text style={[type.small, { color: colors.muted }]}>{q2.sets} × {unit(q2)}</Text>
            </View>
          ))}
        </Card>
      </View>
    );

    return (
      <Screen>
        {header}
        <View style={[{ gap: 14, marginTop: 16 }, wide && { flexDirection: 'row', alignItems: 'flex-start' }]}>
          <View style={[{ gap: 12 }, wide && { flex: 1 }]}>
            {coachBanner}
            {camera}
            <View style={[s.status, { backgroundColor: phase === 'rest' ? colors.inkSoft : paused ? colors.inkSoft : !frame && cur.ex.camera ? colors.inkSoft : issue ? colors.warn : colors.good }]}>
              <Text style={[type.small, { color: colors.onDark, fontWeight: '600' }]}>
                {phase === 'rest' ? `Rest. ${next ? `Next: ${next.ex.name}` : ''}` : paused ? 'Paused' : !cur.ex.camera ? 'Follow the steps and breathe.' : !frame ? 'Get into position…' : issue ? issue.cue : cur.ex.mode === 'hold' ? (frame.inPose ? 'Great, hold it and breathe.' : 'Get into the pose.') : 'Good form. Keep going!'}
              </Text>
            </View>
          </View>
          <View style={wide ? { width: 340 } : undefined}>{side}</View>
        </View>
        <Disclaimer>Rule-based form guidance from camera pose estimation, not medical advice. Stop if you feel pain, dizziness, chest pain or breathlessness.</Disclaimer>
      </Screen>
    );
  }

  // ===== Summary =====
  const summaryCard = summary && (
    <DarkCard style={{ gap: 10 }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}><Check size={18} color={colors.accentFrom} /><Text style={[type.label, { color: colors.accentFrom }]}>{summary.sets ? 'Saved to your health record' : 'Session ended'}</Text></View>
      <Text style={[type.h1, { color: colors.onDark }]}>{summary.title}</Text>
      <Text style={[type.body, { color: colors.onDarkMuted }]}>{summary.sets} sets · {summary.reps} reps · {Math.floor(summary.seconds / 60)}m {summary.seconds % 60}s · about {summary.kcal} kcal (estimate)</Text>
      <PrimaryButton label="Back to trainer" onPress={() => router.back()} />
    </DarkCard>
  );

  // ===== Plan day intro =====
  if (dayParam !== null) {
    return (
      <Screen>
        {header}
        <View style={{ gap: 12, marginTop: 16 }}>
          {coachBanner}
          {summaryCard}
          {plan === undefined ? <Text style={[type.small, { color: colors.muted }]}>Loading your plan…</Text>
            : !planDay ? <Card><Text style={[type.body, { color: colors.ink }]}>No plan yet. Create one on the Gym Trainer tab.</Text></Card>
            : planDay.rest || !dayQueue ? <Card><Text style={[type.body, { color: colors.ink }]}>{DAY_NAMES[dayParam]} is a rest day. A light walk is fine.</Text></Card>
            : !summary && (
              <Card style={{ gap: 10 }}>
                <Text style={[type.label, { color: colors.accent }]}>{DAY_NAMES[dayParam]} · {planDay.minutes} min · {dayQueue.length} exercises</Text>
                <Text style={[type.h2, { color: colors.ink }]}>{planDay.title}</Text>
                {!!planDay.why && <Text style={[type.small, { color: colors.muted }]}>{planDay.why}</Text>}
                {dayQueue.map((q2, i) => (
                  <View key={q2.ex.id + i} style={s.row}>
                    <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{i + 1}. {q2.ex.name}{q2.caution ? '  ⚠' : ''}</Text>
                    <Text style={[type.small, { color: colors.muted }]}>{q2.sets} × {unit(q2)}</Text>
                  </View>
                ))}
                <Text style={[type.small, { color: colors.muted }]}>Set up your camera so your whole body is visible. Each exercise tells you whether to face it or stand side-on.</Text>
                <PrimaryButton label="Start session" onPress={() => start(dayQueue, planDay.title)} />
              </Card>
            )}
        </View>
      </Screen>
    );
  }

  // ===== Library =====
  const ql = q.trim().toLowerCase();
  const list = EXERCISES.filter(e => (cat === 'all' || e.category === cat) && (!ql || e.name.toLowerCase().includes(ql) || e.muscles.some(m => m.includes(ql))));
  const table = (() => {
    const agg = new Map<string, { exercise: string; date: string; reps: number; sets: number; seconds: number }>();
    for (const h of history ?? []) {
      const date = new Date(h.at).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
      const k = `${h.exercise}|${date}`; const a = agg.get(k) ?? { exercise: h.exercise, date, reps: 0, sets: 0, seconds: 0 };
      a.reps += h.reps; a.sets += h.sets; a.seconds += h.seconds; agg.set(k, a);
    }
    return [...agg.values()].slice(0, 30);
  })();

  const detail = picked && (() => {
    const fit = suitability(picked, tags);
    const isReps = picked.mode === 'reps';
    const [lo, hi] = (isReps ? picked.range.reps : picked.range.seconds) ?? [1, 60];
    return (
      <Card style={{ gap: 10, borderColor: colors.accent, borderWidth: 1.5 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
          <Text style={[type.label, { color: colors.accent }]}>{CATEGORIES[picked.category]} · {MODE_LABEL[picked.mode]}</Text>
          <Pressable onPress={() => setPicked(null)}><Text style={[type.small, { color: colors.muted }]}>Close</Text></Pressable>
        </View>
        <Text style={[type.h2, { color: colors.ink }]}>{picked.name}</Text>
        <Text style={[type.small, { color: colors.muted }]}>{picked.muscles.join(', ')} · {picked.level} · {picked.equipment === 'none' ? 'no equipment' : picked.equipment} · {picked.camera ? (picked.view === 'side' ? 'stand side-on to the camera' : 'face the camera') : 'no camera needed'}</Text>
        {fit.level !== 'ok' && <Text style={[type.small, { color: fit.level === 'avoid' ? colors.sos : colors.warn }]}>{fit.level === 'avoid' ? `Not recommended for you (${fit.reasons.join(', ')}). Ask your doctor first.` : `Take care with ${fit.reasons.join(', ')}: go gently, fewer sets.`}</Text>}
        {picked.steps.map((t, i) => <View key={i} style={{ flexDirection: 'row', gap: 8 }}><Text style={[type.small, { color: colors.accent, width: 14 }]}>{i + 1}</Text><Text style={[type.small, { color: colors.fg, flex: 1 }]}>{t}</Text></View>)}
        <Stepper label="Sets" value={cfg.sets} min={picked.range.sets[0]} max={picked.range.sets[1]} onChange={n => setCfg(c => ({ ...c, sets: n }))} />
        <Stepper label={isReps ? 'Reps per set' : picked.mode === 'hold' ? 'Hold (seconds)' : 'Seconds'} value={cfg.target} min={lo} max={hi} step={isReps ? 1 : 5} onChange={n => setCfg(c => ({ ...c, target: n }))} />
        <PrimaryButton label={fit.level === 'avoid' ? 'Not recommended' : 'Start'} disabled={fit.level === 'avoid' || (!GYM_CAMERA_SUPPORTED && picked.camera)}
          onPress={() => start([{ ex: picked, sets: cfg.sets, target: cfg.target, caution: fit.level === 'caution' ? fit.reasons : undefined }], picked.name)} />
        {picked.source && <Text style={[type.small, { color: colors.muted, fontSize: 11 }]}>Detection logic adapted from {picked.source}.</Text>}
      </Card>
    );
  })();

  return (
    <Screen>
      {header}
      <View style={{ gap: 12, marginTop: 16 }}>
        {coachBanner}
        {summaryCard}
        {plan?.days?.length ? (
          <Pressable onPress={() => router.push('/trainer')} style={s.planLink}>
            <Text style={[type.small, { color: colors.ink, flex: 1 }]}>Your weekly plan is on the Gym Trainer tab. Today: <Text style={{ fontWeight: '700' }}>{plan.days[(new Date().getDay() + 6) % 7]?.title}</Text></Text>
          </Pressable>
        ) : null}
        {tags.length > 0 && (
          <View style={s.tags}><Text style={[type.small, { color: colors.ink }]}>Filtered for your health details: <Text style={{ fontWeight: '700' }}>{tags.map(tagLabel).join(', ')}</Text>. Exercises marked "Avoid" can't be started.</Text></View>
        )}
        {picked && (!inline || !list.some(e => e.id === picked.id)) && detail}
        <TextInput value={q} onChangeText={setQ} placeholder="Search exercises or muscles (e.g. squat, hamstrings, yoga)" placeholderTextColor={colors.muted} style={s.search} />
        <View style={s.wrap}>
          <Chip label={`All (${EXERCISES.length})`} active={cat === 'all'} onPress={() => setCat('all')} />
          {Object.entries(CATEGORIES).map(([k, v]) => <Chip key={k} label={`${v} (${EXERCISES.filter(e => e.category === k).length})`} active={cat === k} onPress={() => setCat(k)} />)}
        </View>
        {list.length === 0 ? <Card><Text style={[type.small, { color: colors.muted }]}>No exercises match "{q}".</Text></Card> : (
          <View style={s.grid}>
            {list.map(e => {
              const fit = suitability(e, tags);
              if (inline && picked?.id === e.id) return <View key={e.id} style={{ width: '100%' }}>{detail}</View>; // details open in place
              return (
                <Pressable key={e.id} onPress={() => { setPicked(e); setInline(true); }} style={[s.tile, wide && { width: '32%' }]}>
                  <Text style={[type.label, { color: colors.muted }]}>{CATEGORIES[e.category]}</Text>
                  <Text style={[type.h3, { color: fit.level === 'avoid' ? colors.muted : colors.ink }]}>{e.name}</Text>
                  <Text style={[type.small, { color: colors.muted }]}>{MODE_LABEL[e.mode]} · {e.level}{e.equipment !== 'none' ? ` · ${e.equipment}` : ''}{e.impact === 'high' ? ' · high impact' : ''}</Text>
                  {fit.level !== 'ok' && <Text style={[type.small, { color: fit.level === 'avoid' ? colors.sos : colors.warn, fontWeight: '600' }]}>{fit.level === 'avoid' ? 'Avoid' : 'Caution'}: {fit.reasons.join(', ')}</Text>}
                </Pressable>
              );
            })}
          </View>
        )}
        <SectionTitle eyebrow="Your log" title="Workout history" />
        {history === null ? <Text style={[type.small, { color: colors.muted }]}>Loading…</Text>
          : table.length === 0 ? <Card><Text style={[type.small, { color: colors.muted }]}>No workout history yet. Finished sets appear here.</Text></Card>
          : (
            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <View style={[s.trow, { backgroundColor: colors.surface }]}>
                {['Exercise', 'Reps', 'Sets', 'Time', 'Date'].map((h, i) => <Text key={h} style={[type.label, s.cell, i === 0 && { flex: 2.4 }, { color: colors.muted }]}>{h}</Text>)}
              </View>
              {table.map(r => (
                <View key={r.exercise + r.date} style={s.trow}>
                  <Text style={[type.small, s.cell, { flex: 2.4, color: colors.ink }]}>{r.exercise}</Text>
                  <Text style={[type.small, s.cell, { color: colors.ink }]}>{r.reps || '–'}</Text>
                  <Text style={[type.small, s.cell, { color: colors.ink }]}>{r.sets}</Text>
                  <Text style={[type.small, s.cell, { color: colors.ink }]}>{r.seconds}s</Text>
                  <Text style={[type.small, s.cell, { color: colors.muted }]}>{r.date}</Text>
                </View>
              ))}
            </Card>
          )}
        <Disclaimer>Rule-based form guidance from camera pose estimation, not medical advice. Video stays on your device; only the exercise name and the form issue go to the voice coach. Squat, push-up, lunge, curl and shoulder-press logic adapted from ai-gym-coach by Shradha Khapra.</Disclaimer>
        {!GYM_CAMERA_SUPPORTED && <Text style={[type.small, { color: colors.accent }]}>The live camera coach runs in the web app for now; breathing and timed exercises work here.</Text>}
      </View>
    </Screen>
  );
}

function Stepper({ label, value, onChange, min = 1, max = 50, step = 1 }: { label: string; value: number; onChange: (n: number) => void; min?: number; max?: number; step?: number }) {
  const set = (n: number) => onChange(Math.max(min, Math.min(max, n)));
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
      <Text style={[type.small, { color: colors.fg }]}>{label}</Text>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
        <Pressable onPress={() => set(value - step)} style={s.step} accessibilityLabel={`Decrease ${label}`}><Text style={[type.h3, { color: colors.ink }]}>−</Text></Pressable>
        <Text style={[type.h3, { color: colors.ink, minWidth: 32, textAlign: 'center' }]}>{value}</Text>
        <Pressable onPress={() => set(value + step)} style={s.step} accessibilityLabel={`Increase ${label}`}><Text style={[type.h3, { color: colors.ink }]}>+</Text></Pressable>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  tile: { width: '100%', gap: 4, backgroundColor: colors.card, borderRadius: radius.control, borderWidth: 1, borderColor: colors.line, padding: 14 },
  step: { width: 36, height: 36, borderRadius: 18, borderWidth: 1, borderColor: colors.line, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bg },
  voice: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, height: 36, borderRadius: 18, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.card },
  coach: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: colors.card, borderWidth: 1.5, borderColor: colors.good, borderRadius: radius.control, padding: 14 },
  status: { borderRadius: radius.control, paddingHorizontal: 14, paddingVertical: 12 },
  metric: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  row: { flexDirection: 'row', justifyContent: 'space-between', gap: 8, paddingVertical: 6, borderTopWidth: 1, borderTopColor: colors.line },
  tags: { backgroundColor: colors.surface, borderRadius: radius.control, padding: 12 },
  planLink: { backgroundColor: colors.card, borderRadius: radius.control, borderWidth: 1, borderColor: colors.line, padding: 12 },
  search: { backgroundColor: colors.card, borderRadius: radius.control, borderWidth: 1, borderColor: colors.line, paddingHorizontal: 14, paddingVertical: 12, color: colors.ink, fontSize: 15 },
  trow: { flexDirection: 'row', paddingHorizontal: 14, paddingVertical: 10, borderTopWidth: 1, borderTopColor: colors.line },
  cell: { flex: 1 },
});
