import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { TODAY_WORKOUT } from '../data/content';
import { useHealth } from '../store/health';
import { colors, fonts, type } from '../theme';
import { Close, Pause, Play } from '../components/icons';
import { PrimaryButton, ProgressBar } from '../components/ui';

const REST = 20; const SEC_PER_REP = 3;
type Step = { kind: 'work' | 'rest'; ex: number; set: number; seconds: number };
const steps: Step[] = TODAY_WORKOUT.exercises.flatMap((e, ex) => Array.from({ length: e.sets }).flatMap((_, set) => {
  const work: Step = { kind: 'work', ex, set, seconds: e.seconds ?? (e.reps ?? 10) * SEC_PER_REP };
  return [work, { kind: 'rest' as const, ex, set, seconds: REST }];
})).slice(0, -1);

export default function Workout() {
  const { addWorkout } = useHealth();
  const [i, setI] = useState(0);
  const [left, setLeft] = useState(steps[0].seconds);
  const [running, setRunning] = useState(true);
  const [done, setDone] = useState(false);
  const started = useRef(Date.now());
  const say = (t: string) => { try { Speech.stop(); Speech.speak(t, { rate: 1.0 }); } catch {} };

  const step = steps[i]; const ex = TODAY_WORKOUT.exercises[step.ex];
  useEffect(() => {
    if (step.kind === 'work') say(`${ex.name}. ${ex.reps ? `${ex.reps} reps` : `${ex.seconds} seconds`}. ${ex.cue}`);
    else { const next = TODAY_WORKOUT.exercises[steps[i + 1]?.ex]; say(`Rest. Next: ${next?.name ?? 'finish'}`); }
  }, [i]);

  useEffect(() => {
    if (!running || done) return;
    const t = setInterval(() => setLeft(l => {
      if (l === 11 && step.kind === 'work') say('10 seconds left');
      if (l <= 1) { if (i + 1 >= steps.length) finish(); else { setI(i + 1); return steps[i + 1].seconds; } return 0; }
      return l - 1;
    }), 1000);
    return () => clearInterval(t);
  }, [running, i, done]);

  function finish() {
    setDone(true); setRunning(false); say('Workout complete. Great job.');
    addWorkout({ title: TODAY_WORKOUT.title, minutes: Math.max(1, Math.round((Date.now() - started.current) / 60000)) });
  }

  if (done) return (
    <SafeAreaView style={[s.root, { alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 }]}>
      <Text style={[type.label, { color: colors.accentFrom }]}>Logged to your health record</Text>
      <Text style={[type.hero, { color: colors.onDark, textAlign: 'center' }]}>Workout complete</Text>
      <Text style={[type.body, { color: colors.onDarkMuted, textAlign: 'center' }]}>Your doctor and nutrition plan will take today’s activity into account.</Text>
      <PrimaryButton label="Back to trainer" onPress={() => router.back()} />
    </SafeAreaView>
  );

  return (
    <SafeAreaView style={s.root}>
      <View style={s.top}>
        <Pressable onPress={() => { Speech.stop(); router.back(); }} style={s.x}><Close size={20} color={colors.onDark} /></Pressable>
        <Text style={[type.small, { color: colors.onDarkMuted }]}>Step {i + 1} of {steps.length}</Text>
        <Pressable onPress={finish}><Text style={[type.small, { color: colors.accentFrom }]}>Finish</Text></Pressable>
      </View>
      <View style={{ paddingHorizontal: 24 }}><ProgressBar value={(i + 1) / steps.length} color={colors.accentFrom} track="#3A2A18" /></View>

      <View style={s.center}>
        <Text style={[type.label, { color: step.kind === 'rest' ? colors.good : colors.accentFrom }]}>{step.kind === 'rest' ? 'Rest' : `${ex.muscle} · Set ${step.set + 1} of ${ex.sets}`}</Text>
        <Text style={[type.hero, { color: colors.onDark, textAlign: 'center', marginTop: 8 }]}>{step.kind === 'rest' ? 'Breathe' : ex.name}</Text>
        <Text style={s.timer}>{String(Math.floor(left / 60)).padStart(1, '0')}:{String(left % 60).padStart(2, '0')}</Text>
        <Text style={[type.body, { color: colors.onDarkMuted, textAlign: 'center', maxWidth: 360 }]}>
          {step.kind === 'rest' ? `Next: ${TODAY_WORKOUT.exercises[steps[i + 1]?.ex]?.name ?? 'finish'}` : ex.reps ? `${ex.reps} reps · ${ex.cue}` : ex.cue}
        </Text>
      </View>

      <View style={s.controls}>
        <Pressable onPress={() => { if (i > 0) { setI(i - 1); setLeft(steps[i - 1].seconds); } }} style={s.side}><Text style={[type.small, { color: colors.onDark }]}>Back</Text></Pressable>
        <Pressable onPress={() => setRunning(r => !r)} style={s.play}>{running ? <Pause size={30} color={colors.ink} /> : <Play size={30} color={colors.ink} />}</Pressable>
        <Pressable onPress={() => { if (i + 1 < steps.length) { setI(i + 1); setLeft(steps[i + 1].seconds); } else finish(); }} style={s.side}><Text style={[type.small, { color: colors.onDark }]}>Skip</Text></Pressable>
      </View>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.ink },
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 14 },
  x: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.inkSoft, alignItems: 'center', justifyContent: 'center' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24, gap: 6 },
  timer: { fontFamily: fonts.display, fontSize: 96, color: colors.onDark, marginVertical: 12 },
  controls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 28, paddingBottom: 40 },
  play: { width: 84, height: 84, borderRadius: 42, backgroundColor: colors.onDark, alignItems: 'center', justifyContent: 'center' },
  side: { width: 64, height: 44, borderRadius: 22, borderWidth: 1, borderColor: '#4A3824', alignItems: 'center', justifyContent: 'center' },
});
