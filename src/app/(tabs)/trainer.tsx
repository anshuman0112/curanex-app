import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Card, DarkCard, Eyebrow, PrimaryButton, Screen, SectionTitle, useWide } from '../../components/ui';
import { Dumbbell, Flame, Sparkle } from '../../components/icons';
import { useHealth } from '../../store/health';
import { TODAY_WORKOUT } from '../../data/content';
import { colors, radius, type } from '../../theme';

export default function TrainerScreen() {
  const wide = useWide();
  const h = useHealth();
  const days = Array.from({ length: 7 }).map((_, i) => { const d = new Date(); d.setDate(d.getDate() - d.getDay() + 1 + i); return d; });
  const doneOn = (d: Date) => h.workouts.some(w => new Date(w.at).toDateString() === d.toDateString());
  const streak = (() => { let n = 0; const d = new Date(); if (!doneOn(d)) d.setDate(d.getDate() - 1); while (doneOn(d)) { n++; d.setDate(d.getDate() - 1); } return n; })();
  const total = h.workouts.reduce((a, w) => a + w.minutes, 0);

  return (
    <Screen>
      <View style={{ paddingTop: 12 }}>
        <Eyebrow>Fitness · Daily</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Gym Trainer</Text>
      </View>

      {/* Week strip */}
      <View style={s.week}>
        {days.map(d => {
          const today = d.toDateString() === new Date().toDateString(); const done = doneOn(d);
          return (
            <View key={d.toISOString()} style={[s.day, today && { backgroundColor: colors.ink }]}>
              <Text style={[type.small, { color: today ? colors.onDarkMuted : colors.muted }]}>{d.toLocaleDateString(undefined, { weekday: 'narrow' })}</Text>
              <Text style={[type.h3, { color: today ? colors.onDark : colors.ink }]}>{d.getDate()}</Text>
              <View style={[s.dayDot, done && { backgroundColor: colors.accentFrom }]} />
            </View>
          );
        })}
      </View>

      <View style={[{ gap: 12, marginTop: 12 }, wide && { flexDirection: 'row' }]}>
        {/* Today's workout */}
        <DarkCard style={[{ gap: 14 }, wide && { flex: 1.5 }]}>
          <Text style={[type.label, { color: colors.onDarkMuted }]}>Today · {TODAY_WORKOUT.focus}</Text>
          <Text style={[type.h1, { color: colors.onDark }]}>{TODAY_WORKOUT.title}</Text>
          <Text style={[type.small, { color: colors.onDarkMuted }]}>{TODAY_WORKOUT.minutes} min · {TODAY_WORKOUT.exercises.length} exercises</Text>
          <View style={s.note}><Sparkle size={12} color={colors.accentFrom} /><Text style={[type.small, { color: colors.onDark, flex: 1 }]}>{TODAY_WORKOUT.note}</Text></View>
          <PrimaryButton label="Start guided workout" onPress={() => router.push('/workout')} />
        </DarkCard>
        <View style={[{ gap: 12 }, wide ? { flex: 1 } : { flexDirection: 'row' }]}>
          <Card style={{ flex: 1, gap: 6 }}><Flame size={20} color={colors.accent} /><Text style={[type.h1, { color: colors.ink }]}>{streak}</Text><Text style={[type.small, { color: colors.muted }]}>day streak</Text></Card>
          <Card style={{ flex: 1, gap: 6 }}><Dumbbell size={20} color={colors.accent} /><Text style={[type.h1, { color: colors.ink }]}>{total}</Text><Text style={[type.small, { color: colors.muted }]}>minutes trained</Text></Card>
        </View>
      </View>

      <SectionTitle eyebrow="Plan" title="Exercises" />
      <View style={{ gap: 10 }}>
        {TODAY_WORKOUT.exercises.map((e, i) => (
          <Card key={e.name} style={s.ex}>
            <View style={s.num}><Text style={[type.h3, { color: colors.accent }]}>{i + 1}</Text></View>
            <View style={{ flex: 1 }}>
              <Text style={[type.h3, { color: colors.ink }]}>{e.name}</Text>
              <Text style={[type.small, { color: colors.muted }]}>{e.muscle} · {e.reps ? `${e.sets} × ${e.reps}` : `${e.seconds}s`}</Text>
              <Text style={[type.small, { color: colors.fg, marginTop: 4 }]}>{e.cue}</Text>
            </View>
          </Card>
        ))}
      </View>

      <Pressable style={s.soon}>
        <Text style={[type.label, { color: colors.accent }]}>Coming soon</Text>
        <Text style={[type.h3, { color: colors.ink, marginTop: 4 }]}>AI form check</Text>
        <Text style={[type.small, { color: colors.muted, marginTop: 2 }]}>Your camera counts reps and corrects posture in real time.</Text>
      </Pressable>
    </Screen>
  );
}

const s = StyleSheet.create({
  week: { flexDirection: 'row', gap: 6, marginTop: 16 },
  day: { flex: 1, alignItems: 'center', gap: 2, paddingVertical: 10, borderRadius: radius.control, backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line },
  dayDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: 'transparent', marginTop: 2 },
  note: { flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: colors.inkSoft, borderRadius: radius.control, padding: 10 },
  ex: { flexDirection: 'row', gap: 14, alignItems: 'flex-start' },
  num: { width: 36, height: 36, borderRadius: 18, backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center' },
  soon: { marginTop: 18, borderRadius: radius.cardSm, borderWidth: 1, borderStyle: 'dashed', borderColor: colors.subtle, padding: 18 },
});
