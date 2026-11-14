import { useMemo, useState } from 'react';
import { router } from 'expo-router';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { Card, Chip, DarkCard, Eyebrow, GhostButton, PrimaryButton, Screen, SectionTitle, useWide } from '../../components/ui';
import { Dumbbell, Flame, Sparkle } from '../../components/icons';
import { useHealth } from '../../store/health';
import { TODAY_WORKOUT } from '../../data/content';
import { colors, radius, type } from '../../theme';
import { CATEGORIES, EXERCISES, getExercise, healthTags, tagLabel } from '../../lib/exerciseEngine';
import { DAY_NAMES, EQUIPMENT_OPTIONS, GOAL_OPTIONS, defaultPrefs, latestVitals, profileKey, todayIndex, usePlan } from '../../lib/workoutPlan';
import type { PlanPrefs } from '../../lib/api';

export default function TrainerScreen() {
  const wide = useWide();
  const h = useHealth();
  const { plan, generate, save } = usePlan();
  const [sel, setSel] = useState(todayIndex());
  const [editing, setEditing] = useState(false);
  const [prefs, setPrefs] = useState<PlanPrefs>(() => defaultPrefs(h.profile));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [showExcluded, setShowExcluded] = useState(false);

  const days = Array.from({ length: 7 }).map((_, i) => { const d = new Date(); d.setDate(d.getDate() - todayIndex() + i); return d; });
  const doneOn = (d: Date) => h.workouts.some(w => new Date(w.at).toDateString() === d.toDateString());
  const streak = (() => { let n = 0; const d = new Date(); if (!doneOn(d)) d.setDate(d.getDate() - 1); while (doneOn(d)) { n++; d.setDate(d.getDate() - 1); } return n; })();
  const total = h.workouts.reduce((a, w) => a + w.minutes, 0);
  const tags = useMemo(() => healthTags(h.profile), [h.profile]);
  const vitals = latestVitals(h.watch);
  const stale = !!plan && plan.profileKey !== profileKey(h.profile);
  const day = plan?.days?.[sel];

  async function build() {
    setBusy(true); setErr('');
    try { await generate(h.profile, h.watch, prefs); setEditing(false); setSel(todayIndex()); }
    catch { setErr('Could not reach the Curanex server to build your plan. Check your connection and try again.'); }
    finally { setBusy(false); }
  }

  const builder = (
    <Card style={{ gap: 12 }}>
      <Text style={[type.label, { color: colors.accent }]}>Personal weekly plan</Text>
      <Text style={[type.h2, { color: colors.ink }]}>{plan ? 'Change my plan' : 'Build my weekly plan'}</Text>
      <Text style={[type.small, { color: colors.muted }]}>
        The AI picks only from {EXERCISES.length} vetted exercises (gym, yoga, cardio, mobility, low-impact, breathing). Anything unsafe for your health details is removed first.
      </Text>
      <View style={s.health}>
        <Text style={[type.small, { color: colors.ink }]}>
          Using: {h.profile ? [h.profile.age && `age ${h.profile.age}`, h.profile.weightKg && `${h.profile.weightKg} kg`, h.profile.goal && `goal: ${h.profile.goal}`].filter(Boolean).join(' · ') : 'no profile yet'}
          {tags.length ? ` · ${tags.map(tagLabel).join(', ')}` : ''}
          {vitals?.bp_sys ? ` · BP ${vitals.bp_sys}/${vitals.bp_dia ?? '?'}` : ''}
        </Text>
        {!h.profile && <Pressable onPress={() => router.push('/profile')}><Text style={[type.small, { color: colors.accent, marginTop: 4 }]}>Add your health details for a safer plan →</Text></Pressable>}
      </View>
      <Text style={[type.label, { color: colors.muted }]}>Goal</Text>
      <View style={s.wrap}>{GOAL_OPTIONS.map(g => <Chip key={g.id} label={g.label} active={prefs.goal === g.id} onPress={() => setPrefs(p => ({ ...p, goal: g.id }))} />)}</View>
      <Text style={[type.label, { color: colors.muted }]}>Days per week</Text>
      <View style={s.wrap}>{[2, 3, 4, 5, 6].map(n => <Chip key={n} label={String(n)} active={prefs.days === n} onPress={() => setPrefs(p => ({ ...p, days: n }))} />)}</View>
      <Text style={[type.label, { color: colors.muted }]}>Minutes per session</Text>
      <View style={s.wrap}>{[15, 20, 30, 45, 60].map(n => <Chip key={n} label={`${n}`} active={prefs.minutes === n} onPress={() => setPrefs(p => ({ ...p, minutes: n }))} />)}</View>
      <Text style={[type.label, { color: colors.muted }]}>Level</Text>
      <View style={s.wrap}>{(['beginner', 'intermediate', 'advanced'] as const).map(l => <Chip key={l} label={l[0].toUpperCase() + l.slice(1)} active={prefs.level === l} onPress={() => setPrefs(p => ({ ...p, level: l }))} />)}</View>
      <Text style={[type.label, { color: colors.muted }]}>Equipment you have</Text>
      <View style={s.wrap}>{EQUIPMENT_OPTIONS.map(e => {
        const on = prefs.equipment.includes(e.id);
        return <Chip key={e.id} label={e.label} active={on} onPress={() => setPrefs(p => ({ ...p, equipment: on ? p.equipment.filter(x => x !== e.id) : [...p.equipment, e.id] }))} />;
      })}</View>
      {busy ? (
        <View style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}><ActivityIndicator color={colors.accent} /><Text style={[type.small, { color: colors.muted }]}>Checking your health details and building your week…</Text></View>
      ) : <PrimaryButton label={plan ? 'Rebuild my plan' : 'Build my plan'} onPress={build} />}
      {!!err && <Text style={[type.small, { color: colors.sos }]}>{err}</Text>}
      {plan && !busy && <GhostButton label="Cancel" onPress={() => setEditing(false)} />}
    </Card>
  );

  return (
    <Screen>
      <View style={{ paddingTop: 12 }}>
        <Eyebrow>Fitness · Daily</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Gym Trainer</Text>
      </View>

      {/* Week strip: tap a day to see its plan */}
      <View style={s.week}>
        {days.map((d, i) => {
          const isSel = i === sel; const done = doneOn(d); const pd = plan?.days?.[i];
          return (
            <Pressable key={d.toISOString()} onPress={() => setSel(i)} style={[s.day, isSel && { backgroundColor: colors.ink }]} accessibilityLabel={`${DAY_NAMES[i]} plan`}>
              <Text style={[type.small, { color: isSel ? colors.onDarkMuted : colors.muted }]}>{d.toLocaleDateString(undefined, { weekday: 'narrow' })}</Text>
              <Text style={[type.h3, { color: isSel ? colors.onDark : colors.ink }]}>{d.getDate()}</Text>
              <Text numberOfLines={1} style={[type.small, { fontSize: 10, color: isSel ? colors.onDarkMuted : colors.muted }]}>{pd ? (pd.rest ? 'Rest' : `${pd.minutes}m`) : i === todayIndex() ? 'Today' : ' '}</Text>
              <View style={[s.dayDot, done && { backgroundColor: colors.accentFrom }]} />
            </Pressable>
          );
        })}
      </View>

      {plan === undefined ? <Text style={[type.small, { color: colors.muted, marginTop: 12 }]}>Loading your plan…</Text> : (
        <View style={{ gap: 12, marginTop: 12 }}>
          {stale && !editing && (
            <Pressable onPress={() => setEditing(true)} style={s.warnBox}>
              <Text style={[type.small, { color: colors.ink }]}>Your health details changed since this plan was made. <Text style={{ color: colors.accent, fontWeight: '700' }}>Rebuild it →</Text></Text>
            </Pressable>
          )}
          {plan?.blocked && <View style={[s.warnBox, { borderColor: colors.sos }]}><Text style={[type.body, { color: colors.sos }]}>{plan.blocked}</Text></View>}

          {(!plan || editing) ? builder : (
            <View style={[{ gap: 12 }, wide && { flexDirection: 'row' }]}>
              <DarkCard style={[{ gap: 12 }, wide && { flex: 1.5 }]}>
                <Text style={[type.label, { color: colors.onDarkMuted }]}>{sel === todayIndex() ? 'Today' : DAY_NAMES[sel]} · {day?.rest ? 'Rest day' : day?.focus}</Text>
                <Text style={[type.h1, { color: colors.onDark }]}>{day?.title ?? 'No plan'}</Text>
                {day && !day.rest && <Text style={[type.small, { color: colors.onDarkMuted }]}>{day.minutes} min · {day.items.length} exercises</Text>}
                {!!day?.why && <View style={s.note}><Sparkle size={12} color={colors.accentFrom} /><Text style={[type.small, { color: colors.onDark, flex: 1 }]}>{day.why}</Text></View>}
                {day && !day.rest ? (
                  <>
                    <PrimaryButton label="Start live AI session" onPress={() => router.push(`/gym-coach?day=${sel}`)} />
                    <GhostButton dark label="Guided timer workout" onPress={() => router.push(`/workout?day=${sel}`)} />
                  </>
                ) : <Text style={[type.small, { color: colors.onDark }]}>Recovery day. A gentle walk or the breathing exercises are fine.</Text>}
              </DarkCard>
              <View style={[{ gap: 12 }, wide ? { flex: 1 } : { flexDirection: 'row' }]}>
                <Card style={{ flex: 1, gap: 6 }}><Flame size={20} color={colors.accent} /><Text style={[type.h1, { color: colors.ink }]}>{streak}</Text><Text style={[type.small, { color: colors.muted }]}>day streak</Text></Card>
                <Card style={{ flex: 1, gap: 6 }}><Dumbbell size={20} color={colors.accent} /><Text style={[type.h1, { color: colors.ink }]}>{total}</Text><Text style={[type.small, { color: colors.muted }]}>minutes trained</Text></Card>
              </View>
            </View>
          )}

          {plan && !editing && day && !day.rest && (
            <>
              <SectionTitle eyebrow={`${DAY_NAMES[sel]} plan`} title="Exercises" />
              <View style={{ gap: 10 }}>
                {day.items.map((it, i) => {
                  const e = getExercise(it.id);
                  return (
                    <Pressable key={it.id + i} onPress={() => router.push(`/gym-coach?id=${it.id}`)}>
                      <Card style={s.ex}>
                        <View style={s.num}><Text style={[type.h3, { color: colors.accent }]}>{i + 1}</Text></View>
                        <View style={{ flex: 1 }}>
                          <Text style={[type.h3, { color: colors.ink }]}>{it.name}</Text>
                          <Text style={[type.small, { color: colors.muted }]}>{CATEGORIES[it.category]} · {it.sets} × {it.reps ? `${it.reps} reps` : `${it.seconds}s`}{e ? ` · ${e.muscles.join(', ')}` : ''}</Text>
                          {e && <Text style={[type.small, { color: colors.fg, marginTop: 4 }]}>{e.steps[0]}</Text>}
                          {it.caution && <Text style={[type.small, { color: colors.warn, marginTop: 4 }]}>Take care: {it.caution.join(', ')}. Fewer sets, go gently.</Text>}
                        </View>
                      </Card>
                    </Pressable>
                  );
                })}
              </View>
            </>
          )}

          {plan && !editing && (
            <Card style={{ gap: 8 }}>
              <Text style={[type.label, { color: colors.accent }]}>{plan.source === 'ai' ? 'AI plan · chosen only from vetted exercises' : 'Rule-based plan (AI was unavailable)'}</Text>
              {plan.notes.map((n, i) => <Text key={i} style={[type.small, { color: colors.warn }]}>• {n}</Text>)}
              {plan.tips.map((t, i) => <Text key={'t' + i} style={[type.small, { color: colors.fg }]}>• {t}</Text>)}
              {plan.excluded.length > 0 && (
                <Pressable onPress={() => setShowExcluded(v => !v)}>
                  <Text style={[type.small, { color: colors.ink, fontWeight: '600' }]}>{showExcluded ? '▾' : '▸'} {plan.excluded.length} exercises left out for your safety</Text>
                  {showExcluded && plan.excluded.map(x => <Text key={x.id} style={[type.small, { color: colors.muted }]}>   {x.name}: {x.reason}</Text>)}
                </Pressable>
              )}
              <Text style={[type.small, { color: colors.muted }]}>{plan.disclaimer}</Text>
              <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>
                <GhostButton label="Change plan" onPress={() => { setPrefs(plan.prefs); setEditing(true); }} />
                <GhostButton label="Clear plan" onPress={() => save(null)} />
              </View>
            </Card>
          )}

          {!plan && (
            <Pressable style={s.starter} onPress={() => router.push('/workout')}>
              <Text style={[type.small, { color: colors.muted }]}>No plan yet? Try the starter workout: <Text style={{ color: colors.accent, fontWeight: '600' }}>{TODAY_WORKOUT.title} ({TODAY_WORKOUT.minutes} min) →</Text></Text>
            </Pressable>
          )}
        </View>
      )}

      <Pressable style={s.coachCard} onPress={() => router.push('/gym-coach')} accessibilityRole="button">
        <View style={{ flex: 1, gap: 4 }}>
          <Text style={[type.label, { color: colors.accent }]}>Live · camera + voice</Text>
          <Text style={[type.h2, { color: colors.ink }]}>AI Gym Coach · {EXERCISES.length} exercises</Text>
          <Text style={[type.small, { color: colors.muted }]}>Gym strength, bodyweight, yoga, cardio and weight loss, core, mobility, low-impact and breathing. The camera counts reps, times holds and corrects your form live.</Text>
        </View>
        <View style={s.go}><Text style={[type.h3, { color: colors.onDark }]}>→</Text></View>
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
  wrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  health: { backgroundColor: colors.surface, borderRadius: radius.control, padding: 12 },
  warnBox: { borderWidth: 1.5, borderColor: colors.warn, borderRadius: radius.control, padding: 12, backgroundColor: colors.card },
  starter: { padding: 4 },
  coachCard: { marginTop: 18, flexDirection: 'row', alignItems: 'center', gap: 14, borderRadius: radius.cardSm, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.card, padding: 18 },
  go: { width: 44, height: 44, borderRadius: 22, backgroundColor: colors.ink, alignItems: 'center', justifyContent: 'center' },
});
