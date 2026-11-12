import { router } from 'expo-router';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Circle } from 'react-native-svg';
import { Card, Chip, DarkCard, Eyebrow, GhostButton, Logo, PrimaryButton, ProgressBar, RoundIcon, Screen, SectionTitle, useWide } from '../../components/ui';
import { ArrowUR, Check, Doc, Drop, Dumbbell, Flame, Grid, Leaf, Pill, Shield, Sparkle, Stetho, Camera } from '../../components/icons';
import { bmi, todaysMeals, useHealth } from '../../store/health';
import { colors, fonts, radius, type } from '../../theme';
import models from '../../data/models.json';
import { AreaAlertCard } from '../../components/AreaAlertCard';

const greet = () => { const h = new Date().getHours(); return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening'; };

export default function HomeScreen() {
  const wide = useWide();
  const h = useHealth();
  const p = h.profile!;
  const meals = todaysMeals(h.meals);
  const kcal = meals.reduce((a, m) => a + m.kcal, 0);
  const kcalGoal = 1900;
  const workedOutToday = h.workouts.some(w => new Date(w.at).toDateString() === new Date().toDateString());
  const medsTotal = p.medications.length;
  const medsDone = p.medications.filter(m => h.takenToday.includes(m)).length;

  // Simple, explainable daily score (PLAN-2DAY Block 7)
  const parts = [Math.min(1, kcal / kcalGoal), Math.min(1, h.water / 2500), workedOutToday ? 1 : 0, medsTotal ? medsDone / medsTotal : 1];
  const score = Math.round((parts.reduce((a, b) => a + b, 0) / parts.length) * 100);

  const insight = buildInsight(h, kcal);
  const tierCount = (t: string) => (models as { app_tier: string }[]).filter(m => m.app_tier === t).length;

  return (
    <Screen>
      {/* Top bar */}
      <View style={s.top}>
        <Logo />
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
          <Text style={[type.small, { color: colors.muted }]}>{new Date().toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}</Text>
          <Pressable onPress={() => router.push('/profile')} style={s.avatar} accessibilityLabel="Profile">
            <Text style={{ color: colors.onDark, fontFamily: fonts.semibold }}>{p.name.slice(0, 1).toUpperCase() || 'U'}</Text>
          </Pressable>
        </View>
      </View>

      {/* Hero — mirrors the website hero */}
      <View style={s.hero}>
        <LinearGradient colors={['#160604', '#4A1A0B', '#8A3415', '#3B1407']} locations={[0, 0.45, 0.7, 1]} style={StyleSheet.absoluteFill} />
        <Image source={require('../../../assets/images/hero-hand.png')} resizeMode="contain" style={[s.hand, wide && s.handWide]} />
        <View style={[s.heroBody, wide && { maxWidth: '55%' }]}>
          <Eyebrow dark>{greet()}, {p.name || 'there'}</Eyebrow>
          <Text style={[type.hero, { color: colors.onDark, marginTop: 10 }, wide && { fontSize: 52, lineHeight: 56 }]}>Your Everyday{'\n'}Health Partner</Text>
          <Text style={[type.body, { color: colors.onDarkMuted, marginTop: 10, maxWidth: 380 }]}>
            Ask the AI doctor, follow your meal plan, train safely, and reach help in one tap.
          </Text>
          <View style={{ flexDirection: 'row', gap: 10, marginTop: 18, flexWrap: 'wrap' }}>
            <PrimaryButton label="Talk to the AI Doctor" onPress={() => router.push('/doctor')} />
            <GhostButton dark label="Voice" onPress={() => router.push('/voice')} />
          </View>
        </View>
        <View style={[s.reco, wide && { right: 24, left: undefined, width: 260 }]}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8 }}>
            <Sparkle size={12} color={colors.accentFrom} />
            <Text style={[type.small, { color: colors.onDark, fontFamily: fonts.semibold }]}>AI Recommendation</Text>
          </View>
          {recommendations(h, kcal, workedOutToday).map(r => (
            <View key={r} style={{ flexDirection: 'row', gap: 8, alignItems: 'center', marginTop: 4 }}>
              <View style={s.recoDot} /><Text style={[type.small, { color: colors.onDarkMuted }]}>{r}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Today */}
      <SectionTitle eyebrow="Today" title="Your health at a glance" />
      <View style={[s.row, { flexWrap: wide ? 'nowrap' : 'wrap' }]}>
        <DarkCard style={[s.scoreCard, wide ? { flex: 1.2 } : { width: '100%' }]}>
          <ScoreRing value={score} />
          <View style={{ flex: 1, gap: 4 }}>
            <Text style={[type.label, { color: colors.onDarkMuted }]}>Daily health score</Text>
            <Text style={[type.h2, { color: colors.onDark }]}>{score >= 75 ? 'Strong day' : score >= 45 ? 'On track' : 'Let’s get moving'}</Text>
            <Text style={[type.small, { color: colors.onDarkMuted }]}>Meals, water, activity and medicines combined.</Text>
          </View>
        </DarkCard>
        <View style={[s.grid, wide ? { flex: 2 } : { width: '100%' }]}>
          <Stat icon={<Flame size={18} color={colors.accent} />} label="Calories" value={`${kcal}`} unit={`/ ${kcalGoal}`} progress={kcal / kcalGoal} onPress={() => router.push('/nutrition')} />
          <Stat icon={<Drop size={18} color={colors.accent} />} label="Water" value={`${(h.water / 1000).toFixed(1)}`} unit="/ 2.5 L" progress={h.water / 2500} onPress={() => h.addWater(250)} hint="Tap +250 ml" />
          <Stat icon={<Dumbbell size={18} color={colors.accent} />} label="Workout" value={workedOutToday ? 'Done' : 'Pending'} progress={workedOutToday ? 1 : 0} onPress={() => router.push('/trainer')} />
          <Stat icon={<Pill size={18} color={colors.accent} />} label="Medicines" value={`${medsDone}/${medsTotal}`} progress={medsTotal ? medsDone / medsTotal : 1} />
        </View>
      </View>

      {/* Insight */}
      <Card style={s.insight}>
        <View style={s.insightIcon}><Sparkle size={16} color="#fff" /></View>
        <View style={{ flex: 1, gap: 4 }}>
          <Text style={[type.label, { color: colors.accent }]}>Curanex insight</Text>
          <Text style={[type.body, { color: colors.ink }]}>{insight}</Text>
        </View>
      </Card>
      <AreaAlertCard district={p.district} />

      {/* Medicines */}
      {medsTotal > 0 && (
        <>
          <SectionTitle eyebrow="Medicines" title="Today’s doses" />
          <View style={{ gap: 10 }}>
            {p.medications.map(m => {
              const done = h.takenToday.includes(m);
              return (
                <Pressable key={m} onPress={() => h.toggleMed(m)} style={[s.med, done && { backgroundColor: colors.surface }]}>
                  <Pill size={20} color={colors.accent} />
                  <Text style={[type.body, { flex: 1, color: colors.ink, textDecorationLine: done ? 'line-through' : 'none' }]}>{m}</Text>
                  <View style={[s.check, done && { backgroundColor: colors.ink, borderColor: colors.ink }]}>{done && <Check size={14} color="#fff" stroke={2.4} />}</View>
                </Pressable>
              );
            })}
          </View>
        </>
      )}

      {/* Built-in care — the website's feature cards */}
      <SectionTitle eyebrow="Features" title="Built-in care" />
      <View style={s.careGrid}>
        <CareCard tag="Consultation · 24/7" title="AI Doctor" desc="Chat or talk. Routes your question to the right medical model." chips={['Symptom check', 'Voice', 'Reports']} icon={<Stetho size={26} color={colors.onDark} />} onPress={() => router.push('/doctor')} wide={wide} />
        <CareCard tag="Fitness · Daily" title="Gym Trainer" desc="Workouts that respect your conditions, with voice cues." chips={['Workouts', 'Guided', 'Streaks']} icon={<Dumbbell size={26} color={colors.onDark} />} onPress={() => router.push('/trainer')} wide={wide} />
        <CareCard tag="Nutrition · Personal" title="AI Nutritionist" desc="Indian meal plans matched to your body and health." chips={['Meal plans', 'Snap to log']} icon={<Leaf size={26} color={colors.onDark} />} onPress={() => router.push('/nutrition')} wide={wide} />
        <CareCard tag="Safety · Instant" title="Emergency SOS" desc="Hold to alert contacts with live location and medical ID." chips={['One-tap alert', 'First aid']} icon={<Shield size={26} color={colors.onDark} />} onPress={() => router.push('/sos')} wide={wide} sos />
        <CareCard tag="For doctors · PIN" title="Doctor Desk" desc="A separate desk for doctors: tests, likely diseases, treatment and an AI chat, all from the medical library with sources." chips={['Tests', 'Diagnosis', 'Treatment', 'Ask AI']} icon={<Stetho size={26} color={colors.onDark} />} onPress={() => router.push('/doctor-portal')} wide={wide} full />
      </View>

      {/* Quick actions */}
      <SectionTitle eyebrow="Shortcuts" title="Quick actions" />
      <View style={s.quick}>
        <Quick icon={<Doc size={20} color={colors.ink} />} label="Upload report" onPress={() => router.push({ pathname: '/doctor', params: { q: 'Explain my blood report' } })} />
        <Quick icon={<Camera size={20} color={colors.ink} />} label="Snap a meal" onPress={() => router.push('/nutrition')} />
        <Quick icon={<Shield size={20} color={colors.ink} />} label="Medical ID" onPress={() => router.push('/profile')} />
        <Quick icon={<Grid size={20} color={colors.ink} />} label="AI models" onPress={() => router.push('/models')} />
      </View>

      {/* Model library banner — honest numbers */}
      <Pressable onPress={() => router.push('/models')}>
        <View style={s.models}>
          <View style={{ flex: 1, gap: 6 }}>
            <Eyebrow>AI model library</Eyebrow>
            <Text style={[type.h2, { color: colors.ink }]}>{models.length} open medical models across 17 specialties</Text>
            <Text style={[type.small, { color: colors.muted }]}>{tierCount('A') + tierCount('B')} accept inputs you can give from your phone. See which are active.</Text>
          </View>
          <RoundIcon dark><ArrowUR size={18} color="#fff" /></RoundIcon>
        </View>
      </Pressable>

      {/* Closing CTA from the website */}
      <DarkCard style={{ marginTop: 18, padding: 26, gap: 16 }}>
        <Text style={[type.h1, { color: colors.onDark }]}>Worried about a symptom?{'\n'}Let’s take a look.</Text>
        <PrimaryButton label="Start a consult" onPress={() => router.push('/doctor')} />
      </DarkCard>
      <Text style={[type.small, { color: colors.muted, textAlign: 'center', marginTop: 18 }]}>Curanex offers guidance, not diagnosis. BMI {bmi(p) ?? '—'} · Blood group {p.bloodGroup || '—'}</Text>
    </Screen>
  );
}

function recommendations(h: ReturnType<typeof useHealth>, kcal: number, worked: boolean) {
  const out: string[] = [];
  if (h.water < 2000) out.push(`Drink ${((2500 - h.water) / 1000).toFixed(1)} L more water`);
  if (!worked) out.push('20 min brisk walk');
  if (h.profile?.conditions.some(c => /hypert|bp/i.test(c))) out.push('Keep salt under 5 g today');
  if (kcal === 0) out.push('Log your first meal');
  out.push('Sleep before 10:30 PM');
  return out.slice(0, 3);
}

function buildInsight(h: ReturnType<typeof useHealth>, kcal: number) {
  const p = h.profile!;
  const lastWorkoutDays = h.workouts[0] ? Math.floor((Date.now() - h.workouts[0].at) / 86400000) : null;
  if (p.conditions.some(c => /hypert|bp/i.test(c)) && lastWorkoutDays !== null && lastWorkoutDays >= 1)
    return `It’s been ${lastWorkoutDays} day${lastWorkoutDays > 1 ? 's' : ''} since your last workout. With hypertension, 30 minutes of moderate activity most days can help lower blood pressure, and today’s plan is low-impact.`;
  if (p.conditions.some(c => /diabet/i.test(c)) && kcal > 0) return 'You’ve logged meals today. Pairing carbs with protein and fibre (dal, curd, salad) helps keep sugar steadier.';
  if (h.water < 1000) return 'You’re low on water today. Mild dehydration can feel like tiredness or a headache.';
  return 'Log meals and workouts for a few days and Curanex will spot patterns across your diet, activity and symptoms.';
}

function ScoreRing({ value }: { value: number }) {
  const r = 34, c = 2 * Math.PI * r;
  return (
    <View style={{ width: 84, height: 84, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={84} height={84} style={StyleSheet.absoluteFill}>
        <Circle cx={42} cy={42} r={r} stroke="#3A2A18" strokeWidth={8} fill="none" />
        <Circle cx={42} cy={42} r={r} stroke={colors.accentFrom} strokeWidth={8} fill="none" strokeLinecap="round"
          strokeDasharray={`${c}`} strokeDashoffset={c * (1 - value / 100)} transform="rotate(-90 42 42)" />
      </Svg>
      <Text style={{ fontFamily: fonts.semibold, fontSize: 22, color: colors.onDark }}>{value}</Text>
    </View>
  );
}

function Stat({ icon, label, value, unit, progress, onPress, hint }: { icon: React.ReactNode; label: string; value: string; unit?: string; progress: number; onPress?: () => void; hint?: string }) {
  return (
    <Pressable onPress={onPress} disabled={!onPress} style={({ pressed }) => [s.stat, pressed && { opacity: 0.8 }]}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        {icon}<Text style={[type.small, { color: colors.muted }]}>{hint ?? label}</Text>
      </View>
      <Text style={{ fontFamily: fonts.semibold, fontSize: 22, color: colors.ink, marginTop: 10 }}>
        {value}{unit ? <Text style={[type.small, { color: colors.muted }]}> {unit}</Text> : null}
      </Text>
      <View style={{ marginTop: 10 }}><ProgressBar value={progress} /></View>
    </Pressable>
  );
}

function CareCard({ tag, title, desc, chips, icon, onPress, wide, sos, full }: { tag: string; title: string; desc: string; chips: string[]; icon: React.ReactNode; onPress: () => void; wide: boolean; sos?: boolean; full?: boolean }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [{ width: wide && !full ? '49.2%' : '100%' }, pressed && { opacity: 0.9 }]}>
      <DarkCard style={{ minHeight: 210, justifyContent: 'space-between' }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
          <Text style={[type.label, { color: colors.onDarkMuted }]}>{tag}</Text>
          <View style={s.cardArrow}><ArrowUR size={14} color={colors.onDark} /></View>
        </View>
        <View style={{ alignItems: 'center', marginVertical: 8 }}>
          <View style={[s.careIcon, sos && { backgroundColor: colors.sos }]}>{icon}</View>
        </View>
        <View style={{ gap: 6 }}>
          <Text style={[type.h2, { color: colors.onDark }]}>{title}</Text>
          <Text style={[type.small, { color: colors.onDarkMuted }]}>{desc}</Text>
          <View style={{ flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>{chips.map(c => <Chip key={c} label={c} dark />)}</View>
        </View>
      </DarkCard>
    </Pressable>
  );
}

const Quick = ({ icon, label, onPress }: { icon: React.ReactNode; label: string; onPress: () => void }) => (
  <Pressable onPress={onPress} style={({ pressed }) => [s.quickItem, pressed && { opacity: 0.7 }]}>
    <View style={s.quickIcon}>{icon}</View>
    <Text style={[type.small, { color: colors.ink, fontFamily: fonts.medium, textAlign: 'center' }]}>{label}</Text>
  </Pressable>
);

const s = StyleSheet.create({
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12 },
  avatar: { width: 38, height: 38, borderRadius: 19, backgroundColor: colors.ink, alignItems: 'center', justifyContent: 'center' },
  hero: { borderRadius: radius.card, overflow: 'hidden', minHeight: 500, padding: 22, justifyContent: 'flex-start' },
  hand: { position: 'absolute', width: '150%', height: 300, right: '-25%', bottom: 70, opacity: 0.95 },
  handWide: { width: '70%', height: 420, right: '-6%', bottom: 10 },
  heroBody: { zIndex: 2 },
  reco: { position: 'absolute', left: 16, right: 16, bottom: 16, padding: 14, borderRadius: radius.cardSm, backgroundColor: 'rgba(20,8,4,0.62)', borderWidth: 1, borderColor: 'rgba(246,239,227,0.14)' },
  recoDot: { width: 6, height: 6, borderRadius: 3, borderWidth: 1.5, borderColor: colors.accentFrom },
  row: { flexDirection: 'row', gap: 12 },
  scoreCard: { flexDirection: 'row', alignItems: 'center', gap: 16, borderRadius: radius.cardSm },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  stat: { flexGrow: 1, flexBasis: '46%', backgroundColor: colors.card, borderRadius: radius.cardSm, padding: 14, borderWidth: 1, borderColor: colors.line },
  insight: { flexDirection: 'row', gap: 12, marginTop: 12, alignItems: 'flex-start' },
  insightIcon: { width: 32, height: 32, borderRadius: 16, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  med: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: colors.card, borderRadius: radius.control, padding: 14, borderWidth: 1, borderColor: colors.line },
  check: { width: 24, height: 24, borderRadius: 12, borderWidth: 1.5, borderColor: colors.subtle, alignItems: 'center', justifyContent: 'center' },
  careGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 12, justifyContent: 'space-between' },
  cardArrow: { width: 30, height: 30, borderRadius: 15, backgroundColor: colors.inkSoft, alignItems: 'center', justifyContent: 'center' },
  careIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  quick: { flexDirection: 'row', gap: 10 },
  quickItem: { flex: 1, alignItems: 'center', gap: 8, backgroundColor: colors.card, borderRadius: radius.cardSm, paddingVertical: 14, paddingHorizontal: 6, borderWidth: 1, borderColor: colors.line },
  quickIcon: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center' },
  models: { flexDirection: 'row', alignItems: 'center', gap: 14, marginTop: 18, backgroundColor: colors.surface, borderRadius: radius.card, padding: 22 },
});
