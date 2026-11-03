import { useEffect, useRef, useState } from 'react';
import { Linking, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import * as Location from 'expo-location';
import * as Haptics from 'expo-haptics';
import Svg, { Circle } from 'react-native-svg';
import { Card, Eyebrow, Screen, SectionTitle } from '../../components/ui';
import { Phone, Pin, Shield, Close, Check } from '../../components/icons';
import { useHealth } from '../../store/health';
import { FIRST_AID } from '../../data/content';
import { colors, fonts, radius, type } from '../../theme';

const HOLD_MS = 3000;
const COUNTDOWN = 5;
type Phase = 'idle' | 'holding' | 'countdown' | 'sent';

export default function SOSScreen() {
  const { profile } = useHealth();
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [count, setCount] = useState(COUNTDOWN);
  const [log, setLog] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const start = useRef(0);

  const clear = () => { if (timer.current) clearInterval(timer.current); timer.current = null; };
  useEffect(() => clear, []);

  const onPressIn = () => {
    if (phase !== 'idle') return;
    setPhase('holding'); start.current = Date.now();
    if (Platform.OS !== 'web') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    timer.current = setInterval(() => {
      const p = (Date.now() - start.current) / HOLD_MS; setProgress(Math.min(1, p));
      if (p >= 1) { clear(); beginCountdown(); }
    }, 40);
  };
  const onPressOut = () => { if (phase === 'holding' || timer.current) { if (progress < 1) { clear(); setPhase('idle'); setProgress(0); } } };

  const beginCountdown = () => {
    setPhase('countdown'); setCount(COUNTDOWN);
    if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    let n = COUNTDOWN;
    timer.current = setInterval(() => { n -= 1; setCount(n); if (n <= 0) { clear(); fire(); } }, 1000);
  };
  const cancel = () => { clear(); setPhase('idle'); setProgress(0); setCount(COUNTDOWN); };

  /** Deterministic SOS: location → SMS to contacts with medical ID → dial 112. No AI involved. */
  async function fire() {
    setPhase('sent'); const steps: string[] = [];
    let mapLink = '(location unavailable)';
    try {
      const perm = await Location.requestForegroundPermissionsAsync();
      if (perm.granted) {
        const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
        mapLink = `https://maps.google.com/?q=${pos.coords.latitude},${pos.coords.longitude}`; steps.push('Location captured');
      } else steps.push('Location permission denied, sending without location');
    } catch { steps.push('Could not get location'); }

    const p = profile;
    const body = `EMERGENCY from ${p?.name || 'a Curanex user'}. I need help. Location: ${mapLink}\n` +
      `Medical ID: blood ${p?.bloodGroup || '?'}; conditions: ${p?.conditions.join(', ') || 'none'}; allergies: ${p?.allergies.join(', ') || 'none'}; meds: ${p?.medications.join(', ') || 'none'}`;
    const numbers = (p?.contacts ?? []).map(c => c.phone).filter(Boolean);
    if (numbers.length && Platform.OS !== 'web') {
      const sep = Platform.OS === 'ios' ? '&' : '?';
      await Linking.openURL(`sms:${numbers.join(Platform.OS === 'ios' ? ',' : ';')}${sep}body=${encodeURIComponent(body)}`).catch(() => {});
      steps.push(`Message opened for ${numbers.length} contact${numbers.length > 1 ? 's' : ''}`);
    } else if (Platform.OS === 'web') steps.push('SMS isn’t available in a browser. Call 112 below.');
    else steps.push('No emergency contacts saved');
    setLog(steps);
  }

  const R = 112, C = 2 * Math.PI * R;
  return (
    <Screen>
      <View style={{ paddingTop: 12 }}>
        <Eyebrow>Safety · Instant</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Emergency SOS</Text>
        <Text style={[type.body, { color: colors.muted, marginTop: 4 }]}>Hold the button for 3 seconds. You’ll have 5 seconds to cancel.</Text>
      </View>

      {/* Big button */}
      <View style={s.stage}>
        {phase === 'countdown' ? (
          <View style={{ alignItems: 'center', gap: 16 }}>
            <Text style={[type.label, { color: colors.sos }]}>Alerting in</Text>
            <Text style={{ fontFamily: fonts.display, fontSize: 96, color: colors.sos, lineHeight: 104 }}>{count}</Text>
            <Pressable onPress={cancel} style={s.cancel}><Close size={18} color={colors.ink} /><Text style={[type.body, { fontFamily: fonts.semibold }]}>Cancel</Text></Pressable>
          </View>
        ) : phase === 'sent' ? (
          <View style={{ alignItems: 'center', gap: 12, width: '100%' }}>
            <View style={[s.btn, { backgroundColor: colors.ink }]}><Check size={56} color="#fff" stroke={2.4} /></View>
            {log.map(l => <Text key={l} style={[type.body, { color: colors.ink }]}>• {l}</Text>)}
            <Pressable onPress={() => Linking.openURL('tel:112')} style={[s.call, { backgroundColor: colors.sos }]}>
              <Phone size={20} color="#fff" /><Text style={[type.h3, { color: '#fff' }]}>Call 112 now</Text>
            </Pressable>
            <Pressable onPress={cancel}><Text style={[type.small, { color: colors.muted }]}>I’m safe, reset</Text></Pressable>
          </View>
        ) : (
          <Pressable onPressIn={onPressIn} onPressOut={onPressOut} accessibilityRole="button" accessibilityLabel="Hold for 3 seconds to send SOS">
            <View style={s.ringWrap}>
              <Svg width={260} height={260} style={StyleSheet.absoluteFill}>
                <Circle cx={130} cy={130} r={R} stroke="#F0D5CC" strokeWidth={10} fill="none" />
                <Circle cx={130} cy={130} r={R} stroke={colors.sosDeep} strokeWidth={10} fill="none" strokeLinecap="round"
                  strokeDasharray={`${C}`} strokeDashoffset={C * (1 - progress)} transform="rotate(-90 130 130)" />
              </Svg>
              <View style={[s.btn, phase === 'holding' && { transform: [{ scale: 0.96 }] }]}>
                <Text style={{ fontFamily: fonts.bold, fontSize: 40, color: '#fff', letterSpacing: 2 }}>SOS</Text>
                <Text style={[type.small, { color: '#FFDCD3' }]}>{phase === 'holding' ? 'Keep holding…' : 'Hold 3 seconds'}</Text>
              </View>
            </View>
          </Pressable>
        )}
      </View>

      {/* Quick dial */}
      <SectionTitle eyebrow="Quick dial" title="Call for help" />
      <View style={s.tiles}>
        <Dial label="Emergency" number="112" />
        <Dial label="Ambulance" number="108" />
        {(profile?.contacts ?? []).map(c => <Dial key={c.phone} label={c.relation || 'Contact'} number={c.phone} name={c.name} />)}
      </View>
      <Pressable onPress={() => Linking.openURL('https://www.google.com/maps/search/hospital+near+me')} style={s.hospitals}>
        <Pin size={22} color={colors.ink} />
        <View style={{ flex: 1 }}>
          <Text style={[type.h3, { color: colors.ink }]}>Hospitals near me</Text>
          <Text style={[type.small, { color: colors.muted }]}>Opens Google Maps with directions</Text>
        </View>
      </Pressable>

      {/* Medical ID */}
      <SectionTitle eyebrow="Shared with responders" title="Medical ID" />
      <View style={s.medid}>
        <Row k="Name" v={profile?.name} /><Row k="Blood group" v={profile?.bloodGroup} />
        <Row k="Conditions" v={profile?.conditions.join(', ')} /><Row k="Allergies" v={profile?.allergies.join(', ')} />
        <Row k="Medications" v={profile?.medications.join(', ')} last />
      </View>

      {/* First aid — offline */}
      <SectionTitle eyebrow="Works offline" title="First aid guides" />
      <View style={{ gap: 10 }}>
        {FIRST_AID.map(g => (
          <Card key={g.id} style={{ padding: 0 }}>
            <Pressable onPress={() => setOpen(o => (o === g.id ? null : g.id))} style={{ padding: 16, flexDirection: 'row', alignItems: 'center', gap: 12 }}>
              <View style={s.faIcon}><Shield size={18} color={colors.sos} /></View>
              <View style={{ flex: 1 }}>
                <Text style={[type.h3, { color: colors.ink }]}>{g.title}</Text>
                <Text style={[type.small, { color: colors.muted }]}>{g.when}</Text>
              </View>
              <Text style={[type.h3, { color: colors.muted }]}>{open === g.id ? '–' : '+'}</Text>
            </Pressable>
            {open === g.id && (
              <View style={{ paddingHorizontal: 16, paddingBottom: 16, gap: 8 }}>
                {g.steps.map((st, i) => (
                  <View key={i} style={{ flexDirection: 'row', gap: 10 }}>
                    <Text style={[type.small, { color: colors.sos, fontFamily: fonts.semibold, width: 18 }]}>{i + 1}</Text>
                    <Text style={[type.body, { color: colors.ink, flex: 1 }]}>{st}</Text>
                  </View>
                ))}
              </View>
            )}
          </Card>
        ))}
      </View>
    </Screen>
  );
}

const Dial = ({ label, number, name }: { label: string; number: string; name?: string }) => (
  <Pressable onPress={() => Linking.openURL(`tel:${number}`)} style={({ pressed }) => [s.dial, pressed && { opacity: 0.8 }]}>
    <Phone size={18} color={colors.sos} />
    <Text style={[type.h3, { color: colors.ink, marginTop: 8 }]} numberOfLines={1}>{name ?? number}</Text>
    <Text style={[type.small, { color: colors.muted }]}>{label}</Text>
  </Pressable>
);
const Row = ({ k, v, last }: { k: string; v?: string; last?: boolean }) => (
  <View style={[{ flexDirection: 'row', paddingVertical: 10, gap: 12 }, !last && { borderBottomWidth: 1, borderBottomColor: '#3A2A18' }]}>
    <Text style={[type.small, { color: colors.onDarkMuted, width: 110 }]}>{k}</Text>
    <Text style={[type.body, { color: colors.onDark, flex: 1 }]}>{v || '—'}</Text>
  </View>
);

const s = StyleSheet.create({
  stage: { alignItems: 'center', justifyContent: 'center', minHeight: 320, marginVertical: 16 },
  ringWrap: { width: 260, height: 260, alignItems: 'center', justifyContent: 'center' },
  btn: { width: 200, height: 200, borderRadius: 100, backgroundColor: colors.sos, alignItems: 'center', justifyContent: 'center', gap: 4,
    shadowColor: colors.sos, shadowOpacity: 0.45, shadowRadius: 30, shadowOffset: { width: 0, height: 10 }, elevation: 14 },
  cancel: { flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, paddingHorizontal: 28, paddingVertical: 14, borderRadius: 999 },
  call: { flexDirection: 'row', gap: 10, alignItems: 'center', paddingHorizontal: 28, paddingVertical: 16, borderRadius: 999, marginTop: 6 },
  tiles: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  dial: { flexGrow: 1, flexBasis: '30%', minWidth: 100, backgroundColor: colors.card, borderRadius: radius.cardSm, padding: 14, borderWidth: 1, borderColor: colors.line },
  hospitals: { flexDirection: 'row', alignItems: 'center', gap: 14, marginTop: 10, backgroundColor: colors.surface, borderRadius: radius.cardSm, padding: 16 },
  medid: { backgroundColor: colors.ink, borderRadius: radius.cardSm, paddingHorizontal: 18, paddingVertical: 6 },
  faIcon: { width: 36, height: 36, borderRadius: 18, backgroundColor: '#F7E1DA', alignItems: 'center', justifyContent: 'center' },
});
