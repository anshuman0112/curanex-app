import { useState } from 'react';
import { Image, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { EMPTY_PROFILE, ProfileSection, SECTIONS, sectionValid } from '../components/ProfileForm';
import { GhostButton, Logo, PrimaryButton } from '../components/ui';
import { ArrowL } from '../components/icons';
import { useHealth } from '../store/health';
import { colors, MAX_W, type } from '../theme';

export default function Onboarding() {
  const { setProfile, loadDemo } = useHealth();
  const [step, setStep] = useState(-1); // -1 = welcome
  const [p, setP] = useState(EMPTY_PROFILE);

  if (step === -1) return (
    <View style={{ flex: 1, backgroundColor: colors.ink }}>
      <LinearGradient colors={['#120503', '#3A1408', '#8A3415', '#4A1A0B', '#160604']} locations={[0, 0.2, 0.42, 0.62, 1]} style={StyleSheet.absoluteFill} />
      <Image source={require('../../assets/images/hero-hand.png')} resizeMode="contain" style={s.hand} />
      <SafeAreaView style={{ flex: 1 }}>
        <View style={[s.col, { flex: 1, justifyContent: 'space-between', paddingVertical: 16 }]}>
          <Logo dark />
          <View style={{ gap: 14 }}>
            <Text style={[type.hero, { color: colors.onDark, fontSize: 44, lineHeight: 48 }]}>Intelligent care,{'\n'}always within reach.</Text>
            <Text style={[type.body, { color: colors.onDarkMuted }]}>An AI doctor, nutritionist, trainer and emergency lifeline that share one picture of your health.</Text>
            <PrimaryButton label="Get started" onPress={() => setStep(0)} style={{ marginTop: 8 }} />
            <GhostButton dark label="Explore with a demo profile" onPress={() => { loadDemo(); router.replace('/'); }} style={{ alignSelf: 'flex-start' }} />
            <Text style={[type.small, { color: colors.onDarkMuted, marginTop: 6 }]}>Your data stays on this device unless you use AI features. Curanex gives guidance, not diagnosis.</Text>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );

  const sec = SECTIONS[step];
  const valid = sectionValid(p, sec.key);
  const last = step === SECTIONS.length - 1;
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={[s.col, { paddingVertical: 16, gap: 18 }]} keyboardShouldPersistTaps="handled">
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
            <Pressable onPress={() => setStep(step - 1)} hitSlop={10}><ArrowL size={22} /></Pressable>
            <Text style={[type.small, { color: colors.muted }]}>Step {step + 1} of {SECTIONS.length}</Text>
          </View>
          <View style={{ flexDirection: 'row', gap: 6 }}>
            {SECTIONS.map((_, i) => <View key={i} style={[s.seg, i <= step && { backgroundColor: colors.accent }]} />)}
          </View>
          <View style={{ gap: 6 }}>
            <Text style={[type.h1, { color: colors.ink }]}>{sec.title}</Text>
            <Text style={[type.body, { color: colors.muted }]}>{sec.sub}</Text>
          </View>
          <ProfileSection k={sec.key} p={p} set={setP} />
          <PrimaryButton label={last ? 'Finish setup' : 'Continue'} disabled={!valid}
            onPress={() => { if (last) { setProfile(p); router.replace('/'); } else setStep(step + 1); }} style={{ marginTop: 8 }} />
          {!valid && <Text style={[type.small, { color: colors.muted }]}>{sec.key === 'contacts' ? 'Add at least one emergency contact.' : 'Fill in name, age and sex to continue.'}</Text>}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  col: { width: '100%', maxWidth: Math.min(MAX_W, 560), alignSelf: 'center', paddingHorizontal: 20 },
  hand: { position: 'absolute', top: '12%', left: '-35%', width: '170%', height: 380, opacity: 0.9 },
  seg: { flex: 1, height: 4, borderRadius: 2, backgroundColor: colors.line },
});
