import { useEffect, useRef, useState } from 'react';
import { Animated, Easing, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { Close, Mic, Send } from '../components/icons';
import { EmergencyCard } from '../components/blocks';
import { chat } from '../lib/api';
import { getUserContext, useHealth } from '../store/health';
import { colors, fonts, type } from '../theme';

type Mode = 'idle' | 'listening' | 'thinking' | 'speaking';

/**
 * Voice doctor UI. Recording → /transcribe is wired in the next step
 * (needs expo-audio + the backend). Until then you can type what you'd say,
 * and the reply is spoken aloud with expo-speech.
 */
export default function Voice() {
  const h = useHealth();
  const [mode, setMode] = useState<Mode>('idle');
  const [heard, setHeard] = useState('');
  const [reply, setReply] = useState('');
  const [escalate, setEscalate] = useState<string | null>(null);
  const pulse = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(Animated.timing(pulse, { toValue: 1, duration: mode === 'listening' ? 900 : 2200, easing: Easing.inOut(Easing.ease), useNativeDriver: true }));
    pulse.setValue(0); loop.start(); return () => loop.stop();
  }, [mode]);
  useEffect(() => () => { Speech.stop(); }, []);

  async function ask(text: string) {
    if (!text.trim()) return;
    setMode('thinking'); setReply(''); setEscalate(null);
    const res = await chat({ session_id: 'voice', message: text, context: getUserContext(h), history: [] });
    const esc = res.blocks.find(b => b.type === 'escalate');
    if (esc && esc.type === 'escalate') { setEscalate(esc.reason); setMode('idle'); Speech.speak(`${esc.reason}. Please call 112 now.`); return; }
    const t = res.blocks.filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('\n\n') || 'Please continue in the chat to share details or an image.';
    setReply(t); setMode('speaking');
    Speech.speak(t.replace(/Offline demo:[^\n]*\n*/g, ''), { onDone: () => setMode('idle'), onStopped: () => setMode('idle') });
  }

  const scale = pulse.interpolate({ inputRange: [0, 0.5, 1], outputRange: [1, mode === 'listening' ? 1.12 : 1.05, 1] });
  const label = { idle: 'Tap the mic and speak', listening: 'Listening…', thinking: 'Thinking…', speaking: 'Speaking' }[mode];

  return (
    <SafeAreaView style={s.root}>
      <LinearGradient colors={['#120503', '#3E160A', '#120503']} style={StyleSheet.absoluteFill} />
      <View style={s.top}>
        <Text style={[type.label, { color: colors.onDarkMuted }]}>Voice doctor</Text>
        <Pressable onPress={() => { Speech.stop(); router.back(); }} style={s.x}><Close size={20} color={colors.onDark} /></Pressable>
      </View>

      <View style={s.center}>
        <Animated.View style={[s.orbOuter, { transform: [{ scale }] }]}>
          <LinearGradient colors={[colors.accentFrom, colors.accentTo]} style={s.orb} />
        </Animated.View>
        <Text style={[type.h2, { color: colors.onDark, marginTop: 28 }]}>{label}</Text>
        {heard ? <Text style={[type.body, { color: colors.onDarkMuted, marginTop: 8, textAlign: 'center' }]}>“{heard}”</Text> : null}
        {reply ? <Text style={[type.body, { color: colors.onDark, marginTop: 16, textAlign: 'center', maxWidth: 520 }]} numberOfLines={8}>{reply}</Text> : null}
        {escalate ? <View style={{ marginTop: 16, width: '100%', maxWidth: 520 }}><EmergencyCard reason={escalate} /></View> : null}
      </View>

      <View style={s.bottom}>
        <View style={s.typeRow}>
          <TextInput value={heard} onChangeText={setHeard} placeholder="Type what you would say (speech input coming next)" placeholderTextColor={colors.onDarkMuted} style={s.input} onSubmitEditing={() => ask(heard)} />
          <Pressable onPress={() => ask(heard)} style={s.send}><Send size={16} color="#fff" /></Pressable>
        </View>
        <Pressable onPressIn={() => setMode('listening')} onPressOut={() => setMode(m => (m === 'listening' ? 'idle' : m))} style={s.mic} accessibilityLabel="Hold to talk">
          <Mic size={30} color={colors.ink} />
        </Pressable>
        <Text style={[type.small, { color: colors.onDarkMuted }]}>Hold to talk</Text>
      </View>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.ink },
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 14 },
  x: { width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(246,239,227,0.1)', alignItems: 'center', justifyContent: 'center' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24 },
  orbOuter: { width: 200, height: 200, borderRadius: 100, padding: 18, backgroundColor: 'rgba(207,128,71,0.18)' },
  orb: { flex: 1, borderRadius: 100 },
  bottom: { alignItems: 'center', gap: 12, paddingBottom: 32, paddingHorizontal: 20 },
  typeRow: { flexDirection: 'row', gap: 8, width: '100%', maxWidth: 520, alignItems: 'center' },
  input: { flex: 1, borderWidth: 1, borderColor: '#4A3824', borderRadius: 999, paddingHorizontal: 16, paddingVertical: 10, color: colors.onDark, fontFamily: fonts.body, fontSize: 14 },
  send: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  mic: { width: 84, height: 84, borderRadius: 42, backgroundColor: colors.onDark, alignItems: 'center', justifyContent: 'center' },
});
