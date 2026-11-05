import { useEffect, useRef, useState } from 'react';
import { Animated, Easing, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { requestRecordingPermissionsAsync, RecordingPresets, setAudioModeAsync, useAudioRecorder } from 'expo-audio';
import * as Speech from 'expo-speech';
import { Close, Mic, Send } from '../components/icons';
import { EmergencyCard } from '../components/blocks';
import { Block, transcribe } from '../lib/api';
import { useDoctor } from '../store/doctor';
import { colors, fonts, type } from '../theme';

type Mode = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking';
const MAX_RECORD_MS = 60000;

/** What the app says aloud for a reply. Forms and uploads can't be done by voice, so they are announced and opened in the chat. */
function spokenReply(blocks: Block[]): { text: string; needsChat: boolean } {
  const parts = blocks.filter(b => b.type === 'text').map(b => (b as { text: string }).text.replace(/Offline demo:[^\n]*\n*/g, ''));
  const form = blocks.find(b => b.type === 'form_request');
  const upload = blocks.find(b => b.type === 'upload_request');
  const plan = blocks.find(b => b.type === 'care_plan');
  if (form) parts.push('I need a few details. Please tap Continue in chat and fill in the short form.');
  if (upload) parts.push('I need an image for this. Please tap Continue in chat and upload it.');
  if (plan) parts.push('The full plan, with tests and what to watch for, is in the chat.');
  return { text: parts.join(' ').trim(), needsChat: !!(form || upload || plan) };
}

/** Voice doctor: record, /transcribe, the same /chat as the text screen with via "voice", then a spoken reply. */
export default function Voice() {
  const { send, busy } = useDoctor();
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const [mode, setMode] = useState<Mode>('idle');
  const [heard, setHeard] = useState('');
  const [reply, setReply] = useState('');
  const [error, setError] = useState('');
  const [needsChat, setNeedsChat] = useState(false);
  const [escalate, setEscalate] = useState<string | null>(null);
  const pulse = useRef(new Animated.Value(0)).current;
  const limit = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const loop = Animated.loop(Animated.timing(pulse, { toValue: 1, duration: mode === 'listening' ? 900 : 2200, easing: Easing.inOut(Easing.ease), useNativeDriver: Platform.OS !== 'web' }));
    pulse.setValue(0); loop.start(); return () => loop.stop();
  }, [mode]);
  useEffect(() => () => { Speech.stop(); if (limit.current) clearTimeout(limit.current); }, []);

  async function ask(text: string, language: string | null = null) {
    if (!text.trim() || busy) return;
    setMode('thinking'); setReply(''); setEscalate(null); setError(''); setNeedsChat(false);
    const res = await send(text, undefined, 'voice');
    if (!res) { setMode('idle'); return; }
    const voice = /^hi/i.test(language ?? '') ? 'hi-IN' : 'en-IN';
    const esc = res.blocks.find(b => b.type === 'escalate');
    if (esc && esc.type === 'escalate') { setEscalate(esc.reason); setMode('idle'); Speech.speak(`${esc.reason}. Please call 112 now.`, { language: voice }); return; }
    const said = spokenReply(res.blocks);
    const shown = said.text || 'Please continue in the chat.';
    setReply(shown); setNeedsChat(said.needsChat || !said.text); setMode('speaking');
    Speech.speak(shown, { language: voice, onDone: () => setMode('idle'), onStopped: () => setMode('idle'), onError: () => setMode('idle') });
  }

  async function startRecording() {
    Speech.stop(); setError(''); setReply(''); setEscalate(null); setNeedsChat(false); setHeard('');
    try {
      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) { setError('Microphone access is off. Allow it in your settings, or type below.'); return; }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      recorder.record();
      setMode('listening');
      limit.current = setTimeout(() => { stopRecording(); }, MAX_RECORD_MS);
    } catch { setError('The microphone could not start. Please type below instead.'); setMode('idle'); }
  }

  async function stopRecording() {
    if (limit.current) { clearTimeout(limit.current); limit.current = null; }
    setMode('transcribing');
    try {
      await recorder.stop();
      await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
      if (!recorder.uri) throw new Error('no recording');
      const { text, language } = await transcribe(recorder.uri);
      if (!text) { setError("I didn't catch that. Please try again a little closer to the mic."); setMode('idle'); return; }
      setHeard(text);
      await ask(text, language);
    } catch { setError("Couldn't reach the voice service. Check the connection, or type below."); setMode('idle'); }
  }

  const onMic = () => { if (mode === 'listening') stopRecording(); else if (mode === 'idle' || mode === 'speaking') startRecording(); };
  const scale = pulse.interpolate({ inputRange: [0, 0.5, 1], outputRange: [1, mode === 'listening' ? 1.12 : 1.05, 1] });
  const label = { idle: 'Tap the mic and speak', listening: 'Listening… tap to finish', transcribing: 'Understanding…', thinking: 'Thinking…', speaking: 'Speaking' }[mode];
  const micBusy = mode === 'transcribing' || mode === 'thinking';

  return (
    <SafeAreaView style={s.root}>
      <LinearGradient colors={['#120503', '#3E160A', '#120503']} style={StyleSheet.absoluteFill} />
      <View style={s.top}>
        <Text style={[type.label, { color: colors.onDarkMuted }]}>Voice doctor</Text>
        <Pressable onPress={() => { Speech.stop(); router.back(); }} style={s.x} accessibilityLabel="Close"><Close size={20} color={colors.onDark} /></Pressable>
      </View>

      <View style={s.center}>
        <Animated.View style={[s.orbOuter, { transform: [{ scale }] }]}>
          <LinearGradient colors={[colors.accentFrom, colors.accentTo]} style={s.orb} />
        </Animated.View>
        <Text style={[type.h2, { color: colors.onDark, marginTop: 28 }]}>{label}</Text>
        {heard && mode !== 'listening' ? <Text style={[type.body, { color: colors.onDarkMuted, marginTop: 8, textAlign: 'center' }]}>“{heard}”</Text> : null}
        {reply ? <Text style={[type.body, { color: colors.onDark, marginTop: 16, textAlign: 'center', maxWidth: 520 }]} numberOfLines={8}>{reply}</Text> : null}
        {error ? <Text style={[type.body, { color: colors.onDark, marginTop: 16, textAlign: 'center', maxWidth: 520 }]}>{error}</Text> : null}
        {needsChat ? <Pressable onPress={() => { Speech.stop(); router.navigate('/doctor'); }} style={s.chat}><Text style={[type.body, { color: '#fff' }]}>Continue in chat</Text></Pressable> : null}
        {escalate ? <View style={{ marginTop: 16, width: '100%', maxWidth: 520 }}><EmergencyCard reason={escalate} /></View> : null}
      </View>

      <View style={s.bottom}>
        <View style={s.typeRow}>
          <TextInput value={heard} onChangeText={setHeard} placeholder="Or type what you would say" placeholderTextColor={colors.onDarkMuted} style={s.input} onSubmitEditing={() => ask(heard)} />
          <Pressable onPress={() => ask(heard)} style={s.send} accessibilityLabel="Send"><Send size={16} color="#fff" /></Pressable>
        </View>
        <Pressable onPress={onMic} disabled={micBusy} style={[s.mic, micBusy && { opacity: 0.4 }, mode === 'listening' && { backgroundColor: colors.accentFrom }]} accessibilityLabel={mode === 'listening' ? 'Finish speaking' : 'Start speaking'}>
          <Mic size={30} color={colors.ink} />
        </Pressable>
        <Text style={[type.small, { color: colors.onDarkMuted }]}>{mode === 'listening' ? 'Tap to finish' : 'Tap to talk'}</Text>
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
  chat: { marginTop: 16, backgroundColor: colors.accent, borderRadius: 999, paddingHorizontal: 20, paddingVertical: 10 },
  mic: { width: 84, height: 84, borderRadius: 42, backgroundColor: colors.onDark, alignItems: 'center', justifyContent: 'center' },
});
