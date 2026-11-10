import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';
import { router } from 'expo-router';
import { requestRecordingPermissionsAsync, RecordingPresets, setAudioModeAsync, useAudioRecorder } from 'expo-audio';
import { Card, Disclaimer, Eyebrow, PrimaryButton, Screen } from '../components/ui';
import { ArrowL } from '../components/icons';
import { voiceCheck, VoiceCheckResult } from '../lib/api';
import { useHealth } from '../store/health';
import { colors, fonts, type } from '../theme';

const SECONDS = 6;
const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Experimental voice health check: one sustained "aaah", measured on the server, never stored. */
export default function VoiceCheck() {
  const { profile } = useHealth();
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const [stage, setStage] = useState<'idle' | 'recording' | 'checking'>('idle');
  const [left, setLeft] = useState(SECONDS);
  const [error, setError] = useState('');
  const [result, setResult] = useState<VoiceCheckResult | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const sex = profile?.sex === 'Male' || profile?.sex === 'Female' ? profile.sex : '';

  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  async function start() {
    setError(''); setResult(null);
    try {
      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) { setError('Microphone access is off. Allow it in your settings to use the voice check.'); return; }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      recorder.record();
      setStage('recording'); setLeft(SECONDS);
      let n = SECONDS;
      timer.current = setInterval(() => { n -= 1; setLeft(n); if (n <= 0) finish(); }, 1000);
    } catch { setError('The microphone could not start.'); setStage('idle'); }
  }

  async function finish() {
    if (timer.current) { clearInterval(timer.current); timer.current = null; }
    setStage('checking');
    try {
      await recorder.stop();
      await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
      if (!recorder.uri) throw new Error('The recording was empty. Please try again.');
      setResult(await voiceCheck(recorder.uri, sex));
    } catch (e) { setError(e instanceof Error ? e.message : 'The voice check failed. Please try again.'); }
    setStage('idle');
  }

  return (
    <Screen padBottom={60}>
      <Pressable onPress={() => router.back()} style={{ paddingVertical: 12 }} hitSlop={10}><ArrowL size={22} /></Pressable>
      <Eyebrow>Screening · experimental</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Voice health check</Text>
      <Text style={[type.body, { color: colors.muted, marginTop: 6 }]}>
        Measures how steady your voice is (pitch wobble, loudness wobble and breathiness) and compares it with recordings from people with and without Parkinson’s disease.
      </Text>

      <Card style={{ marginTop: 18, gap: 6 }}>
        <Text style={[type.h3, { color: colors.ink }]}>How to record</Text>
        <Text style={[type.body, { color: colors.ink }]}>1. Sit in a quiet room. Turn off fans and the TV.</Text>
        <Text style={[type.body, { color: colors.ink }]}>2. Hold the phone about 15 cm from your mouth.</Text>
        <Text style={[type.body, { color: colors.ink }]}>3. Take a breath, tap Start and say one long, steady “aaah” at a comfortable pitch until the timer ends ({SECONDS} seconds).</Text>
      </Card>

      {!sex && (
        <Card style={{ marginTop: 12 }}>
          <Text style={[type.body, { color: colors.ink }]}>Voice measures differ between men and women, so the check needs your sex. Add it in your profile first.</Text>
          <Pressable onPress={() => router.push('/profile')} style={{ marginTop: 8 }}><Text style={[type.small, { color: colors.accent, fontFamily: fonts.medium }]}>Open profile</Text></Pressable>
        </Card>
      )}

      <View style={{ marginTop: 20, alignItems: 'flex-start', gap: 10 }}>
        {stage === 'recording' ? (
          <PrimaryButton label={`Keep saying “aaah”… ${left}s`} onPress={finish} icon={false} />
        ) : (
          <PrimaryButton label={result || error ? 'Record again' : 'Start'} onPress={start} disabled={!sex || stage === 'checking'} />
        )}
        {stage === 'checking' && <View style={{ flexDirection: 'row', gap: 8, alignItems: 'center' }}><ActivityIndicator color={colors.accent} /><Text style={[type.small, { color: colors.muted }]}>Measuring your voice…</Text></View>}
        {error ? <Text style={[type.body, { color: colors.warn }]}>{error}</Text> : null}
      </View>

      {result && (
        <Card style={{ marginTop: 20, gap: 8 }}>
          <Eyebrow>Experimental result</Eyebrow>
          <Text style={[type.h3, { color: colors.ink }]}>{result.label}</Text>
          {Object.entries(result.measures).map(([k, v]) => (
            <Text key={k} style={[type.body, { color: colors.ink }]}>{k}: <Text style={{ fontFamily: fonts.medium }}>{v}</Text></Text>
          ))}
          <Text style={[type.small, { color: colors.muted }]}>
            Tested on {result.test.test_people.n} people it never saw: found {pct(result.test.test_people.sensitivity)} of people with Parkinson’s and cleared {pct(result.test.test_people.specificity)} of people without it (AUC {result.test.test_people.auc}). That is below the bar Curanex uses for active models.
          </Text>
          <Text style={[type.small, { color: colors.muted }]}>Trained on: {result.trained_on}.</Text>
          <Text style={[type.small, { color: colors.muted }]}>{result.credit}</Text>
          <Disclaimer>{result.notice}</Disclaimer>
        </Card>
      )}
      <Text style={[type.small, { color: colors.muted, marginTop: 16 }]}>Your recording is measured and deleted straight away. It is never stored.</Text>
    </Screen>
  );
}
