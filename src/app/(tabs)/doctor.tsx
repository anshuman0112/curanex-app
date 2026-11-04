import { useEffect, useRef, useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router, useLocalSearchParams } from 'expo-router';
import { Chip, Disclaimer, RoundIcon } from '../../components/ui';
import { EmergencyCard, FormRequestCard, ModelResultCard, TextBlock, UploadRequestCard } from '../../components/blocks';
import { Mic, Send, Sparkle, Stetho } from '../../components/icons';
import { health, uploadImage } from '../../lib/api';
import { useDoctor } from '../../store/doctor';
import { useHealth } from '../../store/health';
import { QUICK_PROMPTS } from '../../data/content';
import { colors, fonts, MAX_W, radius, type } from '../../theme';


export default function DoctorScreen() {
  const h = useHealth();
  const params = useLocalSearchParams<{ q?: string }>();
  const { msgs, busy, answered, markAnswered, send: sendTurn } = useDoctor();
  const [input, setInput] = useState('');
  const [online, setOnline] = useState<boolean | null>(null);
  const scroll = useRef<React.ComponentRef<typeof ScrollView>>(null);

  useEffect(() => { health().then(setOnline); }, []);
  useEffect(() => { if (params.q) setInput(params.q); }, [params.q]);

  async function send(message: string, extra?: Parameters<typeof sendTurn>[1]) {
    setInput('');
    await sendTurn(message, extra);
  }

  useEffect(() => { setTimeout(() => scroll.current?.scrollToEnd({ animated: true }), 80); }, [msgs, busy]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.bg }} edges={['top']}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <View style={s.column}>
          {/* Header */}
          <View style={s.header}>
            <View style={{ flex: 1 }}>
              <Text style={[type.h1, { color: colors.ink }]}>AI Doctor</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 2 }}>
                <View style={[s.status, { backgroundColor: online ? colors.good : online === false ? colors.warn : colors.subtle }]} />
                <Text style={[type.small, { color: colors.muted }]}>{online ? 'Connected · model router active' : online === false ? 'Offline demo · backend not connected' : 'Connecting…'}</Text>
              </View>
            </View>
            <RoundIcon dark onPress={() => router.push('/voice')}><Mic size={18} color="#fff" /></RoundIcon>
          </View>
          <Disclaimer />
        </View>

        <ScrollView ref={scroll} style={{ flex: 1 }} contentContainerStyle={[s.column, { paddingVertical: 16, gap: 14 }]} keyboardShouldPersistTaps="handled">
          {msgs.length === 0 && (
            <View style={s.empty}>
              <View style={s.emptyIcon}><Stetho size={28} color="#fff" /></View>
              <Text style={[type.h2, { color: colors.ink, textAlign: 'center' }]}>How are you feeling, {h.profile?.name || 'today'}?</Text>
              <Text style={[type.body, { color: colors.muted, textAlign: 'center', maxWidth: 420 }]}>
                Describe a symptom, upload a report or image, or just ask a health question. When a trained medical model fits, I’ll run it and explain the result.
              </Text>
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
                {QUICK_PROMPTS.map(q => <Chip key={q} label={q} onPress={() => send(q)} />)}
              </View>
            </View>
          )}

          {msgs.map(m => m.role === 'user' ? (
            <View key={m.id} style={s.user}><Text style={[type.body, { color: colors.onDark }]}>{m.text}</Text></View>
          ) : (
            <View key={m.id} style={{ gap: 8 }}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                <Sparkle size={12} />
                <Text style={[type.small, { color: colors.muted }]}>{m.used?.length ? `Analyzed by ${m.used.join(', ')}` : 'Curanex AI'}</Text>
              </View>
              {m.blocks?.map((b, i) => {
                const key = `${m.id}-${i}`;
                switch (b.type) {
                  case 'text': return <TextBlock key={key} text={b.text} />;
                  case 'escalate': return <EmergencyCard key={key} reason={b.reason} />;
                  case 'model_result': return <ModelResultCard key={key} block={b} />;
                  case 'form_request': return <FormRequestCard key={key} block={b} done={answered[key]}
                    onSubmit={v => { markAnswered(key); send('', { form_values: v, model_id: b.model_id, label: Object.entries(v).filter(([, x]) => x).map(([k, x]) => `${k}: ${x}`).join(' · ') }); }} />;
                  case 'upload_request': return <UploadRequestCard key={key} block={b} done={answered[key]}
                    onSubmit={async uri => { const id = await uploadImage(uri); markAnswered(key); await send('', { image_id: id, model_id: b.model_id, label: '📎 Image uploaded' }); }} />;
                }
              })}
            </View>
          ))}
          {busy && <View style={[s.typing]}><Text style={[type.small, { color: colors.muted }]}>Curanex is thinking…</Text></View>}
        </ScrollView>

        {/* Composer */}
        <View style={[s.column, { paddingBottom: 96 }]}>
          <View style={s.composer}>
            <TextInput value={input} onChangeText={setInput} placeholder="Describe how you feel…" placeholderTextColor={colors.subtle}
              style={s.input} multiline onSubmitEditing={() => send(input)} blurOnSubmit={Platform.OS === 'web'} />
            <Pressable onPress={() => send(input)} disabled={!input.trim() || busy} style={[s.send, (!input.trim() || busy) && { opacity: 0.4 }]} accessibilityLabel="Send">
              <Send size={18} color="#fff" stroke={2.2} />
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  column: { width: '100%', maxWidth: MAX_W - 240, alignSelf: 'center', paddingHorizontal: 16 },
  header: { flexDirection: 'row', alignItems: 'center', paddingTop: 12, paddingBottom: 10 },
  status: { width: 8, height: 8, borderRadius: 4 },
  empty: { alignItems: 'center', gap: 10, paddingVertical: 24 },
  emptyIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center', marginBottom: 6 },
  user: { alignSelf: 'flex-end', backgroundColor: colors.ink, borderRadius: radius.cardSm, borderTopRightRadius: 6, paddingHorizontal: 14, paddingVertical: 10, maxWidth: '82%' },
  typing: { alignSelf: 'flex-start', backgroundColor: colors.surface, borderRadius: 999, paddingHorizontal: 14, paddingVertical: 8 },
  composer: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, backgroundColor: colors.card, borderRadius: 26, borderWidth: 1, borderColor: colors.line, padding: 6, paddingLeft: 16 },
  input: { flex: 1, fontFamily: fonts.body, fontSize: 15, color: colors.ink, maxHeight: 120, paddingVertical: 10 },
  send: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
});
