import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router, useLocalSearchParams } from 'expo-router';
import * as DocumentPicker from 'expo-document-picker';
import { Chip, Disclaimer, RoundIcon } from '../../components/ui';
import { CarePlanCard, EmergencyCard, FormRequestCard, ModelResultCard, TextBlock, UploadRequestCard } from '../../components/blocks';
import { Close, Doc, Mic, Plus, Send, Sparkle, Stetho } from '../../components/icons';
import { fetchModels, health, identifyUpload, uploadImage } from '../../lib/api';
import { useDoctor } from '../../store/doctor';
import { useHealth } from '../../store/health';
import { QUICK_PROMPTS } from '../../data/content';
import { colors, fonts, MAX_W, radius, type } from '../../theme';

type LiveModel = { id: string; name: string; integrated: boolean; verified?: boolean; upload_kind?: string | null; own_screen?: string };
const UPLOAD_LABEL: Record<string, string> = { chest_xray: 'Chest X-ray', knee_xray: 'Knee X-ray', bone_xray: 'Arm or hand X-ray', brain_mri: 'Brain MRI', blood_cell: 'Blood cell', endoscopy: 'Endoscopy image', dental_photo: 'Teeth photo', skin_photo: 'Skin photo', fundus: 'Eye image', ultrasound: 'Ultrasound image', dermoscopy: 'Dermoscope image' };

export default function DoctorScreen() {
  const h = useHealth();
  const params = useLocalSearchParams<{ q?: string }>();
  const { msgs, busy, answered, markAnswered, send: sendTurn, clear } = useDoctor();
  const [input, setInput] = useState('');
  const [online, setOnline] = useState<boolean | null>(null);
  const [menu, setMenu] = useState(false);
  const [live, setLive] = useState<LiveModel[] | null>(null);
  const [reading, setReading] = useState(false);
  const [attachError, setAttachError] = useState('');
  const scroll = useRef<React.ComponentRef<typeof ScrollView>>(null);

  useEffect(() => { health().then(setOnline); }, []);
  useEffect(() => { if (params.q) setInput(params.q); }, [params.q]);

  async function send(message: string, extra?: Parameters<typeof sendTurn>[1]) {
    setInput('');
    await sendTurn(message, extra);
  }

  useEffect(() => { if (menu) fetchModels<LiveModel>().then(m => setLive(m ? m.filter(x => x.integrated) : [])); }, [menu]);

  function choose(m: LiveModel) {
    if (m.own_screen) { setMenu(false); router.push(m.own_screen as '/voice-check'); return; }  // e.g. the voice check records on its own screen
    setMenu(false); setAttachError('');
    sendTurn('', { model_id: m.id, label: m.upload_kind ? `Check my ${(UPLOAD_LABEL[m.upload_kind] ?? 'image').toLowerCase()}` : m.name });
  }

  async function attachReport() {
    setAttachError('');
    const res = await DocumentPicker.getDocumentAsync({ type: ['application/pdf', 'image/*', 'application/dicom', 'application/zip'], copyToCacheDirectory: true });
    if (res.canceled) return;
    const file = res.assets[0];
    setReading(true);
    try {
      const got = await identifyUpload(file.uri, file.name, file.mimeType ?? 'application/octet-stream');
      if (got.route === 'report') {
        setMenu(false);
        await sendTurn('Please explain my report.', { report_text: got.text, label: `📄 ${file.name}` });
      } else if (got.model_id) {
        setMenu(false);
        await sendTurn(`This is my ${got.kind}. Please check it.`, { image_id: got.image_id, model_id: got.model_id, label: `📎 ${got.kind[0].toUpperCase()}${got.kind.slice(1)}` });
      } else {
        // no checked model for this file (ECG, CT, spine X-ray, anything else): straight to the AI, clearly labelled
        setMenu(false);
        const what = got.kind === 'unknown' || got.kind === 'other' ? 'medical image' : got.kind;
        const seen = got.findings.length ? got.findings.join('; ') : 'the AI image reader could not describe it clearly';
        await sendTurn(`I uploaded a ${what}${got.body_part ? ` (${got.body_part})` : ''}. Curanex has no checked model for this kind of file, so an AI image reader looked at it and noted: ${seen}. ` +
          'Explain in simple words what this may mean, say clearly that it is not a diagnosis, and tell me which doctor or test I should get.',
          { label: `📎 ${what[0].toUpperCase()}${what.slice(1)} (AI reading)` });
      }
    } catch (e) { setAttachError(e instanceof Error ? e.message : 'That file could not be read.'); }
    setReading(false);
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
            {msgs.length > 0 && <Pressable onPress={clear} hitSlop={8} style={{ marginRight: 12 }} accessibilityLabel="Clear this chat"><Text style={[type.small, { color: colors.muted }]}>Clear chat</Text></Pressable>}
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
                  case 'care_plan': return <CarePlanCard key={key} block={b} />;
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
          {menu && (
            <View style={s.menu}>
              <View style={s.menuHead}>
                <Text style={[type.h3, { color: colors.ink }]}>Upload or run a check</Text>
                <Pressable onPress={() => setMenu(false)} accessibilityLabel="Close menu"><Close size={18} /></Pressable>
              </View>
              <Pressable onPress={attachReport} disabled={reading} style={s.reportBtn} accessibilityLabel="Upload a report, X-ray, scan or photo">
                {reading ? <ActivityIndicator color={colors.accent} /> : <Doc size={18} color={colors.accent} />}
                <View style={{ flex: 1 }}>
                  <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{reading ? 'Looking at your file…' : 'Upload anything'}</Text>
                  <Text style={[type.small, { color: colors.muted }]}>Report, X-ray, ECG, MRI, scan or photo. I’ll work out what it is and check it.</Text>
                </View>
              </Pressable>
              {attachError ? <Text style={[type.small, { color: colors.warn }]}>{attachError}</Text> : null}
              {live === null && <ActivityIndicator color={colors.accent} />}
              {live?.length === 0 && <Text style={[type.small, { color: colors.muted }]}>Checks aren’t available right now because the server isn’t connected.</Text>}
              {!!live?.some(m => m.upload_kind) && <Text style={[type.label, { color: colors.muted }]}>Upload an image</Text>}
              <View style={s.chips}>{live?.filter(m => m.upload_kind).map(m => <Chip key={m.id} label={['fundus', 'skin_photo', 'dental_photo', 'ct_scan'].includes(m.upload_kind ?? '') ? m.name : UPLOAD_LABEL[m.upload_kind ?? ''] ?? m.name} onPress={() => choose(m)} />)}</View>
              {!!live?.some(m => !m.upload_kind) && <Text style={[type.label, { color: colors.muted }]}>Quick checks</Text>}
              <View style={s.chips}>{live?.filter(m => !m.upload_kind).map(m => <Chip key={m.id} label={m.name} onPress={() => choose(m)} />)}</View>
            </View>
          )}
          <View style={s.composer}>
            <Pressable onPress={() => setMenu(v => !v)} style={s.attach} accessibilityLabel="Upload or run a check"><Plus size={20} color={colors.ink} /></Pressable>
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
  composer: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, backgroundColor: colors.card, borderRadius: 26, borderWidth: 1, borderColor: colors.line, padding: 6 },
  attach: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center' },
  menu: { backgroundColor: colors.card, borderRadius: radius.cardSm, borderWidth: 1, borderColor: colors.line, padding: 14, gap: 10, marginBottom: 10 },
  menuHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  reportBtn: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: colors.surface, borderRadius: radius.cardSm, padding: 12 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  input: { flex: 1, fontFamily: fonts.body, fontSize: 15, color: colors.ink, maxHeight: 120, paddingVertical: 10 },
  send: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
});
