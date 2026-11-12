import { useState } from 'react';
import { ActivityIndicator, Image, Linking, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { router } from 'expo-router';
import type { Block } from '../lib/api';
import { colors, fonts, radius, type } from '../theme';
import { Camera, Phone, Shield, Sparkle } from './icons';
import { Chip, PrimaryButton, ProgressBar } from './ui';

/** Assistant text bubble */
export const TextBlock = ({ text }: { text: string }) => (
  <View style={s.bot}><Text style={[type.body, { color: colors.ink }]}>{text}</Text></View>
);

/** Red-flag escalation — deterministic, shown instead of an AI reply. */
export function EmergencyCard({ reason }: { reason: string }) {
  return (
    <View style={s.emergency}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
        <Shield size={20} color="#fff" />
        <Text style={[type.label, { color: '#FFD9CF' }]}>Emergency</Text>
      </View>
      <Text style={[type.h2, { color: '#fff', marginTop: 8 }]}>{reason}: get help now</Text>
      <Text style={[type.body, { color: '#FFE6DF', marginTop: 6 }]}>This could be serious. Don’t wait for an online answer. Call emergency services, and keep the person still and calm.</Text>
      <View style={{ flexDirection: 'row', gap: 10, marginTop: 16, flexWrap: 'wrap' }}>
        <Pressable onPress={() => Linking.openURL('tel:112')} style={s.emBtn}>
          <Phone size={18} color={colors.sos} /><Text style={[type.body, { color: colors.sos, fontFamily: fonts.semibold }]}>Call 112</Text>
        </Pressable>
        <Pressable onPress={() => router.push('/sos')} style={[s.emBtn, { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#FFC8BB' }]}>
          <Text style={[type.body, { color: '#fff', fontFamily: fonts.semibold }]}>Alert my contacts</Text>
        </Pressable>
      </View>
    </View>
  );
}

/** Dynamic form built from the model's input schema. Pre-filled values come from the health profile. */
export function FormRequestCard({ block, onSubmit, done }: { block: Extract<Block, { type: 'form_request' }>; onSubmit: (v: Record<string, string>) => void; done?: boolean }) {
  const [vals, setVals] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(block.defaults ?? {}).map(([k, v]) => [k, String(v)])));
  const missing = block.fields.filter(f => !f.optional && !vals[f.key]?.trim());
  return (
    <View style={s.tool}>
      <Text style={[type.label, { color: colors.accent }]}>Needs a few details</Text>
      <Text style={[type.h3, { color: colors.ink, marginTop: 4 }]}>{block.title}</Text>
      {block.defaults && <Text style={[type.small, { color: colors.muted, marginTop: 4 }]}>Filled in from your report. Please check each value against it before running.</Text>}
      {block.prefilled && (
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {Object.entries(block.prefilled).map(([k, v]) => <Chip key={k} label={`${k.toUpperCase()} ${v} · from profile`} />)}
        </View>
      )}
      <View style={{ gap: 10, marginTop: 12 }}>
        {block.fields.map(f => (
          <View key={f.key} style={{ gap: 4 }}>
            <Text style={[type.small, { color: colors.muted }]}>{f.label}{f.unit ? ` (${f.unit})` : ''}{f.optional ? ' · optional' : ''}</Text>
            {f.options ? (
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6 }}>
                {f.options.map(o => <Chip key={o} label={o} active={vals[f.key] === o} onPress={done ? undefined : () => setVals(v => ({ ...v, [f.key]: o }))} />)}
              </View>
            ) : (
            <TextInput editable={!done} value={vals[f.key] ?? ''} onChangeText={t => setVals(v => ({ ...v, [f.key]: t }))}
              keyboardType={f.type === 'number' ? 'decimal-pad' : 'default'} placeholder={f.type === 'number' ? '0' : ''} placeholderTextColor={colors.subtle} style={s.input} />
            )}
          </View>
        ))}
      </View>
      <Text style={[type.small, { color: colors.muted, marginTop: 10 }]}>Don’t know a value? Skip this and I’ll suggest which test to take.</Text>
      {!done && <PrimaryButton label="Run check" disabled={missing.length > 0} onPress={() => onSubmit(vals)} style={{ marginTop: 12 }} />}
    </View>
  );
}

const KIND_LABEL: Record<string, string> = { skin_photo: 'Skin photo', dental_photo: 'Teeth photo', chest_xray: 'Chest X-ray', knee_xray: 'Knee X-ray', bone_xray: 'Arm or hand X-ray', brain_mri: 'Brain MRI image', blood_cell: 'Blood cell image', endoscopy: 'Endoscopy image', ct_scan: 'CT image', fundus: 'Eye (fundus) image', report: 'Lab report', ultrasound: 'Ultrasound image', dermoscopy: 'Dermoscope image' };

/** Kinds where a hospital scan file (DICOM, or a .zip of one series) can be uploaded instead of a picture. */
const SCAN_KINDS = new Set(['ct_scan', 'brain_mri', 'chest_xray', 'knee_xray', 'bone_xray', 'ultrasound']);

export function UploadRequestCard({ block, onSubmit, done }: { block: Extract<Block, { type: 'upload_request' }>; onSubmit: (uri: string) => Promise<void>; done?: boolean }) {
  const [uri, setUri] = useState<string | null>(null);
  const [scanName, setScanName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const analyse = async (image: string) => {
    setBusy(true); setError('');
    try { await onSubmit(image); } catch (e) { setError(e instanceof Error ? e.message : 'That image could not be used.'); setUri(null); setScanName(null); }
    setBusy(false);
  };
  const pick = async (camera: boolean) => {
    const opts: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.8 };
    const res = camera && Platform.OS !== 'web' ? await ImagePicker.launchCameraAsync(opts) : await ImagePicker.launchImageLibraryAsync(opts);
    if (!res.canceled) { setUri(res.assets[0].uri); setScanName(null); }
  };
  const pickScan = async () => {
    const res = await DocumentPicker.getDocumentAsync({ type: '*/*', copyToCacheDirectory: true });
    if (res.canceled) return;
    const f = res.assets[0];
    if (!/\.(dcm|dicom|zip)$/i.test(f.name) && !/dicom|zip/i.test(f.mimeType ?? '') && !/^[^.]+$/.test(f.name)) { setError('Choose a DICOM file (.dcm) or a .zip of one series.'); return; }
    setError(''); setUri(f.uri); setScanName(f.name);
  };
  return (
    <View style={s.tool}>
      <Text style={[type.label, { color: colors.accent }]}>Image needed</Text>
      <Text style={[type.h3, { color: colors.ink, marginTop: 4 }]}>{KIND_LABEL[block.kind] ?? 'Image'}</Text>
      <Text style={[type.small, { color: colors.muted, marginTop: 4 }]}>{block.instructions}</Text>
      {uri && scanName ? <Text style={[type.body, { color: colors.ink, marginTop: 12 }]}>🗂 {scanName}: only the image is kept; names and IDs in the file are removed.</Text>
        : uri ? <Image source={{ uri }} style={s.preview} /> : (
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginTop: 12 }}>
          {Platform.OS !== 'web' && <Pressable disabled={done} onPress={() => pick(true)} style={s.pickBtn}><Camera size={18} /><Text style={[type.small, { fontFamily: fonts.medium }]}>Take photo</Text></Pressable>}
          <Pressable disabled={done} onPress={() => pick(false)} style={s.pickBtn}><Text style={[type.small, { fontFamily: fonts.medium }]}>Choose file</Text></Pressable>
          {SCAN_KINDS.has(block.kind) && <Pressable disabled={done} onPress={pickScan} style={s.pickBtn}><Text style={[type.small, { fontFamily: fonts.medium }]}>Scan file (DICOM)</Text></Pressable>}
        </View>
      )}
      {uri && !done && (
        <View style={{ flexDirection: 'row', gap: 10, marginTop: 12, alignItems: 'center' }}>
          <PrimaryButton label={busy ? 'Analysing…' : 'Analyse image'} disabled={busy} onPress={() => analyse(uri)} />
          <Pressable onPress={() => { setUri(null); setScanName(null); }}><Text style={[type.small, { color: colors.muted }]}>{scanName ? 'Choose another' : 'Retake'}</Text></Pressable>
          {busy && <ActivityIndicator color={colors.accent} />}
        </View>
      )}
      {error ? <Text style={[type.small, { color: colors.warn, marginTop: 10 }]}>{error}</Text> : null}
    </View>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <View style={{ gap: 6, marginTop: 14 }}><Text style={[type.label, { color: colors.muted }]}>{title}</Text>{children}</View>;
}

/** Symptom guidance. Everything here has passed the backend's safety checks; the disclaimer always shows. */
export function CarePlanCard({ block }: { block: Extract<Block, { type: 'care_plan' }> }) {
  const urgent = block.urgency === 'doctor_today' || block.urgency === 'emergency';
  return (
    <View style={s.tool}>
      <View style={[s.urgency, { backgroundColor: urgent ? colors.warn : colors.surface }]}>
        <Text style={[type.body, { color: urgent ? '#fff' : colors.ink, fontFamily: fonts.medium }]}>{block.urgency_label}</Text>
      </View>
      <Section title="Possible causes">
        {block.causes.map(c => <View key={c.name}>
          <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{c.name}{c.likelihood ? ` · ${c.likelihood}` : ''}</Text>
          <Text style={[type.small, { color: colors.muted }]}>{c.why}</Text>
        </View>)}
      </Section>
      {block.tests.length > 0 && <Section title="Tests to ask about">
        {block.tests.map(t => <View key={t.name}>
          <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{t.name}</Text>
          <Text style={[type.small, { color: colors.muted }]}>{t.why}</Text>
        </View>)}
      </Section>}
      {(block.medicines.length > 0 || !!block.medicine_note) && <Section title="Medicines to ask a pharmacist about">
        {block.medicines.map(m => <View key={m.name}>
          <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{m.name}</Text>
          <Text style={[type.small, { color: colors.muted }]}>{m.use}{m.caution ? ` ${m.caution}` : ''}</Text>
        </View>)}
        {block.medicines.length > 0 && <Text style={[type.small, { color: colors.muted }]}>Ask your pharmacist or doctor how much is right for you. Check the label for allergies.</Text>}
        {block.medicine_note ? <Text style={[type.small, { color: colors.ink }]}>{block.medicine_note}</Text> : null}
      </Section>}
      {block.self_care.length > 0 && <Section title="What you can do now">
        {block.self_care.map(x => <Text key={x} style={[type.small, { color: colors.ink }]}>• {x}</Text>)}
      </Section>}
      {block.see_doctor_if.length > 0 && <Section title="See a doctor if">
        {block.see_doctor_if.map(x => <Text key={x} style={[type.small, { color: colors.ink }]}>• {x}</Text>)}
      </Section>}
      {block.sources.length > 0 && <Section title="Read more">
        {block.sources.map(l => <Pressable key={l.url} onPress={() => Linking.openURL(l.url)}><Text style={[type.small, { color: colors.accent }]}>{l.title} · MedlinePlus</Text></Pressable>)}
      </Section>}
      <Text style={[type.small, { color: colors.muted, marginTop: 14 }]}>{block.basis} In an emergency, call 112.</Text>
    </View>
  );
}

export function ModelResultCard({ block }: { block: Extract<Block, { type: 'model_result' }> }) {
  const [open, setOpen] = useState(false);
  const probs = Object.entries(block.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <View style={s.result}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, flex: 1 }}>
          <Sparkle size={12} color={colors.accentFrom} />
          <Text style={[type.small, { color: colors.onDarkMuted }]} numberOfLines={1}>{block.model_name}</Text>
        </View>
        <View style={s.badge}><Text style={[type.label, { color: colors.ink, fontSize: 9 }]}>{block.demo ? 'Demo data' : 'Not a diagnosis'}</Text></View>
      </View>
      {block.flag && !block.demo && (
        <View style={[s.flag, { backgroundColor: block.flag === 'green' ? colors.good : colors.warn }]} accessibilityRole="text">
          <Text style={[type.small, { color: '#fff', fontFamily: fonts.medium }]}>
            {block.flag === 'green' ? 'No problems found in these results' : 'Please see a doctor about this result'}
          </Text>
        </View>
      )}
      <Text style={[type.h2, { color: colors.onDark, marginTop: 10 }]}>{block.label}</Text>
      {block.verified === false && !block.demo && <Text style={[type.small, { color: colors.accentFrom, marginTop: 4 }]}>Unverified: accuracy not proven, see limitations</Text>}
      {block.warnings?.map(w => <Text key={w} style={[type.small, { color: colors.accentFrom, marginTop: 4 }]}>{w}</Text>)}
      {block.heatmap_url && (
        <View style={{ marginTop: 12 }}>
          <Image source={{ uri: block.heatmap_url }} style={s.heatmap} accessibilityLabel="Heatmap of where the model looked" resizeMode="contain" />
          <Text style={[type.small, { color: colors.onDarkMuted, marginTop: 4 }]}>Orange shows where the model looked for: {block.explanation?.heatmap_finding}</Text>
        </View>
      )}
      <View style={{ gap: 10, marginTop: 12 }}>
        {probs.map(([k, v]) => (
          <View key={k} style={{ gap: 4 }}>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
              <Text style={[type.small, { color: colors.onDark }]}>{k}</Text>
              <Text style={[type.small, { color: colors.onDarkMuted }]}>{Math.round(v * 100)}%</Text>
            </View>
            <ProgressBar value={v} color={colors.accentFrom} track="#3A2A18" />
          </View>
        ))}
      </View>
      {!!block.explanation?.top_factors?.length && (
        <Text style={[type.small, { color: colors.onDarkMuted, marginTop: 12 }]}>
          What drove this: {block.explanation.top_factors.map(([name, v]) => `${name} (${v >= 0 ? 'raised' : 'lowered'} it)`).join(', ')}
        </Text>
      )}
      <Pressable onPress={() => setOpen(o => !o)} style={{ marginTop: 14 }}>
        <Text style={[type.small, { color: colors.accentFrom, fontFamily: fonts.medium }]}>{open ? 'Hide' : 'Show'} model details & limitations</Text>
      </Pressable>
      {open && (
        <View style={{ marginTop: 8, gap: 4 }}>
          <Text style={[type.small, { color: colors.onDarkMuted }]}>Trained on: {block.trained_on}</Text>
          <Text style={[type.small, { color: colors.onDarkMuted }]}>Limitations: {block.limitations}</Text>
          {block.credit && <Text style={[type.small, { color: colors.onDarkMuted }]}>{block.credit}</Text>}
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  urgency: { borderRadius: radius.cardSm, paddingHorizontal: 12, paddingVertical: 10 },
  heatmap: { width: '100%', maxWidth: 360, aspectRatio: 1, borderRadius: radius.cardSm, alignSelf: 'center' },
  bot: { backgroundColor: colors.card, borderRadius: radius.cardSm, borderTopLeftRadius: 6, padding: 14, borderWidth: 1, borderColor: colors.line, maxWidth: 620 },
  emergency: { backgroundColor: colors.sos, borderRadius: radius.cardSm, padding: 18, maxWidth: 620 },
  emBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#fff', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 999 },
  tool: { backgroundColor: colors.card, borderRadius: radius.cardSm, padding: 16, borderWidth: 1, borderColor: colors.accentFrom, maxWidth: 620 },
  input: { backgroundColor: colors.bg, borderRadius: radius.control, paddingHorizontal: 12, paddingVertical: 10, fontFamily: fonts.body, fontSize: 15, color: colors.ink, borderWidth: 1, borderColor: colors.line },
  pickBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 14, paddingVertical: 10, borderRadius: 999, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.bg },
  preview: { width: '100%', height: 200, borderRadius: radius.control, marginTop: 12, backgroundColor: colors.surface },
  result: { backgroundColor: colors.ink, borderRadius: radius.cardSm, padding: 18, maxWidth: 620 },
  badge: { backgroundColor: colors.surface, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 4 },
  flag: { alignSelf: 'flex-start', borderRadius: 999, paddingHorizontal: 12, paddingVertical: 6, marginTop: 12 },
});
