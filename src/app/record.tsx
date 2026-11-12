// Patient health record: save a case, share it with a doctor (consent, can be withdrawn), and read "Your doctor has advised".
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import { AuthCard } from '../components/AuthCard';
import { Card, Chip, Disclaimer, Eyebrow, Field, GhostButton, PrimaryButton, Screen } from '../components/ui';
import { api, RecordCase, useSession } from '../lib/accounts';
import { colors, fonts, type } from '../theme';

export default function Record() {
  const s = useSession('patient');
  const [cases, setCases] = useState<RecordCase[] | null>(null);
  const [title, setTitle] = useState(''); const [symptoms, setSymptoms] = useState(''); const [report, setReport] = useState('');
  const [shareTo, setShareTo] = useState<Record<number, number>>({});
  const [doctors, setDoctors] = useState<{ id: number; name: string; specialty: string; clinic: string }[] | null>(null);
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const load = useCallback(async () => {
    if (!s.token) return;
    api<{ id: number; name: string; specialty: string; clinic: string }[]>('/records/doctors', s.token).then(setDoctors).catch(() => setDoctors([]));
    try { setCases(await api<RecordCase[]>('/records/cases', s.token)); } catch (e) { setError(e instanceof Error ? e.message : 'Could not load your record.'); }
  }, [s.token]);
  useEffect(() => { load(); }, [load]);

  async function act(f: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await f(); await load(); } catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.'); } finally { setBusy(false); }
  }
  const reportValues = () => Object.fromEntries(report.split('\n').map(l => l.split(':')).filter(p => p.length >= 2 && p[0].trim()).map(p => [p[0].trim(), p.slice(1).join(':').trim()]));

  if (!s.ready) return <Screen><ActivityIndicator color={colors.accent} /></Screen>;
  return (
    <Screen>
      <Eyebrow>My health record</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Share with a doctor</Text>
      <Text style={[type.small, { color: colors.muted, marginTop: 6 }]}>Your record is stored encrypted. Only doctors you share a case with can see it, and you can stop sharing at any time.</Text>
      {!s.user ? <AuthCard role="patient" title="Log in to your record" onDone={s.save} /> : s.user.role === 'doctor' ? (
        <Card style={{ marginTop: 16, gap: 8 }}><Text style={[type.body, { color: colors.ink }]}>This is a doctor account. Use the doctor portal.</Text><GhostButton label="Log out" onPress={s.logout} /></Card>
      ) : (
        <>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
            <Text style={[type.small, { color: colors.muted }]}>Logged in as {s.user.name}</Text>
            <GhostButton label="Log out" onPress={s.logout} />
          </View>
          <Card style={{ marginTop: 12, gap: 12 }}>
            <Text style={[type.h3, { color: colors.ink }]}>New case</Text>
            <Field label="Title" value={title} onChangeText={setTitle} placeholder="e.g. Fever and cough" />
            <Field label="How you feel" value={symptoms} onChangeText={setSymptoms} multiline placeholder="e.g. fever for 3 days, a lot of cough" />
            <Field label="Report values (optional, one per line as name: value)" value={report} onChangeText={setReport} multiline />
            <PrimaryButton label={busy ? 'Saving…' : 'Save case'} disabled={busy || title.trim().length < 2 || symptoms.trim().length < 3}
              onPress={() => act(async () => { await api('/records/cases', s.token, 'POST', { title, symptoms, ...(report.trim() ? { report: reportValues() } : {}) }); setTitle(''); setSymptoms(''); setReport(''); })} />
          </Card>
          {!!error && <Text style={[type.small, { color: colors.sos, marginTop: 8 }]}>{error}</Text>}
          {cases === null && !error && <ActivityIndicator color={colors.accent} style={{ marginTop: 16 }} />}
          {cases && !cases.length && <Text style={[type.small, { color: colors.muted, marginTop: 16 }]}>No cases yet. Save one above to share it with your doctor.</Text>}
          {cases?.map(c => (
            <Card key={c.id} style={{ marginTop: 12, gap: 8 }}>
              <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{c.title}</Text>
              <Text style={[type.small, { color: colors.fg }]}>{c.symptoms}</Text>
              {c.doctor_plans.map(p => (
                <View key={p.confirmed} style={{ backgroundColor: colors.surface, borderRadius: 12, padding: 12, gap: 4 }}>
                  <Text style={[type.label, { color: colors.good }]}>Your doctor has advised · {p.doctor_name}</Text>
                  <Text style={[type.small, { color: colors.ink, fontFamily: fonts.medium }]}>{p.diagnosis}</Text>
                  {!!p.tests.length && <Text style={[type.small, { color: colors.fg }]}>Tests: {p.tests.join(', ')}</Text>}
                  {!!p.medicines.length && <Text style={[type.small, { color: colors.fg }]}>Medicines: {p.medicines.join(', ')}</Text>}
                  {!!p.advice && <Text style={[type.small, { color: colors.fg }]}>{p.advice}</Text>}
                  {!!p.referral && <Text style={[type.small, { color: colors.fg }]}>Referral: {p.referral}</Text>}
                </View>
              ))}
              {c.shared_with.map(d => (
                <View key={d.id} style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text style={[type.small, { color: colors.muted }]}>Shared with {d.name}</Text>
                  <Chip label="Stop sharing" onPress={() => act(() => api(`/records/cases/${c.id}/share/${d.id}`, s.token, 'DELETE'))} />
                </View>
              ))}
              <Text style={[type.label, { color: colors.muted }]}>Choose a verified doctor</Text>
              {doctors === null && <ActivityIndicator color={colors.accent} />}
              {doctors && !doctors.length && <Text style={[type.small, { color: colors.muted }]}>No verified doctors yet. Please check again later.</Text>}
              <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>
                {doctors?.map(d => <Chip key={d.id} label={`${/^dr\.?\s/i.test(d.name) ? "" : "Dr "}${d.name}${d.specialty ? ` · ${d.specialty}` : ''}`} active={shareTo[c.id] === d.id} onPress={() => setShareTo({ ...shareTo, [c.id]: d.id })} />)}
              </View>
              <PrimaryButton label="Share with this doctor" disabled={busy || !shareTo[c.id]}
                onPress={() => act(() => api(`/records/cases/${c.id}/share`, s.token, 'POST', { doctor_id: shareTo[c.id] }))} />
            </Card>
          ))}
          <Disclaimer>Your doctor's advice is shown exactly as they confirmed it. In an emergency, call 108.</Disclaimer>
        </>
      )}
    </Screen>
  );
}
