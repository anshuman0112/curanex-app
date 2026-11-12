// Doctor workspace (doctor-only). Follows the agreed workflow: symptoms -> tests -> report -> likely diseases -> treatment.
// Every statement comes from the Curanex library with its source; doses are quoted from official drug labels.
import { useState } from 'react';
import { ActivityIndicator, Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { Card, Chip, Disclaimer, Eyebrow, Field, PrimaryButton, Screen } from '../components/ui';
import { Case, DiffReply, findDiseases, Patient, planTreatment, Source, suggestTests, TestsReply, TreatReply } from '../lib/doctorApi';
import { colors, fonts, type } from '../theme';

const YN = ['Yes', 'No'] as const;
const list = (s: string) => s.split(',').map(x => x.trim()).filter(Boolean);

function Sources({ items }: { items: Source[] }) {
  return (
    <View style={{ gap: 2, marginTop: 4 }}>
      {items.map(s => (
        <Pressable key={s.id} onPress={() => Linking.openURL(s.url)} accessibilityRole="link">
          <Text style={[type.small, { color: colors.accent }]}>{s.title} - {s.section} ({s.source})</Text>
        </Pressable>
      ))}
    </View>
  );
}

function Banner({ text, tone }: { text: string; tone: 'sos' | 'warn' }) {
  return (
    <View style={[s.banner, { backgroundColor: tone === 'sos' ? colors.sos : colors.warn }]}>
      <Text style={[type.small, { color: '#fff', fontFamily: fonts.medium }]}>{text}</Text>
    </View>
  );
}

function Status({ busy, error }: { busy: boolean; error: string }) {
  if (busy) return <View style={{ flexDirection: 'row', gap: 8, alignItems: 'center' }}><ActivityIndicator color={colors.accent} /><Text style={[type.small, { color: colors.muted }]}>Searching the library and checking every statement…</Text></View>;
  return error ? <Text style={[type.small, { color: colors.sos }]}>{error}</Text> : null;
}

function Choice({ label, value, options, onChange }: { label: string; value?: string; options: readonly string[]; onChange: (v: string) => void }) {
  return (
    <View style={{ gap: 6 }}>
      <Text style={[type.label, { color: colors.muted }]}>{label}</Text>
      <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>{options.map(o => <Chip key={o} label={o} active={value === o} onPress={() => onChange(o)} />)}</View>
    </View>
  );
}

export default function DoctorPortal() {
  const [key, setKey] = useState('');
  const [age, setAge] = useState(''); const [sex, setSex] = useState<string>(); const [pregnant, setPregnant] = useState<string>();
  const [weight, setWeight] = useState(''); const [kidney, setKidney] = useState<string>(); const [liver, setLiver] = useState<string>();
  const [allergies, setAllergies] = useState(''); const [meds, setMeds] = useState('');
  const [symptoms, setSymptoms] = useState(''); const [history, setHistory] = useState(''); const [report, setReport] = useState('');
  const [tests, setTests] = useState<TestsReply>(); const [diff, setDiff] = useState<DiffReply>(); const [treat, setTreat] = useState<TreatReply>();
  const [busy, setBusy] = useState(''); const [error, setError] = useState(''); const [open, setOpen] = useState<string>();

  const patient = (): Patient => ({
    ...(age ? { age: Number(age) } : {}), ...(sex ? { sex: sex.toLowerCase() } : {}), ...(pregnant ? { pregnant: pregnant === 'Yes' } : {}),
    ...(weight ? { weight_kg: Number(weight) } : {}), ...(kidney ? { kidney_disease: kidney === 'Yes' } : {}), ...(liver ? { liver_disease: liver === 'Yes' } : {}),
    allergies: list(allergies), current_medicines: list(meds),
  });
  const reportValues = () => Object.fromEntries(report.split('\n').map(l => l.split(':')).filter(p => p.length >= 2 && p[0].trim()).map(p => [p[0].trim(), p.slice(1).join(':').trim()]));
  const theCase = (): Case => ({ patient: patient(), symptoms, ...(history ? { history } : {}), ...(report.trim() ? { report: reportValues() } : {}) });

  async function run<T>(step: string, f: () => Promise<T>, set: (v: T) => void) {
    if (!key) { setError('Enter the doctor key first.'); return; }
    if (symptoms.trim().length < 3) { setError('Describe the symptoms first.'); return; }
    setBusy(step); setError('');
    try { set(await f()); } catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.'); } finally { setBusy(''); }
  }

  return (
    <Screen>
      <Eyebrow>Doctor workspace</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Curanex for doctors</Text>
      <Text style={[type.small, { color: colors.muted, marginTop: 6 }]}>
        Answers come only from the Curanex library (StatPearls, MedlinePlus, official drug labels) and every statement is checked against its source. You decide.
      </Text>

      <Card style={{ marginTop: 16, gap: 12 }}>
        <Field label="Doctor key" value={key} onChangeText={setKey} secureTextEntry autoCapitalize="none" placeholder="Given by the Curanex admin" />
      </Card>

      <Card style={{ marginTop: 12, gap: 12 }}>
        <Text style={[type.h3, { color: colors.ink }]}>1. Patient and symptoms</Text>
        <View style={s.row}>
          <View style={{ flex: 1 }}><Field label="Age" value={age} onChangeText={setAge} keyboardType="numeric" /></View>
          <View style={{ flex: 1 }}><Field label="Weight (kg)" value={weight} onChangeText={setWeight} keyboardType="numeric" /></View>
        </View>
        <Choice label="Sex" value={sex} options={['Male', 'Female']} onChange={setSex} />
        {sex === 'Female' && <Choice label="Pregnant" value={pregnant} options={YN} onChange={setPregnant} />}
        <View style={s.row}>
          <View style={{ flex: 1 }}><Choice label="Kidney disease" value={kidney} options={YN} onChange={setKidney} /></View>
          <View style={{ flex: 1 }}><Choice label="Liver disease" value={liver} options={YN} onChange={setLiver} /></View>
        </View>
        <Field label="Allergies (comma separated)" value={allergies} onChangeText={setAllergies} placeholder="e.g. penicillin" />
        <Field label="Current medicines (comma separated)" value={meds} onChangeText={setMeds} />
        <Field label="What the patient says" value={symptoms} onChangeText={setSymptoms} multiline placeholder="e.g. fever for 3 days, a lot of cough, difficulty breathing" />
        <Field label="History (optional)" value={history} onChangeText={setHistory} multiline />
        <PrimaryButton label={busy === 'tests' ? 'Working…' : 'Suggest tests'} disabled={!!busy} onPress={() => run('tests', () => suggestTests(key, theCase()), setTests)} />
      </Card>

      {busy === 'tests' || (error && !busy && !tests) ? <View style={{ marginTop: 10 }}><Status busy={busy === 'tests'} error={error} /></View> : null}
      {tests && (
        <Card style={{ marginTop: 12, gap: 10 }}>
          {tests.emergency && <Banner tone="sos" text={`Emergency sign: ${tests.emergency}`} />}
          {!!tests.missing_info.length && <Banner tone="warn" text={`Please add: ${tests.missing_info.join(', ')}`} />}
          <Text style={[type.h3, { color: colors.ink }]}>Suggested tests</Text>
          {!tests.tests.length && <Text style={[type.small, { color: colors.muted }]}>{tests.note ?? 'No test suggestion was supported by the library. Try describing the symptoms in more detail.'}</Text>}
          {tests.tests.map(t => (
            <View key={t.test} style={s.item}>
              <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{t.test}</Text>
              <Text style={[type.small, { color: colors.fg }]}>{t.why}</Text>
              <Sources items={t.sources} />
            </View>
          ))}
          {tests.dropped > 0 && <Text style={[type.small, { color: colors.muted }]}>{tests.dropped} statement(s) were removed because the library did not support them.</Text>}
        </Card>
      )}

      <Card style={{ marginTop: 12, gap: 12 }}>
        <Text style={[type.h3, { color: colors.ink }]}>2. Report results</Text>
        <Field label="One result per line, as name: value" value={report} onChangeText={setReport} multiline placeholder={'WBC: 15,800 (high)\nSpO2: 93%\nChest X-ray: right lower lobe consolidation'} />
        <PrimaryButton label={busy === 'diff' ? 'Working…' : 'Find likely diseases'} disabled={!!busy} onPress={() => run('diff', () => findDiseases(key, theCase()), v => { setDiff(v); setTreat(undefined); })} />
      </Card>
      {busy === 'diff' && <View style={{ marginTop: 10 }}><Status busy error="" /></View>}
      {diff && (
        <Card style={{ marginTop: 12, gap: 10 }}>
          {diff.emergency && <Banner tone="sos" text={`Emergency sign: ${diff.emergency}`} />}
          <Text style={[type.h3, { color: colors.ink }]}>Likely diseases (supported by the library)</Text>
          {!diff.supported.length && <Text style={[type.small, { color: colors.muted }]}>None of the suggested diseases was supported by the library for these findings.</Text>}
          {diff.supported.map(d => (
            <View key={d.disease} style={s.item}>
              <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{d.disease}</Text>
              <Text style={[type.small, { color: colors.muted }]}>{d.support} supporting statement(s)</Text>
              {d.evidence.slice(0, 3).map(e => <View key={e.text}><Text style={[type.small, { color: colors.fg }]}>• {e.text}</Text><Sources items={e.sources} /></View>)}
              <PrimaryButton label={busy === `treat:${d.disease}` ? 'Working…' : 'Treatment for this'} disabled={!!busy} style={{ marginTop: 6 }}
                onPress={() => run(`treat:${d.disease}`, () => planTreatment(key, theCase(), d.disease), setTreat)} />
            </View>
          ))}
          {!!diff.also_consider.length && (
            <View style={{ gap: 4 }}>
              <Text style={[type.label, { color: colors.muted }]}>Also consider (not confirmed by the library, no treatment given)</Text>
              <Text style={[type.small, { color: colors.fg }]}>{diff.also_consider.join(' · ')}</Text>
            </View>
          )}
        </Card>
      )}

      {busy.startsWith('treat:') && <View style={{ marginTop: 10 }}><Status busy error="" /></View>}
      {error && !busy && (diff || tests) ? <View style={{ marginTop: 10 }}><Status busy={false} error={error} /></View> : null}
      {treat && (
        <Card style={{ marginTop: 12, gap: 10 }}>
          <Text style={[type.h3, { color: colors.ink }]}>3. Treatment: {treat.disease}</Text>
          {!!treat.missing_info.length && <Banner tone="warn" text={`Add these patient details first: ${treat.missing_info.join(', ')}`} />}
          {treat.plan.map(p => <View key={p.text} style={s.item}><Text style={[type.small, { color: colors.fg }]}>{p.text}</Text><Sources items={p.sources} /></View>)}
          {!!treat.referral.length && <Text style={[type.label, { color: colors.muted }]}>Surgery or specialist</Text>}
          {treat.referral.map(r => <View key={r.text} style={s.item}><Text style={[type.small, { color: colors.fg }]}>{r.text}</Text><Sources items={r.sources} /></View>)}
          {!!treat.medicines.length && <Text style={[type.label, { color: colors.muted }]}>Medicines (doses quoted from the official label)</Text>}
          {treat.medicines.map(m => (
            <View key={m.label.medicine} style={s.item}>
              <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{m.label.medicine}</Text>
              <Text style={[type.small, { color: colors.fg }]}>{m.why}</Text>
              <Sources items={m.sources} />
              {m.label.warnings?.map(w => <Banner key={w} tone="sos" text={w} />)}
              {!m.label.found ? <Text style={[type.small, { color: colors.muted }]}>{m.label.note}</Text> : (
                <>
                  <Pressable onPress={() => setOpen(open === m.label.medicine ? undefined : m.label.medicine)}>
                    <Text style={[type.small, { color: colors.accent, fontFamily: fonts.medium }]}>{open === m.label.medicine ? 'Hide' : 'Show'} official label: dose and warnings</Text>
                  </Pressable>
                  {open === m.label.medicine && m.label.sections?.map((sec, i) => (
                    <View key={`${sec.section}${i}`} style={{ gap: 2 }}>
                      <Text style={[type.label, { color: colors.muted }]}>{sec.section}</Text>
                      <Text style={[type.small, { color: colors.fg }]}>{sec.text}</Text>
                    </View>
                  ))}
                  {m.label.label_url && <Pressable onPress={() => Linking.openURL(m.label.label_url!)}><Text style={[type.small, { color: colors.accent }]}>Open the label on DailyMed</Text></Pressable>}
                  <Text style={[type.small, { color: colors.muted }]}>{m.label.note}</Text>
                </>
              )}
            </View>
          ))}
          {!treat.plan.length && !treat.medicines.length && !treat.missing_info.length && <Text style={[type.small, { color: colors.muted }]}>{treat.note ?? 'The library did not support a treatment plan for this diagnosis.'}</Text>}
          {treat.dropped > 0 && <Text style={[type.small, { color: colors.muted }]}>{treat.dropped} statement(s) were removed because the library did not support them.</Text>}
        </Card>
      )}
      <Disclaimer>Decision support for doctors only. Check every suggestion against the patient and local guidelines.</Disclaimer>
    </Screen>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: 'row', gap: 12 },
  item: { gap: 4, paddingVertical: 8, borderTopWidth: 1, borderTopColor: colors.line },
  banner: { borderRadius: 12, paddingHorizontal: 12, paddingVertical: 8 },
});
