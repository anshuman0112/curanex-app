// Doctor workspace (doctor-only). Follows the agreed workflow: symptoms -> tests -> report -> likely diseases -> treatment.
// Every statement comes from the Curanex library with its source; doses are quoted from official drug labels.
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { AuthCard } from '../components/AuthCard';
import { Card, Chip, Disclaimer, Eyebrow, Field, GhostButton, PrimaryButton, Screen } from '../components/ui';
import { api, RecordCase, useSession } from '../lib/accounts';
import { askAI, AskReply, Case, ChatTurn, DiffReply, DoctorAuth, findDiseases, checkPin, Patient, planTreatment, Source, suggestTests, TestsReply, TreatReply } from '../lib/doctorApi';
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
  const session = useSession('doctor');
  const [key, setKey] = useState('');
  const [cases, setCases] = useState<RecordCase[] | null>(null); const [caseId, setCaseId] = useState<number>();
  const [planDx, setPlanDx] = useState(''); const [planTests, setPlanTests] = useState(''); const [planMeds, setPlanMeds] = useState('');
  const [planAdvice, setPlanAdvice] = useState(''); const [planRef, setPlanRef] = useState(''); const [planMsg, setPlanMsg] = useState('');
  const [age, setAge] = useState(''); const [sex, setSex] = useState<string>(); const [pregnant, setPregnant] = useState<string>();
  const [weight, setWeight] = useState(''); const [kidney, setKidney] = useState<string>(); const [liver, setLiver] = useState<string>();
  const [allergies, setAllergies] = useState(''); const [meds, setMeds] = useState('');
  const [symptoms, setSymptoms] = useState(''); const [history, setHistory] = useState(''); const [report, setReport] = useState('');
  const [chat, setChat] = useState<{ q: string; a?: AskReply; error?: string }[]>([]); const [question, setQuestion] = useState(''); const [asking, setAsking] = useState(false);
  const [pinOk, setPinOk] = useState(false); const [pinMsg, setPinMsg] = useState(''); const [checking, setChecking] = useState(false);
  const [tests, setTests] = useState<TestsReply>(); const [diff, setDiff] = useState<DiffReply>(); const [treat, setTreat] = useState<TreatReply>();
  const [busy, setBusy] = useState(''); const [error, setError] = useState(''); const [open, setOpen] = useState<string>();

  const patient = (): Patient => ({
    ...(age ? { age: Number(age) } : {}), ...(sex ? { sex: sex.toLowerCase() } : {}), ...(pregnant ? { pregnant: pregnant === 'Yes' } : {}),
    ...(weight ? { weight_kg: Number(weight) } : {}), ...(kidney ? { kidney_disease: kidney === 'Yes' } : {}), ...(liver ? { liver_disease: liver === 'Yes' } : {}),
    allergies: list(allergies), current_medicines: list(meds),
  });
  const reportValues = () => Object.fromEntries(report.split('\n').map(l => l.split(':')).filter(p => p.length >= 2 && p[0].trim()).map(p => [p[0].trim(), p.slice(1).join(':').trim()]));
  const approved = session.user?.role === 'doctor' && session.user.status === 'approved';
  const auth = (): DoctorAuth => (approved ? { token: session.token, key } : { key });
  const canUse = pinOk;
  async function enterPin() {
    setChecking(true); setPinMsg('');
    try { setPinOk(await checkPin(key)); setPinMsg('Wrong PIN.'); } catch (e) { setPinMsg(e instanceof Error ? e.message : 'Could not check the PIN.'); } finally { setChecking(false); }
  }
  const loadCases = useCallback(async () => {
    if (!approved || !session.token) return;
    try { setCases(await api<RecordCase[]>('/doctor/cases', session.token)); } catch { setCases([]); }
  }, [approved, session.token]);
  useEffect(() => { loadCases(); }, [loadCases]);
  function openCase(c: RecordCase) {
    const p = (c.patient ?? {}) as Record<string, unknown>;
    setCaseId(c.id); setSymptoms(c.symptoms); setHistory(c.history ?? '');
    setReport(Object.entries(c.report ?? {}).map(([k, v]) => `${k}: ${v}`).join('\n'));
    if (typeof p.age === 'number') setAge(String(p.age));
    if (typeof p.sex === 'string') setSex(p.sex.toLowerCase().startsWith('f') ? 'Female' : 'Male');
    setTests(undefined); setDiff(undefined); setTreat(undefined); setPlanMsg('');
  }
  async function confirmPlan() {
    if (!caseId || !session.token || planDx.trim().length < 2) { setPlanMsg('Open a shared case and write the diagnosis first.'); return; }
    try {
      await api(`/doctor/cases/${caseId}/plan`, session.token, 'POST', { diagnosis: planDx, tests: list(planTests), medicines: list(planMeds), advice: planAdvice, referral: planRef || null });
      setPlanMsg('Plan confirmed. The patient can now see it in their record.'); loadCases();
    } catch (e) { setPlanMsg(e instanceof Error ? e.message : 'Could not save the plan.'); }
  }
  async function sendQuestion() {
    const q = question.trim();
    if (q.length < 2 || asking) return;
    if (!canUse) { setError('Log in with an approved doctor account first.'); return; }
    if (symptoms.trim().length < 3) { setError("Write the patient's symptoms first (or open a shared case), then ask."); return; }
    const history: ChatTurn[] = chat.flatMap(t => [{ role: 'doctor' as const, text: t.q }, ...(t.a ? [{ role: 'ai' as const, text: [...t.a.answer.map(x => x.text), ...t.a.general].join(' ') }] : [])]);
    setChat(c => [...c, { q }]); setQuestion(''); setAsking(true);
    try { const a = await askAI(auth(), theCase(), q, history); setChat(c => c.map((t, i) => (i === c.length - 1 ? { ...t, a } : t))); }
    catch (e) { setChat(c => c.map((t, i) => (i === c.length - 1 ? { ...t, error: e instanceof Error ? e.message : 'No answer.' } : t))); }
    finally { setAsking(false); }
  }
  const theCase = (): Case => ({ patient: patient(), symptoms, ...(history ? { history } : {}), ...(report.trim() ? { report: reportValues() } : {}) });

  async function run<T>(step: string, f: () => Promise<T>, set: (v: T) => void) {
    if (!canUse) { setError('Log in with an approved doctor account first.'); return; }
    if (symptoms.trim().length < 3) { setError('Describe the symptoms first.'); return; }
    setBusy(step); setError('');
    try { set(await f()); } catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.'); } finally { setBusy(''); }
  }

  if (!pinOk) return (
    <Screen>
      <Eyebrow>Doctor workspace</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Doctor Desk</Text>
      <Card style={{ marginTop: 16, gap: 12 }}>
        <Text style={[type.h3, { color: colors.ink }]}>Enter the doctor PIN</Text>
        <Field label="PIN" value={key} onChangeText={setKey} secureTextEntry keyboardType="number-pad" autoCapitalize="none" onSubmitEditing={enterPin} />
        {!!pinMsg && <Text style={[type.small, { color: colors.sos }]}>{pinMsg}</Text>}
        <PrimaryButton label={checking ? 'Checking…' : 'Open doctor portal'} disabled={checking || key.length < 4} onPress={enterPin} />
      </Card>
      <Disclaimer>For doctors only. Patients use the AI Doctor in the Curanex app.</Disclaimer>
    </Screen>
  );

  return (
    <Screen>
      <Eyebrow>Doctor workspace</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Doctor Desk</Text>
      <Text style={[type.small, { color: colors.muted, marginTop: 6 }]}>
        Answers come only from the Curanex library (StatPearls, MedlinePlus, official drug labels) and every statement is checked against its source. You decide.
      </Text>

      {!session.ready ? <ActivityIndicator color={colors.accent} style={{ marginTop: 16 }} /> : !session.user ? (
        <>
          <Card style={{ marginTop: 16 }}><Text style={[type.small, { color: colors.muted }]}>The AI workspace below is open. Log in only to see cases patients shared with you and send them your plan.</Text></Card>
          <AuthCard role="doctor" title="Doctor log in (optional)" onDone={session.save} />
        </>
      ) : (
        <Card style={{ marginTop: 16, gap: 8 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={[type.small, { color: colors.muted }]}>{session.user.name} · {session.user.role === 'doctor' ? session.user.status : 'not a doctor account'}</Text>
            <GhostButton label="Log out" onPress={session.logout} />
          </View>
          {session.user.role === 'doctor' && session.user.status === 'pending' && <Banner tone="warn" text="Your account is waiting for approval by the Curanex team. You'll get access once it is approved." />}
          {session.user.role === 'doctor' && !['approved', 'pending'].includes(session.user.status) && <Banner tone="sos" text={`This account is ${session.user.status}. Contact the Curanex team.`} />}
          {approved && (
            <View style={{ gap: 6 }}>
              <Text style={[type.label, { color: colors.muted }]}>Cases shared with you</Text>
              {cases === null && <ActivityIndicator color={colors.accent} />}
              {cases && !cases.length && <Text style={[type.small, { color: colors.muted }]}>No patient has shared a case with you yet. You can still use the workspace below for a patient in front of you.</Text>}
              <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>{cases?.map(c => <Chip key={c.id} label={`${c.title} (#${c.id})`} active={caseId === c.id} onPress={() => openCase(c)} />)}</View>
            </View>
          )}
        </Card>
      )}

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
        <PrimaryButton label={busy === 'tests' ? 'Working…' : 'Suggest tests'} disabled={!!busy} onPress={() => run('tests', () => suggestTests(auth(), theCase()), setTests)} />
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
        <PrimaryButton label={busy === 'diff' ? 'Working…' : 'Find likely diseases'} disabled={!!busy} onPress={() => run('diff', () => findDiseases(auth(), theCase()), v => { setDiff(v); setTreat(undefined); })} />
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
                onPress={() => run(`treat:${d.disease}`, () => planTreatment(auth(), theCase(), d.disease), setTreat)} />
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
          {!!treat.cautions?.length && <Text style={[type.label, { color: colors.sos }]}>Cautions for this patient</Text>}
          {treat.cautions?.map(c => <View key={c.text} style={s.item}><Text style={[type.small, { color: colors.sos }]}>{c.text}</Text><Sources items={c.sources} /></View>)}
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
      <Card style={{ marginTop: 12, gap: 10 }}>
        <Text style={[type.h3, { color: colors.ink }]}>Ask the AI about this patient</Text>
        <Text style={[type.small, { color: colors.muted }]}>Ask anything about the case above, e.g. "Is TB likely?", "Which antibiotic if allergic to penicillin?", "When should I refer?". Answers come from the medical library with sources; doses are never written by the AI.</Text>
        {chat.map((t, i) => (
          <View key={i} style={{ gap: 6 }}>
            <View style={{ alignSelf: 'flex-end', backgroundColor: colors.ink, borderRadius: 14, paddingVertical: 8, paddingHorizontal: 12, maxWidth: '85%' }}><Text style={[type.small, { color: '#fff' }]}>{t.q}</Text></View>
            <View style={[s.item, { maxWidth: '95%' }]}>
              {!t.a && !t.error && <ActivityIndicator color={colors.accent} />}
              {!!t.error && <Text style={[type.small, { color: colors.sos }]}>{t.error}</Text>}
              {!!t.a?.emergency && <Banner tone="sos" text={t.a.emergency} />}
              {t.a?.answer.map(x => <View key={x.text} style={{ gap: 2 }}><Text style={[type.small, { color: colors.fg }]}>{x.text}</Text><Sources items={x.sources} /></View>)}
              {!!t.a?.general.length && <Text style={[type.label, { color: colors.muted }]}>General AI knowledge (not found in the library, check before use)</Text>}
              {t.a?.general.map(g => <Text key={g} style={[type.small, { color: colors.muted }]}>• {g}</Text>)}
              {!!t.a?.note && <Text style={[type.small, { color: colors.muted }]}>{t.a.note}</Text>}
            </View>
          </View>
        ))}
        <Field label="Your question" value={question} onChangeText={setQuestion} multiline placeholder="Type your question about this patient" />
        <PrimaryButton label={asking ? 'Thinking…' : 'Ask'} disabled={asking || question.trim().length < 2} onPress={sendQuestion} />
      </Card>
      {approved && (
        <Card style={{ marginTop: 12, gap: 10 }}>
          <Text style={[type.h3, { color: colors.ink }]}>4. Confirm your plan for the patient</Text>
          <Text style={[type.small, { color: colors.muted }]}>{caseId ? `Case #${caseId}. Only what you confirm here is shown to the patient.` : 'Open a shared case above to send the patient your plan.'}</Text>
          <Field label="Diagnosis" value={planDx} onChangeText={setPlanDx} />
          <Field label="Tests (comma separated)" value={planTests} onChangeText={setPlanTests} />
          <Field label="Medicines with your instructions (comma separated)" value={planMeds} onChangeText={setPlanMeds} />
          <Field label="Advice to the patient" value={planAdvice} onChangeText={setPlanAdvice} multiline />
          <Field label="Referral (optional)" value={planRef} onChangeText={setPlanRef} />
          {!!planMsg && <Text style={[type.small, { color: planMsg.startsWith('Plan confirmed') ? colors.good : colors.sos }]}>{planMsg}</Text>}
          <PrimaryButton label="Confirm and send to patient" disabled={!caseId} onPress={confirmPlan} />
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
