// Login / sign-up card shared by the patient record, doctor portal and admin screens.
import { useState } from 'react';
import { Text, View } from 'react-native';
import { login, signup, User } from '../lib/accounts';
import { colors, type } from '../theme';
import { Card, Chip, Field, PrimaryButton } from './ui';

export function AuthCard({ role, title, onDone, allowSignup = true }: { role: 'patient' | 'doctor' | 'admin'; title: string; onDone: (token: string, user: User) => void; allowSignup?: boolean }) {
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState(''); const [password, setPassword] = useState(''); const [name, setName] = useState('');
  const [clinic, setClinic] = useState(''); const [specialty, setSpecialty] = useState(''); const [phone, setPhone] = useState('');
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  async function submit() {
    if (!email.includes('@') || password.length < 8) { setError('Enter your email and a password of at least 8 characters.'); return; }
    if (mode === 'signup' && name.trim().length < 2) { setError('Enter your name.'); return; }
    setBusy(true); setError('');
    try {
      const r = mode === 'login' ? await login(email, password)
        : await signup({ email, password, name, role: role === 'doctor' ? 'doctor' : 'patient', ...(role === 'doctor' ? { clinic, specialty, phone } : {}) });
      onDone(r.token, r.user);
    } catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.'); } finally { setBusy(false); }
  }
  return (
    <Card style={{ gap: 12, marginTop: 16 }}>
      <Text style={[type.h3, { color: colors.ink }]}>{title}</Text>
      {allowSignup && <View style={{ flexDirection: 'row', gap: 8 }}>
        <Chip label="Log in" active={mode === 'login'} onPress={() => setMode('login')} />
        <Chip label="Create account" active={mode === 'signup'} onPress={() => setMode('signup')} />
      </View>}
      {mode === 'signup' && <Field label="Full name" value={name} onChangeText={setName} />}
      <Field label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" />
      <Field label="Password" value={password} onChangeText={setPassword} secureTextEntry />
      {mode === 'signup' && role === 'doctor' && <>
        <Field label="Clinic or hospital" value={clinic} onChangeText={setClinic} />
        <Field label="Specialty" value={specialty} onChangeText={setSpecialty} />
        <Field label="Phone (for verification)" value={phone} onChangeText={setPhone} keyboardType="phone-pad" />
        <Text style={[type.small, { color: colors.muted }]}>New doctor accounts are checked and approved by the Curanex team by hand before they can be used.</Text>
      </>}
      {!!error && <Text style={[type.small, { color: colors.sos }]}>{error}</Text>}
      <PrimaryButton label={busy ? 'Please wait…' : mode === 'login' ? 'Log in' : 'Create account'} disabled={busy} onPress={submit} />
    </Card>
  );
}
