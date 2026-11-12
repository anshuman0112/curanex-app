// Admin (owner only): approve, reject or suspend doctor accounts by hand.
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import { AuthCard } from '../components/AuthCard';
import { Card, Chip, Eyebrow, GhostButton, Screen } from '../components/ui';
import { api, useSession, User } from '../lib/accounts';
import { colors, fonts, type } from '../theme';

type Doc = User & { created: number };

export default function Admin() {
  const s = useSession('admin');
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [error, setError] = useState('');
  const load = useCallback(async () => {
    if (!s.token) return;
    try { setDocs(await api<Doc[]>('/admin/doctors', s.token)); setError(''); } catch (e) { setError(e instanceof Error ? e.message : 'Could not load doctors.'); }
  }, [s.token]);
  useEffect(() => { load(); }, [load]);
  async function decide(id: number, action: 'approve' | 'reject' | 'suspend') {
    try { await api(`/admin/doctors/${id}/${action}`, s.token, 'POST'); await load(); } catch (e) { setError(e instanceof Error ? e.message : 'Action failed.'); }
  }
  if (!s.ready) return <Screen><ActivityIndicator color={colors.accent} /></Screen>;
  return (
    <Screen>
      <Eyebrow>Admin</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Doctor approvals</Text>
      {!s.user ? <AuthCard role="admin" title="Admin log in" allowSignup={false} onDone={s.save} /> : s.user.role !== 'admin' ? (
        <Card style={{ marginTop: 16, gap: 8 }}>
          <Text style={[type.body, { color: colors.ink }]}>This account is not the admin.</Text>
          <GhostButton label="Log out" onPress={s.logout} />
        </Card>
      ) : (
        <>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
            <Text style={[type.small, { color: colors.muted }]}>Logged in as {s.user.email}</Text>
            <GhostButton label="Log out" onPress={s.logout} />
          </View>
          {!!error && <Text style={[type.small, { color: colors.sos, marginTop: 8 }]}>{error}</Text>}
          {docs === null && !error && <ActivityIndicator color={colors.accent} style={{ marginTop: 16 }} />}
          {docs && !docs.length && <Text style={[type.small, { color: colors.muted, marginTop: 16 }]}>No doctor accounts yet.</Text>}
          {docs?.map(d => (
            <Card key={d.id} style={{ marginTop: 12, gap: 6 }}>
              <Text style={[type.body, { color: colors.ink, fontFamily: fonts.medium }]}>{d.name} · {d.status}</Text>
              <Text style={[type.small, { color: colors.fg }]}>{d.email}{d.details?.phone ? ` · ${d.details.phone}` : ''}</Text>
              <Text style={[type.small, { color: colors.muted }]}>{[d.details?.clinic, d.details?.specialty].filter(Boolean).join(' · ') || 'No clinic details given'}</Text>
              <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                {d.status !== 'approved' && <Chip label="Approve" onPress={() => decide(d.id, 'approve')} />}
                {d.status === 'pending' && <Chip label="Reject" onPress={() => decide(d.id, 'reject')} />}
                {d.status === 'approved' && <Chip label="Suspend" onPress={() => decide(d.id, 'suspend')} />}
              </View>
            </Card>
          ))}
        </>
      )}
    </Screen>
  );
}
