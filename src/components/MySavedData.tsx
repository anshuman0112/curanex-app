// "My saved data": everything Curanex stored for this patient in the cloud (chats, uploads, reports, check results, doctor plans).
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Linking, Pressable, Text, View } from 'react-native';
import { api } from '../lib/accounts';
import { colors, fonts, type } from '../theme';
import { Card, Chip } from './ui';

type Row = Record<string, unknown> & { id: number; created_at: string };
type Data = { cloud: boolean; chats: Row[]; uploads: Row[]; reports: Row[]; model_results: Row[]; cases: Row[]; doctor_plans: Row[] };
const TABS = [['chats', 'Chats'], ['uploads', 'Uploads'], ['reports', 'Reports'], ['model_results', 'Check results'], ['doctor_plans', 'Doctor plans']] as const;
const when = (s: string) => new Date(s).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const str = (v: unknown) => (v == null ? '' : String(v));

export function MySavedData({ token }: { token: string }) {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<(typeof TABS)[number][0]>('chats');
  const load = useCallback(async () => {
    setError('');
    try { setData(await api<Data>('/me/data', token)); } catch (e) { setError(e instanceof Error ? e.message : 'Could not load your saved data.'); }
  }, [token]);
  useEffect(() => { load(); }, [load]);

  const rows = data ? (data[tab] as Row[]) : [];
  return (
    <Card style={{ marginTop: 16, gap: 10 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text style={[type.h3, { color: colors.ink }]}>My saved data</Text>
        <Pressable onPress={load}><Text style={[type.small, { color: colors.accent }]}>Refresh</Text></Pressable>
      </View>
      <Text style={[type.small, { color: colors.muted }]}>While you are logged in, your AI Doctor chats, uploads, reports and check results are saved privately to your account.</Text>
      {!data && !error && <ActivityIndicator color={colors.accent} />}
      {!!error && <Text style={[type.small, { color: colors.sos }]}>{error}</Text>}
      {data && !data.cloud && <Text style={[type.small, { color: colors.muted }]}>Cloud saving is not switched on for this server.</Text>}
      {data && (
        <>
          <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>
            {TABS.map(([k, label]) => <Chip key={k} label={`${label} (${(data[k] as Row[]).length})`} active={tab === k} onPress={() => setTab(k)} />)}
          </View>
          {!rows.length && <Text style={[type.small, { color: colors.muted }]}>Nothing saved here yet.</Text>}
          {rows.slice(0, 30).map(r => (
            <View key={r.id} style={{ borderTopWidth: 1, borderColor: colors.line, paddingTop: 8, gap: 2 }}>
              <Text style={[type.small, { color: colors.muted }]}>{when(r.created_at)}</Text>
              {tab === 'chats' && <Text style={[type.small, { color: colors.fg }]}><Text style={{ fontFamily: fonts.medium }}>{r.role === 'user' ? 'You: ' : 'Curanex: '}</Text>{str(r.content).slice(0, 300)}</Text>}
              {tab === 'uploads' && <Text style={[type.small, { color: colors.fg }]}>{str(r.kind) || 'Image'}{r.body_part ? ` · ${str(r.body_part)}` : ''}{r.model_id ? ' · checked by a model' : ''}</Text>}
              {tab === 'reports' && <Text style={[type.small, { color: colors.fg }]}>{str(r.text).slice(0, 200)}</Text>}
              {tab === 'model_results' && <Text style={[type.small, { color: colors.fg }]}><Text style={{ fontFamily: fonts.medium }}>{str(r.model_name)}: </Text>{str(r.label)}{r.flag ? ` (${str(r.flag)})` : ''}</Text>}
              {tab === 'doctor_plans' && <Text style={[type.small, { color: colors.fg }]}>{str((r.plan as Record<string, unknown>)?.diagnosis)} · {str((r.plan as Record<string, unknown>)?.doctor_name)}</Text>}
              {typeof r.url === 'string' && <Pressable onPress={() => Linking.openURL(r.url as string)}><Text style={[type.small, { color: colors.accent }]}>Open file</Text></Pressable>}
            </View>
          ))}
        </>
      )}
    </Card>
  );
}
