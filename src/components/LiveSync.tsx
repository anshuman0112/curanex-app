// Automatic live watch sync on iPhone: Nothing X -> Apple Health -> an iPhone Shortcut (runs by itself every hour) -> Curanex.
// Shows the user's private sync key + post link for the Shortcut, and the newest live readings (polled while this screen is open).
import { useCallback, useEffect, useState } from 'react';
import { Linking, Platform, Pressable, Text, View } from 'react-native';
import { router } from 'expo-router';
import { api, useSession } from '../lib/accounts';
import { API_URL } from '../lib/api';
import { DaySummary, useHealth, WatchAlert } from '../store/health';
import { colors, fonts, type } from '../theme';
import { AuthCard } from './AuthCard';
import { Card, GhostButton, PrimaryButton } from './ui';

type Live = { heart_rate?: number; spo2?: number; steps?: number; resting_hr?: number; sleep_h?: number; hrv_ms?: number; at?: number };
type State = { days: DaySummary[]; live: Live | null; alerts: WatchAlert[]; via?: string; sources?: string[] };

const ago = (t?: number) => {
  if (!t) return '';
  const m = Math.round((Date.now() / 1000 - t) / 60);
  return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
};


// iPhone Shortcuts app links (documented URL scheme): open the app / start a new shortcut / run the user's "Curanex Sync" now
function openUrl(url: string) {
  if (Platform.OS === 'web' && typeof window !== 'undefined') window.location.href = url;
  else Linking.openURL(url).catch(() => {});
}
const SHORTCUT_NAME = 'Curanex Sync';

function copy(text: string) {
  if (Platform.OS === 'web' && typeof navigator !== 'undefined' && navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
}

export function LiveSync() {
  const s = useSession('patient');
  const h = useHealth();
  const [key, setKey] = useState(''); const [msg, setMsg] = useState('');
  const [live, setLive] = useState<State | null>(null);
  const postUrl = `${API_URL}/v1/wearables/live`;
  const [check, setCheck] = useState('');
  async function checkKey() {
    setCheck('Checking…');
    try {
      const r = await fetch(postUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key }) });
      // a valid key with no readings answers 422 "No readings"; an unknown key answers 403
      setCheck(r.status === 422 ? '✅ Your key works and the link is reachable. Now run the Shortcut.' : r.status === 403 ? '❌ Key not recognised. Tap "Get my sync key" again and copy the new one.' : `Unexpected answer (${r.status}).`);
    } catch { setCheck('❌ Could not reach Curanex. Is the laptop on and the link still working?'); }
  }

  const poll = useCallback(async () => {
    if (!s.token) return;
    try {
      const st = await api<State>('/v1/wearables/state', s.token);
      setLive(st);
      if (st.days.length) h.setWatch({ days: st.days, sources: st.sources ?? [], via: st.via ?? 'iPhone Shortcut (live)', syncedAt: Date.now(), alerts: st.alerts });
    } catch { /* server not reachable right now */ }
  }, [s.token]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { poll(); const t = setInterval(poll, 20000); return () => clearInterval(t); }, [poll]);

  async function makeKey() {
    setMsg('');
    try { setKey((await api<{ key: string }>('/v1/wearables/device-key', s.token, 'POST')).key); }
    catch (e) { setMsg(e instanceof Error ? e.message : 'Could not make a key.'); }
  }

  if (!s.ready) return null;
  if (!s.token) return (
    <Card style={{ marginTop: 12, gap: 4 }}>
      <Text style={[type.h3, { color: colors.ink }]}>Automatic live sync</Text>
      <Text style={[type.small, { color: colors.fg }]}>Log in or create a free account here, so your watch readings go to your own private account. Then you can get your sync key.</Text>
      <AuthCard role="patient" title="Log in or create account" onDone={s.save} />
    </Card>
  );

  const l = live?.live;
  return (
    <Card style={{ marginTop: 12, gap: 10 }}>
      <Text style={[type.h3, { color: colors.ink }]}>Automatic live sync</Text>
      {l ? (
        <View style={{ backgroundColor: colors.surface, borderRadius: 12, padding: 12, gap: 4 }}>
          <Text style={[type.label, { color: colors.good }]}>● Live from your watch · {ago(l.at)}</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 14 }}>
            {l.heart_rate != null && <Text style={[type.body, { color: colors.ink, fontFamily: fonts.semibold }]}>❤ {Math.round(l.heart_rate)} bpm</Text>}
            {l.spo2 != null && <Text style={[type.body, { color: colors.ink, fontFamily: fonts.semibold }]}>O₂ {Math.round(l.spo2)}%</Text>}
            {l.steps != null && <Text style={[type.body, { color: colors.ink, fontFamily: fonts.semibold }]}>👣 {Math.round(l.steps).toLocaleString()}</Text>}
            {l.resting_hr != null && <Text style={[type.body, { color: colors.ink }]}>Resting {Math.round(l.resting_hr)} bpm</Text>}
            {l.sleep_h != null && <Text style={[type.body, { color: colors.ink }]}>Sleep {l.sleep_h} h</Text>}
          </View>
        </View>
      ) : <Text style={[type.small, { color: colors.muted }]}>No live readings yet. Set up the Shortcut below; readings appear here by themselves.</Text>}

      <Text style={[type.small, { color: colors.fg }]}>
        Your iPhone's Shortcuts app sends your watch readings here every hour, by itself. You set it up once.
      </Text>
      {!key ? <PrimaryButton label="Get my sync key" onPress={makeKey} /> : (
        <View style={{ gap: 8 }}>
          <Text style={[type.label, { color: colors.muted }]}>1. Post link (URL)</Text>
          <Pressable onPress={() => copy(postUrl)}><Text selectable style={[type.small, { color: colors.accent }]}>{postUrl}  (tap to copy)</Text></Pressable>
          <Text style={[type.label, { color: colors.muted }]}>2. Your private key (keep it secret)</Text>
          <Pressable onPress={() => copy(key)}><Text selectable style={[type.small, { color: colors.accent }]}>{key}  (tap to copy)</Text></Pressable>
          <Text style={[type.small, { color: colors.muted }]}>Making a new key stops the old one.</Text>
          <Text style={[type.label, { color: colors.muted, marginTop: 6 }]}>3. Set up the Shortcut</Text>
          <PrimaryButton label="Open Shortcuts app" onPress={() => openUrl('shortcuts://create-shortcut')} />
          <Text style={[type.small, { color: colors.muted }]}>It opens a new shortcut. Name it "{SHORTCUT_NAME}", add the 5 actions from the steps, paste the link and key above.</Text>
          <GhostButton label="Check my key" onPress={checkKey} />
          {!!check && <Text style={[type.small, { color: check.startsWith('✅') ? colors.good : colors.fg }]}>{check}</Text>}
          <GhostButton label={`Run "${SHORTCUT_NAME}" now`} onPress={() => openUrl(`shortcuts://run-shortcut?name=${encodeURIComponent(SHORTCUT_NAME)}`)} />
          <GhostButton label="Open my shortcuts" onPress={() => openUrl('shortcuts://')} />
        </View>
      )}
      {!!msg && <Text style={[type.small, { color: colors.sos }]}>{msg}</Text>}
    </Card>
  );
}
