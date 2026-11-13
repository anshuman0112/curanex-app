// Connect your watch: Curanex has no band of its own. It reads what the user's watch app already collected:
// iPhone -> watch app (e.g. Nothing X) shares to Apple Health -> upload the Health app export here (free).
// Android -> watch app shares to Health Connect -> Curanex Android app reads it live.
import { useState } from 'react';
import { ActivityIndicator, Platform, Pressable, Text, View } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { router } from 'expo-router';
import { Card, Chip, Disclaimer, Eyebrow, GhostButton, PrimaryButton, Screen } from '../components/ui';
import { syncWatchDays, uploadHealthExport, WatchReply } from '../lib/api';
import { liveUnavailableReason, openHealthConnect, readHealthConnect } from '../lib/watch';
import { useHealth } from '../store/health';
import { colors, fonts, type } from '../theme';

type Phone = 'iphone' | 'android';
const BRANDS = ['Nothing / CMF', 'Noise', 'Samsung', 'Amazfit', 'boAt / Fire-Boltt', 'Fitbit / Pixel', 'Other'] as const;
type Brand = (typeof BRANDS)[number];

// How each brand's own app shares data with the phone's health store
const APP: Record<Brand, { iphone: string; android: string }> = {
  'Nothing / CMF': { iphone: 'Open Nothing X, open your watch settings and turn on Apple Health sync. In the Health permission screen, turn on all categories.',
    android: 'Open Nothing X (or CMF Watch), go to your watch settings and turn on Health Connect. Allow all data types.' },
  Noise: { iphone: 'Open NoiseFit > Profile > Connect Apple Health, and allow all categories.',
    android: 'Open NoiseFit > Profile > Connect Google Fit. Then in Google Fit > Profile > Settings, turn on Health Connect sync.' },
  Samsung: { iphone: 'Galaxy Watches do not work with iPhone.', android: 'Samsung Health already shares with Health Connect: Samsung Health > Settings > Health Connect, allow all.' },
  Amazfit: { iphone: 'Open Zepp > Profile > Add accounts > Apple Health, and allow all.', android: 'Open Zepp > Profile > Add accounts > Health Connect, and allow all.' },
  'boAt / Fire-Boltt': { iphone: 'In your watch app (boAt Crest / Da Fit / Fire-Boltt), open Settings > Apple Health and turn it on.',
    android: 'In your watch app, open Settings > Google Fit or Health Connect and turn it on.' },
  'Fitbit / Pixel': { iphone: 'Fitbit does not write to Apple Health. Use an Android phone, or another watch.', android: 'Fitbit / Pixel Watch: Fitbit app > Settings > Health Connect, turn it on.' },
  Other: { iphone: 'In your watch app, look for "Apple Health" or "Health" in Settings and turn on sharing.', android: 'In your watch app, look for "Health Connect" or "Google Fit" in Settings and turn on sharing.' },
};

const fmt = (v?: number, unit = '') => (v == null ? '–' : `${Math.round(v * 10) / 10}${unit}`);

export default function Devices() {
  const h = useHealth();
  const [phone, setPhone] = useState<Phone>(Platform.OS === 'android' ? 'android' : 'iphone');
  const [brand, setBrand] = useState<Brand>('Nothing / CMF');
  const [busy, setBusy] = useState(''); const [error, setError] = useState('');
  const w = h.watch;

  const save = (r: WatchReply) => h.setWatch({ days: r.days, sources: r.sources, via: r.via, syncedAt: Date.now(), alerts: r.alerts });

  async function upload() {
    setError('');
    // only the export file types: iPhone Safari then opens Files directly (no "Photo Library / Take Photo")
    const res = await DocumentPicker.getDocumentAsync({ type: ['application/zip', 'application/x-zip-compressed', '.zip', 'text/xml', 'application/xml', '.xml'], copyToCacheDirectory: true });
    if (res.canceled) return;
    const f = res.assets[0];
    setBusy('Reading your health export… big files can take a minute.');
    try { save(await uploadHealthExport(f.uri, f.name, f.mimeType ?? 'application/zip')); }
    catch (e) { setError(e instanceof Error ? e.message : 'That file could not be read.'); }
    setBusy('');
  }

  async function connectAndroid() {
    setError(''); setBusy('Reading your watch data from Health Connect…');
    try {
      const live = await readHealthConnect();
      if (!live.days.length) throw new Error('Health Connect has no watch data yet. Check that your watch app shares with Health Connect, then sync again.');
      try { save(await syncWatchDays(live.days, live.apps)); }
      catch { h.setWatch({ days: live.days, sources: live.apps, via: 'Health Connect', syncedAt: Date.now(), alerts: [] }); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not read Health Connect.'); }
    setBusy('');
  }

  const liveReason = liveUnavailableReason();
  const last = w?.days.slice(-7).reverse() ?? [];

  return (
    <Screen>
      <Eyebrow>Smartwatch</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>Connect your watch</Text>
      <Text style={[type.small, { color: colors.muted, marginTop: 6 }]}>
        Curanex reads what your own watch app already collected (steps, heart rate, sleep, oxygen). No extra band needed.
      </Text>

      {w && (
        <Card style={{ marginTop: 16, gap: 10 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={[type.h3, { color: colors.ink }]}>Your watch data</Text>
            <GhostButton label="Disconnect" onPress={() => h.setWatch(null)} />
          </View>
          <Text style={[type.small, { color: colors.muted }]}>From {w.sources.join(', ') || 'your watch'} via {w.via} · updated {new Date(w.syncedAt).toLocaleString()}</Text>
          {w.alerts.map(a => (
            <View key={a.text} style={{ backgroundColor: a.level === 'urgent' ? '#FBE4DF' : colors.surface, borderRadius: 12, padding: 10 }}>
              <Text style={[type.small, { color: a.level === 'urgent' ? colors.sos : colors.ink }]}>{a.text}</Text>
              {a.level === 'urgent' && <Pressable onPress={() => router.push('/sos')}><Text style={[type.small, { color: colors.sos, fontFamily: fonts.semibold, marginTop: 4 }]}>Open SOS</Text></Pressable>}
            </View>
          ))}
          <View style={{ flexDirection: 'row', borderBottomWidth: 1, borderColor: colors.line, paddingBottom: 6 }}>
            {['Day', 'Steps', 'Rest HR', 'Sleep', 'SpO2'].map(x => <Text key={x} style={[type.label, { color: colors.muted, flex: 1 }]}>{x}</Text>)}
          </View>
          {last.map(d => (
            <View key={d.date} style={{ flexDirection: 'row' }}>
              <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{d.date.slice(5)}</Text>
              <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{d.steps != null ? Math.round(d.steps).toLocaleString() : '–'}</Text>
              <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{fmt(d.hr_resting)}</Text>
              <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{fmt(d.sleep_h, ' h')}</Text>
              <Text style={[type.small, { color: colors.ink, flex: 1 }]}>{d.spo2_avg != null ? `${d.spo2_avg}%` : '–'}</Text>
            </View>
          ))}
          <Text style={[type.small, { color: colors.muted }]}>The AI Doctor now uses these readings. Watch readings are not medical measurements.</Text>
        </Card>
      )}

      <Card style={{ marginTop: 16, gap: 12 }}>
        <Text style={[type.h3, { color: colors.ink }]}>1. Your phone</Text>
        <View style={{ flexDirection: 'row', gap: 8 }}>
          <Chip label="iPhone" active={phone === 'iphone'} onPress={() => setPhone('iphone')} />
          <Chip label="Android" active={phone === 'android'} onPress={() => setPhone('android')} />
        </View>
        <Text style={[type.h3, { color: colors.ink }]}>2. Your watch</Text>
        <View style={{ flexDirection: 'row', gap: 8, flexWrap: 'wrap' }}>{BRANDS.map(b => <Chip key={b} label={b} active={brand === b} onPress={() => setBrand(b)} />)}</View>
        <Text style={[type.h3, { color: colors.ink }]}>3. Let your watch app share its data</Text>
        <Text style={[type.small, { color: colors.fg }]}>{APP[brand][phone]}</Text>

        {phone === 'iphone' ? (
          <>
            <Text style={[type.h3, { color: colors.ink }]}>4. Send it to Curanex</Text>
            <Text style={[type.small, { color: colors.fg }]}>On your iPhone open the Health app, tap your picture (top right), then Export All Health Data, then Export. Save the file (export.zip) to Files. Then tap the button below and choose export.zip.</Text>
            <PrimaryButton label={busy ? 'Reading…' : w ? 'Upload a new export' : 'Upload Health export'} disabled={!!busy} onPress={upload} />
            <Text style={[type.small, { color: colors.muted }]}>Free. Upload again any time to refresh. Live sync on iPhone needs a paid Apple developer account, so Curanex uses the export.</Text>
          </>
        ) : (
          <>
            <Text style={[type.h3, { color: colors.ink }]}>4. Connect Curanex to Health Connect</Text>
            {liveReason ? <Text style={[type.small, { color: colors.warn }]}>{liveReason}</Text> : (
              <>
                <PrimaryButton label={busy ? 'Syncing…' : w ? 'Sync now' : 'Connect'} disabled={!!busy} onPress={connectAndroid} />
                <Pressable onPress={openHealthConnect}><Text style={[type.small, { color: colors.accent }]}>Open Health Connect settings</Text></Pressable>
              </>
            )}
          </>
        )}
        {!!busy && <View style={{ flexDirection: 'row', gap: 8, alignItems: 'center' }}><ActivityIndicator color={colors.accent} /><Text style={[type.small, { color: colors.muted }]}>{busy}</Text></View>}
        {!!error && <Text style={[type.small, { color: colors.sos }]}>{error}</Text>}
      </Card>
      <Disclaimer>Watch readings help Curanex give better guidance. They are not a diagnosis. In an emergency call 112.</Disclaimer>
    </Screen>
  );
}
