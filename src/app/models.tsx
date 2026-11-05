import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { router } from 'expo-router';
import { Card, Chip, Eyebrow, Screen } from '../components/ui';
import { ArrowL } from '../components/icons';
import raw from '../data/models.json';
import * as ImagePicker from 'expo-image-picker';
import { ExperimentalModel, ExperimentalResult, fetchExperimental, fetchModels, runExperimental } from '../lib/api';
import { colors, fonts, radius, type } from '../theme';

type Metrics = { accuracy?: number | null; sensitivity?: number | null; specificity?: number | null; auc?: number | null; eval_set?: string };
type M = {
  id: string; categories: string[]; tasks: string[]; input: string; app_tier: 'A' | 'B' | 'C'; status: string;
  integrated: boolean; verified?: boolean; name?: string; trained_on?: string; limitations?: string; credit?: string; metrics?: Metrics | null; description?: string; conditions?: string[];
};
const saved = raw as unknown as M[];

const BADGE = {
  active: { label: 'Active', bg: colors.ink, fg: colors.onDark },
  unverified: { label: 'Unverified', bg: colors.surface, fg: colors.accent },
  soon: { label: 'Coming soon', bg: colors.surface, fg: colors.ink },
  clinic: { label: 'Clinic-grade', bg: 'transparent', fg: colors.muted },
};
const state = (m: M) => (m.integrated ? (m.verified ? 'active' : 'unverified') : m.app_tier === 'C' ? 'clinic' : 'soon') as keyof typeof BADGE;
const pct = (v?: number | null) => (v == null ? '–' : `${Math.round(v * 100)}%`);
const conditions = (m: M) => m.conditions ?? m.tasks;
const count = (n: number) => `${n} condition${n === 1 ? '' : 's'}`;

export default function ModelLibrary() {
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState<'all' | 'phone' | 'active' | 'experimental'>('all');
  const [experimental, setExperimental] = useState<ExperimentalModel[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, ExperimentalResult | string>>({});
  const [models, setModels] = useState<M[]>(saved);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetchModels<M>().then(list => {
      if (!alive) return;
      if (list?.length) { setModels(list); setLive(true); }
      setLoading(false);
    });
    fetchExperimental().then(list => { if (alive) setExperimental(list); });
    return () => { alive = false; };
  }, []);
  async function tryModel(id: string) {
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.9 });
    if (picked.canceled) return;
    setBusy(id);
    try { const r = await runExperimental(id, picked.assets[0].uri); setResults(v => ({ ...v, [id]: r })); }
    catch { setResults(v => ({ ...v, [id]: 'That image could not be checked. Please try another.' })); }
    setBusy(null);
  }
  const cats = useMemo(() => {
    const list = models.filter(m =>
      (filter === 'all' || filter === 'experimental' || (filter === 'phone' ? m.app_tier !== 'C' : m.integrated)) &&
      (!q || (m.tasks.join(' ') + (m.name ?? '') + m.categories.join(' ')).toLowerCase().includes(q.toLowerCase())));
    const map = new Map<string, M[]>();
    list.forEach(m => { const c = m.categories[0]; map.set(c, [...(map.get(c) ?? []), m]); });
    return [...map.entries()];
  }, [q, filter, models]);
  const active = models.filter(m => m.integrated).length;
  const covered = models.filter(m => m.integrated).reduce((n, m) => n + conditions(m).length, 0);

  return (
    <Screen padBottom={60}>
      <Pressable onPress={() => router.back()} style={{ paddingVertical: 12 }} hitSlop={10}><ArrowL size={22} /></Pressable>
      <Eyebrow>AI model library</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>{active} active models</Text>
      <Text style={[type.body, { color: colors.muted, marginTop: 6 }]}>
        Checking {count(covered)} today. Curanex’s AI doctor guides you on any symptom, and uses a tested model when one fits.
      </Text>

      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 }}>
        {loading && <ActivityIndicator size="small" color={colors.accent} />}
        <Text style={[type.small, { color: colors.muted }]}>{loading ? 'Checking live status…' : live ? 'Live status from the Curanex backend' : 'Offline: showing the saved list, status may be out of date'}</Text>
      </View>

      <TextInput value={q} onChangeText={setQ} placeholder="Search a condition, e.g. diabetes, skin, TB" placeholderTextColor={colors.subtle} style={s.search} />
      <View style={{ flexDirection: 'row', gap: 8, marginTop: 10 }}>
        <Chip label="All" active={filter === 'all'} onPress={() => setFilter('all')} />
        <Chip label="Phone-friendly" active={filter === 'phone'} onPress={() => setFilter('phone')} />
        <Chip label="Active" active={filter === 'active'} onPress={() => setFilter('active')} />
        <Chip label={`Experimental ${experimental.length}`} active={filter === 'experimental'} onPress={() => setFilter('experimental')} />
      </View>

      {filter === 'experimental' && (
        <View style={{ marginTop: 20, gap: 8 }}>
          <Text style={[type.small, { color: colors.ink }]}>Research demos trained on tiny test images. They aren’t used for patient results.</Text>
          {experimental.length === 0 && <Text style={[type.body, { color: colors.muted }]}>No experimental models are available right now.</Text>}
          {experimental.map(m => {
            const r = results[m.id];
            return (
              <Card key={m.id} style={s.card}>
                <Text style={[type.h3, { color: colors.ink }]}>{m.title}</Text>
                <Text style={[type.small, { color: colors.muted }]}>Checks {m.labels.length} types · Accuracy {pct(m.test.accuracy)} · AUC {m.test.auc} on {m.test.n} test images</Text>
                <Pressable onPress={() => tryModel(m.id)} disabled={busy === m.id} style={{ marginTop: 8 }}>
                  <Text style={[type.small, { color: colors.accent, fontFamily: fonts.medium }]}>{busy === m.id ? 'Checking…' : 'Try with an image'}</Text>
                </Pressable>
                {typeof r === 'string' && <Text style={[type.small, { color: colors.warn, marginTop: 6 }]}>{r}</Text>}
                {r && typeof r !== 'string' && <Text style={[type.small, { color: colors.ink, marginTop: 6 }]}>{r.top.map(([k, v]) => `${k} ${pct(v)}`).join(' · ')}. {r.notice}</Text>}
              </Card>
            );
          })}
        </View>
      )}
      {filter !== 'experimental' && cats.length === 0 && <Text style={[type.body, { color: colors.muted, marginTop: 24 }]}>{filter === 'active' ? 'No models are connected yet. They turn active as the backend integrates them.' : 'No models match.'}</Text>}
      {filter !== 'experimental' && cats.map(([cat, list]) => (
        <View key={cat} style={{ marginTop: 24 }}>
          <Text style={[type.label, { color: colors.accent, marginBottom: 10 }]}>{cat} · {list.length}</Text>
          <View style={{ gap: 8 }}>
            {list.map(m => {
              const b = BADGE[state(m)];
              const isActive = m.integrated;
              return (
                <Pressable key={m.id} onPress={() => setOpen(o => (o === m.id ? null : m.id))}>
                  <Card style={s.card}>
                    <View style={s.row}>
                      <View style={{ flex: 1, gap: 2 }}>
                        <Text style={[type.h3, { color: colors.ink }]} numberOfLines={1}>{m.name ?? m.tasks[0]}</Text>
                        <Text style={[type.small, { color: colors.muted }]} numberOfLines={1}>{m.input} · {count(conditions(m).length)}</Text>
                      </View>
                      <View style={[s.badge, { backgroundColor: b.bg, borderColor: state(m) === 'clinic' ? colors.line : b.bg }]}>
                        <Text style={[type.small, { color: b.fg, fontFamily: fonts.medium, fontSize: 11 }]}>{b.label}</Text>
                      </View>
                    </View>
                    {open === m.id && (
                      <View style={{ gap: 6, marginTop: 10 }}>
                        <Text style={[type.small, { color: colors.ink }]}>{isActive ? `Checks for: ${conditions(m).join(', ')}.` : `Planned check for ${m.tasks.join(', ')}.`}</Text>
                        {isActive && m.metrics && <Text style={[type.small, { color: colors.accent }]}>Accuracy {pct(m.metrics.accuracy)} · Catches {pct(m.metrics.sensitivity)} of cases · Avoids {pct(m.metrics.specificity)} of false alarms</Text>}
                        {isActive && <Text style={[type.small, { color: colors.muted }]}>Screening only, not a diagnosis.</Text>}
                        {isActive && m.credit && <Text style={[type.small, { color: colors.muted }]}>{m.credit}</Text>}
                      </View>
                    )}
                  </Card>
                </Pressable>
              );
            })}
          </View>
        </View>
      ))}
    </Screen>
  );
}

const s = StyleSheet.create({
  search: { marginTop: 18, backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, borderRadius: radius.pill, paddingHorizontal: 18, paddingVertical: 12, fontFamily: fonts.body, fontSize: 15, color: colors.ink },
  card: { paddingVertical: 12 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  badge: { borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4, borderWidth: 1 },
});
