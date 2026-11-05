import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Linking, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { router } from 'expo-router';
import { Card, Chip, Eyebrow, Screen } from '../components/ui';
import { ArrowL, ArrowUR } from '../components/icons';
import raw from '../data/models.json';
import { fetchModels } from '../lib/api';
import { colors, fonts, radius, type } from '../theme';

type Metrics = { accuracy?: number | null; sensitivity?: number | null; specificity?: number | null; auc?: number | null; eval_set?: string };
type M = {
  id: string; repo: string; url: string | null; categories: string[]; tasks: string[]; input: string; app_tier: 'A' | 'B' | 'C'; status: string;
  integrated: boolean; verified?: boolean; name?: string; trained_on?: string; limitations?: string; credit?: string; metrics?: Metrics | null;
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

export default function ModelLibrary() {
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState<'all' | 'phone' | 'active'>('all');
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
    return () => { alive = false; };
  }, []);
  const cats = useMemo(() => {
    const list = models.filter(m =>
      (filter === 'all' || (filter === 'phone' ? m.app_tier !== 'C' : m.integrated)) &&
      (!q || (m.tasks.join(' ') + m.repo + m.categories.join(' ')).toLowerCase().includes(q.toLowerCase())));
    const map = new Map<string, M[]>();
    list.forEach(m => { const c = m.categories[0]; map.set(c, [...(map.get(c) ?? []), m]); });
    return [...map.entries()];
  }, [q, filter, models]);
  const active = models.filter(m => m.integrated).length;
  const phone = models.filter(m => m.app_tier !== 'C').length;

  return (
    <Screen padBottom={60}>
      <Pressable onPress={() => router.back()} style={{ paddingVertical: 12 }} hitSlop={10}><ArrowL size={22} /></Pressable>
      <Eyebrow>AI model library</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>{models.length} open medical models</Text>
      <Text style={[type.body, { color: colors.muted, marginTop: 6 }]}>
        Curanex’s AI doctor guides you on any symptom, and uses a tested model when one fits. AI guidance is general advice, not a tested model. {active} active · {phone} accept phone-friendly input · {models.length - phone} need hospital equipment.
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
      </View>

      {cats.length === 0 && <Text style={[type.body, { color: colors.muted, marginTop: 24 }]}>{filter === 'active' ? 'No models are connected yet. They turn active as the backend integrates them.' : 'No models match.'}</Text>}
      {cats.map(([cat, list]) => (
        <View key={cat} style={{ marginTop: 24 }}>
          <Text style={[type.label, { color: colors.accent, marginBottom: 10 }]}>{cat} · {list.length}</Text>
          <View style={{ gap: 8 }}>
            {list.map(m => {
              const b = BADGE[state(m)];
              const isActive = m.integrated;
              return (
                <Pressable key={m.id} onPress={() => (isActive ? setOpen(o => (o === m.id ? null : m.id)) : m.url && Linking.openURL(m.url))}>
                  <Card style={s.card}>
                    <View style={s.row}>
                      <View style={{ flex: 1, gap: 2 }}>
                        <Text style={[type.h3, { color: colors.ink }]} numberOfLines={1}>{m.name ?? m.tasks[0]}</Text>
                        <Text style={[type.small, { color: colors.muted }]} numberOfLines={1}>{m.input} · {m.repo}</Text>
                        {isActive && m.metrics && <Text style={[type.small, { color: colors.accent }]}>Catches {pct(m.metrics.sensitivity)} · avoids false alarms {pct(m.metrics.specificity)} · AUC {m.metrics.auc ?? '–'}</Text>}
                      </View>
                      <View style={[s.badge, { backgroundColor: b.bg, borderColor: state(m) === 'clinic' ? colors.line : b.bg }]}>
                        <Text style={[type.small, { color: b.fg, fontFamily: fonts.medium, fontSize: 11 }]}>{b.label}</Text>
                      </View>
                      {!isActive && <ArrowUR size={16} color={colors.muted} />}
                    </View>
                    {open === m.id && (
                      <View style={{ gap: 6, marginTop: 10 }}>
                        {m.metrics?.eval_set && <Text style={[type.small, { color: colors.muted }]}>Measured on: {m.metrics.eval_set}</Text>}
                        {m.trained_on && <Text style={[type.small, { color: colors.muted }]}>Trained on: {m.trained_on}</Text>}
                        {m.limitations && <Text style={[type.small, { color: colors.ink }]}>Limitations: {m.limitations}</Text>}
                        {m.credit && <Text style={[type.small, { color: colors.muted }]}>{m.credit}</Text>}
                        {m.url && <Pressable onPress={() => Linking.openURL(m.url as string)}><Text style={[type.small, { color: colors.accent, fontFamily: fonts.medium }]}>View source repository</Text></Pressable>}
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
