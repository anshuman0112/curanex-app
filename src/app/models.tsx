import { useMemo, useState } from 'react';
import { Linking, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { router } from 'expo-router';
import { Card, Chip, Eyebrow, Screen } from '../components/ui';
import { ArrowL, ArrowUR } from '../components/icons';
import raw from '../data/models.json';
import { colors, fonts, radius, type } from '../theme';

type M = { id: string; repo: string; url: string | null; categories: string[]; tasks: string[]; input: string; app_tier: 'A' | 'B' | 'C'; status: string; integrated: boolean; demo_candidate: boolean };
const models = raw as M[];

const BADGE = {
  active: { label: 'Active', bg: colors.ink, fg: colors.onDark },
  soon: { label: 'Available soon', bg: colors.surface, fg: colors.ink },
  clinic: { label: 'Clinic-grade', bg: 'transparent', fg: colors.muted },
};
const state = (m: M) => (m.integrated ? 'active' : m.app_tier === 'C' ? 'clinic' : 'soon') as keyof typeof BADGE;

export default function ModelLibrary() {
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState<'all' | 'phone' | 'active'>('all');
  const cats = useMemo(() => {
    const list = models.filter(m =>
      (filter === 'all' || (filter === 'phone' ? m.app_tier !== 'C' : m.integrated)) &&
      (!q || (m.tasks.join(' ') + m.repo + m.categories.join(' ')).toLowerCase().includes(q.toLowerCase())));
    const map = new Map<string, M[]>();
    list.forEach(m => { const c = m.categories[0]; map.set(c, [...(map.get(c) ?? []), m]); });
    return [...map.entries()];
  }, [q, filter]);
  const active = models.filter(m => m.integrated).length;
  const phone = models.filter(m => m.app_tier !== 'C').length;

  return (
    <Screen padBottom={60}>
      <Pressable onPress={() => router.back()} style={{ paddingVertical: 12 }} hitSlop={10}><ArrowL size={22} /></Pressable>
      <Eyebrow>AI model library</Eyebrow>
      <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>{models.length} open medical models</Text>
      <Text style={[type.body, { color: colors.muted, marginTop: 6 }]}>
        Curanex’s AI doctor routes your question to a specialist model when one fits. {active} active · {phone} accept phone-friendly input · {models.length - phone} need hospital equipment.
      </Text>

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
              return (
                <Pressable key={m.id} onPress={() => m.url && Linking.openURL(m.url)}>
                  <Card style={s.row}>
                    <View style={{ flex: 1, gap: 2 }}>
                      <Text style={[type.h3, { color: colors.ink }]} numberOfLines={1}>{m.tasks[0]}</Text>
                      <Text style={[type.small, { color: colors.muted }]} numberOfLines={1}>{m.input} · {m.repo}</Text>
                    </View>
                    <View style={[s.badge, { backgroundColor: b.bg, borderColor: state(m) === 'clinic' ? colors.line : b.bg }]}>
                      <Text style={[type.small, { color: b.fg, fontFamily: fonts.medium, fontSize: 11 }]}>{b.label}</Text>
                    </View>
                    <ArrowUR size={16} color={colors.muted} />
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
  row: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 12 },
  badge: { borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4, borderWidth: 1 },
});
