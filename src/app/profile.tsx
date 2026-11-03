import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { router } from 'expo-router';
import { ProfileSection, SECTIONS } from '../components/ProfileForm';
import { Card, GhostButton, PrimaryButton, Screen } from '../components/ui';
import { ArrowR, Close, Grid } from '../components/icons';
import { bmi, useHealth } from '../store/health';
import { API_URL } from '../lib/api';
import { colors, type } from '../theme';

export default function ProfileScreen() {
  const h = useHealth();
  const [p, setP] = useState(h.profile!);
  return (
    <Screen padBottom={60}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12 }}>
        <Text style={[type.h1, { color: colors.ink }]}>Profile</Text>
        <Pressable onPress={() => router.back()} hitSlop={10}><Close size={22} /></Pressable>
      </View>
      <Text style={[type.small, { color: colors.muted }]}>BMI {bmi(p) ?? '—'} · stored on this device</Text>

      {SECTIONS.map(sec => (
        <Card key={sec.key} style={{ marginTop: 16, gap: 14, backgroundColor: colors.bg }}>
          <Text style={[type.h2, { color: colors.ink }]}>{sec.title}</Text>
          <ProfileSection k={sec.key} p={p} set={setP} />
        </Card>
      ))}
      <PrimaryButton label="Save changes" onPress={() => { h.setProfile(p); router.back(); }} style={{ marginTop: 20 }} />

      <Pressable onPress={() => router.push('/models')}>
        <Card style={{ marginTop: 20, flexDirection: 'row', alignItems: 'center', gap: 12 }}>
          <Grid size={20} color={colors.accent} />
          <Text style={[type.h3, { color: colors.ink, flex: 1 }]}>AI model library</Text>
          <ArrowR size={18} />
        </Card>
      </Pressable>

      <Card style={{ marginTop: 12, gap: 10 }}>
        <Text style={[type.h3, { color: colors.ink }]}>Data & privacy</Text>
        <Text style={[type.small, { color: colors.muted }]}>Backend: {API_URL}</Text>
        <View style={{ flexDirection: 'row', gap: 10, flexWrap: 'wrap' }}>
          <GhostButton label="Load demo data" onPress={() => { h.loadDemo(); router.back(); }} />
          <GhostButton label="Delete all my data" onPress={() => { h.reset(); router.replace('/onboarding'); }} />
        </View>
      </Card>
    </Screen>
  );
}
