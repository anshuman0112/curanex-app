import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';
import { router } from 'expo-router';
import { Card } from './ui';
import { AreaAlert, fetchAlert } from '../lib/api';
import { colors, fonts, type } from '../theme';

/** "Health alert in your area" on Home: dengue-season weather for the district in the profile. */
export function AreaAlertCard({ district }: { district?: string }) {
  const [alert, setAlert] = useState<AreaAlert | string | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    let alive = true;
    setAlert(null);
    if (district?.trim()) fetchAlert(district.trim()).then(a => { if (alive) setAlert(a); });
    return () => { alive = false; };
  }, [district]);

  if (!district?.trim()) return (
    <Pressable onPress={() => router.push('/profile')}>
      <Card style={{ marginTop: 14, gap: 4 }}>
        <Text style={[type.label, { color: colors.accent }]}>Health alert in your area</Text>
        <Text style={[type.body, { color: colors.ink }]}>Add your district in your profile to see dengue-season weather alerts for your area.</Text>
      </Card>
    </Pressable>
  );
  return (
    <Card style={{ marginTop: 14, gap: 6 }}>
      <Text style={[type.label, { color: colors.accent }]}>Health alert in your area</Text>
      {alert === null && <View style={{ flexDirection: 'row', gap: 8, alignItems: 'center' }}><ActivityIndicator size="small" color={colors.accent} /><Text style={[type.small, { color: colors.muted }]}>Checking weather for {district}…</Text></View>}
      {typeof alert === 'string' && <Text style={[type.body, { color: colors.muted }]}>{alert}</Text>}
      {alert && typeof alert !== 'string' && (
        <>
          <Text style={[type.h3, { color: colors.ink }]}>{alert.headline}</Text>
          <Text style={[type.small, { color: colors.muted }]}>
            {alert.place} · last 4 weeks: {alert.weather.temperature_c}°C, {alert.weather.humidity_pct}% humidity, {alert.weather.rain_4_weeks_mm} mm rain
          </Text>
          {alert.advice.slice(0, open ? undefined : 1).map(a => <Text key={a} style={[type.body, { color: colors.ink }]}>• {a}</Text>)}
          {open && alert.estimate && (
            <Text style={[type.small, { color: colors.muted }]}>Dengue-week estimate: {alert.estimate.level}. {alert.estimate.accuracy_note}</Text>
          )}
          {open && <Text style={[type.small, { color: colors.muted }]}>{alert.notice} {alert.source}.</Text>}
          <Pressable onPress={() => setOpen(o => !o)} hitSlop={8}>
            <Text style={[type.small, { color: colors.accent, fontFamily: fonts.medium }]}>{open ? 'Show less' : 'What to do'}</Text>
          </Pressable>
        </>
      )}
    </Card>
  );
}
