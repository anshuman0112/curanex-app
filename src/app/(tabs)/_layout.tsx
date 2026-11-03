import { Redirect, Tabs } from 'expo-router';
import { Pressable, StyleSheet, Text, View, ActivityIndicator } from 'react-native';
import type { ComponentProps } from 'react';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useHealth } from '../../store/health';
import { colors, fonts } from '../../theme';
import { Home, Stetho, Leaf, Dumbbell } from '../../components/icons';

type BottomTabBarProps = Parameters<NonNullable<ComponentProps<typeof Tabs>['tabBar']>>[0];

const ITEMS: Record<string, { label: string; Icon?: typeof Home }> = {
  index: { label: 'Home', Icon: Home },
  doctor: { label: 'Doctor', Icon: Stetho },
  sos: { label: 'SOS' },
  nutrition: { label: 'Nutrition', Icon: Leaf },
  trainer: { label: 'Trainer', Icon: Dumbbell },
};

/** Floating dark dock with a raised SOS button in the centre. Same on phone and web. */
function Dock({ state, emitter, navigateToTab }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  return (
    <View pointerEvents="box-none" style={[s.wrap, { paddingBottom: Math.max(insets.bottom, 12) }]}>
      <View style={s.dock}>
        {state.routes.map((route, i) => {
          const item = ITEMS[route.name]; if (!item) return null;
          const focused = state.index === i;
          const go = () => { const e = emitter.emit({ type: 'tabPress', target: route.key, canPreventDefault: true }); if (!focused && !e.defaultPrevented) navigateToTab(route.key); };
          if (route.name === 'sos') return (
            <Pressable key={route.key} onPress={go} accessibilityRole="button" accessibilityLabel="Emergency SOS" style={s.sosSlot}>
              <View style={[s.sos, focused && { borderColor: colors.onDark }]}><Text style={s.sosText}>SOS</Text></View>
            </Pressable>
          );
          const { Icon } = item;
          return (
            <Pressable key={route.key} onPress={go} accessibilityRole="tab" accessibilityState={{ selected: focused }} style={s.item}>
              {Icon && <Icon size={21} color={focused ? colors.accentFrom : colors.onDarkMuted} />}
              <Text style={[s.label, { color: focused ? colors.onDark : colors.onDarkMuted }]}>{item.label}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

export default function TabsLayout() {
  const { ready, profile } = useHealth();
  if (!ready) return <View style={{ flex: 1, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center' }}><ActivityIndicator color={colors.accent} /></View>;
  if (!profile) return <Redirect href="/onboarding" />;
  return (
    <Tabs tabBar={p => <Dock {...p} />} screenOptions={{ headerShown: false, sceneStyle: { backgroundColor: colors.bg } }}>
      <Tabs.Screen name="index" />
      <Tabs.Screen name="doctor" />
      <Tabs.Screen name="sos" />
      <Tabs.Screen name="nutrition" />
      <Tabs.Screen name="trainer" />
    </Tabs>
  );
}

const s = StyleSheet.create({
  wrap: { position: 'absolute', left: 0, right: 0, bottom: 0, alignItems: 'center', paddingHorizontal: 12 },
  dock: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.ink, borderRadius: 999, paddingHorizontal: 8, height: 66, width: '100%', maxWidth: 480,
    shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 18, shadowOffset: { width: 0, height: 8 }, elevation: 12 },
  item: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 3, height: '100%' },
  label: { fontFamily: fonts.medium, fontSize: 10.5 },
  sosSlot: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  sos: { width: 62, height: 62, borderRadius: 31, marginTop: -26, backgroundColor: colors.sos, alignItems: 'center', justifyContent: 'center', borderWidth: 4, borderColor: colors.bg,
    shadowColor: colors.sos, shadowOpacity: 0.5, shadowRadius: 12, shadowOffset: { width: 0, height: 4 }, elevation: 10 },
  sosText: { color: '#fff', fontFamily: fonts.bold, fontSize: 14, letterSpacing: 1 },
});
