import { ReactNode } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, TextInputProps, TextStyle, View, ViewStyle, StyleProp, useWindowDimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, fonts, radius, type, MAX_W } from '../theme';
import { ArrowUR, Sparkle } from './icons';

export const T = ({ style, children, ...p }: { style?: StyleProp<TextStyle>; children: ReactNode; numberOfLines?: number }) => (
  <Text style={[type.body, { color: colors.fg }, style]} {...p}>{children}</Text>
);

/** Page wrapper: cream background, centered column on wide web screens, room for the floating tab bar. */
export function Screen({ children, scroll = true, padBottom = 120, style }: { children: ReactNode; scroll?: boolean; padBottom?: number; style?: ViewStyle }) {
  const inner = <View style={[s.column, style]}>{children}</View>;
  return (
    <SafeAreaView style={s.safe} edges={['top']}>
      {scroll ? (
        <ScrollView contentContainerStyle={{ paddingBottom: padBottom }} showsVerticalScrollIndicator={false}>{inner}</ScrollView>
      ) : inner}
    </SafeAreaView>
  );
}

export const useWide = () => useWindowDimensions().width >= 820;

/** "• The Platform" style eyebrow label from the site. */
export const Eyebrow = ({ children, dark }: { children: ReactNode; dark?: boolean }) => (
  <View style={s.eyebrow}>
    <View style={[s.dot, dark && { backgroundColor: colors.onDarkMuted }]} />
    <Text style={[type.small, { color: dark ? colors.onDarkMuted : colors.muted }]}>{children}</Text>
  </View>
);

export const Card = ({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) => (
  <View style={[s.card, style]}>{children}</View>
);
export const DarkCard = ({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) => (
  <View style={[s.dark, style]}>{children}</View>
);

export const Chip = ({ label, dark, active, onPress, style }: { label: string; dark?: boolean; active?: boolean; onPress?: () => void; style?: StyleProp<ViewStyle> }) => (
  <Pressable onPress={onPress} disabled={!onPress} style={({ pressed }) => [
    s.chip,
    dark ? { borderColor: '#4A3824' } : { borderColor: colors.line, backgroundColor: colors.card },
    active && { backgroundColor: colors.ink, borderColor: colors.ink },
    pressed && { opacity: 0.7 }, style,
  ]}>
    <Text style={[type.small, { color: active || dark ? colors.onDark : colors.fg, fontFamily: fonts.medium }]}>{label}</Text>
  </Pressable>
);

/** Copper gradient pill with a round arrow — the site's primary CTA. */
export function PrimaryButton({ label, onPress, icon = true, style, disabled }: { label: string; onPress?: () => void; icon?: boolean; style?: StyleProp<ViewStyle>; disabled?: boolean }) {
  return (
    <Pressable onPress={onPress} disabled={disabled} style={({ pressed }) => [{ opacity: disabled ? 0.5 : pressed ? 0.85 : 1, alignSelf: 'flex-start' }, style]}>
      <LinearGradient colors={[colors.accentFrom, colors.accentTo]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={s.primary}>
        <Text style={[type.body, { color: '#fff', fontFamily: fonts.semibold }]}>{label}</Text>
        {icon && <View style={s.primaryIcon}><ArrowUR size={16} color={colors.accent} stroke={2.2} /></View>}
      </LinearGradient>
    </Pressable>
  );
}

export const GhostButton = ({ label, onPress, dark, style }: { label: string; onPress?: () => void; dark?: boolean; style?: StyleProp<ViewStyle> }) => (
  <Pressable onPress={onPress} style={({ pressed }) => [s.ghost, { borderColor: dark ? '#5A4630' : colors.line }, pressed && { opacity: 0.7 }, style]}>
    <Text style={[type.body, { color: dark ? colors.onDark : colors.ink, fontFamily: fonts.medium }]}>{label}</Text>
  </Pressable>
);

export const RoundIcon = ({ children, onPress, dark, size = 40, style }: { children: ReactNode; onPress?: () => void; dark?: boolean; size?: number; style?: StyleProp<ViewStyle> }) => (
  <Pressable onPress={onPress} hitSlop={8} style={({ pressed }) => [{ width: size, height: size, borderRadius: size / 2, alignItems: 'center', justifyContent: 'center', backgroundColor: dark ? colors.ink : colors.card, borderWidth: dark ? 0 : 1, borderColor: colors.line }, pressed && { opacity: 0.7 }, style]}>
    {children}
  </Pressable>
);

export const Field = ({ label, hint, ...p }: TextInputProps & { label: string; hint?: string }) => (
  <View style={{ gap: 6 }}>
    <Text style={[type.label, { color: colors.muted }]}>{label}</Text>
    <TextInput placeholderTextColor={colors.subtle} style={s.input} {...p} />
    {hint ? <Text style={[type.small, { color: colors.muted }]}>{hint}</Text> : null}
  </View>
);

export const Logo = ({ dark }: { dark?: boolean }) => (
  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
    <Sparkle size={18} />
    <Text style={{ fontFamily: fonts.semibold, fontSize: 18, color: dark ? colors.onDark : colors.ink }}>Curanex</Text>
  </View>
);

export const Disclaimer = ({ children }: { children?: ReactNode }) => (
  <View style={s.disclaimer}>
    <Text style={[type.small, { color: colors.fg }]}>
      {children ?? 'Curanex gives health guidance, not a diagnosis. In an emergency, use SOS or call 112.'}
    </Text>
  </View>
);

export const SectionTitle = ({ eyebrow, title, action }: { eyebrow?: string; title: string; action?: ReactNode }) => (
  <View style={{ flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between', marginTop: 28, marginBottom: 14 }}>
    <View style={{ flex: 1, gap: 6 }}>
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <Text style={[type.h2, { color: colors.ink }]}>{title}</Text>
    </View>
    {action}
  </View>
);

export const ProgressBar = ({ value, color = colors.accent, track = colors.surface }: { value: number; color?: string; track?: string }) => (
  <View style={{ height: 6, borderRadius: 3, backgroundColor: track, overflow: 'hidden' }}>
    <View style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, height: '100%', backgroundColor: color, borderRadius: 3 }} />
  </View>
);

const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  column: { width: '100%', maxWidth: MAX_W, alignSelf: 'center', paddingHorizontal: 16 },
  eyebrow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.muted },
  card: { backgroundColor: colors.card, borderRadius: radius.cardSm, padding: 18, borderWidth: 1, borderColor: colors.line },
  dark: { backgroundColor: colors.ink, borderRadius: radius.card, padding: 20 },
  chip: { paddingHorizontal: 12, paddingVertical: 7, borderRadius: radius.pill, borderWidth: 1 },
  primary: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingLeft: 18, paddingRight: 6, paddingVertical: 6, borderRadius: radius.pill, minHeight: 46 },
  primaryIcon: { width: 34, height: 34, borderRadius: 17, backgroundColor: '#fff', alignItems: 'center', justifyContent: 'center' },
  ghost: { paddingHorizontal: 18, minHeight: 46, borderRadius: radius.pill, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  input: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, borderRadius: radius.control, paddingHorizontal: 14, paddingVertical: 12, fontFamily: fonts.body, fontSize: 15, color: colors.ink },
  disclaimer: { backgroundColor: colors.surface, borderRadius: radius.control, paddingHorizontal: 14, paddingVertical: 10 },
});
