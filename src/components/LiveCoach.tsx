// Native (Expo Go): in-browser MediaPipe isn't available, so live form check is web-only for now.
// The web implementation is LiveCoach.web.tsx.
import { Text, View } from 'react-native';
import { colors, radius, type } from '../theme';

export type LiveCoachProps = {
  exercise: string;
  target?: number;
  active: boolean;                 // false during rest: camera stays on, counting pauses
  onReps?: (reps: number) => void;
  say: (text: string) => void;
};

export const LIVE_COACH_SUPPORTED = false;

export default function LiveCoach(_: LiveCoachProps) {
  return (
    <View style={{ backgroundColor: colors.inkSoft, borderRadius: radius.cardSm, padding: 16 }}>
      <Text style={[type.small, { color: colors.onDarkMuted, textAlign: 'center' }]}>Live form check works in the web app for now.</Text>
    </View>
  );
}
