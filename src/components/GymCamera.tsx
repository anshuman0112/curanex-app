// Native (Expo Go): the in-browser pose model is web-only for now. Web version: GymCamera.web.tsx.
import { Text, View } from 'react-native';
import { colors, radius, type } from '../theme';
import type { EngineFrame, Exercise, ExerciseEngine } from '../lib/exerciseEngine';

export type GymCameraProps = {
  exercise: Exercise;
  engine: ExerciseEngine;          // owned by the screen; swapped per exercise / set
  active: boolean;                 // false during rest: skeleton still drawn, nothing counted
  onFrame: (f: EngineFrame) => void;
  onError?: (msg: string) => void;
};
export const GYM_CAMERA_SUPPORTED = false;

export default function GymCamera(_: GymCameraProps) {
  return (
    <View style={{ backgroundColor: colors.ink, borderRadius: radius.cardSm, padding: 24, aspectRatio: 4 / 3, justifyContent: 'center' }}>
      <Text style={[type.body, { color: colors.onDark, textAlign: 'center' }]}>The live camera coach runs in the web app for now. Open Curanex in Chrome on your laptop or phone.</Text>
    </View>
  );
}
