import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, ActivityIndicator } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useFonts, Onest_400Regular, Onest_500Medium, Onest_600SemiBold, Onest_700Bold } from '@expo-google-fonts/onest';
import { PlayfairDisplay_500Medium, PlayfairDisplay_600SemiBold } from '@expo-google-fonts/playfair-display';
import { HealthProvider } from '../store/health';
import { DoctorProvider } from '../store/doctor';
import { colors } from '../theme';

export default function RootLayout() {
  const [loaded] = useFonts({ Onest_400Regular, Onest_500Medium, Onest_600SemiBold, Onest_700Bold, PlayfairDisplay_500Medium, PlayfairDisplay_600SemiBold });
  if (!loaded) return <View style={{ flex: 1, backgroundColor: colors.ink, alignItems: 'center', justifyContent: 'center' }}><ActivityIndicator color={colors.accentFrom} /></View>;
  return (
    <SafeAreaProvider>
      <HealthProvider>
        <DoctorProvider>
        <StatusBar style="dark" />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.bg } }}>
          <Stack.Screen name="(tabs)" />
          <Stack.Screen name="onboarding" options={{ gestureEnabled: false }} />
          <Stack.Screen name="profile" options={{ presentation: 'modal' }} />
          <Stack.Screen name="models" />
          <Stack.Screen name="voice" options={{ presentation: 'fullScreenModal', animation: 'fade' }} />
          <Stack.Screen name="workout" options={{ presentation: 'fullScreenModal' }} />
        </Stack>
        </DoctorProvider>
      </HealthProvider>
    </SafeAreaProvider>
  );
}
