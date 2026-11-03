import { useState } from 'react';
import { Image, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import Svg, { Circle } from 'react-native-svg';
import { Card, Chip, DarkCard, Eyebrow, PrimaryButton, ProgressBar, Screen, SectionTitle, useWide } from '../../components/ui';
import { Camera, Check, Drop, Plus } from '../../components/icons';
import { todaysMeals, useHealth } from '../../store/health';
import { MEAL_PLAN } from '../../data/content';
import { colors, fonts, radius, type } from '../../theme';

const GOAL = { kcal: 1900, protein: 80, carbs: 230, fat: 60 };

export default function NutritionScreen() {
  const wide = useWide();
  const h = useHealth();
  const meals = todaysMeals(h.meals);
  const sum = (k: 'kcal' | 'protein' | 'carbs' | 'fat') => meals.reduce((a, m) => a + m[k], 0);
  const [photo, setPhoto] = useState<string | null>(null);
  const conditions = h.profile?.conditions ?? [];
  const adapted = conditions.length ? `Adapted for ${conditions.join(', ').toLowerCase()} · ${h.profile?.diet || 'any diet'}` : `${h.profile?.diet || 'Balanced'} plan`;

  const snap = async () => {
    const opts: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.7 };
    const res = Platform.OS === 'web' ? await ImagePicker.launchImageLibraryAsync(opts) : await ImagePicker.launchCameraAsync(opts);
    if (!res.canceled) setPhoto(res.assets[0].uri);
  };

  return (
    <Screen>
      <View style={{ paddingTop: 12 }}>
        <Eyebrow>Nutrition · Personal</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>AI Nutritionist</Text>
      </View>

      <View style={[{ gap: 12, marginTop: 16 }, wide && { flexDirection: 'row' }]}>
        {/* Calories + macros */}
        <DarkCard style={[{ flexDirection: 'row', alignItems: 'center', gap: 18 }, wide && { flex: 1.3 }]}>
          <Ring value={sum('kcal') / GOAL.kcal} label={`${sum('kcal')}`} sub={`of ${GOAL.kcal} kcal`} />
          <View style={{ flex: 1, gap: 12 }}>
            <Macro name="Protein" v={sum('protein')} g={GOAL.protein} />
            <Macro name="Carbs" v={sum('carbs')} g={GOAL.carbs} />
            <Macro name="Fat" v={sum('fat')} g={GOAL.fat} />
          </View>
        </DarkCard>
        {/* Water */}
        <Card style={[{ gap: 12 }, wide && { flex: 1 }]}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <View style={{ flexDirection: 'row', gap: 8, alignItems: 'center' }}><Drop size={18} color={colors.accent} /><Text style={[type.h3, { color: colors.ink }]}>Water</Text></View>
            <Text style={[type.small, { color: colors.muted }]}>{(h.water / 1000).toFixed(2)} / 2.5 L</Text>
          </View>
          <View style={{ flexDirection: 'row', gap: 6, flexWrap: 'wrap' }}>
            {Array.from({ length: 10 }).map((_, i) => (
              <View key={i} style={[s.glass, i < Math.round(h.water / 250) && { backgroundColor: colors.accent, borderColor: colors.accent }]} />
            ))}
          </View>
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <Pressable onPress={() => h.addWater(250)} style={s.waterBtn}><Plus size={16} color="#fff" /><Text style={[type.small, { color: '#fff', fontFamily: fonts.semibold }]}>250 ml</Text></Pressable>
            <Pressable onPress={() => h.addWater(-250)} style={[s.waterBtn, { backgroundColor: colors.surface }]}><Text style={[type.small, { color: colors.ink }]}>Undo</Text></Pressable>
          </View>
        </Card>
      </View>

      {/* Snap to log */}
      <Pressable onPress={snap} style={s.snap}>
        <View style={s.snapIcon}><Camera size={22} color="#fff" /></View>
        <View style={{ flex: 1 }}>
          <Text style={[type.h3, { color: colors.ink }]}>Snap to log</Text>
          <Text style={[type.small, { color: colors.muted }]}>Photograph your plate. Curanex recognises Indian dishes and estimates portions.</Text>
        </View>
      </Pressable>
      {photo && <SnapResult uri={photo} onAdd={() => { h.addMeal({ name: 'Rice, dal tadka, aloo gobi', time: 'Snapped', kcal: 540, protein: 16, carbs: 82, fat: 15 }); setPhoto(null); }} onClose={() => setPhoto(null)} />}

      {/* Plan */}
      <SectionTitle eyebrow="Today’s plan" title="Meals for you" action={<Chip label="Regenerate" onPress={() => {}} />} />
      <Text style={[type.small, { color: colors.muted, marginTop: -6, marginBottom: 12 }]}>{adapted}</Text>
      <View style={s.planGrid}>
        {MEAL_PLAN.map(m => {
          const logged = meals.some(x => x.name === m.name);
          return (
            <Card key={m.time} style={[s.plan, { width: wide ? '49%' : '100%' }]}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={[type.label, { color: colors.accent }]}>{m.time}</Text>
                {m.tag ? <Text style={[type.small, { color: colors.good }]}>{m.tag}</Text> : null}
              </View>
              <Text style={[type.h3, { color: colors.ink, marginTop: 6 }]}>{m.name}</Text>
              <Text style={[type.small, { color: colors.muted, marginTop: 4 }]}>{m.kcal} kcal · P {m.protein}g · C {m.carbs}g · F {m.fat}g</Text>
              <Pressable disabled={logged} onPress={() => h.addMeal({ name: m.name, time: m.time, kcal: m.kcal, protein: m.protein, carbs: m.carbs, fat: m.fat })}
                style={[s.logBtn, logged && { backgroundColor: colors.ink, borderColor: colors.ink }]}>
                {logged ? <Check size={14} color="#fff" stroke={2.4} /> : <Plus size={14} />}
                <Text style={[type.small, { fontFamily: fonts.medium, color: logged ? '#fff' : colors.ink }]}>{logged ? 'Logged' : 'I ate this'}</Text>
              </Pressable>
            </Card>
          );
        })}
      </View>

      {/* Log */}
      <SectionTitle eyebrow="Diary" title="Logged today" />
      {meals.length === 0 ? <Text style={[type.body, { color: colors.muted }]}>Nothing logged yet. Snap a photo or tap “I ate this”.</Text> : (
        <Card style={{ paddingVertical: 4 }}>
          {meals.map((m, i) => (
            <View key={m.id} style={[{ flexDirection: 'row', paddingVertical: 12, alignItems: 'center' }, i < meals.length - 1 && { borderBottomWidth: 1, borderBottomColor: colors.line }]}>
              <View style={{ flex: 1 }}>
                <Text style={[type.body, { color: colors.ink }]}>{m.name}</Text>
                <Text style={[type.small, { color: colors.muted }]}>{m.time} · {new Date(m.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</Text>
              </View>
              <Text style={[type.body, { fontFamily: fonts.semibold, color: colors.ink }]}>{m.kcal} kcal</Text>
            </View>
          ))}
        </Card>
      )}
    </Screen>
  );
}

function SnapResult({ uri, onAdd, onClose }: { uri: string; onAdd: () => void; onClose: () => void }) {
  return (
    <Card style={{ marginTop: 12, gap: 12 }}>
      <View style={{ flexDirection: 'row', gap: 12 }}>
        <Image source={{ uri }} style={{ width: 88, height: 88, borderRadius: 14 }} />
        <View style={{ flex: 1, gap: 4 }}>
          <View style={{ flexDirection: 'row', gap: 6, alignItems: 'center' }}>
            <Text style={[type.label, { color: colors.accent }]}>Detected</Text>
            <View style={s.demo}><Text style={[type.label, { fontSize: 9, color: colors.ink }]}>Demo</Text></View>
          </View>
          <Text style={[type.body, { color: colors.ink }]}>Rice (1 cup) · Dal tadka (1 bowl) · Aloo gobi (½ cup)</Text>
          <Text style={[type.small, { color: colors.muted }]}>≈ 540 kcal · P 16g · C 82g · F 15g</Text>
        </View>
      </View>
      <Text style={[type.small, { color: colors.muted }]}>Sample result. With the backend connected, the food model reads your actual photo, and you can edit items before saving.</Text>
      <View style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}>
        <PrimaryButton label="Add to diary" onPress={onAdd} icon={false} />
        <Pressable onPress={onClose}><Text style={[type.small, { color: colors.muted }]}>Discard</Text></Pressable>
      </View>
    </Card>
  );
}

function Ring({ value, label, sub }: { value: number; label: string; sub: string }) {
  const r = 46, c = 2 * Math.PI * r;
  return (
    <View style={{ width: 112, height: 112, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={112} height={112} style={StyleSheet.absoluteFill}>
        <Circle cx={56} cy={56} r={r} stroke="#3A2A18" strokeWidth={10} fill="none" />
        <Circle cx={56} cy={56} r={r} stroke={colors.accentFrom} strokeWidth={10} fill="none" strokeLinecap="round" strokeDasharray={`${c}`} strokeDashoffset={c * (1 - Math.min(1, value))} transform="rotate(-90 56 56)" />
      </Svg>
      <Text style={{ fontFamily: fonts.semibold, fontSize: 24, color: colors.onDark }}>{label}</Text>
      <Text style={[type.small, { color: colors.onDarkMuted, fontSize: 11 }]}>{sub}</Text>
    </View>
  );
}
const Macro = ({ name, v, g }: { name: string; v: number; g: number }) => (
  <View style={{ gap: 4 }}>
    <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
      <Text style={[type.small, { color: colors.onDark }]}>{name}</Text>
      <Text style={[type.small, { color: colors.onDarkMuted }]}>{v} / {g} g</Text>
    </View>
    <ProgressBar value={v / g} color={colors.accentFrom} track="#3A2A18" />
  </View>
);

const s = StyleSheet.create({
  glass: { width: 22, height: 30, borderRadius: 6, borderWidth: 1.5, borderColor: colors.subtle },
  waterBtn: { flexDirection: 'row', gap: 6, alignItems: 'center', backgroundColor: colors.accent, paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999 },
  snap: { flexDirection: 'row', alignItems: 'center', gap: 14, marginTop: 12, backgroundColor: colors.surface, borderRadius: radius.cardSm, padding: 16 },
  snapIcon: { width: 48, height: 48, borderRadius: 24, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
  planGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, justifyContent: 'space-between' },
  plan: { gap: 0 },
  logBtn: { flexDirection: 'row', gap: 6, alignItems: 'center', alignSelf: 'flex-start', marginTop: 12, paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, borderWidth: 1, borderColor: colors.line },
  demo: { backgroundColor: colors.surface, borderRadius: 999, paddingHorizontal: 6, paddingVertical: 2 },
});
