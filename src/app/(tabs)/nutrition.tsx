import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Image, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import Svg, { Circle } from 'react-native-svg';
import { Card, Chip, DarkCard, Eyebrow, PrimaryButton, ProgressBar, Screen, SectionTitle, useWide } from '../../components/ui';
import { Camera, Check, Drop, Plus } from '../../components/icons';
import { todaysMeals, useHealth } from '../../store/health';
import { MEAL_PLAN } from '../../data/content';
import { analyseFood, FoodResult, getMealPlan, getTargets, MealPlan, Nutrients, PlanMeal, Targets } from '../../lib/api';
import { colors, fonts, radius, type } from '../../theme';

const FALLBACK = { kcal: 1900, protein_g: 80, carbs_g: 230, fat_g: 60, fibre_g: 30, iron_mg: 19, calcium_mg: 1000, vitamin_c_mg: 80 };
// nutrient rows shown for a photo (label, unit)
const ROWS: [keyof Nutrients, string, string][] = [
  ['kcal', 'Calories', 'kcal'], ['protein_g', 'Protein', 'g'], ['carbs_g', 'Carbohydrates', 'g'], ['fat_g', 'Fat', 'g'], ['fibre_g', 'Fibre', 'g'],
  ['sugar_g', 'Sugar', 'g'], ['iron_mg', 'Iron', 'mg'], ['calcium_mg', 'Calcium', 'mg'], ['vitamin_c_mg', 'Vitamin C', 'mg'], ['vitamin_a_ug', 'Vitamin A', 'µg'],
  ['vitamin_b12_ug', 'Vitamin B12', 'µg'], ['vitamin_d_ug', 'Vitamin D', 'µg'], ['folate_ug', 'Folate', 'µg'], ['sodium_mg', 'Sodium', 'mg'], ['potassium_mg', 'Potassium', 'mg'],
];
const fmt = (v?: number) => (v == null ? '–' : v >= 100 ? Math.round(v).toString() : (Math.round(v * 10) / 10).toString());

export default function NutritionScreen() {
  const wide = useWide();
  const h = useHealth();
  const meals = todaysMeals(h.meals);
  const sumMacro = (k: 'kcal' | 'protein' | 'carbs' | 'fat') => Math.round(meals.reduce((a, m) => a + m[k], 0));
  const sumN = (k: string) => meals.reduce((a, m) => a + (m.nutrients?.[k] ?? 0), 0);
  const [targets, setTargets] = useState<Targets | null>(null);
  const goal = targets ?? FALLBACK;

  const [photo, setPhoto] = useState<string | null>(null);
  const [food, setFood] = useState<FoodResult | null>(null);
  const [reading, setReading] = useState(false);
  const [snapError, setSnapError] = useState('');

  const [plan, setPlan] = useState<MealPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState('');

  const conditions = h.profile?.conditions ?? [];
  const adapted = conditions.length ? `Adapted for ${conditions.join(', ').toLowerCase()} · ${h.profile?.diet || 'any diet'}` : `${h.profile?.diet || 'Balanced'} plan`;

  const makePlan = useCallback(async () => {
    if (!h.profile) return;
    setPlanning(true); setPlanError('');
    try {
      const p = await getMealPlan(h.profile);
      if (p.error) setPlanError(p.error); else setPlan(p);
      setTargets(p.targets);
    } catch { setPlanError('The meal planner needs the Curanex server. Showing a sample plan.'); }
    setPlanning(false);
  }, [h.profile]);

  useEffect(() => { if (h.profile) getTargets(h.profile).then(t => t && setTargets(t)); }, [h.profile]);
  useEffect(() => { if (h.profile && !plan && !planning) makePlan(); }, [h.profile]);  // eslint-disable-line react-hooks/exhaustive-deps

  const snap = async () => {
    setSnapError('');
    const opts: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.7 };
    let res: ImagePicker.ImagePickerResult;
    if (Platform.OS === 'web') res = await ImagePicker.launchImageLibraryAsync(opts);
    else {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      res = perm.granted ? await ImagePicker.launchCameraAsync(opts) : await ImagePicker.launchImageLibraryAsync(opts);
    }
    if (res.canceled) return;
    const uri = res.assets[0].uri;
    setPhoto(uri); setFood(null); setReading(true);
    try {
      const out = await analyseFood(uri);
      if (out.error) setSnapError(out.error);
      else if (!out.is_food) setSnapError("I couldn't see any food in that photo. Try a clear photo of your plate from above.");
      else setFood(out);
    } catch (e) { setSnapError(e instanceof Error ? e.message : 'That photo could not be read.'); }
    setReading(false);
  };

  const addSnap = () => {
    if (!food) return;
    const t = food.totals;
    h.addMeal({ name: food.items.map(i => i.name).join(', ').slice(0, 120), time: 'Snapped', kcal: Math.round(t.kcal ?? 0), protein: Math.round(t.protein_g ?? 0),
      carbs: Math.round(t.carbs_g ?? 0), fat: Math.round(t.fat_g ?? 0), nutrients: t as Record<string, number> });
    setPhoto(null); setFood(null);
  };

  const planMeals: PlanMeal[] = plan?.meals.length ? plan.meals : MEAL_PLAN.map(m => ({ time: m.time, name: m.name, why: m.tag ?? '', kcal: m.kcal, protein: m.protein, carbs: m.carbs, fat: m.fat }));

  return (
    <Screen>
      <View style={{ paddingTop: 12 }}>
        <Eyebrow>Nutrition · Personal</Eyebrow>
        <Text style={[type.h1, { color: colors.ink, marginTop: 6 }]}>AI Nutritionist</Text>
      </View>

      <View style={[{ gap: 12, marginTop: 16 }, wide && { flexDirection: 'row' }]}>
        <DarkCard style={[{ flexDirection: 'row', alignItems: 'center', gap: 18 }, wide && { flex: 1.3 }]}>
          <Ring value={sumMacro('kcal') / goal.kcal} label={`${sumMacro('kcal')}`} sub={`of ${goal.kcal} kcal`} />
          <View style={{ flex: 1, gap: 12 }}>
            <Macro name="Protein" v={sumMacro('protein')} g={goal.protein_g} />
            <Macro name="Carbs" v={sumMacro('carbs')} g={goal.carbs_g} />
            <Macro name="Fat" v={sumMacro('fat')} g={goal.fat_g} />
          </View>
        </DarkCard>
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

      {/* Vitamins and minerals today, from snapped meals */}
      <Card style={{ marginTop: 12, gap: 10 }}>
        <Text style={[type.h3, { color: colors.ink }]}>Vitamins and minerals today</Text>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 14 }}>
          {([['fibre_g', 'Fibre', 'g'], ['iron_mg', 'Iron', 'mg'], ['calcium_mg', 'Calcium', 'mg'], ['vitamin_c_mg', 'Vitamin C', 'mg']] as const).map(([k, label, unit]) => (
            <View key={k} style={{ width: wide ? '22%' : '46%', gap: 4 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={[type.small, { color: colors.ink }]}>{label}</Text>
                <Text style={[type.small, { color: colors.muted }]}>{fmt(sumN(k))} / {goal[k]} {unit}</Text>
              </View>
              <ProgressBar value={sumN(k) / goal[k]} />
            </View>
          ))}
        </View>
        <Text style={[type.small, { color: colors.muted }]}>Counted from meals you snap. Daily amounts: {targets?.source ?? 'ICMR-NIN 2020 recommended amounts for Indian adults.'}</Text>
      </Card>

      {/* Snap a meal */}
      <Pressable onPress={snap} disabled={reading} style={s.snap}>
        <View style={s.snapIcon}>{reading ? <ActivityIndicator color="#fff" /> : <Camera size={22} color="#fff" />}</View>
        <View style={{ flex: 1 }}>
          <Text style={[type.h3, { color: colors.ink }]}>{reading ? 'Reading your plate…' : 'Snap a meal'}</Text>
          <Text style={[type.small, { color: colors.muted }]}>Photograph your plate. Curanex names each dish and estimates calories, protein, carbs, fat, fibre, iron, calcium and vitamins.</Text>
        </View>
      </Pressable>
      {!!snapError && <Text style={[type.small, { color: colors.warn, marginTop: 8 }]}>{snapError}</Text>}
      {photo && (reading || food) && <SnapResult uri={photo} food={food} onAdd={addSnap} onClose={() => { setPhoto(null); setFood(null); }} />}

      {/* Plan */}
      <SectionTitle eyebrow="Today’s plan" title="Meals for you" action={<Chip label={planning ? 'Planning…' : 'Regenerate'} onPress={() => !planning && makePlan()} />} />
      <Text style={[type.small, { color: colors.muted, marginTop: -6, marginBottom: 12 }]}>
        {plan?.meals.length ? `${adapted} · about ${goal.kcal} kcal a day` : planning ? 'Making a plan for your body, diet and health…' : `Sample plan. ${planError}`}
      </Text>
      {planning && !plan && <ActivityIndicator color={colors.accent} style={{ marginBottom: 12 }} />}
      <View style={s.planGrid}>
        {planMeals.map(m => {
          const logged = meals.some(x => x.name === m.name);
          return (
            <Card key={`${m.time}-${m.name}`} style={[s.plan, { width: wide ? '49%' : '100%' }]}>
              <Text style={[type.label, { color: colors.accent }]}>{m.time}</Text>
              <Text style={[type.h3, { color: colors.ink, marginTop: 6 }]}>{m.name}</Text>
              {!!m.why && <Text style={[type.small, { color: colors.good, marginTop: 4 }]}>{m.why}</Text>}
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
      {!!plan?.tips.length && (
        <Card style={{ marginTop: 12, gap: 6 }}>
          <Text style={[type.h3, { color: colors.ink }]}>Tips for you</Text>
          {plan.tips.map(t => <Text key={t} style={[type.small, { color: colors.fg }]}>• {t}</Text>)}
        </Card>
      )}

      {/* Log */}
      <SectionTitle eyebrow="Diary" title="Logged today" />
      {meals.length === 0 ? <Text style={[type.body, { color: colors.muted }]}>Nothing logged yet. Snap a photo or tap “I ate this”.</Text> : (
        <Card style={{ paddingVertical: 4 }}>
          {meals.map((m, i) => (
            <View key={m.id} style={[{ flexDirection: 'row', paddingVertical: 12, alignItems: 'center' }, i < meals.length - 1 && { borderBottomWidth: 1, borderBottomColor: colors.line }]}>
              <View style={{ flex: 1 }}>
                <Text style={[type.body, { color: colors.ink }]}>{m.name}</Text>
                <Text style={[type.small, { color: colors.muted }]}>{m.time} · {new Date(m.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · P {m.protein}g · C {m.carbs}g · F {m.fat}g</Text>
              </View>
              <Text style={[type.body, { fontFamily: fonts.semibold, color: colors.ink }]}>{m.kcal} kcal</Text>
            </View>
          ))}
        </Card>
      )}
    </Screen>
  );
}

function SnapResult({ uri, food, onAdd, onClose }: { uri: string; food: FoodResult | null; onAdd: () => void; onClose: () => void }) {
  return (
    <Card style={{ marginTop: 12, gap: 12 }}>
      <View style={{ flexDirection: 'row', gap: 12 }}>
        <Image source={{ uri }} style={{ width: 88, height: 88, borderRadius: 14 }} />
        <View style={{ flex: 1, gap: 4 }}>
          <Text style={[type.label, { color: colors.accent }]}>{food ? `Detected · ${food.confidence ?? 'medium'} confidence` : 'Reading…'}</Text>
          {!food && <ActivityIndicator color={colors.accent} style={{ alignSelf: 'flex-start' }} />}
          {food?.items.map(i => <Text key={i.name} style={[type.small, { color: colors.ink }]}>{i.name} · {i.portion} · {fmt(i.kcal)} kcal</Text>)}
        </View>
      </View>
      {food && (
        <>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
            {ROWS.map(([k, label, unit]) => (
              <View key={k} style={s.nutrient}>
                <Text style={[type.small, { color: colors.muted, fontSize: 11 }]}>{label}</Text>
                <Text style={[type.body, { color: colors.ink, fontFamily: fonts.semibold }]}>{fmt(food.totals[k])} <Text style={[type.small, { color: colors.muted }]}>{unit}</Text></Text>
              </View>
            ))}
          </View>
          <Text style={[type.small, { color: colors.muted }]}>{food.note}</Text>
          <View style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}>
            <PrimaryButton label="Add to diary" onPress={onAdd} icon={false} />
            <Pressable onPress={onClose}><Text style={[type.small, { color: colors.muted }]}>Discard</Text></Pressable>
          </View>
        </>
      )}
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
  nutrient: { backgroundColor: colors.surface, borderRadius: 12, paddingHorizontal: 10, paddingVertical: 6, minWidth: 96 },
});
