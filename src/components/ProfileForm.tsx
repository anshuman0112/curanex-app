import { useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import type { Profile } from '../store/health';
import { colors, fonts, radius, type } from '../theme';
import { Chip, Field } from './ui';
import { Close, Plus } from './icons';

export const EMPTY_PROFILE: Profile = { name: '', age: '', sex: '', heightCm: '', weightKg: '', bloodGroup: '', conditions: [], allergies: [], medications: [], diet: '', goal: '', contacts: [] };
const BLOOD = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'];
const COMMON_CONDITIONS = ['Diabetes', 'Hypertension', 'Thyroid', 'PCOS', 'Asthma', 'Heart disease'];
const DIETS: Profile['diet'][] = ['Vegetarian', 'Non-veg', 'Vegan', 'Jain', 'Eggetarian'];
const GOALS: Profile['goal'][] = ['Lose weight', 'Build muscle', 'Stay fit', 'Manage a condition'];

export type Section = 'basics' | 'health' | 'lifestyle' | 'contacts';
export const SECTIONS: { key: Section; title: string; sub: string }[] = [
  { key: 'basics', title: 'About you', sub: 'Used to personalise advice and pre-fill medical checks.' },
  { key: 'health', title: 'Your health', sub: 'Conditions, allergies and medicines shape every answer.' },
  { key: 'lifestyle', title: 'Food & goals', sub: 'Your nutritionist and trainer plan around these.' },
  { key: 'contacts', title: 'Emergency contacts', sub: 'SOS alerts these people with your location.' },
];

export function sectionValid(p: Profile, k: Section) {
  if (k === 'basics') return !!p.name.trim() && !!p.age && !!p.sex;
  if (k === 'contacts') return p.contacts.length > 0;
  return true;
}

export function ProfileSection({ k, p, set }: { k: Section; p: Profile; set: (p: Profile) => void }) {
  const up = (patch: Partial<Profile>) => set({ ...p, ...patch });
  if (k === 'basics') return (
    <View style={{ gap: 14 }}>
      <Field label="Name" value={p.name} onChangeText={name => up({ name })} placeholder="Your first name" />
      <View style={{ flexDirection: 'row', gap: 10 }}>
        <View style={{ flex: 1 }}><Field label="Age" value={p.age} onChangeText={age => up({ age: age.replace(/\D/g, '') })} keyboardType="number-pad" placeholder="34" /></View>
        <View style={{ flex: 1 }}><Field label="Height (cm)" value={p.heightCm} onChangeText={heightCm => up({ heightCm: heightCm.replace(/\D/g, '') })} keyboardType="number-pad" placeholder="170" /></View>
        <View style={{ flex: 1 }}><Field label="Weight (kg)" value={p.weightKg} onChangeText={weightKg => up({ weightKg: weightKg.replace(/[^\d.]/g, '') })} keyboardType="decimal-pad" placeholder="70" /></View>
      </View>
      <Picker label="Sex" options={['Female', 'Male', 'Other']} value={p.sex} onChange={v => up({ sex: v as Profile['sex'] })} />
      <Picker label="Blood group" options={BLOOD} value={p.bloodGroup} onChange={bloodGroup => up({ bloodGroup })} />
    </View>
  );
  if (k === 'health') return (
    <View style={{ gap: 18 }}>
      <TagInput label="Conditions" values={p.conditions} onChange={conditions => up({ conditions })} suggestions={COMMON_CONDITIONS} placeholder="Add a condition" />
      <TagInput label="Allergies" values={p.allergies} onChange={allergies => up({ allergies })} suggestions={['Penicillin', 'Peanuts', 'Lactose', 'Sulfa']} placeholder="Add an allergy" />
      <TagInput label="Current medicines" values={p.medications} onChange={medications => up({ medications })} placeholder="e.g. Metformin 500 mg" />
    </View>
  );
  if (k === 'lifestyle') return (
    <View style={{ gap: 18 }}>
      <Picker label="Diet" options={DIETS as string[]} value={p.diet} onChange={v => up({ diet: v as Profile['diet'] })} />
      <Picker label="Main goal" options={GOALS as string[]} value={p.goal} onChange={v => up({ goal: v as Profile['goal'] })} />
    </View>
  );
  return <Contacts p={p} up={up} />;
}

function Contacts({ p, up }: { p: Profile; up: (x: Partial<Profile>) => void }) {
  const [c, setC] = useState({ name: '', phone: '', relation: '' });
  return (
    <View style={{ gap: 14 }}>
      {p.contacts.map((x, i) => (
        <View key={i} style={s.contact}>
          <View style={{ flex: 1 }}>
            <Text style={[type.h3, { color: colors.ink }]}>{x.name}</Text>
            <Text style={[type.small, { color: colors.muted }]}>{x.relation} · {x.phone}</Text>
          </View>
          <Pressable onPress={() => up({ contacts: p.contacts.filter((_, j) => j !== i) })} hitSlop={8}><Close size={18} color={colors.muted} /></Pressable>
        </View>
      ))}
      <Field label="Contact name" value={c.name} onChangeText={name => setC({ ...c, name })} placeholder="Priya" />
      <View style={{ flexDirection: 'row', gap: 10 }}>
        <View style={{ flex: 1.4 }}><Field label="Phone" value={c.phone} onChangeText={phone => setC({ ...c, phone })} keyboardType="phone-pad" placeholder="+91…" /></View>
        <View style={{ flex: 1 }}><Field label="Relation" value={c.relation} onChangeText={relation => setC({ ...c, relation })} placeholder="Sister" /></View>
      </View>
      <Pressable disabled={!c.name || !c.phone} onPress={() => { up({ contacts: [...p.contacts, c] }); setC({ name: '', phone: '', relation: '' }); }}
        style={[s.add, (!c.name || !c.phone) && { opacity: 0.4 }]}>
        <Plus size={16} /><Text style={[type.body, { fontFamily: fonts.medium }]}>Add contact</Text>
      </Pressable>
    </View>
  );
}

function Picker({ label, options, value, onChange }: { label: string; options: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <View style={{ gap: 8 }}>
      <Text style={[type.label, { color: colors.muted }]}>{label}</Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
        {options.map(o => <Chip key={o} label={o} active={value === o} onPress={() => onChange(o)} />)}
      </View>
    </View>
  );
}

function TagInput({ label, values, onChange, suggestions = [], placeholder }: { label: string; values: string[]; onChange: (v: string[]) => void; suggestions?: string[]; placeholder: string }) {
  const [t, setT] = useState('');
  const add = (v: string) => { const x = v.trim(); if (x && !values.includes(x)) onChange([...values, x]); setT(''); };
  return (
    <View style={{ gap: 8 }}>
      <Text style={[type.label, { color: colors.muted }]}>{label}</Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
        {values.map(v => <Chip key={v} label={`${v}  ×`} active onPress={() => onChange(values.filter(x => x !== v))} />)}
        {suggestions.filter(x => !values.includes(x)).map(x => <Chip key={x} label={`+ ${x}`} onPress={() => add(x)} />)}
      </View>
      <View style={s.tagRow}>
        <TextInput value={t} onChangeText={setT} onSubmitEditing={() => add(t)} placeholder={placeholder} placeholderTextColor={colors.subtle} style={s.tagInput} />
        <Pressable onPress={() => add(t)} disabled={!t.trim()} style={[s.tagAdd, !t.trim() && { opacity: 0.4 }]}><Plus size={16} color="#fff" /></Pressable>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  contact: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.card, borderRadius: radius.control, padding: 14, borderWidth: 1, borderColor: colors.line },
  add: { flexDirection: 'row', gap: 8, alignItems: 'center', justifyContent: 'center', paddingVertical: 12, borderRadius: 999, borderWidth: 1, borderColor: colors.ink },
  tagRow: { flexDirection: 'row', gap: 8, alignItems: 'center' },
  tagInput: { flex: 1, backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, borderRadius: radius.control, paddingHorizontal: 14, paddingVertical: 11, fontFamily: fonts.body, fontSize: 15, color: colors.ink },
  tagAdd: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.accent, alignItems: 'center', justifyContent: 'center' },
});
