// Static content bundled with the app (works offline).
// TODO before release: have a clinician review first-aid text and cite the source in-app.

export type Guide = { id: string; title: string; when: string; steps: string[] };
export const FIRST_AID: Guide[] = [
  { id: 'cpr', title: 'CPR (adult)', when: 'Not breathing or only gasping', steps: [
    'Call 112 or ask someone to call, and get an AED if one is nearby.',
    'Place the heel of your hand in the centre of the chest, other hand on top.',
    'Push hard and fast: about 5 cm deep, 100–120 pushes a minute.',
    'Let the chest come fully back up between pushes.',
    'Keep going until help arrives or the person starts breathing.'] },
  { id: 'choking', title: 'Choking', when: 'Cannot speak, cough or breathe', steps: [
    'Ask “Are you choking?” Encourage coughing if they can.',
    'Give up to 5 firm back blows between the shoulder blades.',
    'Then up to 5 abdominal thrusts (stand behind, fist above the navel, pull in and up).',
    'Alternate back blows and thrusts. Call 112 if it does not clear.'] },
  { id: 'heart', title: 'Heart attack signs', when: 'Chest pain spreading to arm, jaw or back', steps: [
    'Call 112 immediately.',
    'Sit them down and keep them calm.',
    'Loosen tight clothing.',
    'If they become unresponsive and stop breathing normally, start CPR.'] },
  { id: 'stroke', title: 'Stroke: think FAST', when: 'Sudden weakness or confusion', steps: [
    'Face: is one side drooping?', 'Arms: can they raise both?', 'Speech: is it slurred or strange?',
    'Time: call 112 now and note when symptoms started.'] },
  { id: 'bleeding', title: 'Severe bleeding', when: 'Bleeding that won’t stop', steps: [
    'Press firmly on the wound with a clean cloth.', 'Keep pressing; add more cloth on top, do not remove the first.',
    'Raise the injured part if possible.', 'Call 112 if bleeding is heavy or does not slow.'] },
  { id: 'burns', title: 'Burns', when: 'Heat, chemical or electrical burn', steps: [
    'Cool under cool running water for 20 minutes.', 'Remove jewellery near the burn unless stuck.',
    'Cover loosely with cling film or a clean cloth.', 'Do not use ice, butter or toothpaste. Get help for large burns.'] },
];

export type Exercise = { name: string; muscle: string; sets: number; reps?: number; seconds?: number; cue: string };
export const TODAY_WORKOUT: { title: string; focus: string; minutes: number; note: string; exercises: Exercise[] } = {
  title: 'Low-impact strength', focus: 'Full body', minutes: 24,
  note: 'Adapted for blood pressure: no breath-holding, steady pace.',
  exercises: [
    { name: 'March in place', muscle: 'Warm-up', sets: 1, seconds: 60, cue: 'Lift knees gently, swing arms.' },
    { name: 'Bodyweight squat', muscle: 'Legs', sets: 3, reps: 12, cue: 'Sit back, chest up, knees over toes.' },
    { name: 'Wall push-up', muscle: 'Chest', sets: 3, reps: 10, cue: 'Body straight, breathe out as you push.' },
    { name: 'Glute bridge', muscle: 'Glutes', sets: 3, reps: 12, cue: 'Squeeze at the top for one second.' },
    { name: 'Bird dog', muscle: 'Core', sets: 2, reps: 8, cue: 'Opposite arm and leg, keep hips level.' },
    { name: 'Slow walk + stretch', muscle: 'Cool-down', sets: 1, seconds: 120, cue: 'Breathe slowly through the nose.' },
  ],
};

export type PlanMeal = { time: string; name: string; kcal: number; protein: number; carbs: number; fat: number; tag?: string };
export const MEAL_PLAN: PlanMeal[] = [
  { time: 'Breakfast', name: 'Vegetable oats upma + curd', kcal: 340, protein: 14, carbs: 48, fat: 9, tag: 'Low sodium' },
  { time: 'Lunch', name: 'Brown rice, rajma, cucumber salad', kcal: 560, protein: 21, carbs: 86, fat: 12 },
  { time: 'Snack', name: 'Roasted chana + guava', kcal: 190, protein: 9, carbs: 28, fat: 4, tag: 'High fibre' },
  { time: 'Dinner', name: 'Moong dal chilla + mint chutney', kcal: 420, protein: 24, carbs: 46, fat: 13 },
];

export const QUICK_PROMPTS = ["I'm always thirsty and tired", 'Here is my chest X-ray', 'I have a rash on my arm', 'Explain my blood report'];
