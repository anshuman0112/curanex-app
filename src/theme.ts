// Design tokens carried over from the Curanex website (:root in index.html)
export const colors = {
  bg: '#F6EFE3',          // cream page background
  card: '#FFFFFF',
  ink: '#221709',          // dark cards, primary text
  fg: '#2B2013',
  inkSoft: '#3A2A18',      // raised surface on dark
  muted: '#9B8A6F',
  subtle: '#C3B294',
  line: '#E2D5BD',
  surface: '#EFE5D1',
  surface2: '#E4D7BD',
  accent: '#B15F2C',
  accentFrom: '#CF8047',
  accentTo: '#97501F',
  onDark: '#F6EFE3',
  onDarkMuted: '#A8957A',
  sos: '#C2341F',          // warm emergency red that fits the palette
  sosDeep: '#8E1F10',
  good: '#5E7B3A',
  warn: '#C98A1B',
};

export const radius = { card: 32, cardSm: 20, control: 14, pill: 999 };
export const space = (n: number) => n * 4;

export const fonts = {
  body: 'Onest_400Regular',
  medium: 'Onest_500Medium',
  semibold: 'Onest_600SemiBold',
  bold: 'Onest_700Bold',
  display: 'PlayfairDisplay_500Medium',
  displayBold: 'PlayfairDisplay_600SemiBold',
};

export const type = {
  hero: { fontFamily: fonts.display, fontSize: 38, lineHeight: 42, letterSpacing: -0.5 },
  h1: { fontFamily: fonts.display, fontSize: 30, lineHeight: 36, letterSpacing: -0.3 },
  h2: { fontFamily: fonts.semibold, fontSize: 22, lineHeight: 28, letterSpacing: -0.3 },
  h3: { fontFamily: fonts.semibold, fontSize: 17, lineHeight: 22 },
  body: { fontFamily: fonts.body, fontSize: 15, lineHeight: 22 },
  small: { fontFamily: fonts.body, fontSize: 13, lineHeight: 18 },
  label: { fontFamily: fonts.medium, fontSize: 11, lineHeight: 14, letterSpacing: 1.2, textTransform: 'uppercase' as const },
};

export const MAX_W = 1080;
