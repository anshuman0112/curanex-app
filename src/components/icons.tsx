import Svg, { Path, Circle, Rect } from 'react-native-svg';

type P = { size?: number; color?: string; stroke?: number };
const S = ({ size = 22, children }: { size?: number; children: React.ReactNode }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">{children}</Svg>
);
const st = (c: string, w = 1.8) => ({ stroke: c, strokeWidth: w, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const });

export const Sparkle = ({ size = 22, color = '#B15F2C' }: P) => (
  <Svg width={size} height={size} viewBox="0 0 24 24"><Path d="M12 1.5c.6 5.6 4.9 9.9 10.5 10.5-5.6.6-9.9 4.9-10.5 10.5C11.4 16.9 7.1 12.6 1.5 12 7.1 11.4 11.4 7.1 12 1.5Z" fill={color} /></Svg>
);
export const ArrowUR = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M7 17 17 7M8 7h9v9" {...st(color, stroke)} /></S>;
export const ArrowR = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M4 12h16m-6-6 6 6-6 6" {...st(color, stroke)} /></S>;
export const ArrowL = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M20 12H4m6-6-6 6 6 6" {...st(color, stroke)} /></S>;
export const Home = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1v-9.5Z" {...st(color, stroke)} /></S>;
export const Stetho = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M6 3v5a5 5 0 0 0 10 0V3M11 13v2a5 5 0 0 0 10 0v-1" {...st(color, stroke)} /><Circle cx="21" cy="12" r="2" {...st(color, stroke)} /></S>;
export const Leaf = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M5 19c0-9 6-14 15-14 0 9-5 15-14 15M5 19l7-7" {...st(color, stroke)} /></S>;
export const Dumbbell = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M6 7v10M3 9.5v5M18 7v10M21 9.5v5M6 12h12" {...st(color, stroke)} /></S>;
export const Phone = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M5 3h4l2 5-2.5 1.5a11 11 0 0 0 6 6L16 13l5 2v4a2 2 0 0 1-2 2A17 17 0 0 1 3 5a2 2 0 0 1 2-2Z" {...st(color, stroke)} /></S>;
export const Pin = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 21s-7-6.2-7-12a7 7 0 0 1 14 0c0 5.8-7 12-7 12Z" {...st(color, stroke)} /><Circle cx="12" cy="9" r="2.5" {...st(color, stroke)} /></S>;
export const Mic = ({ size, color = '#221709', stroke }: P) => <S size={size}><Rect x="9" y="3" width="6" height="11" rx="3" {...st(color, stroke)} /><Path d="M5 11a7 7 0 0 0 14 0M12 18v3" {...st(color, stroke)} /></S>;
export const Send = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M5 12h14M13 6l6 6-6 6" {...st(color, stroke)} /></S>;
export const Camera = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M4 8h3l2-3h6l2 3h3v11H4V8Z" {...st(color, stroke)} /><Circle cx="12" cy="13" r="3.5" {...st(color, stroke)} /></S>;
export const Plus = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 5v14M5 12h14" {...st(color, stroke)} /></S>;
export const Check = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="m5 12.5 4.5 4.5L19 7.5" {...st(color, stroke)} /></S>;
export const Close = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M6 6l12 12M18 6 6 18" {...st(color, stroke)} /></S>;
export const Drop = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z" {...st(color, stroke)} /></S>;
export const Heart = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 20s-8-4.7-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 9c0 6.3-8 11-8 11Z" {...st(color, stroke)} /></S>;
export const Grid = ({ size, color = '#221709', stroke }: P) => <S size={size}><Rect x="4" y="4" width="6.5" height="6.5" rx="1.5" {...st(color, stroke)} /><Rect x="13.5" y="4" width="6.5" height="6.5" rx="1.5" {...st(color, stroke)} /><Rect x="4" y="13.5" width="6.5" height="6.5" rx="1.5" {...st(color, stroke)} /><Rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.5" {...st(color, stroke)} /></S>;
export const Play = ({ size, color = '#221709' }: P) => <S size={size}><Path d="M8 5.5v13l10.5-6.5L8 5.5Z" fill={color} /></S>;
export const Pause = ({ size, color = '#221709' }: P) => <S size={size}><Rect x="6.5" y="5" width="4" height="14" rx="1" fill={color} /><Rect x="13.5" y="5" width="4" height="14" rx="1" fill={color} /></S>;
export const Doc = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M6 3h8l4 4v14H6V3Z M14 3v4h4M9 12h6M9 16h6" {...st(color, stroke)} /></S>;
export const Pill = ({ size, color = '#221709', stroke }: P) => <S size={size}><Rect x="3" y="8" width="18" height="8" rx="4" transform="rotate(-35 12 12)" {...st(color, stroke)} /><Path d="m9.5 8.5 5 7" {...st(color, stroke)} /></S>;
export const Shield = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.2 7.5 9.5 4.3-1.3 7.5-4.9 7.5-9.5V6L12 3Z" {...st(color, stroke)} /></S>;
export const Flame = ({ size, color = '#221709', stroke }: P) => <S size={size}><Path d="M12 21a6 6 0 0 0 6-6c0-4-3-6-4-10-2 2-3 4-3 6-1-1-1.5-2-1.5-3C7 10 6 12.5 6 15a6 6 0 0 0 6 6Z" {...st(color, stroke)} /></S>;
