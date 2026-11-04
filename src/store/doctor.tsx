import { createContext, ReactNode, useContext, useMemo, useRef, useState } from 'react';
import { Block, chat, ChatResponse } from '../lib/api';
import { getUserContext, useHealth } from './health';

/** One conversation shared by the AI Doctor chat and the voice screen, so either can pick up where the other stopped. */
export type Msg = { id: string; role: 'user' | 'assistant'; text?: string; blocks?: Block[]; used?: string[] };
export type Extra = { form_values?: Record<string, string>; image_id?: string; model_id?: string; report_text?: string; label?: string };
type Turn = { role: 'user' | 'assistant'; content: string };

type DoctorStore = {
  msgs: Msg[];
  busy: boolean;
  answered: Record<string, boolean>;
  markAnswered: (key: string) => void;
  send: (message: string, extra?: Extra, via?: 'text' | 'voice') => Promise<ChatResponse | null>;
};

const Ctx = createContext<DoctorStore | null>(null);
const sessionId = Math.random().toString(36).slice(2);
const newId = () => Math.random().toString(36).slice(2);

function toHistory(msgs: Msg[]): Turn[] {
  return msgs.flatMap((m): Turn[] => m.role === 'user' ? [{ role: 'user', content: m.text ?? '' }]
    : (m.blocks ?? []).filter(b => b.type === 'text').map(b => ({ role: 'assistant', content: (b as { text: string }).text })));
}

export function DoctorProvider({ children }: { children: ReactNode }) {
  const h = useHealth();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [answered, setAnswered] = useState<Record<string, boolean>>({});
  const latest = useRef(msgs);
  const health = useRef(h);
  health.current = h;

  const update = (fn: (m: Msg[]) => Msg[]) => { latest.current = fn(latest.current); setMsgs(latest.current); };

  const store = useMemo<DoctorStore>(() => ({
    msgs, busy, answered,
    markAnswered: key => setAnswered(a => ({ ...a, [key]: true })),
    send: async (message, extra, via = 'text') => {
      if (!message.trim() && !extra) return null;
      const { label, ...rest } = extra ?? {};
      const history = toHistory(latest.current);
      update(m => [...m, { id: newId(), role: 'user', text: label ?? message }]);
      setBusy(true);
      try {
        const res = await chat({ session_id: sessionId, message, context: getUserContext(health.current), history, via, ...rest });
        const used = res.blocks.filter(b => b.type === 'model_result').map(b => (b as { model_name: string }).model_name);
        update(m => [...m, { id: newId(), role: 'assistant', blocks: res.blocks, used }]);
        return res;
      } finally { setBusy(false); }
    },
  }), [msgs, busy, answered]);

  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

export function useDoctor(): DoctorStore {
  const store = useContext(Ctx);
  if (!store) throw new Error('useDoctor must be used inside DoctorProvider');
  return store;
}
