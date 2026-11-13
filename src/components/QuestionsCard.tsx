// Doctor-style follow-up questions: tap one answer per question, then send them all at once.
import { useState } from 'react';
import { Text, View } from 'react-native';
import { colors, type } from '../theme';
import { Card, Chip, PrimaryButton } from './ui';

export function QuestionsCard({ questions, onSubmit, done }: { questions: { q: string; options: string[] }[]; onSubmit: (text: string) => void; done?: boolean }) {
  const [picked, setPicked] = useState<Record<number, string>>({});
  const answered = Object.keys(picked).length;
  return (
    <Card style={{ gap: 14 }}>
      {questions.map((item, i) => (
        <View key={i} style={{ gap: 8 }}>
          <Text style={[type.body, { color: colors.ink }]}>{i + 1}. {item.q}</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
            {item.options.map(o => <Chip key={o} label={o} active={picked[i] === o} onPress={() => !done && setPicked({ ...picked, [i]: o })} />)}
          </View>
        </View>
      ))}
      {done ? <Text style={[type.small, { color: colors.muted }]}>Answers sent.</Text> : (
        <PrimaryButton label={answered < questions.length ? `Send answers (${answered}/${questions.length})` : 'Send answers'} disabled={!answered}
          onPress={() => onSubmit(questions.map((item, i) => `${item.q} ${picked[i] ?? 'Not sure'}`).join('\n'))} />
      )}
    </Card>
  );
}
