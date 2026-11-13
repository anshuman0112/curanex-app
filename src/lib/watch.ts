// Live watch sync needs the Curanex Android app (Health Connect). On web and iPhone the watch data comes from the
// Apple Health export upload instead (see app/devices.tsx). The Android version of this file is watch.android.ts.
import type { DaySummary } from '../store/health';

export type LiveRead = { days: DaySummary[]; apps: string[] };

/** Why live sync is not available here (null when it is). */
export function liveUnavailableReason(): string | null {
  return 'Live sync works in the Curanex Android app (Health Connect). On iPhone or the website, use the Apple Health export upload below.';
}

export async function readHealthConnect(): Promise<LiveRead> {
  throw new Error(liveUnavailableReason() ?? 'Not available.');
}

export function openHealthConnect(): void {}
