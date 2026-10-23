import { afterEach, describe, expect, it, vi } from 'vitest';
import { createDeltaBatcher, scheduleFrame, type Schedule } from './deltaBatcher';

// A scheduler the test fires by hand, standing in for one animation frame
const manualSchedule = () => {
  const queue: { callback: () => void; cancelled: boolean }[] = [];
  const schedule: Schedule = (callback) => {
    const entry = { callback, cancelled: false };
    queue.push(entry);
    return () => {
      entry.cancelled = true;
    };
  };
  const frame = () => queue.splice(0).forEach((e) => !e.cancelled && e.callback());
  return { schedule, frame, scheduled: () => queue.filter((e) => !e.cancelled).length };
};

describe('createDeltaBatcher', () => {
  it('combines every chunk pushed within a frame into one emit', () => {
    const emit = vi.fn();
    const { schedule, frame, scheduled } = manualSchedule();
    const batcher = createDeltaBatcher(emit, schedule);
    batcher.push('Hel');
    batcher.push('lo ');
    batcher.push('world');
    expect(emit).not.toHaveBeenCalled();
    expect(scheduled()).toBe(1);
    frame();
    expect(emit).toHaveBeenCalledTimes(1);
    expect(emit).toHaveBeenCalledWith('Hello world');
  });

  it('schedules again for chunks that arrive after a frame', () => {
    const emit = vi.fn();
    const { schedule, frame } = manualSchedule();
    const batcher = createDeltaBatcher(emit, schedule);
    batcher.push('a');
    frame();
    batcher.push('b');
    frame();
    expect(emit.mock.calls).toEqual([['a'], ['b']]);
  });

  it('flush emits what is queued at once and cancels the pending frame', () => {
    const emit = vi.fn();
    const { schedule, frame } = manualSchedule();
    const batcher = createDeltaBatcher(emit, schedule);
    batcher.push('tail');
    batcher.flush();
    expect(emit).toHaveBeenCalledWith('tail');
    frame();
    expect(emit).toHaveBeenCalledTimes(1);
  });

  it('flush with nothing queued emits nothing', () => {
    const emit = vi.fn();
    createDeltaBatcher(emit, manualSchedule().schedule).flush();
    expect(emit).not.toHaveBeenCalled();
  });

  it('discard drops the queue and its frame', () => {
    const emit = vi.fn();
    const { schedule, frame } = manualSchedule();
    const batcher = createDeltaBatcher(emit, schedule);
    batcher.push('gone');
    batcher.discard();
    frame();
    batcher.flush();
    expect(emit).not.toHaveBeenCalled();
  });

  it('ignores empty chunks', () => {
    const emit = vi.fn();
    const { schedule, scheduled } = manualSchedule();
    createDeltaBatcher(emit, schedule).push('');
    expect(scheduled()).toBe(0);
  });
});

describe('scheduleFrame', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('falls back to a 16 ms timeout when requestAnimationFrame is unavailable', () => {
    vi.useFakeTimers();
    vi.stubGlobal('requestAnimationFrame', undefined);
    const callback = vi.fn();
    scheduleFrame(callback);
    vi.advanceTimersByTime(15);
    expect(callback).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('cancels the fallback timeout', () => {
    vi.useFakeTimers();
    vi.stubGlobal('requestAnimationFrame', undefined);
    const callback = vi.fn();
    scheduleFrame(callback)();
    vi.advanceTimersByTime(100);
    expect(callback).not.toHaveBeenCalled();
  });

  it('uses requestAnimationFrame when it exists', () => {
    const raf = vi.fn(() => 7);
    const caf = vi.fn();
    vi.stubGlobal('requestAnimationFrame', raf);
    vi.stubGlobal('cancelAnimationFrame', caf);
    const cancel = scheduleFrame(() => undefined);
    expect(raf).toHaveBeenCalledTimes(1);
    cancel();
    expect(caf).toHaveBeenCalledWith(7);
  });
});
