//! Standard MIDI File (format 0) serialization of decoded notes, matching the
//! reference `note_event2midi` (mido-based) byte for byte: same event
//! ordering, channel assignment, running status and end-of-track handling.

use candle_transformers::models::muscriptor::tokenizer::{Note, DRUM_PROGRAM};

const TICKS_PER_BEAT: u32 = 480;
const TEMPO_US: f64 = 500_000.; // 120 bpm
const VELOCITY: u8 = 100;

#[derive(Debug, Clone)]
struct MidiEvent {
    is_drum: bool,
    program: i32,
    time: f64,
    /// 1 for onset, 0 for offset.
    velocity: u8,
    pitch: u8,
}

fn second2tick(second: f64) -> i64 {
    (second * TICKS_PER_BEAT as f64 * 1e6 / TEMPO_US).round_ties_even() as i64
}

fn push_varlen(out: &mut Vec<u8>, mut value: u64) {
    let mut bytes = vec![(value & 0x7f) as u8];
    value >>= 7;
    while value > 0 {
        bytes.push((value & 0x7f) as u8 | 0x80);
        value >>= 7;
    }
    bytes.reverse();
    out.extend(bytes);
}

/// Serialize notes to a single-track (format 0) MIDI file.
pub fn notes_to_midi_bytes(notes: &[Note]) -> Vec<u8> {
    // Notes to on/off events; drums get only an onset here...
    let mut events = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        events.push(MidiEvent {
            is_drum: note.is_drum,
            program: note.program,
            time: note.onset,
            velocity: 1,
            pitch: note.pitch,
        });
        if !note.is_drum {
            events.push(MidiEvent {
                is_drum: false,
                program: note.program,
                time: note.offset,
                velocity: 0,
                pitch: note.pitch,
            });
        }
    }
    // ...and a synthetic offset 10ms after each drum hit.
    let drum_offsets: Vec<MidiEvent> = events
        .iter()
        .filter(|ev| ev.is_drum)
        .map(|ev| MidiEvent {
            time: ev.time + 0.01,
            velocity: 0,
            ..ev.clone()
        })
        .collect();
    events.extend(drum_offsets);
    events.sort_by(|a, b| {
        a.time
            .total_cmp(&b.time)
            .then(a.is_drum.cmp(&b.is_drum))
            .then(a.program.cmp(&b.program))
            .then(a.velocity.cmp(&b.velocity))
            .then(a.pitch.cmp(&b.pitch))
    });

    let mut track: Vec<u8> = Vec::new();
    let mut running_status: Option<u8> = None;
    let mut push_message = |track: &mut Vec<u8>, delta: u64, status: u8, data: &[u8]| {
        push_varlen(track, delta);
        if running_status != Some(status) {
            track.push(status);
            running_status = Some(status);
        }
        track.extend_from_slice(data);
    };

    // Programs are assigned channels 0-8, 10-15 in order of first appearance;
    // drums always use channel 9. Overflow falls back to channel 15.
    let mut program_to_channel: std::collections::HashMap<i32, u8> = Default::default();
    let mut available_channels: std::collections::VecDeque<u8> = (0..9).chain(10..16).collect();
    let mut drums_initialized = false;
    let mut current_tick = 0i64;
    for ev in &events {
        let absolute_tick = second2tick(ev.time);
        let mut delta_tick = (absolute_tick - current_tick).max(0) as u64;
        current_tick = current_tick.max(absolute_tick);

        let channel = match program_to_channel.get(&ev.program) {
            Some(&ch) => ch,
            None if ev.program == DRUM_PROGRAM || ev.is_drum => {
                if !drums_initialized {
                    push_message(&mut track, delta_tick, 0xc9, &[0]);
                    delta_tick = 0;
                    drums_initialized = true;
                }
                9
            }
            None => {
                let ch = available_channels.pop_front().unwrap_or(15);
                program_to_channel.insert(ev.program, ch);
                push_message(&mut track, delta_tick, 0xc0 | ch, &[ev.program as u8]);
                delta_tick = 0;
                ch
            }
        };

        let (status_nibble, velocity) = if ev.velocity > 0 {
            (0x90u8, VELOCITY)
        } else {
            (0x80u8, 0)
        };
        push_message(
            &mut track,
            delta_tick,
            status_nibble | channel,
            &[ev.pitch, velocity],
        );
    }
    // End of track meta event.
    track.extend_from_slice(&[0x00, 0xff, 0x2f, 0x00]);

    let mut out = Vec::with_capacity(track.len() + 22);
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // format 0
    out.extend_from_slice(&1u16.to_be_bytes()); // one track
    out.extend_from_slice(&(TICKS_PER_BEAT as u16).to_be_bytes());
    out.extend_from_slice(b"MTrk");
    out.extend_from_slice(&(track.len() as u32).to_be_bytes());
    out.extend_from_slice(&track);
    out
}
