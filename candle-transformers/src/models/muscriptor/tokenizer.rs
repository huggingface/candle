//! MT3-style MIDI token vocabulary and the streaming token → note-event decoder.
//!
//! The vocabulary layout is fixed: `PAD`, `EOS`, `UNK`, then `shift`
//! (`max_shift_steps` values), `pitch` (128), `velocity` (2), `tie`,
//! `program` (130) and `drum` (128) ranges.

use std::collections::HashMap;

pub const MAX_SHIFT_STEPS: u32 = 1001;
/// 3 special tokens + 1001 shift + 128 pitch + 2 velocity + 1 tie + 130
/// program + 128 drum.
pub const VOCAB_SIZE: usize = 1393;
pub const EOS_ID: u32 = 1;
pub const DRUM_PROGRAM: i32 = 128;
pub const MINIMUM_NOTE_DURATION_SEC: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Pad,
    Eos,
    Unk,
    Shift(u32),
    Pitch(u8),
    Velocity(u8),
    Tie,
    Program(i32),
    Drum(u8),
}

pub fn decode_token(token: u32) -> Option<Event> {
    let mut t = token;
    match t {
        0 => return Some(Event::Pad),
        1 => return Some(Event::Eos),
        2 => return Some(Event::Unk),
        _ => t -= 3,
    }
    if t < MAX_SHIFT_STEPS {
        return Some(Event::Shift(t));
    }
    t -= MAX_SHIFT_STEPS;
    if t < 128 {
        return Some(Event::Pitch(t as u8));
    }
    t -= 128;
    if t < 2 {
        return Some(Event::Velocity(t as u8));
    }
    t -= 2;
    if t < 1 {
        return Some(Event::Tie);
    }
    t -= 1;
    if t < 130 {
        return Some(Event::Program(t as i32));
    }
    t -= 130;
    if t < 128 {
        return Some(Event::Drum(t as u8));
    }
    None
}

/// The `MT3_FULL_PLUS` grouping of General MIDI programs, as `(group_id,
/// programs)` pairs, followed by singleton groups for every unassigned
/// program in `0..128` (matching the reference tokenizer's
/// `misc_programs="SINGLETON_GROUPS"`, `is_mt3=True` construction).
pub fn group_program_map() -> Vec<Vec<i32>> {
    let mut groups: Vec<Vec<i32>> = vec![
        vec![0, 1, 3, 6, 7],
        vec![2, 4, 5],
        (8..16).collect(),
        (16..24).collect(),
        vec![24, 25],
        vec![26, 27, 28],
        vec![29, 30, 31],
        vec![32, 35],
        vec![33, 34, 36, 37, 38, 39],
        vec![40],
        vec![41],
        vec![42],
        vec![43],
        vec![46],
        vec![47],
        vec![48, 49, 44, 45],
        vec![50, 51],
        vec![52, 53, 54],
        vec![55],
        vec![56, 59],
        vec![57],
        vec![58],
        vec![60],
        vec![61, 62, 63],
        vec![64, 65],
        vec![66],
        vec![67],
        vec![68],
        vec![69],
        vec![70],
        vec![71],
        (72..80).collect(),
        (80..88).collect(),
        (88..96).collect(),
        vec![100],
        vec![101],
    ];
    let assigned: std::collections::HashSet<i32> = groups.iter().flatten().copied().collect();
    for p in 0..128 {
        if !assigned.contains(&p) {
            groups.push(vec![p]);
        }
    }
    groups
}

/// Human-readable names of the `MT3_FULL_PLUS` instrument groups, in group-id
/// order. `("drums", 36)` names the first misc singleton group; the group ids
/// index the model's learned conditioning classes and must not change.
pub const GROUP_NAMES: [(&str, u32); 35] = [
    ("acoustic_piano", 0),
    ("electric_piano", 1),
    ("chromatic_percussion", 2),
    ("organ", 3),
    ("acoustic_guitar", 4),
    ("clean_electric_guitar", 5),
    ("distorted_electric_guitar", 6),
    ("acoustic_bass", 7),
    ("electric_bass", 8),
    ("violin", 9),
    ("viola", 10),
    ("cello", 11),
    ("contrabass", 12),
    ("orchestral_harp", 13),
    ("timpani", 14),
    ("string_ensemble", 15),
    ("synth_strings", 16),
    ("voice", 17),
    ("orchestra_hit", 18),
    ("trumpet", 19),
    ("trombone", 20),
    ("tuba", 21),
    ("french_horn", 22),
    ("brass_section", 23),
    ("soprano_and_alto_sax", 24),
    ("tenor_sax", 25),
    ("baritone_sax", 26),
    ("oboe", 27),
    ("english_horn", 28),
    ("bassoon", 29),
    ("clarinet", 30),
    ("flutes", 31),
    ("synth_lead", 32),
    ("synth_pad", 33),
    ("drums", 36),
];

/// Map a decoded program number to a readable instrument name. The decoded
/// program is always the first program of its group; unassigned programs
/// surface as `program_<n>`.
pub fn instrument_for_program(program: i32) -> String {
    if program == DRUM_PROGRAM {
        return "drums".to_string();
    }
    let groups = group_program_map();
    for (name, gid) in GROUP_NAMES.iter() {
        if let Some(programs) = groups.get(*gid as usize) {
            if programs.first() == Some(&program) {
                return name.to_string();
            }
        }
    }
    format!("program_{program}")
}

/// Inverse of [`instrument_for_program`] for MIDI assembly.
pub fn program_for_instrument(instrument: &str) -> Option<i32> {
    if instrument == "drums" {
        return Some(DRUM_PROGRAM);
    }
    if let Some(n) = instrument.strip_prefix("program_") {
        return n.parse().ok();
    }
    let groups = group_program_map();
    GROUP_NAMES
        .iter()
        .find(|(name, _)| *name == instrument)
        .and_then(|(_, gid)| groups.get(*gid as usize))
        .and_then(|programs| programs.first().copied())
}

/// Resolve loosely-typed instrument tokens to conditioning class ids.
/// Matching is case-insensitive; a non-exact token may be a substring that
/// matches exactly one group name.
pub fn instrument_class_ids(names: &[impl AsRef<str>]) -> std::result::Result<Vec<u32>, String> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        let t = name.as_ref().trim().to_lowercase();
        if let Some((_, gid)) = GROUP_NAMES.iter().find(|(n, _)| *n == t) {
            ids.push(*gid);
            continue;
        }
        let hits: Vec<&(&str, u32)> = GROUP_NAMES.iter().filter(|(n, _)| n.contains(&t)).collect();
        match hits.as_slice() {
            [(_, gid)] => ids.push(*gid),
            [] => {
                let valid: Vec<&str> = GROUP_NAMES.iter().map(|(n, _)| *n).collect();
                return Err(format!(
                    "unknown instrument name {t:?}; valid names: {}",
                    valid.join(", ")
                ));
            }
            _ => {
                let matches: Vec<&str> = hits.iter().map(|(n, _)| *n).collect();
                return Err(format!(
                    "ambiguous instrument name {t:?}: matches {}",
                    matches.join(", ")
                ));
            }
        }
    }
    Ok(ids)
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoteStart {
    pub pitch: u8,
    pub start_time: f64,
    pub index: usize,
    pub instrument: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoteEvent {
    Start(NoteStart),
    End { end_time: f64, start_index: usize },
}

/// Streaming decoder from model tokens to [`NoteEvent`]s.
///
/// Feed each segment with [`TokenDecoder::start_chunk`] followed by its tokens
/// (EOS already stripped) through [`TokenDecoder::push`], then call
/// [`TokenDecoder::finish`] once at the end of the stream. Every emitted
/// `Start` is matched by exactly one later `End` with the same index.
///
/// Each chunk begins with a *tie prologue* — `(program, pitch)` pairs for
/// notes sustained from the previous chunk, terminated by a `tie` token —
/// after which open notes not in the tie set are closed at the boundary.
#[derive(Debug, Clone, Default)]
pub struct TokenDecoder {
    open_notes: HashMap<(i32, u8), NoteStart>,
    next_index: usize,
    seek_time: f64,
    next_seek_time: Option<f64>,
    start_tick: i64,
    tick_state: i64,
    program_state: Option<i32>,
    velocity_state: Option<u8>,
    in_prologue: bool,
    skip_rest: bool,
    tie_set: std::collections::HashSet<(i32, u8)>,
    chunk_started: bool,
    frame_rate: f64,
}

impl TokenDecoder {
    pub fn new(frame_rate: usize) -> Self {
        Self {
            frame_rate: frame_rate as f64,
            ..Default::default()
        }
    }

    fn mint(&mut self, pitch: u8, start_time: f64, instrument: String) -> NoteStart {
        let ev = NoteStart {
            pitch,
            start_time,
            index: self.next_index,
            instrument,
        };
        self.next_index += 1;
        ev
    }

    fn close_all_at_boundary(&mut self, out: &mut Vec<NoteEvent>) {
        let mut open: Vec<_> = self.open_notes.drain().map(|(_, v)| v).collect();
        open.sort_by_key(|ev| ev.index);
        for ev in open {
            out.push(NoteEvent::End {
                end_time: self.seek_time,
                start_index: ev.index,
            });
        }
    }

    pub fn start_chunk(
        &mut self,
        seek_time: f64,
        next_seek_time: Option<f64>,
        out: &mut Vec<NoteEvent>,
    ) {
        // A previous chunk that never closed its tie prologue is malformed:
        // every still-open note ends at that chunk's boundary.
        if self.chunk_started && self.in_prologue {
            self.close_all_at_boundary(out);
        }
        self.seek_time = seek_time;
        self.next_seek_time = next_seek_time;
        self.start_tick = (seek_time * self.frame_rate).round_ties_even() as i64;
        self.tick_state = self.start_tick;
        self.program_state = None;
        self.velocity_state = None;
        self.in_prologue = true;
        self.skip_rest = false;
        self.tie_set.clear();
        self.chunk_started = true;
    }

    pub fn push(&mut self, token: u32, out: &mut Vec<NoteEvent>) {
        let event = match decode_token(token) {
            Some(event) => event,
            None => return,
        };

        if self.in_prologue {
            match event {
                Event::Tie => {
                    // End of the tie section: close prior notes not sustained here.
                    self.in_prologue = false;
                    self.velocity_state = None;
                    let mut to_close: Vec<_> = self
                        .open_notes
                        .iter()
                        .filter(|(key, _)| !self.tie_set.contains(key))
                        .map(|(key, ev)| (*key, ev.index))
                        .collect();
                    to_close.sort_by_key(|&(_, index)| index);
                    for (key, index) in to_close {
                        self.open_notes.remove(&key);
                        out.push(NoteEvent::End {
                            end_time: self.seek_time,
                            start_index: index,
                        });
                    }
                }
                Event::Shift(_) => {
                    // No tie token: the chunk is malformed. Close all open
                    // notes at the boundary and drop the rest of the chunk.
                    self.in_prologue = false;
                    self.skip_rest = true;
                    let seek_time = self.seek_time;
                    let mut open: Vec<_> = self.open_notes.drain().map(|(_, v)| v.index).collect();
                    open.sort_unstable();
                    for index in open {
                        out.push(NoteEvent::End {
                            end_time: seek_time,
                            start_index: index,
                        });
                    }
                }
                Event::Program(p) => self.program_state = Some(p),
                Event::Pitch(pitch) => {
                    if let Some(program) = self.program_state {
                        self.tie_set.insert((program, pitch));
                    }
                }
                _ => {}
            }
            return;
        }

        if self.skip_rest {
            return;
        }

        match event {
            Event::Shift(v) => {
                if v > 0 {
                    self.tick_state = self.start_tick + v as i64;
                }
            }
            Event::Program(p) => self.program_state = Some(p),
            Event::Velocity(v) => self.velocity_state = Some(v),
            Event::Drum(pitch) => {
                let time = self.tick_state as f64 / self.frame_rate;
                if self.next_seek_time.is_none_or(|next| time < next) {
                    let start = self.mint(pitch, time, "drums".to_string());
                    let index = start.index;
                    out.push(NoteEvent::Start(start));
                    out.push(NoteEvent::End {
                        end_time: time + MINIMUM_NOTE_DURATION_SEC,
                        start_index: index,
                    });
                }
            }
            Event::Pitch(pitch) => {
                let (Some(program), Some(velocity)) = (self.program_state, self.velocity_state)
                else {
                    return;
                };
                let time = self.tick_state as f64 / self.frame_rate;
                if self.next_seek_time.is_some_and(|next| time >= next) {
                    return;
                }
                let key = (program, pitch);
                if let Some(open) = self.open_notes.remove(&key) {
                    out.push(NoteEvent::End {
                        end_time: time,
                        start_index: open.index,
                    });
                }
                if velocity > 0 {
                    let start = self.mint(pitch, time, instrument_for_program(program));
                    self.open_notes.insert(key, start.clone());
                    out.push(NoteEvent::Start(start));
                }
            }
            _ => {}
        }
    }

    /// End of stream: close anything still open. A well-formed final chunk
    /// uses the minimum-duration fallback; a chunk that ended mid-prologue
    /// closes at its boundary.
    pub fn finish(&mut self, out: &mut Vec<NoteEvent>) {
        if self.chunk_started && self.in_prologue {
            self.close_all_at_boundary(out);
        } else {
            let mut open: Vec<_> = self.open_notes.drain().map(|(_, v)| v).collect();
            open.sort_by_key(|ev| ev.index);
            for ev in open {
                out.push(NoteEvent::End {
                    end_time: ev.start_time + MINIMUM_NOTE_DURATION_SEC,
                    start_index: ev.index,
                });
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub is_drum: bool,
    /// MIDI program number (0-127); 128 for drums.
    pub program: i32,
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
}

fn sort_notes(notes: &mut [Note]) {
    notes.sort_by(|a, b| {
        a.onset
            .total_cmp(&b.onset)
            .then(a.is_drum.cmp(&b.is_drum))
            .then(a.program.cmp(&b.program))
            .then(a.pitch.cmp(&b.pitch))
            .then(a.offset.total_cmp(&b.offset))
    });
}

/// Reassemble [`Note`]s from a `NoteEvent` stream and apply the reference
/// note-cleanup pass (minimum durations, per-channel overlap trimming, final
/// sort).
pub fn events_to_notes(events: &[NoteEvent]) -> Vec<Note> {
    let mut notes = Vec::new();
    let mut open: HashMap<usize, Note> = HashMap::new();
    for ev in events {
        match ev {
            NoteEvent::Start(start) => {
                let is_drum = start.instrument == "drums";
                let program = if is_drum {
                    DRUM_PROGRAM
                } else {
                    program_for_instrument(&start.instrument).unwrap_or(0)
                };
                open.insert(
                    start.index,
                    Note {
                        is_drum,
                        program,
                        onset: start.start_time,
                        offset: start.start_time, // patched by the matching End
                        pitch: start.pitch,
                    },
                );
            }
            NoteEvent::End {
                end_time,
                start_index,
            } => {
                if let Some(mut note) = open.remove(start_index) {
                    note.offset = *end_time;
                    notes.push(note);
                }
            }
        }
    }
    validate_notes(&mut notes);
    trim_overlapping_notes(notes)
}

/// Enforce a minimum note duration (`validate_notes(fix=True)`).
fn validate_notes(notes: &mut [Note]) {
    for note in notes.iter_mut() {
        if note.onset > note.offset {
            note.offset = note.offset.max(note.onset + MINIMUM_NOTE_DURATION_SEC);
        } else if !note.is_drum && note.offset - note.onset < MINIMUM_NOTE_DURATION_SEC {
            note.offset = note.onset + MINIMUM_NOTE_DURATION_SEC;
        }
    }
}

/// Truncate overlapping same-(program, pitch, drum) notes at the next onset
/// and drop the ones that end up empty; sorts the result.
fn trim_overlapping_notes(notes: Vec<Note>) -> Vec<Note> {
    if notes.len() <= 1 {
        return notes;
    }
    let mut channels: HashMap<(i32, u8, bool), Vec<Note>> = HashMap::new();
    for note in notes {
        channels
            .entry((note.program, note.pitch, note.is_drum))
            .or_default()
            .push(note);
    }
    let mut trimmed = Vec::new();
    for (_, mut channel_notes) in channels {
        channel_notes.sort_by(|a, b| a.onset.total_cmp(&b.onset));
        for i in 1..channel_notes.len() {
            if channel_notes[i - 1].offset > channel_notes[i].onset {
                channel_notes[i - 1].offset = channel_notes[i].onset;
            }
        }
        trimmed.extend(channel_notes.into_iter().filter(|n| n.onset < n.offset));
    }
    sort_notes(&mut trimmed);
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_ranges() {
        assert_eq!(decode_token(0), Some(Event::Pad));
        assert_eq!(decode_token(EOS_ID), Some(Event::Eos));
        assert_eq!(decode_token(2), Some(Event::Unk));
        assert_eq!(decode_token(3), Some(Event::Shift(0)));
        assert_eq!(
            decode_token(3 + MAX_SHIFT_STEPS - 1),
            Some(Event::Shift(1000))
        );
        assert_eq!(decode_token(1004), Some(Event::Pitch(0)));
        assert_eq!(decode_token(1131), Some(Event::Pitch(127)));
        assert_eq!(decode_token(1132), Some(Event::Velocity(0)));
        assert_eq!(decode_token(1133), Some(Event::Velocity(1)));
        assert_eq!(decode_token(1134), Some(Event::Tie));
        assert_eq!(decode_token(1135), Some(Event::Program(0)));
        assert_eq!(decode_token(1264), Some(Event::Program(129)));
        assert_eq!(decode_token(1265), Some(Event::Drum(0)));
        assert_eq!(decode_token(1392), Some(Event::Drum(127)));
        assert_eq!(decode_token(VOCAB_SIZE as u32), None);
    }

    fn token(event: Event) -> u32 {
        (0..VOCAB_SIZE as u32)
            .find(|&t| decode_token(t) == Some(event))
            .unwrap()
    }

    #[test]
    fn instrument_group_round_trip() {
        for (name, _) in GROUP_NAMES.iter() {
            let program = program_for_instrument(name).unwrap();
            assert_eq!(instrument_for_program(program), *name, "{name}");
        }
        assert_eq!(instrument_for_program(DRUM_PROGRAM), "drums");
        assert_eq!(program_for_instrument("program_97"), Some(97));
        assert_eq!(instrument_class_ids(&["ACOUSTIC_PIANO"]), Ok(vec![0]));
        assert_eq!(instrument_class_ids(&["timp"]), Ok(vec![14]));
        assert!(instrument_class_ids(&["piano"]).is_err()); // ambiguous
        assert!(instrument_class_ids(&["theremin"]).is_err());
    }

    /// A well-formed chunk: empty tie prologue, one note opened and closed.
    #[test]
    fn decode_simple_note() {
        let mut decoder = TokenDecoder::new(100);
        let mut out = Vec::new();
        decoder.start_chunk(0.0, Some(5.0), &mut out);
        for t in [
            token(Event::Tie),
            token(Event::Shift(50)),
            token(Event::Program(0)),
            token(Event::Velocity(1)),
            token(Event::Pitch(60)),
            token(Event::Shift(150)),
            token(Event::Velocity(0)),
            token(Event::Pitch(60)),
        ] {
            decoder.push(t, &mut out);
        }
        decoder.finish(&mut out);
        assert_eq!(
            out,
            vec![
                NoteEvent::Start(NoteStart {
                    pitch: 60,
                    start_time: 0.5,
                    index: 0,
                    instrument: "acoustic_piano".to_string(),
                }),
                NoteEvent::End {
                    end_time: 1.5,
                    start_index: 0,
                },
            ]
        );
    }

    /// A note sustained through the tie prologue survives into the next
    /// chunk; one not in the tie set is closed at the boundary.
    #[test]
    fn decode_tie_prologue() {
        let mut decoder = TokenDecoder::new(100);
        let mut out = Vec::new();
        decoder.start_chunk(0.0, Some(5.0), &mut out);
        for t in [
            token(Event::Tie),
            token(Event::Program(0)),
            token(Event::Velocity(1)),
            token(Event::Pitch(60)),
            token(Event::Pitch(64)),
        ] {
            decoder.push(t, &mut out);
        }
        assert_eq!(out.len(), 2); // both notes open
        decoder.start_chunk(5.0, None, &mut out);
        for t in [
            token(Event::Program(0)),
            token(Event::Pitch(60)), // only pitch 60 is tied over
            token(Event::Tie),
            token(Event::Shift(100)),
            token(Event::Velocity(0)),
            token(Event::Pitch(60)),
        ] {
            decoder.push(t, &mut out);
        }
        decoder.finish(&mut out);
        assert_eq!(
            &out[2..],
            &[
                // pitch 64 not tied: closed at the 5 s boundary.
                NoteEvent::End {
                    end_time: 5.0,
                    start_index: 1,
                },
                NoteEvent::End {
                    end_time: 6.0,
                    start_index: 0,
                },
            ]
        );
    }

    /// A chunk whose prologue never emits `tie` (first event is a shift) is
    /// malformed: all open notes close at the boundary, the rest is dropped.
    #[test]
    fn decode_malformed_chunk() {
        let mut decoder = TokenDecoder::new(100);
        let mut out = Vec::new();
        decoder.start_chunk(0.0, Some(5.0), &mut out);
        for t in [
            token(Event::Tie),
            token(Event::Program(40)),
            token(Event::Velocity(1)),
            token(Event::Pitch(70)),
        ] {
            decoder.push(t, &mut out);
        }
        decoder.start_chunk(5.0, None, &mut out);
        decoder.push(token(Event::Shift(10)), &mut out);
        // Note events after the malformed prologue are dropped.
        for t in [
            token(Event::Program(0)),
            token(Event::Velocity(1)),
            token(Event::Pitch(50)),
        ] {
            decoder.push(t, &mut out);
        }
        decoder.finish(&mut out);
        assert_eq!(
            out,
            vec![
                NoteEvent::Start(NoteStart {
                    pitch: 70,
                    start_time: 0.0,
                    index: 0,
                    instrument: "violin".to_string(),
                }),
                NoteEvent::End {
                    end_time: 5.0,
                    start_index: 0,
                },
            ]
        );
    }

    /// Overlapping same-pitch notes are trimmed at the next onset and the
    /// minimum note duration is enforced.
    #[test]
    fn note_cleanup() {
        let events = vec![
            NoteEvent::Start(NoteStart {
                pitch: 60,
                start_time: 0.0,
                index: 0,
                instrument: "acoustic_piano".to_string(),
            }),
            NoteEvent::Start(NoteStart {
                pitch: 60,
                start_time: 1.0,
                index: 1,
                instrument: "acoustic_piano".to_string(),
            }),
            NoteEvent::End {
                end_time: 2.0,
                start_index: 0, // overlaps the second onset
            },
            NoteEvent::End {
                end_time: 1.0, // zero length: extended to the minimum
                start_index: 1,
            },
        ];
        let notes = events_to_notes(&events);
        assert_eq!(notes.len(), 2);
        assert_eq!((notes[0].onset, notes[0].offset), (0.0, 1.0));
        assert_eq!(
            (notes[1].onset, notes[1].offset),
            (1.0, 1.0 + MINIMUM_NOTE_DURATION_SEC)
        );
    }
}
