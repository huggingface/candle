use crate::*;
use std::collections::HashMap;

/// Per-turn metadata. The session event/text budgets bound retained identities.
#[derive(Default)]
pub(super) struct EventMetadata {
    segments: HashMap<String, (u64, bool)>,
    next_segment: u64,
    audio_end: u64,
}
impl EventMetadata {
    pub(super) fn normalize(
        &mut self,
        event: AudioEvent,
        structured: bool,
    ) -> Result<(AudioEvent, usize)> {
        let (event, extra_bytes) = match event {
            AudioEvent::Transcript { turn, transcript } if structured => {
                let id = loop {
                    let id = format!("candle:{turn}:{}", self.next_segment);
                    self.next_segment = self.next_segment.checked_add(1).ok_or_else(|| {
                        Error::LimitExceeded("transcript segment counter overflow".into())
                    })?;
                    if !self.segments.contains_key(&id) {
                        break id;
                    }
                };
                let size = id.len();
                (
                    AudioEvent::TranscriptSegment {
                        turn,
                        transcript,
                        status: SegmentStatus::Final,
                        metadata: SegmentMetadata {
                            segment_id: id,
                            ..Default::default()
                        },
                    },
                    size,
                )
            }
            AudioEvent::Audio { turn, audio } if structured => (
                AudioEvent::AudioChunk {
                    turn,
                    audio,
                    start_frame: self.audio_end,
                },
                0,
            ),
            event => (event, 0),
        };
        match &event {
            AudioEvent::TranscriptSegment {
                status, metadata, ..
            } => {
                if metadata.segment_id.trim().is_empty() {
                    return Err(Error::InvalidInput("transcript segment ID is empty".into()));
                }
                match self.segments.get(&metadata.segment_id) {
                    None if metadata.revision != 0 || *status == SegmentStatus::Revision => {
                        return Err(Error::InvalidInput(
                            "new transcript segments must start at revision zero as partial or final"
                                .into(),
                        ));
                    }
                    Some((previous, _)) if previous.checked_add(1) != Some(metadata.revision) => {
                        return Err(Error::InvalidInput(
                            "transcript revisions must increase by one for an existing segment"
                                .into(),
                        ));
                    }
                    Some((_, true)) if *status != SegmentStatus::Revision => {
                        return Err(Error::InvalidInput(
                            "a final transcript can only be corrected by a revision".into(),
                        ));
                    }
                    _ => {}
                }
                self.segments.insert(
                    metadata.segment_id.clone(),
                    (metadata.revision, *status != SegmentStatus::Partial),
                );
            }
            AudioEvent::AudioChunk {
                audio, start_frame, ..
            } => {
                if *start_frame < self.audio_end {
                    return Err(Error::InvalidInput(
                        "audio chunk clocks overlap or regress".into(),
                    ));
                }
                self.audio_end = start_frame
                    .checked_add(audio.frames() as u64)
                    .ok_or_else(|| Error::InvalidInput("audio chunk clock overflow".into()))?;
            }
            AudioEvent::Audio { audio, .. } => {
                self.audio_end = self
                    .audio_end
                    .checked_add(audio.frames() as u64)
                    .ok_or_else(|| Error::InvalidInput("audio output clock overflow".into()))?;
            }
            _ => {}
        }
        Ok((event, extra_bytes))
    }
}
