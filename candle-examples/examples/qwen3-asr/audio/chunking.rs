//! Long-audio chunking utilities.
//!
//! The official implementation uses an energy-based boundary search. This module
//! is a faithful port of:
//! `../../../Qwen3-ASR/qwen_asr/inference/utils.py:split_audio_into_chunks`.

use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub wav: Vec<f32>,
    pub offset_sec: f32,
}

const MIN_ASR_INPUT_SECONDS: f32 = 0.5;
const DEFAULT_SEARCH_EXPAND_SEC: f32 = 5.0;
const DEFAULT_MIN_WINDOW_MS: f32 = 100.0;

pub fn split_audio_into_chunks(
    wav: &[f32],
    sr: u32,
    max_chunk_sec: f32,
) -> Result<Vec<AudioChunk>> {
    if sr == 0 {
        bail!("invalid sample_rate=0");
    }
    if !max_chunk_sec.is_finite() || max_chunk_sec <= 0.0 {
        bail!("max_chunk_sec must be finite and > 0");
    }

    let total_len = wav.len();
    let total_sec = total_len as f32 / sr as f32;
    if total_sec <= max_chunk_sec {
        return Ok(vec![AudioChunk {
            wav: wav.to_vec(),
            offset_sec: 0.0,
        }]);
    }

    let max_len = (max_chunk_sec * sr as f32) as usize;
    let max_len = max_len.max(1);

    // The official stack uses large `max_chunk_sec` values (>= 180s) so the boundary search window
    // is always local. If callers request very small chunk sizes (e.g., 1s), the default 5s
    // expansion can dominate, turning the search into a global "pick the first minimum" and
    // producing pathological micro-chunks. Clamp expansion relative to the target chunk size.
    let expand = (DEFAULT_SEARCH_EXPAND_SEC * sr as f32) as usize;
    let expand = expand.min(max_len / 2);
    let win_raw = ((DEFAULT_MIN_WINDOW_MS / 1000.0) * sr as f32) as usize;
    let win = win_raw.max(4);

    let mut chunks: Vec<AudioChunk> = Vec::new();
    let mut start = 0usize;
    let mut offset_sec = 0.0f32;

    while total_len.saturating_sub(start) > max_len {
        let cut = start.saturating_add(max_len);

        let left = cut.saturating_sub(expand).max(start);
        let right = cut.saturating_add(expand).min(total_len);

        let boundary = if right.saturating_sub(left) <= win {
            cut
        } else {
            let seg = &wav[left..right];
            let seg_abs: Vec<f32> = seg.iter().map(|x| x.abs()).collect();

            let seg_len = seg_abs.len();
            if seg_len <= win {
                cut
            } else {
                let mut window_sum: f32 = seg_abs[..win].iter().sum();
                let mut min_sum = window_sum;
                let mut min_pos = 0usize;

                for pos in 1..=seg_len - win {
                    window_sum += seg_abs[pos + win - 1];
                    window_sum -= seg_abs[pos - 1];
                    if window_sum < min_sum {
                        min_sum = window_sum;
                        min_pos = pos;
                    }
                }

                let local = &seg_abs[min_pos..min_pos + win];
                let mut inner_pos = 0usize;
                let mut inner_min = local[0];
                for (idx, &v) in local.iter().enumerate().skip(1) {
                    if v < inner_min {
                        inner_min = v;
                        inner_pos = idx;
                    }
                }

                left + min_pos + inner_pos
            }
        };

        let min_boundary = start.saturating_add(1);
        let mut boundary = boundary;
        if boundary < min_boundary {
            boundary = min_boundary;
        }
        if boundary > total_len {
            boundary = total_len;
        }

        let chunk_wav = wav[start..boundary].to_vec();
        chunks.push(AudioChunk {
            wav: chunk_wav,
            offset_sec,
        });

        let advanced = boundary.saturating_sub(start);
        offset_sec += advanced as f32 / sr as f32;
        start = boundary;
    }

    let tail = wav[start..total_len].to_vec();
    chunks.push(AudioChunk {
        wav: tail,
        offset_sec,
    });

    let min_len = (MIN_ASR_INPUT_SECONDS * sr as f32) as usize;
    if min_len > 0 {
        for c in &mut chunks {
            if c.wav.len() < min_len {
                c.wav.resize(min_len, 0.0);
            }
        }
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::split_audio_into_chunks;

    #[test]
    fn test_split_audio_into_chunks_short_no_chunking() -> anyhow::Result<()> {
        let sr = 10u32;
        let wav: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let chunks = split_audio_into_chunks(&wav, sr, 5.0)?;
        if chunks.len() != 1 {
            anyhow::bail!("expected 1 chunk, got {}", chunks.len());
        }
        if chunks[0].offset_sec != 0.0 {
            anyhow::bail!("expected offset_sec=0.0, got {}", chunks[0].offset_sec);
        }
        if chunks[0].wav != wav {
            anyhow::bail!("expected chunk wav to equal input wav");
        }
        Ok(())
    }

    #[test]
    fn test_split_audio_into_chunks_reconstructs_original_and_pads_short_tail() -> anyhow::Result<()>
    {
        let sr = 10u32;
        let max_chunk_sec = 3.0;

        // 31 samples => 3.1s; should split once.
        // Place a unique minimal-energy window at the end to force a stable boundary.
        let mut wav = vec![1.0f32; 31];
        for x in &mut wav[27..] {
            *x = 0.0;
        }

        let chunks = split_audio_into_chunks(&wav, sr, max_chunk_sec)?;
        if chunks.len() != 2 {
            anyhow::bail!("expected 2 chunks, got {}", chunks.len());
        }

        let start0 = (chunks[0].offset_sec * sr as f32).round() as usize;
        let start1 = (chunks[1].offset_sec * sr as f32).round() as usize;
        if start0 != 0 {
            anyhow::bail!("expected first offset to map to start=0, got {start0}");
        }
        if start1 != 27 {
            anyhow::bail!("expected second offset to map to start=27, got {start1}");
        }

        let end0 = start1;
        let end1 = wav.len();

        let orig0 = &wav[start0..end0];
        let got0 = chunks[0].wav.get(..orig0.len()).ok_or_else(|| {
            anyhow::anyhow!(
                "chunk0 shorter than expected: chunk_len={} orig_len={}",
                chunks[0].wav.len(),
                orig0.len()
            )
        })?;
        if got0 != orig0 {
            anyhow::bail!("chunk0 does not match original segment");
        }

        let orig1 = &wav[start1..end1];
        let got1 = chunks[1].wav.get(..orig1.len()).ok_or_else(|| {
            anyhow::anyhow!(
                "chunk1 shorter than expected: chunk_len={} orig_len={}",
                chunks[1].wav.len(),
                orig1.len()
            )
        })?;
        if got1 != orig1 {
            anyhow::bail!("chunk1 original region does not match");
        }

        // Tail should be padded to at least 0.5s => min_len=5 at sr=10.
        if chunks[1].wav.len() != 5 {
            anyhow::bail!(
                "expected padded tail chunk len=5, got {}",
                chunks[1].wav.len()
            );
        }
        if chunks[1].wav[orig1.len()] != 0.0 {
            anyhow::bail!("expected padding to be zeros");
        }

        Ok(())
    }

    #[test]
    fn test_split_audio_into_chunks_silence_small_chunk_does_not_microchunk() -> anyhow::Result<()>
    {
        // Regression test: when `DEFAULT_SEARCH_EXPAND_SEC` dominates the requested chunk size,
        // a naive boundary search can pick the first minimum and advance by 1 sample repeatedly.
        //
        // For a constant-energy signal (silence), we should still make progress in sensible
        // increments (except for the final tail), rather than producing hundreds of tiny chunks.
        let sr = 10u32;
        let wav = vec![0.0f32; 30]; // 3.0s at 10Hz
        let chunks = split_audio_into_chunks(&wav, sr, 1.0)?;

        // With the expansion clamp, we expect chunk lengths:
        // - 4 chunks of 0.5s (5 samples) plus a 1.0s tail (10 samples).
        let got_lens: Vec<usize> = chunks.iter().map(|c| c.wav.len()).collect();
        let want_lens = vec![5usize, 5, 5, 5, 10];
        if got_lens != want_lens {
            anyhow::bail!("unexpected chunk lengths: got={got_lens:?} want={want_lens:?}");
        }

        // Ensure offsets match the expected starts.
        let got_offsets: Vec<usize> = chunks
            .iter()
            .map(|c| (c.offset_sec * sr as f32).round() as usize)
            .collect();
        let want_offsets = vec![0usize, 5, 10, 15, 20];
        if got_offsets != want_offsets {
            anyhow::bail!("unexpected chunk offsets: got={got_offsets:?} want={want_offsets:?}");
        }

        // Ensure reconstruction: concatenating the unpadded regions equals the original.
        let mut recon: Vec<f32> = Vec::with_capacity(wav.len());
        for (i, c) in chunks.iter().enumerate() {
            let start = got_offsets[i];
            let end = got_offsets.get(i + 1).copied().unwrap_or(wav.len());
            let orig = wav
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("original slice out of bounds"))?;
            let got = c
                .wav
                .get(..orig.len())
                .ok_or_else(|| anyhow::anyhow!("chunk shorter than original segment"))?;
            if got != orig {
                anyhow::bail!("chunk {i} does not match original segment");
            }
            recon.extend_from_slice(got);
        }
        if recon != wav {
            anyhow::bail!("reconstructed wav does not match input");
        }

        Ok(())
    }
}
