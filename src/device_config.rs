use cpal::{SampleRate, SupportedOutputConfigs};
use log::info;

pub fn get_wanted_device_config(
    output_configs: SupportedOutputConfigs,
    sample_rate: u32,
    channels: u32,
    buffer_size_in_samples: u32,
) -> Option<cpal::StreamConfig> {
    let mut must_have_config = None;

    let buffer_size_in_bytes = buffer_size_in_samples * channels * (size_of::<f32>() as u32);

    for config in output_configs {
        info!("Supported output config: {:?}", config);

        if must_have_config.is_some() {
            continue;
        }

        let sr_min = config.min_sample_rate();
        let sr_max = config.max_sample_rate();

        let sr_buffer_size = match config.buffer_size().clone() {
            cpal::SupportedBufferSize::Range { min, max } => {
                if min <= buffer_size_in_bytes && buffer_size_in_bytes <= max {
                    Some(buffer_size_in_bytes)
                } else {
                    None
                }
            }

            cpal::SupportedBufferSize::Unknown => None,
        };

        // Usually we will have 48k sample rate and 2 channels. Hardcoding it for that.
        if sr_min == SampleRate(sample_rate)
            && sr_max == SampleRate(sample_rate)
            && sr_buffer_size.is_some()
        {
            must_have_config = Some(cpal::StreamConfig {
                channels: channels as u16,
                sample_rate: SampleRate(sample_rate),
                buffer_size: cpal::BufferSize::Fixed(buffer_size_in_samples),
            });
        }
    }

    return must_have_config;
}
