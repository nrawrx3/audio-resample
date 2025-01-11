use cpal::{
    traits::{DeviceTrait, HostTrait},
    SampleRate, SupportedInputConfigs, SupportedOutputConfigs, SupportedStreamConfig,
    SupportedStreamConfigRange,
};
use log::info;
use std::io::{self, Write};

use crate::constants;

pub fn find_hardcoded_stream_config(
    stream_configs: &mut dyn Iterator<Item = SupportedStreamConfigRange>,
    sample_rate: usize,
    channels: usize,
    buffer_size_in_samples: usize,
) -> Option<cpal::StreamConfig> {
    let mut must_have_config = None;

    let buffer_size_in_bytes = buffer_size_in_samples * channels * size_of::<f32>();

    for config in stream_configs {
        info!("Supported config for device {:?}", config);

        if must_have_config.is_some() {
            continue;
        }

        if channels != config.channels() as usize {
            continue;
        }

        let sr_min = config.min_sample_rate();
        let sr_max = config.max_sample_rate();

        let sr_buffer_size = match config.buffer_size().clone() {
            cpal::SupportedBufferSize::Range { min, max } => {
                if min <= buffer_size_in_bytes as u32 && buffer_size_in_bytes as u32 <= max {
                    Some(buffer_size_in_bytes)
                } else {
                    None
                }
            }

            cpal::SupportedBufferSize::Unknown => None,
        };

        // Usually we will have 48k sample rate and 2 channels. Hardcoding it for that.
        if sr_min <= SampleRate(sample_rate as u32)
            && sr_max >= SampleRate(sample_rate as u32)
            && sr_buffer_size.is_some()
        {
            must_have_config = Some(cpal::StreamConfig {
                channels: channels as u16,
                sample_rate: SampleRate(sample_rate as u32),
                buffer_size: cpal::BufferSize::Fixed(buffer_size_in_samples as u32),
            });
        }
    }

    return must_have_config;
}

pub fn select_audio_capture_device() -> cpal::Device {
    let host = cpal::default_host();

    let input_devices: Vec<_> = host.input_devices().unwrap().collect();

    println!("Available input devices:");

    for (i, device) in input_devices.iter().enumerate() {
        println!("{}: {}", i, device.name().unwrap());
    }

    let input_device = select_device_menu("input", &input_devices);

    println!(
        "Selected input device: {}",
        input_devices[input_device].name().unwrap()
    );

    input_devices[input_device].clone()
}

fn select_device_menu(kind: &str, devices: &[cpal::Device]) -> usize {
    if devices.is_empty() {
        panic!("No {} devices available!", kind);
    }

    loop {
        print!("Select a {} device (0-{}): ", kind, devices.len() - 1);
        let _ = io::stdout().flush();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim().parse::<usize>() {
            Ok(i) if i < devices.len() => return i,
            _ => {
                println!("Invalid selection");
                continue;
            }
        }
    }
}

pub fn select_suitable_stream_config(
    stream_configs: &mut dyn Iterator<Item = SupportedStreamConfigRange>,
) -> cpal::StreamConfig {
    let all_configs = stream_configs.collect::<Vec<_>>();

    // Iterate over all the supported configs and print them and let the user select one.
    for (i, config) in all_configs.iter().enumerate() {
        println!("{}: {:?}", i, config);
    }

    println!("Select a config: ");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    let selected_config_index = input.trim().parse::<usize>().unwrap();

    let selected_config = all_configs[selected_config_index].clone();

    println!("Selected config: {:?}", selected_config);

    // We just need the F32 pcm formats.

    if selected_config.sample_format() == cpal::SampleFormat::F32 {
        return cpal::StreamConfig {
            // channels: constants::MIN_INPUT_DEVICE_CHANNELS as u16,
            channels: selected_config.channels(),
            sample_rate: selected_config.max_sample_rate(),
            buffer_size: cpal::BufferSize::Fixed(
                constants::MIN_INPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES as u32,
            ),
        };
    } else {
        panic!("Selected config does not satisfy minimum requirements");
    }
}
