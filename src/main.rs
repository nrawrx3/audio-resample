use std::fmt;

use clap::{arg, command, Arg, Command};
use constants::{OUTPUT_DEVICE_BUFFER_SIZE_MS, OUTPUT_DEVICE_CHANNELS, OUTPUT_DEVICE_SAMPLE_RATE};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use device_config::get_wanted_device_config;
use log::info;
use rubato::{FftFixedInOut, Resampler};

mod constants;
mod cpal_thread;
mod device_config;

fn new_sample_count(
    original_sample_rate: u32,
    original_sample_count: u32,
    target_sample_rate: u32,
) -> u32 {
    (original_sample_count as f32 / original_sample_rate as f32 * target_sample_rate as f32) as u32
}

// In the following code a frame refers to the N-tuple of a single sample. So
// sample rate is same as frame rate. When We have a data structure xxx_frames,
// it means a Vec of Vec<f32>.
fn append_frames(channel_buffers: &mut [Vec<f32>], additional: &[Vec<f32>], nbr_frames: usize) {
    channel_buffers
        .iter_mut()
        .zip(additional.iter())
        .for_each(|(b, a)| b.extend_from_slice(&a[..nbr_frames]));
}

// Function to write non-interleaved frames to a wav file.
fn write_frames_to_wav(
    channel_buffers: &[Vec<f32>],
    filename: &str,
    num_frames_to_delay: usize,
    sample_rate: u32,
) {
    let mut writer = hound::WavWriter::create(
        filename,
        hound::WavSpec {
            channels: channel_buffers.len() as u16,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .expect("error while creating wav file");

    let num_frames = num_frames_to_delay as usize + channel_buffers[0].len();

    for sample_index in 0..num_frames_to_delay {
        for _ in channel_buffers.iter() {
            writer
                .write_sample(0.0)
                .expect("error while writing sample");
        }
    }

    for sample_index in num_frames_to_delay..num_frames {
        for channel_buffer in channel_buffers.iter() {
            writer
                .write_sample(channel_buffer[sample_index - num_frames_to_delay])
                .expect("error while writing sample");
        }
    }
}

fn wav_file_resampler_app(file_path: &str) {
    let mut reader = hound::WavReader::open(file_path).expect("error while opening wav file");

    let wav_spec = reader.spec();

    info!("Wav spec: {:?}", wav_spec);

    // Collect into interleaved samples first.
    let wav_samples_flat = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect::<Vec<f32>>();

    info!("Wav samples length: {}", wav_samples_flat.len());

    // Convert to non-interleaved samples. This assumes we have OUTPUT_DEVICE_CHANNELS in the input audio already. This is a resampling example, not a channel mixing example. :)
    let mut all_input_frames: Vec<Vec<f32>> =
        vec![
            Vec::<f32>::with_capacity(wav_samples_flat.len() / wav_spec.channels as usize);
            OUTPUT_DEVICE_CHANNELS as usize
        ];

    for channel_number in 0..OUTPUT_DEVICE_CHANNELS {
        // Loop through the samples and put them into the non-interleaved buffer.
        for i in 0..(wav_samples_flat.len() / wav_spec.channels as usize) {
            let pc_index = i * (wav_spec.channels as usize) + channel_number as usize;

            all_input_frames[channel_number as usize].push(wav_samples_flat[pc_index]);
        }
    }

    let input_frame_count = wav_samples_flat.len() as u32 / wav_spec.channels as u32;
    let output_frame_count = new_sample_count(
        wav_spec.sample_rate,
        input_frame_count,
        OUTPUT_DEVICE_SAMPLE_RATE,
    );

    // Create a non-interleaved buffer for the resampled output.
    let mut all_output_frames = vec![
        Vec::<f32>::with_capacity(output_frame_count as usize);
        OUTPUT_DEVICE_CHANNELS as usize
    ];

    // We can arbitrarily choose a chunk size here. Since we will later be doing audio streams, we will have a chunk size of 10 ms worth of samples (due to libwebrtc).
    let input_chunk_size_in_samples =
        OUTPUT_DEVICE_SAMPLE_RATE / 1000 * OUTPUT_DEVICE_BUFFER_SIZE_MS;
    let output_chunk_size_in_samples = new_sample_count(
        wav_spec.sample_rate,
        input_chunk_size_in_samples,
        OUTPUT_DEVICE_SAMPLE_RATE,
    );

    let mut resampler = FftFixedInOut::<f32>::new(
        wav_spec.sample_rate as usize,
        OUTPUT_DEVICE_SAMPLE_RATE as usize,
        input_chunk_size_in_samples as usize,
        wav_spec.channels as usize,
    )
    .expect("error while creating resampler");

    let new_clip_length_in_samples = new_sample_count(
        wav_spec.sample_rate,
        wav_samples_flat.len() as u32 / wav_spec.channels as u32,
        OUTPUT_DEVICE_SAMPLE_RATE,
    );

    let resampler_delay = resampler.output_delay();

    info!("Resampler delay: {}", resampler_delay);
    info!("Original sample rate: {}", wav_spec.sample_rate);
    info!("New sample rate: {}", OUTPUT_DEVICE_SAMPLE_RATE);
    info!(
        "Input clip length: {}",
        wav_samples_flat.len() as u32 / wav_spec.channels as u32
    );
    info!("Output clip length: {}", new_clip_length_in_samples);
    info!("Resampling chunk size: {}", input_chunk_size_in_samples);

    // In a loop, process the next chunk and store it into an output buffer.
    // ...

    // The input buffer is of type Vec<&[f32]>, where the outer Vec is the channels and the inner &[f32] is the samples of corresponding channel. The slices will be trimmed from beginning as the resampler processes them.
    let mut input_frame_slices: Vec<&[f32]> = all_input_frames.iter().map(|v| &v[..]).collect();

    // Maximum number of output samples the resampler can return in a single call.
    let mut output_frame_slices =
        vec![vec![0.0f32; resampler.output_frames_max()]; OUTPUT_DEVICE_CHANNELS as usize];

    let mut input_frames_next_count = resampler.input_frames_next();

    while input_frame_slices[0].len() > input_frames_next_count {
        let (num_in_samples, num_out_samples) = resampler
            .process_into_buffer(&input_frame_slices, &mut output_frame_slices, None)
            .unwrap();

        // Trim the input slices.
        for chan in input_frame_slices.iter_mut() {
            *chan = &chan[num_in_samples..];
        }

        append_frames(
            &mut all_output_frames,
            &output_frame_slices,
            num_out_samples,
        );

        input_frames_next_count = resampler.input_frames_next();
    }

    // Remaining partial chunk.
    if input_frame_slices[0].len() > 0 {
        let (_num_in_samples, num_out_samples) = resampler
            .process_partial_into_buffer(Some(&input_frame_slices), &mut output_frame_slices, None)
            .unwrap();

        append_frames(
            &mut all_output_frames,
            &output_frame_slices,
            num_out_samples,
        );
    }

    info!("Done resampling");

    let filename = fmt::format(format_args!(
        "audio/resampled_{}_kHz.wav",
        OUTPUT_DEVICE_SAMPLE_RATE
    ));

    // Write the resampled frames to a wav file.
    write_frames_to_wav(
        &all_output_frames,
        &filename,
        resampler_delay,
        OUTPUT_DEVICE_SAMPLE_RATE,
    );
}

fn main() {
    env_logger::init();

    // let (stop_cpal_tx, stop_cpal_rx) = std::sync::mpsc::channel();

    // let cpal_thread = cpal_thread::start_cpal_playback_thread(stop_cpal_rx);

    // cpal_thread.join().expect("error while joining thread");

    let cmd= Command::new("audio-resampler")
        .about("Resample audio file or stream")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("file")
            .about("Resample audio file")
            .arg(arg!(<FILE> "Path to the audio file to resample"))
            .arg_required_else_help(true)
        );

    let matches = cmd.get_matches();

    match matches.subcommand() {
        Some(("file", file_matches)) => {
            let file_path = file_matches.get_one::<String>("FILE").unwrap();
            wav_file_resampler_app(file_path);
        }
        _ => {
            panic!("Unknown subcommand");
        }
    }
}
