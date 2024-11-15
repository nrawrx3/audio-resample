use std::path::Iter;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{error, info};
use ringbuf::traits::Producer;

use crate::{
    constants::{
        INPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES, INPUT_DEVICE_CHANNELS, INPUT_DEVICE_SAMPLE_RATE, OUTPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES, OUTPUT_DEVICE_CHANNELS, OUTPUT_DEVICE_SAMPLE_RATE
    },
    device_config::find_suitable_stream_config,
};

pub fn start_cpal_playback_thread(
    stop_playback_thread: std::sync::mpsc::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    let mut supported_configs = device
        .supported_output_configs()
        .expect("error while querying configs");

    let config = find_suitable_stream_config(
        &mut supported_configs as &mut dyn Iterator<Item = _>,
        OUTPUT_DEVICE_SAMPLE_RATE,
        OUTPUT_DEVICE_CHANNELS,
        OUTPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES,
    )
    .expect("no config found that matches the wanted parameters");

    info!("Using config: {:?}", config);
    info!("Playing on device: {}", device.name().unwrap());
    std::thread::spawn(move || {
        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut i = 0;
                    while i < data.len() {
                        data[i] = 1.0;
                        i += 1;
                    }
                },
                move |err| {
                    eprintln!("an error occurred on stream: {}", err);
                },
                None,
            )
            .expect("error while building output stream");

        if let Err(err) = stream.play() {
            eprintln!("an error occurred while starting the stream: {}", err);
        }

        if stop_playback_thread.recv().is_ok() {
            info!("stopping playback thread");
        }

        // std::thread::park();
    })
}

pub fn start_cpal_capture_thread<T: Producer<Item = f32> + Send + 'static>(
    stop_capture_thread: std::sync::mpsc::Receiver<()>,
    mut pcm_producer: T,
) -> std::thread::JoinHandle<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no input device available");

    info!("Capturing on device: {}", device.name().unwrap());

    let mut supported_configs = device
        .supported_input_configs()
        .expect("error while querying configs");

    let config = find_suitable_stream_config(
        &mut supported_configs as &mut dyn Iterator<Item = _>,
        INPUT_DEVICE_SAMPLE_RATE,
        INPUT_DEVICE_CHANNELS,
        INPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES,
    ).expect("no config found that matches the wanted parameters");

    std::thread::spawn(move ||  {
        let stream = device
        .build_input_stream(&config, move |data: &[f32], _| {
            // Push all the samples into the ringbuffer.
            for &pcm in data {
                if pcm_producer.try_push(pcm).is_err() {
                    error!("ringbuffer is full, dropping samples");
                }
            }
        }, move |err| {
            error!("an error occurred on stream: {}", err);
        }, None);
    })
}
