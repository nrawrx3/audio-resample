use core::slice;
use std::{f32::consts::PI, time::Duration};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use device_config::get_wanted_device_config;
use log::info;

mod device_config;

fn start_cpal_playback_thread(
    stop_playback_thread: std::sync::mpsc::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    let supported_configs = device
        .supported_output_configs()
        .expect("error while querying configs");

    let sample_rate = 48000;
    let channels = 2;
    let buffer_size_ms = 10;
    let buffer_size_samples = (sample_rate / 1000) * buffer_size_ms;

    let config = get_wanted_device_config(
        supported_configs,
        sample_rate,
        channels,
        buffer_size_samples,
    )
    .expect("no config found that matches the wanted parameters");

    info!("Using config: {:?}", config);
    info!("Playing on device: {}", device.name().unwrap());

    std::thread::spawn(move || {
        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    print!(".");
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

        // if stop_playback_thread.recv().is_ok() {
        //     info!("stopping playback thread");
        // }

        std::thread::park();
    })
}

fn main() {
    env_logger::init();

    let (stop_cpal_tx, stop_cpal_rx) = std::sync::mpsc::channel();

    let t = start_cpal_playback_thread(stop_cpal_rx);
    t.join().expect("error while joining thread");
}
