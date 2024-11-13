use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::info;

use crate::{
    constants::{
        OUTPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES, OUTPUT_DEVICE_CHANNELS, OUTPUT_DEVICE_SAMPLE_RATE,
    },
    device_config::get_wanted_device_config,
};

pub fn start_cpal_playback_thread(
    stop_playback_thread: std::sync::mpsc::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    let supported_configs = device
        .supported_output_configs()
        .expect("error while querying configs");

    let config = get_wanted_device_config(
        supported_configs,
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
