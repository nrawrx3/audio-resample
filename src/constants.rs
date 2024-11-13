// Configuration for the output device. This is what every audio stream will be resampled to.
pub const OUTPUT_DEVICE_SAMPLE_RATE: u32 = 48000;
pub const OUTPUT_DEVICE_CHANNELS: u32 = 2;
pub const OUTPUT_DEVICE_BUFFER_SIZE_MS: u32 = 10;
pub const OUTPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES: u32 =
    (OUTPUT_DEVICE_SAMPLE_RATE / 1000) * OUTPUT_DEVICE_BUFFER_SIZE_MS;
pub const OUTPUT_DEVICE_BUFFER_SIZE_IN_BYTES: u32 =
    OUTPUT_DEVICE_BUFFER_SIZE_IN_SAMPLES * OUTPUT_DEVICE_CHANNELS * size_of::<f32>() as u32;
