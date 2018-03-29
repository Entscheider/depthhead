/// Use a trained forest to predict the in real time the head pose using the kinect as input.

extern crate depthhead;
extern crate depthhead_example_utils;
extern crate image;
extern crate clap;
extern crate freenectrs;


extern crate serde_json;
#[macro_use]
extern crate log;
extern crate env_logger;

use depthhead_example_utils::{headwin, imgwin, VirtualKeyCode};
use std::time::Duration;

use freenectrs::freenect;
use depthhead::hough::prediction::HoughPrediction;
use clap::{Arg, App};
use std::sync::Arc;
use depthhead::types::{DepthImage, IntrinsicMatrix};

#[macro_use]
extern crate error_chain;
error_chain!{
    foreign_links {
        Serde(serde_json::Error);
        IO(std::io::Error);
        Freenect(freenect::FreenectError);
    }
}

use std::sync;
use std::thread;

/// Messages that are send to the prediction thread
enum ParTask {
    Midp(Arc<DepthImage>),
    Mask(Arc<DepthImage>),
    Break,
}

/// Does prediction using another thread.
struct ParPredict {
    midp_rot: sync::Arc<sync::Mutex<Option<([f32; 3], [f32; 3])>>>,
    mask: sync::Arc<sync::Mutex<Option<(Arc<DepthImage>,
                                        image::ImageBuffer<image::Luma<u8>, Vec<u8>>)>>>,
    sender: sync::mpsc::SyncSender<ParTask>,
    join: Option<thread::JoinHandle<()>>,
}

impl ParPredict {
    fn init(forest: HoughPrediction, sluggish: bool, maxguess: bool) -> ParPredict {
        let midp_rot = sync::Arc::new(sync::Mutex::new(None));
        let mask = sync::Arc::new(sync::Mutex::new(None));
        let mask2 = mask.clone();
        let midp_rot2 = midp_rot.clone();
        let (s, r) = sync::mpsc::sync_channel(1);
        let join = thread::spawn(move || {
            // Stores the latest middle points and rotation guess.
            // This value can be used for the next prediction as an initial guess.
            let mut latest_midp = [0.0, 0.0, 0.0];
            let mut latest_rot = None;

            'f: for task in r.iter() {
                match task {
                    // We should predict the head pose
                    ParTask::Midp(img) => {
                        // Using default kinect intrinsic
                        let intrinsic = IntrinsicMatrix::default_kinect_intrinsic();

                        // If we want to ignore the previous pose estimation and look
                        // for the maximum value for an initial guess.
                        let res = if maxguess {
                            forest.predict_parameter_parallel(img, &intrinsic, None, None)
                            //forest.predict_parameter(img, &intrinsic, None, None)
                        } else {
                            // The previous estimated point should have a z-distance from more than
                            // 500mm to be a valid one.
                            let latest_midp_option = if latest_midp[2] > 500.0 {
                                Some(latest_midp)
                            } else {
                                None
                            };
                            forest.predict_parameter_parallel(img, &intrinsic, latest_midp_option, latest_rot)
                            //forest.predict_parameter(img, &intrinsic, latest_midp_option, latest_rot)
                        };

                        let mid3 = res.mid_point;

                        // If sluggish is set, then update only if not too far away from
                        // the previous guess
                        if !sluggish ||
                           (mid3[0] - latest_midp[0]).abs() + (mid3[1] - latest_midp[1]).abs() +
                           (mid3[2] - latest_midp[2]).abs() < 100.0 ||
                           latest_midp[2] < 500.0 {
                            latest_midp = mid3;
                        }
                        let rot = res.rotation;
                        latest_rot = Some(rot);
                        info!("Predicted: Midpoint {:?} - Rotation {:?}", mid3, rot);
                        let rot = [rot[0] as f32, rot[1] as f32, rot[2] as f32];
                        let mid3 = [mid3[0] as f32, mid3[1] as f32, mid3[2] as f32];
                        *midp_rot.lock().unwrap() = Some((mid3, rot));
                    }
                    // Predict a mask
                    ParTask::Mask(img) => {
                        let res = forest.predict_mask(img.clone());
                        *mask.lock().unwrap() = Some((img, res));
                    }
                    // Cancel thread
                    ParTask::Break => break 'f,
                }
            }
        });

        ParPredict {
            midp_rot: midp_rot2,
            mask: mask2,
            sender: s,
            join: Some(join),
        }
    }

    fn predict_midp(&self, img: Arc<DepthImage>) -> Option<()> {
        match self.sender.try_send((ParTask::Midp(img))) {
            Err(sync::mpsc::TrySendError::Full(_)) => None,
            Err(x) => panic!(x),
            Ok(x) => Some(x),
        }
    }

    fn predict_mask(&self, img: Arc<DepthImage>) -> Option<()> {
        match self.sender.try_send((ParTask::Mask(img))) {
            Err(sync::mpsc::TrySendError::Full(_)) => None,
            Err(x) => panic!(x),
            Ok(x) => Some(x),
        }
    }

    fn midp_rot(&self) -> Option<([f32; 3], [f32; 3])> {
        self.midp_rot.lock().unwrap().take()
    }

    fn mask(&self) -> Option<(Arc<DepthImage>, image::ImageBuffer<image::Luma<u8>, Vec<u8>>)> {
        self.mask.lock().unwrap().take()
    }

    fn wait(mut self) -> thread::Result<()> {
        self.sender.send(ParTask::Break).unwrap();
        self.join.take().unwrap().join()
    }
}

impl Drop for ParPredict {
    fn drop(&mut self) {
        if let Some(joiner) = self.join.take() {
            self.sender.send(ParTask::Break).unwrap();
            joiner.join().unwrap()
        }
    }
}


fn load_tree(path: &str) -> Result<HoughPrediction> {
    use std::fs::File;
    use std::io::Read;
    let mut json = String::new();
    let mut jsonfile = File::open(path)?;
    jsonfile.read_to_string(&mut json)?;
    Ok(serde_json::from_str(&json)?)
}


pub fn main(){
    use std::io::{stderr, Write};
    use std::process::exit;
    if let Err(e) = main_(){
        let mut out = stderr();
        let mut out = out.lock();
        out.write_fmt(format_args!("Error: {}", e)).unwrap();
        exit(-1);
    }
}

pub fn main_() -> Result<()> {
    use std::u16;
    use std::str::FromStr;
    env_logger::init();
    let args = App::new("Live Prediction")
        .arg(Arg::with_name("trained")
            .short("t")
            .long("trained")
            .required(true)
            .takes_value(true)
            .help("Filename of learned tree"))
        .arg(Arg::with_name("threshold")
            .short("s")
            .long("threshold")
            .takes_value(true)
            .help("Threshold for max depth"))
        .arg(Arg::with_name("Mask").long("mask").help("Show only predicted mask"))
        .arg(Arg::with_name("sluggish")
            .long("sluggish")
            .help("Update the prediction result only if not too far away from previous prediction"))
        .arg(Arg::with_name("prevguess")
            .long("prevguess")
            .help("Guess mean shift position with help of an previous guess instead of using the maximum value"))
        .arg(Arg::with_name("motor")
             .long("with-motor")
            .help("Initialize Kinect with support for its motor. You can use the up and down key then to move the kinect."))
        .get_matches();


    let intrinsic = IntrinsicMatrix::default_kinect_intrinsic();
    let threshold =
        args.value_of("threshold").and_then(|x| u16::from_str(*&x).ok()).unwrap_or(1300);
    let forest_path = args.value_of("trained").ok_or("Invalid parameter for learned forest")?;
    let forest = load_tree(&*forest_path)?;

    let ctx = match args.is_present("motor"){
        true => freenect::FreenectContext::init_with_video_motor()?,
        false => freenect::FreenectContext::init_with_video()?
    };
    let device = ctx.open_device(0)?;

    device.set_depth_mode(freenect::FreenectResolution::Medium,
                                freenect::FreenectDepthFormat::MM)?;
    device.set_video_mode(freenect::FreenectResolution::Medium,
                                freenect::FreenectVideoFormat::Rgb)?;

    let dstream = device.depth_stream()?;
    let vstream = device.video_stream()?;

    let mut vwin = headwin::HeadWin::new("Live RGB", (IntrinsicMatrix::default_kinect_intrinsic().0).0);
    let mut vimg = image::RgbaImage::new(640, 480);
    let mut vimg_new = None;//image::RgbaImage::new(640,480);

    let mut dwin = imgwin::ImgWindow::new("Live Depth");
    let mut dimg = image::RgbaImage::new(640, 480);

    ctx.spawn_process_thread().unwrap();
    let sluggish = args.is_present("sluggish");
    let maxguess = !args.is_present("prevguess");
    let predictor = ParPredict::init(forest, sluggish, maxguess);

    let make_mask = args.is_present("Mask");
    let mut depth_img = Arc::new(DepthImage::new(640, 480));

    let mut inphandler = InputHandler {
        device: &device,
        is_closed: false,
    };
    loop {
        let _ = imgwin::FixWaitTimer::new(Duration::from_millis(1000 / 25));

        if let Ok((data, _ /* timestamp */)) = dstream.receiver.try_recv() {
            depth_img = Arc::new(DepthImage::from_fn(640, 480, |x, y| {
                let idx = y * 640 + x;
                let val = data[idx as usize];
                let val = if val > threshold { 0 } else { val };
                image::Luma([val])
            }));
            if make_mask {
                let _ = predictor.predict_mask(depth_img.clone());
                if let Some((depth, mask)) = predictor.mask() {
                    dimg = image::RgbaImage::from_fn(640, 480, |x, y| {
                        let val = depth[(x, y)].data[0];
                        let val = if val > 255 { 255u8 } else { val as u8 };
                        let prob = mask[(x, y)].data[0];
                        let color = val - ((val as u16) * (prob as u16) / 255u16) as u8;
                        image::Rgba([val, color, color, 255])
                    });
                }
            } else {
                dimg = image::RgbaImage::from_fn(640, 480, |x, y| {
                    let val = depth_img[(x, y)].data[0];
                    let val = if val > 600 { val - 600 } else { 0 };
                    let val = val / 2;
                    let val = if val > 255 { 255u8 } else { val as u8 };
                    image::Rgba([val, val, val, 255])
                });

            }
        }

        if let Ok((data, _ /* timestamp */)) = vstream.receiver.try_recv() {
            if make_mask || predictor.predict_midp(depth_img.clone()).is_some() {
                vimg = vimg_new.take().unwrap_or(vimg);
                vimg_new = Some(image::RgbaImage::from_fn(640, 480, |x, y| {
                    let idx = 3 * (y * 640 + x) as usize;
                    let (r, g, b) = (data[idx], data[idx + 1], data[idx + 2]);
                    image::Rgba([r, g, b, 255])
                }));
            }
            let r = 10;
            if let Some((midp, rot)) = predictor.midp_rot() {
                vwin.update_transformation(rot.into(), midp.into(), 2.0);
                let mid2 = intrinsic.space_to_img_coord(midp);
                let (midx, midy) = (mid2.0[0] as u32, mid2.0[1] as u32);
                for y in 0..r {
                    for x in 0..r {
                        let x = x + midx;
                        let y = y + midy;
                        if x < r / 2 || y < r / 2 {
                            continue;
                        }
                        let x = x - r / 2;
                        let y = y - r / 2;
                        if x >= vimg.width() || y >= vimg.height() {
                            continue;
                        }
                        vimg[(x, y)] = image::Rgba([0, 0, 255, 255]);
                    }
                }
            }
        }
        vwin.update_image(vimg.clone());
        dwin.set_img(dimg.clone());
        vwin.redraw();
        dwin.redraw();
        dwin.check_for_event(&mut inphandler);
        vwin.check_for_event(&mut inphandler);
        if inphandler.is_closed {
            break;
        }
    }

    predictor.wait().unwrap();
    ctx.stop_process_thread().unwrap();
    Ok(())
}

struct InputHandler<'a, 'b: 'a> {
    device: &'a freenect::FreenectDevice<'a, 'b>,
    is_closed: bool,
}


macro_rules! ptry {
    ($x: expr) => (
        match $x{
            Ok(x) => x,
            Err(r) => {error!("Error: {}", r); return }
        }
        )
}

impl<'a, 'b> imgwin::EventHandler for InputHandler<'a, 'b> {
    fn close_event(&mut self) {
        self.is_closed = true;
    }
    fn key_event(&mut self, inp: Option<VirtualKeyCode>) {
        if let Some(code) = inp {
            match code {
                VirtualKeyCode::Up => {
                    ptry!(self.device.set_tilt_degree(ptry!(self.device.get_tilt_degree()) + 10.0))
                }
                VirtualKeyCode::Down => {
                    ptry!(self.device.set_tilt_degree(ptry!(self.device.get_tilt_degree()) - 10.0))
                }

                VirtualKeyCode::Q => self.is_closed = true,
                _ => (),
            }
        }
    }
}
