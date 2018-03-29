/// Program which show the data of the biwi database

extern crate depthhead;
extern crate depthhead_example_utils;
extern crate image;
#[macro_use]
extern crate error_chain;
use depthhead::db_reader as db;
use depthhead::db_reader::reader::DepthReader;
use std::env;

use depthhead_example_utils::imgwin::{ImgWindow, FixWaitTimer};
use std::time::Duration;

error_chain!{
    foreign_links {
        Biwi(db::biwi::BiwiReadError);
    }
}

macro_rules! eprintln {
    ($e: expr, $($args: expr)*) => {
        use std::io::{stderr, Write};
        let mut out = stderr();
        let mut out = out.lock();
        out.write_fmt(format_args!($e, $($args,)*)).unwrap();
    };
}

pub fn main(){
    use std::process::exit;
    if let Err(e) = main_(){
        eprintln!("Error: {}",e);
        exit(-1);
    }
}

pub fn main_()  -> Result<()>{
    use std::str::FromStr;
    let mut args = env::args();
    args.next().unwrap();
    let rootpath = match args.next() {
        Some(x) => x,
        _ => {
            bail!("Not enough args: needed root-path to db, may number of person");
        }
    };
    let person_nr = args.next().and_then(|x| usize::from_str(&*x).ok()).unwrap_or(1);


    // Read the data
    let reader = db::biwi::BiwiReader::new(format!("{}/head_pose_masks", rootpath),
                                           format!("{}/hpdb", rootpath),
                                           format!("{}/db_annotations", rootpath));
    if reader.is_valid()? {
        println!("Valid data");
    } else {
        bail!("Invalid data");
    }

    let person_count = reader.person_count()?;
    println!("Found {} persons", person_count);

    if person_count == 0 {
        bail!("No person found");
    }
    // Data for the specific person
    let mut it = reader.person(person_nr)?;

    // use opengl for viewing
    let mut mask_win = ImgWindow::new("Mask".to_string());
    let mut depth_win = ImgWindow::new("Depth".to_string());

    let mut next_truth = || {
        if let Some(data) = it.next() {
            let data = match data{
                Ok(x) => x,
                Err(e) => {eprintln!("Error: {}", e); return None}
            };
            let trans = data.trans;
            let depth_img = data.depth;
            let mask_img = data.mask;
            let mask_img = image::RgbaImage::from_fn(mask_img.width(), mask_img.height(), |x, y| {
                let val = mask_img[(x, y)];
                let dist_to_center = (trans.flat_x() - x as f32).powi(2) +
                                     (trans.flat_y() - y as f32).powi(2);
                let dist_to_center = dist_to_center.sqrt();
                if val.data[0] == 0 {
                    image::Rgba([0u8, 0, 0, 255])
                } else {
                    image::Rgba([255u8, dist_to_center as u8, dist_to_center as u8, 255])
                }
            });
            // Convert depth image to rgb for viewing
            let depth_img =
                image::RgbaImage::from_fn(depth_img.width(), depth_img.height(), |x, y| {
                    let val = depth_img[(x, y)];
                    let val = val.data[0];
                    let val8 = if val > 600 { val - 600 } else { 0 };
                    let val8 = val8 / 2;
                    let val8 = if val8 > 255 { 255 } else { val8 };
                    let val8 = val8 as u8;
                    let dist_to_center = (trans.flat_x() - x as f32).powi(2) +
                                         (trans.flat_y() - y as f32).powi(2);
                    let dist_to_center = dist_to_center.sqrt();
                    image::Rgba([val8,
                                 if val > 255 { val8 } else { 0 },
                                 if dist_to_center > 3.0 { val8 } else { 50 },
                                 255])
                });
            Some((depth_img, mask_img))
        } else {
            None
        }
    };

    mask_win.redraw();
    depth_win.redraw();

    // Main-Loop
    loop {
        let _ = FixWaitTimer::new(Duration::from_millis(1000 / 25));
        if mask_win.check_for_close() {
            break;
        }
        if depth_win.check_for_close() {
            break;
        }

        if let Some((depth_img, mask_img)) = next_truth() {
            mask_win.set_img(mask_img);
            depth_win.set_img(depth_img);
        } else {
            break;
        }

        mask_win.redraw();
        depth_win.redraw();
    }
    Ok(())

}
