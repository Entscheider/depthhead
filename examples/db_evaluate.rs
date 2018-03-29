/// This program generates evaluation files to get the performance

extern crate depthhead;

extern crate image;
extern crate clap;
extern crate regex;
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate error_chain;

extern crate serde_json;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use depthhead::hough::prediction::HoughPrediction;
use depthhead::db_reader::biwi::{BiwiReader, BiwiReadError, BResult};
use depthhead::db_reader::reader::{DepthTrue, DepthReader};

use clap::{Arg, App};

use std::fs::File;
use std::io::{Write, Read, Result as IOResult};

macro_rules! eprintln {
    ($e: expr, $($args: expr)*) => {
        use std::io::{stderr, Write};
        let mut out = stderr();
        let mut out = out.lock();
        out.write_fmt(format_args!($e, $($args,)*)).unwrap();
    };
}

error_chain!{
    foreign_links {
        Biwi(BiwiReadError);
        IO(std::io::Error);
        Serde(serde_json::Error);
    }
}

pub fn main(){
    use std::process::exit;
    if let Err(e) = main_(){
        eprintln!("Error: {}",e);
        exit(-1);
    }
}

macro_rules! otryr {
    ($x: expr, $res: expr, $($s: expr),*) => {
        match $x{
            Some(x) => x,
            None => {warn!($($s,)*); return $res}
        }
    }
}


fn main_() -> Result<()> {
    env_logger::init();
    let args = App::new("db_evaluator")
        .arg(Arg::with_name("trained")
            .short("t")
            .long("trained")
            .required(true)
            .takes_value(true)
            .help("Filename of learned tree"))
        .arg(Arg::with_name("db")
            .short("d")
            .long("data")
            .required(true)
            .takes_value(true)
            .help("Path to the database"))
        .arg(Arg::with_name("out")
            .short("o")
            .long("output")
            .required(true)
            .takes_value(true)
            .help("Directory to save the evaluation result)"))
        .arg(Arg::with_name("persons")
            .short("n")
            .long("person_nrs")
            .required(true)
            .takes_value(true)
            .help("Number of the persons to evaluate, e.g. 1-3,5,7"))
        .arg(Arg::with_name("2d")
            .short("2")
            .long("2d")
            .help("Use 2D-Hough Image for prediction"))
        .get_matches();

    let rootpath = args.value_of("db").ok_or("Invalid parameter for data path")?;
    let learned_tree = args.value_of("trained").ok_or("Invalid parameter for learned forest")?;
    let out_dir = args.value_of("out").ok_or("Invalid parameter for output filename")?;
    let out_dir = std::path::PathBuf::from(out_dir);
    if !out_dir.exists() {
        std::fs::create_dir_all(&out_dir)?;
    } else if out_dir.is_file() {
        bail!("Expected {:?} to be a directory or not to exists, found file",
               out_dir);
    }
    let persons = args.value_of("persons").and_then(|x| parse_nrs(x)).ok_or("Invalid argument for number of persons")?;
    let from2d = args.is_present("2d");

    let reader = BiwiReader::new(format!("{}/head_pose_masks", rootpath),
                                 format!("{}/hpdb", rootpath),
                                 format!("{}/db_annotations", rootpath));
    if !reader.is_valid()?{
        bail!("Invalid Data");
    }

    let personcount = reader.person_count()?;
    info!("Found {} persons", personcount);

    for &j in persons.iter() {
        if j > personcount {
            bail!("Person nr {} does not exit", j);
        } else if j == 0 {
            bail!("Person nr has to start with 1 not with 0");
        }
    }

    let tree = load_tree(learned_tree)?;


    let mut res = EvaluationResult::new(persons,
                                        // Versuchen absoluten String zu bekommen als Metainfo der Evaulation
                                        std::fs::canonicalize(learned_tree)
                                            .ok()
                                            .and_then(|x| x.to_str().map(|x| x.to_string()))
                                            .unwrap_or(learned_tree.to_string()));
    info!("Start evaluation");
    res.evaluate(&reader, &tree, from2d)?;
    info!("Finish evaluation sucessfull");

    info!("Saving result to {:?}", out_dir);


    let json = serde_json::to_string(&res)?;
    let out_eval_filename = out_dir.join("evaluation.json");

    match save_text_file(out_eval_filename, json.as_bytes()) {
        Ok(()) => {
           info!("Evaluation saved sucessfully");
           info!("Create html-files");
           macro_rules! extract {
               ($name: expr) => {
                try!(save_text_file(out_dir.join($name),include_bytes!(concat!("eval_files/",$name))));
               }
           }
           extract!("overview.html");
           extract!("milligram.min.css");
           extract!("milligram.min.css.map");
           extract!("plotly-latest.min.js");
           extract!("vue.min.js");
       }
        Err(r) => {
            error!("Error: {}", r);
            info!("Error occurs while saving, print result to stdout instead");
            println!("{}", json);
        }
    }
    Ok(())
}

pub fn save_text_file<P: AsRef<std::path::Path>>(filename: P, data: &[u8]) -> IOResult<()> {
    let mut file = File::create(filename)?;
    file.write_all(data)?;
    Ok(())
}
fn parse_nrs(s: &str) -> Option<Vec<usize>> {
    use std::str::FromStr;
    lazy_static!{
        static ref RE: regex::Regex = regex::Regex::new(r"(\d+-\d+|\d+)").unwrap();
        static ref LIST: regex::Regex = regex::Regex::new(r"(\d+)-(\d+)").unwrap();
    }
    let mut res: Vec<usize> = vec![];
    for caps in RE.captures_iter(s) {
        let item = &caps[1];
        debug!("Item: {}", item);
        if LIST.is_match(item) {
            let caps = LIST.captures(s).unwrap();
            let beg = otryr!(usize::from_str(&caps[1]).ok(),
                             None,
                             "Not a number {}",
                             &caps[1]);
            let end = otryr!(usize::from_str(&caps[2]).ok(),
                             None,
                             "Not a number {}",
                             &caps[2]);
            if beg > end {
                error!("Number {} is bigger than {}", beg, end);
                return None;
            }
            for i in beg..end + 1 {
                res.push(i);
            }
        } else {
            res.push(otryr!(usize::from_str(item).ok(), None, "Not a number {}", item));
        }
    }
    debug!("Parsed numbers: {:?}", res);
    return Some(res);
}

fn load_tree(path: &str) -> Result<HoughPrediction> {
    let mut json = String::new();
    let mut jsonfile = File::open(path)?;
    jsonfile.read_to_string(&mut json)?;
    Ok(serde_json::from_str(&json)?)
}


#[derive(Debug, Serialize, Deserialize)]
struct EvaluationResult {
    // Which person was used for evaluating
    persons: Vec<usize>,
    // Path to the used random forest
    trained_tree_path: String,
    // A vector with the number of the person and the evaulation results
    res: Vec<(usize, EvalEntrie)>,
}

impl EvaluationResult {
    fn new<I>(persons: Vec<usize>, trained_tree_path: I) -> EvaluationResult
        where I: Into<String>
    {
        EvaluationResult {
            persons: persons,
            trained_tree_path: trained_tree_path.into(),
            res: vec![],
        }
    }

    fn evaluate(&mut self,
                reader: &BiwiReader,
                tree: &HoughPrediction,
                from2d: bool)
                -> Result<()> {
        let len = self.persons.len();
        for (i, person) in self.persons.iter().enumerate() {
            println!("Evaluate person {} - {}/{} ({}%)",
                     person,
                     i + 1,
                     len,
                     (i * 100) / len);
            let mut entry = EvalEntrie::new();
            entry.eval(reader.person(*person)?, tree, from2d)?;
            self.res.push((*person, entry));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct EvalEntrie {
    guess_midpoint: Vec<[f32; 3]>,
    guess_rot: Vec<[f32; 3]>,
    truth_midpoint: Vec<[f32; 3]>,
    truth_rot: Vec<[f32; 3]>,
}

impl EvalEntrie {
    fn new() -> EvalEntrie {
        EvalEntrie {
            guess_midpoint: vec![],
            guess_rot: vec![],
            truth_midpoint: vec![],
            truth_rot: vec![],
        }
    }
    fn eval<I>(&mut self, iter: I, tree: &HoughPrediction, from2dhough: bool) -> BResult<()>
        where I: Iterator<Item = BResult<DepthTrue>>
    {
        use std::sync::Arc;
        // radial to degree
        macro_rules! r2d {
                ($x: expr) => (($x * 180.0 / 3.14159) as f32)
            }
        let len = iter.size_hint().1;
        for (i, element) in iter.enumerate() {
            print!("\rEvaluate image {} of {}",
                   i + 1,
                   len.map(|x| format!("{}", x)).unwrap_or("?".to_string()));
            std::io::stdout().flush().unwrap();
            let truth: DepthTrue = try!(element);
            let img = Arc::new(truth.depth);
            let res = if !from2dhough {
                tree.predict_parameter(img, &truth.intrinsic, None, None)
                //tree.predict_parameter_parallel(img, &truth.intrinsic, None, None)
            } else {
                tree.predict_parameter_from2dhough(img, &truth.intrinsic)
            };
            let guess_mid = res.mid_point;
            let guess_rot = res.rotation;
            let guess_rot = [r2d!(guess_rot[0]), r2d!(guess_rot[1]), r2d!(guess_rot[2])];
            let truth_mid = truth.trans.pos3d;
            let truth_rot = truth.trans.rot;
            self.guess_midpoint.push(guess_mid);
            self.guess_rot.push(guess_rot);
            self.truth_midpoint.push(truth_mid);
            self.truth_rot.push(truth_rot);
        }
        println!("");
        Ok(())
    }
}

