# Depth-Head

This repo implements the depth based head pose regression described in the paper 
"Multiview Facial Landmark Localization in RGB-D
Images via Hierarchical Regression With Binary Patterns" from Zhang et al (see [here](https://zhzhanp.github.io/) for the website).
A hough forest is implemented for that (see "Class-specific Hough forests for object detection" from J. Gall und V. Lempitsky).
The code is written in rust and uses [stamm](https://github.com/Entscheider/stamm) for implementing the forest.

Note: Only head pose regression using a hough forest is implemented. The facial landmark localization as well as the pose regression using a GBDT are missing.

# Screenshot

![](https://raw.githubusercontent.com/Entscheider/depthhead/master/img/screenshot.jpg)

# Training

To run the example applications you can use the pretrained forest (the the releases).
You can also train your own. To do that, first of all training data are necessary.
You can download these files from [the biwi database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html).
More precisely, the following must be downloaded:

- Data
- Binary ground truth files
- Masks used to select positive patches

They should all be extracted into one directory so that they fulfill the following layout: 

```text
|
|- db_annotations
|-- 01
|-- 02
|-- ...
|- head_pose_masks
|-- 01
|-- 02 
|-- ..
|- hpdb
|-- 01
|-- 02
|-- ..
|-- 01.obj
|-- ..
```

# Compiling

To compile the library you should call:

```bash
cargo build --release
```

If you want to start one of the examples, run:

```bash
cargo run --example name_of_the_example --release
```

where `name_of_the_example` is one of the following:

* biwiread: Example which shows the data of the biwi database
* db_evaluate: Example which evaluate a trained forest with test data
* db_prediction: Use a trained forest and test it against the biwi database.
* hough_tree_trainer: Train a forest using the biwi database
* live_prediction: Use a trained forest for a live demonstration. (Requires a kinect)
* show_hough: Show a hough forest result using a single depth image.

Note: You want to use the `--release` flag because the perfomance is horrible otherwise.
You may also want to add `--features reduce_bound_check` to reduce the bound checks for images.

# Trained Forest

A trained forest can be found in the release tab of this repo.

# Code example

Here is a small example for using a trained hough forest.
You should use serde and serde_json for loading.

```rust
// First read the trained forest
let mut jsonfile = File::open(path_to_json)?;
let mut json = String::new();
jsonfile.read_to_string(&mut json)?;
let predictor: HoughPrediction = serde_json::from_str(json.as_str())?;

// We need a intrinsic matrix for prediction. 
// Since we mainly use kinect depth images as source, we want the default intrinsic for that
let intrinsic = IntrinsicMatrix::default_kinect_intrinsic();

// Assuming `img` is a depth image (e.g. from the kinect or the biwi database)
// we can now predict the head pose
let result = predictor.predict_parameter_parallel(img, &intrinsic, None, None);

// `result` gives us the center of the head and it's rotation:
let midpoint_3d: [f32: 3] = result.mid_point;
let rotation: [f64; 3] = result.rotation;
```

For more examples (e.g. train a forest or using the biwi-database) you may want to study the examples directory.

# Issues

If you run a example successfully, you may notice the rotation is not very accurate.
Despite several training iterations and code reviews the problem could not be solved.
If you have a solution, you're welcome to share it.


# License
This code is distributed under the terms of the Apache License, Version 2.0.