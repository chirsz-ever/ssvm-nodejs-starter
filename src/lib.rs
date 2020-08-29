use lazy_static::lazy_static;
use wasm_bindgen::prelude::*;

extern crate nalgebra as na;

use na::{U1, U10};
use typenum::{U200, U784};

macro_rules! include_transmute {
    ($file:expr) => {
        &core::mem::transmute(*include_bytes!($file))
    };
}

const N: usize = 200;
const IMGSZ: usize = 784;

type Matrix<C, R> = na::MatrixMN<f32, C, R>;
type UN = U200;
type UIMGSZ = U784;

fn relu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

static W0_RAW: &[f32; IMGSZ * N] = unsafe { include_transmute!("nn_data/w0.bin") };
static B0_RAW: &[f32; 1 * N] = unsafe { include_transmute!("nn_data/b0.bin") };
static W1_RAW: &[f32; N * N] = unsafe { include_transmute!("nn_data/w1.bin") };
static B1_RAW: &[f32; 1 * N] = unsafe { include_transmute!("nn_data/b1.bin") };
static W2_RAW: &[f32; N * 10] = unsafe { include_transmute!("nn_data/w2.bin") };
static B2_RAW: &[f32; 1 * 10] = unsafe { include_transmute!("nn_data/b2.bin") };

struct NNet {
    w0: Matrix<UIMGSZ, UN>,
    b0: Matrix<U1, UN>,
    w1: Matrix<UN, UN>,
    b1: Matrix<U1, UN>,
    w2: Matrix<UN, U10>,
    b2: Matrix<U1, U10>,
}

impl NNet {
    pub fn new() -> Self {
        NNet {
            w0: Matrix::<UIMGSZ, UN>::from_row_slice(W0_RAW),
            b0: Matrix::<U1, UN>::from_row_slice(B0_RAW),
            w1: Matrix::<UN, UN>::from_row_slice(W1_RAW),
            b1: Matrix::<U1, UN>::from_row_slice(B1_RAW),
            w2: Matrix::<UN, U10>::from_row_slice(W2_RAW),
            b2: Matrix::<U1, U10>::from_row_slice(B2_RAW),
        }
    }

    fn recognize_number(&self, image_data: &[f32]) -> usize {
        let x = Matrix::<U1, UIMGSZ>::from_row_slice(image_data);
        let l0 = (x * &self.w0 + &self.b0).map(relu);
        let l1 = (l0 * &self.w1 + &self.b1).map(relu);
        let onehotv = l1 * &self.w2 + &self.b2;
        onehotv.transpose().imax()
    }
}

lazy_static! {
    static ref GNNET: NNet = NNet::new();
}

#[wasm_bindgen]
pub fn recognize_number(image_data: &[u8]) -> i32 {
    let image_data: Vec<f32> = image_data.iter().map(|&c| c as f32 / 255.0).collect();
    GNNET.recognize_number(&image_data) as _
}
