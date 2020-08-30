use lazy_static::lazy_static;
use rulinalg::matrix::BaseMatrix;
use wasm_bindgen::prelude::*;

type Matrix = rulinalg::matrix::Matrix<f32>;

macro_rules! include_transmute {
    ($file:expr) => {
        &core::mem::transmute(*include_bytes!($file))
    };
}

const N: usize = 200;
const IMGSZ: usize = 784;

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

trait MyTrait<T> {
    fn map(&self, f: impl FnMut(T) -> T) -> Self;
    fn imax(&self) -> usize;
}

impl MyTrait<f32> for Matrix {
    fn map(&self, f: impl FnMut(f32) -> f32) -> Self {
        Matrix::new(
            self.rows(),
            self.cols(),
            self.iter().cloned().map(f).collect::<Vec<f32>>(),
        )
    }

    fn imax(&self) -> usize {
        let data: &[f32] = self.data();
        let mut imax = 0;
        let mut xmax = data[0];
        for (i, &x) in data.iter().enumerate() {
            if x > xmax {
                xmax = x;
                imax = i;
            }
        }
        imax
    }
}

struct NNet {
    w0: Matrix,
    b0: Matrix,
    w1: Matrix,
    b1: Matrix,
    w2: Matrix,
    b2: Matrix,
}

impl NNet {
    pub fn new() -> Self {
        NNet {
            w0: Matrix::new(IMGSZ, N, &W0_RAW[..]),
            b0: Matrix::new(1, N, &B0_RAW[..]),
            w1: Matrix::new(N, N, &W1_RAW[..]),
            b1: Matrix::new(1, N, &B1_RAW[..]),
            w2: Matrix::new(N, 10, &W2_RAW[..]),
            b2: Matrix::new(1, 10, &B2_RAW[..]),
        }
    }

    pub fn recognize_number(&self, image_data: &[f32]) -> usize {
        let x = Matrix::new(1, IMGSZ, image_data);
        let l0 = (&x * &self.w0 + &self.b0).map(relu);
        let l1 = (&l0 * &self.w1 + &self.b1).map(relu);
        let onehotv = l1 * &self.w2 + &self.b2;
        onehotv.imax()
    }
}

lazy_static! {
    static ref GNNET: NNet = NNet::new();
}

#[wasm_bindgen]
pub fn recognize_number(image_data_base64: &str) -> String {
    let image_data = base64::decode(image_data_base64).unwrap();
    assert_eq!(image_data.len(), 784);
    let image_data: Vec<f32> = image_data.iter().map(|&c| c as f32 / 255.0).collect();
    GNNET.recognize_number(&image_data).to_string()
}
