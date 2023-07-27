use four_bar::{efd::na, plot2d::*, *};
use std::f64::consts::PI;

fn fft_recon<C>(path: C, harmonic: usize, pt: usize) -> Vec<[f64; 2]>
where
    C: efd::Curve<[f64; 2]>,
{
    use rustfft::{num_complex::Complex, num_traits::Zero as _};
    let mut data = path
        .as_curve()
        .iter()
        .map(|&[re, im]| Complex { re, im })
        .collect::<Vec<_>>();
    let len = data.len();
    assert!(pt >= len);
    let mut plan = rustfft::FftPlanner::new();
    plan.plan_fft_forward(len).process(&mut data);
    // Remove high frequency
    let n1 = harmonic / 2;
    if harmonic != len {
        let n2 = n1 + harmonic % 2;
        data.iter_mut()
            .take(len - n1)
            .skip(n2)
            .for_each(|c| c.set_zero());
    }
    // Change point number
    data.splice(n1..n1, std::iter::repeat(Complex::zero()).take(pt - len));
    plan.plan_fft_inverse(pt).process(&mut data);
    data.into_iter()
        .map(|c| c / len as f64)
        .map(|Complex { re, im }| [re, im])
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path examples:
    // ./5bar.csv
    // ./curvature.csv
    // ../four-bar-rs/syn-examples/crunode.closed.ron
    // ../four-bar-rs/syn-examples/cusp.closed.ron
    // ../four-bar-rs/syn-examples/heart.closed.ron
    // ../four-bar-rs/syn-examples/bow.open.ron
    // ../four-bar-rs/syn-examples/slice.open.csv
    // ../four-bar-rs/syn-examples/waterdrop.open.ron
    let path = {
        let Some(path) = std::env::args().nth(1) else {
            panic!("Please input path");
        };
        match std::path::Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
        {
            Some("csv") => csv::parse_csv(std::fs::File::open(path)?)?,
            Some("ron") => {
                let fb = ron::de::from_reader::<_, FourBar>(std::fs::File::open(path)?)?;
                fb.curve(180)
            }
            _ => panic!("Unsupported file type"),
        }
    };
    let is_open = false;
    let pt = 90;
    let efd_time = std::time::Instant::now();
    let efd = efd::Efd2::from_curve_harmonic(&path, is_open, 10);
    let harmonic = efd.harmonic();
    dbg!(harmonic, efd_time.elapsed());
    let p_efd = if is_open {
        efd.generate_half(pt)
    } else {
        efd.generate(pt)
    };
    println!("efd-err = {}", efd::curve_diff(&path, &p_efd));

    let fd_time = std::time::Instant::now();
    let p_fd = if is_open {
        let fd_path = path
            .iter()
            .chain(path.iter().rev().skip(1))
            .copied()
            .collect::<Vec<_>>();
        fft_recon(fd_path, harmonic * 2, pt)
    } else {
        fft_recon(&path, harmonic * 2, pt)
    };
    dbg!(fd_time.elapsed());
    println!("fd-err = {}", efd::curve_diff(&path, &p_fd));

    let harmonic = 7;
    let p_efd_fit = {
        let efd_fitting_time = std::time::Instant::now();
        let theta =
            na::RowDVector::from_fn(path.len(), |_, i| i as f64 / (path.len() - 1) as f64 * PI);
        let ax = na::MatrixXx2::from_row_iterator(path.len(), path.iter().flatten().copied());
        let mut a = na::DMatrix::zeros(2 * harmonic, path.len());
        for r in 0..harmonic {
            let n = (r + 1) as f64;
            a.row_mut(r).copy_from(&(n * &theta).map(f64::cos));
            a.row_mut(r + harmonic)
                .copy_from(&(n * &theta).map(f64::sin));
        }
        let a2 = a.transpose();
        let omega = &a * a2;
        let y = a * ax;
        let x = omega.lu().solve(&y).unwrap().transpose();
        let coeffs = efd::Coeff2::from_rows(&[
            x.row(0).columns(0, harmonic),
            x.row(1).columns(0, harmonic),
            x.row(0).columns(harmonic, harmonic),
            x.row(1).columns(harmonic, harmonic),
        ]);
        let efd2 = efd::Efd2::try_from_coeffs_unnorm(coeffs).unwrap();
        dbg!(efd_fitting_time.elapsed());
        efd2.generate_half(pt)
    };
    println!("efd-fit-err = {}", efd::curve_diff(&path, &p_efd_fit));

    let harmonic = 1;
    let p_fd_fit = {
        let p = harmonic as isize;
        let harmonic = p as usize * 2 + 1;
        let fd_fitting_time = std::time::Instant::now();
        let z =
            na::RowDVector::from_fn(path.len(), |_, i| na::Complex::new(path[i][0], path[i][1]));
        let theta = na::RowDVector::from_fn(path.len(), |_, i| {
            na::Complex::from(i as f64 / (path.len() - 1) as f64 * PI)
        });
        let omega = na::DMatrix::from_fn(harmonic, harmonic, |m, k| {
            (&theta * na::Complex::from(k as f64 - m as f64) * na::Complex::i())
                .map(na::Complex::exp)
                .sum()
        });
        let y = na::DVector::from_fn(harmonic, |m, _| {
            let ey = (&theta * ((m as isize - p) as f64 * -na::Complex::i())).map(na::Complex::exp);
            (&z * ey.transpose())[0]
        });
        let x = omega.lu().solve(&y).unwrap();
        dbg!(fd_fitting_time.elapsed());
        let theta = na::RowDVector::from_fn(pt, |_, i| {
            na::Complex::from(i as f64 / (pt - 1) as f64 * PI)
        });
        let ec = {
            let p =
                na::DVector::from_fn(harmonic, |i, _| na::Complex::from((i as isize - p) as f64));
            (p * theta * na::Complex::i()).map(na::Complex::exp)
        };
        (x.transpose() * ec)
            .column_iter()
            .map(|c| [c[0].re, c[0].im])
            .collect::<Vec<_>>()
    };
    println!("fd-fit-err = {}", efd::curve_diff(&path, &p_fd_fit));

    // Plot
    let b = SVGBackend::new("test.svg", (1600, 800));
    let (root_l, root_r) = b.into_drawing_area().split_horizontally(800);
    let fig = Figure::new()
        .grid(false)
        .font(45.)
        .legend(LegendPos::LR)
        .add_line("Target", path, Style::Circle, RED);
    fig.clone()
        .add_line("EFD", p_efd, Style::Line, BLUE)
        .plot(root_l)?;
    fig.clone()
        .add_line("FD", p_fd, Style::Line, BLACK)
        .plot(root_r)?;
    Ok(())
}
