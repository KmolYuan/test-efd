use four_bar::{efd::na, plot::*, *};
use std::f64::consts::PI;

fn fft_recon<C>(path: C, pt: usize) -> Vec<[f64; 2]>
where
    C: efd::Curve<2>,
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
    // Fourier analysis
    let mut lut = data.iter().map(|c| c.norm_sqr()).collect::<Vec<_>>();
    lut.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    // cumsum
    lut.iter_mut().reduce(|prev, next| {
        *next += *prev;
        next
    });
    let total_power = lut[lut.len() - 1];
    lut.iter_mut().for_each(|x| *x /= total_power);
    let harmonic = match lut.binary_search_by(|x| x.partial_cmp(&0.99999).unwrap()) {
        Ok(h) | Err(h) => h + 1,
    };
    println!("FD harmonic = {harmonic}");
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
    // ../four-bar-rs/test-fb/crunode.closed.ron
    // ../four-bar-rs/test-fb/cusp.closed.ron
    // ../four-bar-rs/test-fb/heart.closed.ron
    // ../four-bar-rs/test-fb/bow.open.ron
    // ../four-bar-rs/test-fb/slice.open.csv
    // ../four-bar-rs/test-fb/waterdrop.open.ron
    let (path, is_open) = {
        let Some(path) = std::env::args().nth(1) else {
            panic!("Please input path");
        };
        let path = std::path::Path::new(&path);
        let mode = std::path::Path::new(path.file_stem().unwrap())
            .extension()
            .unwrap()
            .to_str()
            .unwrap();
        let is_open = mode != "closed";
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .expect("Unsupported file type");
        let file = std::fs::File::open(path)?;
        let path = match ext {
            "csv" => csv::from_reader(file)?,
            "ron" => ron::de::from_reader::<_, FourBar>(file)?.curve(180),
            _ => panic!("Unsupported file type"),
        };
        (path, is_open)
    };
    let pt = if is_open {
        path.len() * 2
    } else {
        path.len() * 4
    };
    let efd_time = std::time::Instant::now();
    let efd = efd::Efd2::from_curve(&path, is_open);
    let harmonic = efd.harmonic();
    println!("EFD harmonic = {harmonic}");
    println!("EFD time spent = {:?}", efd_time.elapsed());
    let _p_efd = efd.recon(pt);

    let fd_time = std::time::Instant::now();
    let p_fd = if is_open {
        let fd_path = path
            .iter()
            .chain(path.iter().rev().skip(1))
            .copied()
            .collect::<Vec<_>>();
        fft_recon(fd_path, pt)
    } else {
        fft_recon(&path, pt)
    };
    println!("FD time spent = {:?}", fd_time.elapsed());

    let harmonic = 5;
    let _p_efd_fit = {
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
        let coeffs = std::iter::zip(
            &x.row(0).columns(0, harmonic),
            std::iter::zip(
                &x.row(1).columns(0, harmonic),
                std::iter::zip(
                    &x.row(0).columns(harmonic, harmonic),
                    &x.row(1).columns(harmonic, harmonic),
                ),
            ),
        )
        .map(|(a, (b, (c, d)))| na::SMatrix::from([[*a, *b], [*c, *d]]))
        .collect();
        let efd2 = efd::Efd2::from_coeffs_unchecked(coeffs);
        println!("EFD fitting time spent = {:?}", efd_fitting_time.elapsed());
        efd2.recon(pt)
    };

    let harmonic = 5;
    let _p_fd_fit = {
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
        println!("FD fitting time spent = {:?}", fd_fitting_time.elapsed());
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

    // Plot
    let b = SVGBackend::new("test-5bar.svg", (800, 800));
    // let (root_l, root_r) = b.into_drawing_area().split_horizontally(800);
    let fig = fb::Figure::new(None)
        .grid(false)
        .font(45.)
        .legend(LegendPos::LL)
        .add_line("Target", path, Style::Circle, RED);
    fig.clone()
        .add_line("FD", p_fd, Style::Line, BLUE)
        .plot(b)?;
    // fig.clone()
    //     .add_line("S-TPCF", p_efd_fit, Style::DottedLine, BLUE)
    //     .plot(root_r)?;
    Ok(())
}
