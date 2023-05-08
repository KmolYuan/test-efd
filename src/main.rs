use four_bar::{efd, efd::na, plot2d, plot2d::IntoDrawingArea, FourBar};
use std::f64::consts::PI;

fn fft_recon(path: &[[f64; 2]], harmonic: usize) -> Vec<[f64; 2]> {
    use rustfft::{num_complex::Complex, num_traits::Zero as _};

    let mut data = path
        .iter()
        .map(|&[re, im]| Complex { re, im })
        .collect::<Vec<_>>();
    let len = data.len();
    let mut plan = rustfft::FftPlanner::new();
    plan.plan_fft_forward(len).process(&mut data);
    let n1 = harmonic / 2;
    let n2 = n1 + harmonic % 2;
    data.iter_mut()
        .take(len - n1)
        .skip(n2)
        .for_each(|c| c.set_zero());
    plan.plan_fft_inverse(len).process(&mut data);
    data.into_iter()
        .map(|c| c / len as f64)
        .map(|Complex { re, im }| [re, im])
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const PATH: &[&str] = &[
        "../four-bar-rs/syn-examples/crunode.closed.ron",
        "../four-bar-rs/syn-examples/cusp.closed.ron",
        "../four-bar-rs/syn-examples/heart.closed.ron",
        "../four-bar-rs/syn-examples/c-shape.open.ron",
        "../four-bar-rs/syn-examples/sharp.open.ron",
    ];
    let fb = ron::from_str::<FourBar>(&std::fs::read_to_string(PATH[4])?)?;
    let path = fb.curve(360);

    // Drect method
    let harmonic = 19;
    // let harmonic = path.len() / 2;
    let efd_time = std::time::Instant::now();
    let efd = efd::Efd2::from_curve_harmonic(&path, true, harmonic);
    dbg!(efd_time.elapsed());

    let fd_time = std::time::Instant::now();
    let fft_recon = fft_recon(&path, harmonic * 2);
    dbg!(fd_time.elapsed());
    let path_recon = efd.generate(180);

    let efd_fitting_recon = {
        let mut path = efd.generate_norm_in(path.len(), PI);
        let rot = efd::Transform2::new([0.; 2], na::UnitComplex::new(10f64.to_radians()), 1.);
        rot.transform_inplace(&mut path);
        let efd_fitting_time = std::time::Instant::now();
        let theta = na::RowDVector::from_fn(path.len(), |_, i| i as f64 / path.len() as f64 * PI);
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
            x.row(0).columns(harmonic, harmonic),
            x.row(1).columns(0, harmonic),
            x.row(1).columns(harmonic, harmonic),
        ]);
        let efd2 = efd::Efd2::try_from_coeffs_unnorm(coeffs).unwrap();
        dbg!(efd_fitting_time.elapsed());
        dbg!(efd2.coeffs()[(0, 0)].hypot(efd2.coeffs()[(2, 0)]));
        efd2.generate_half(360)
    };

    let fd_fitting_recon = {
        let p = harmonic as isize;
        let fd_fitting_time = std::time::Instant::now();
        let z =
            na::RowDVector::from_fn(path.len(), |_, i| na::Complex::new(path[i][0], path[i][1]));
        let theta = na::RowDVector::from_fn(path.len(), |_, i| {
            na::Complex::from(i as f64 / path.len() as f64 * PI)
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
        let n = 360;
        let theta = na::RowDVector::from_fn(n, |_, i| na::Complex::from(i as f64 / n as f64 * PI));
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
    let b = plot2d::SVGBackend::new("test.svg", (800 * 3, 800));
    let mut roots = b.into_drawing_area().split_evenly((1, 3));
    let [root1, root2, root3] = [roots.remove(0), roots.remove(0), roots.remove(0)];
    plot2d::plot(
        root1,
        [("", path.as_slice())],
        plot2d::Opt::from(&fb)
            .grid(false)
            .axis(false)
            .scale_bar(true),
    )?;
    let opt = plot2d::Opt::new()
        .grid(false)
        .font(20.)
        .dot(true)
        .legend(plot2d::LegendPos::MM);
    plot2d::plot(
        root2,
        [
            ("Original", path.as_slice()),
            ("EFD Reconstructed", path_recon.as_slice()),
            ("EFD Fitting Reconstructed", efd_fitting_recon.as_slice()),
        ],
        opt.clone(),
    )?;
    plot2d::plot(
        root3,
        [
            ("Original", path.as_slice()),
            ("FD Reconstructed", fft_recon.as_slice()),
            ("FD Fitting Reconstructed", fd_fitting_recon.as_slice()),
        ],
        opt,
    )?;
    Ok(())
}
