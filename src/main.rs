use four_bar::*;

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
    let fb = ron::from_str::<FourBar>(&std::fs::read_to_string(PATH[2])?)?;
    // let mut fb = FourBar::example();
    // *fb.a_mut() = 10f64.to_radians();
    // *fb.inv_mut() = true;
    let path = fb.curve(360);
    let path_closed = curve::closed_lin(&path);
    let efd = efd::Efd2::from_curve_harmonic(&path_closed, None).unwrap();
    let harmonic = efd.harmonic();
    let harmonic2 = harmonic * 2;
    let fft_recon = fft_recon(&path_closed[..path_closed.len() - 1], harmonic2);
    let path_recon = efd.generate(180);
    let b = plot2d::SVGBackend::new("test.svg", (800, 800));
    plot2d::plot(
        b,
        [
            ("Original", path.as_slice()),
            ("EFD Reconstructed", path_recon.as_slice()),
            ("FD Reconstructed", fft_recon.as_slice()),
        ],
        plot2d::Opt::from(None)
            .scale_bar(10.)
            .axis(false)
            .grid(false)
            .dot(true)
            .font(35.)
            .stroke(10),
    )?;
    Ok(())
}
