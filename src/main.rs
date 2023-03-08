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
        .map(|Complex { re, im }| [re / len as f64, im / len as f64])
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fb = ron::from_str::<FourBar>(&std::fs::read_to_string(
        "../four-bar-rs/syn-examples/crunode.closed.ron",
    )?)?;
    let path = fb.curve(360);
    let efd = efd::Efd2::from_curve_harmonic(curve::closed_lin(&path), None).unwrap();
    let harmonic = efd.harmonic();
    let fft_recon = fft_recon(&path, harmonic);
    let path_recon = efd.generate(180);
    let b = plot2d::SVGBackend::new("test.svg", (800, 800));
    plot2d::plot(
        b,
        [
            ("Original", path.as_slice()),
            (
                &format!("EFD Reconstructed ({harmonic} harmonics)"),
                path_recon.as_slice(),
            ),
            (
                &format!("FD Reconstructed ({harmonic} harmonics)"),
                fft_recon.as_slice(),
            ),
        ],
        plot2d::Opt::from(None)
            .grid(false)
            .axis(false)
            .dot(true)
            .stroke(4),
    )?;
    Ok(())
}
